"""
train/4_smolvla_finetune.py

Stage 4: SmolVLA 微调（端到端动作预测）

数据流：
  LeRobotTactileDataset
    ├─ observation.images.realsense_rgb  [T, 3, 480, 640]  → SmolVLM vision encoder（冻结）
    ├─ observation.images.side           [T, 3, 480, 640]  → SmolVLM 额外视角（可选）
    ├─ depth_1ch                         [T, 1, 480, 640]  → 本脚本不用（Stage 3 已对齐）
    ├─ tactile_grid                      [T, 2, 16, 16]    → DualTactileGridEncoder → Projector
    ├─ language_instruction              str                → SmolVLM text encoder（冻结）
    └─ action                            [action_dim]       → MSE 监督目标

模型：
  TactileVLAAdapter（overfit/models.py）
    ├─ SmolVLM（冻结）: 提取 vision + text tokens
    ├─ DualTactileGridEncoder（可加载 Stage 3 权重）: 提取触觉 token
    ├─ TactileMLPProjector: 映射到 VLA hidden_size
    ├─ concat: [vision_tokens, tactile_tokens] → LLM backbone
    └─ ActionHead: mean-pool → Linear → action_pred [B, action_dim]

训练策略：
  - SmolVLM 全冻结
  - tactile_encoder: 可加载 Stage 3 预训练权重，小学习率微调
  - projector + action_head: 正常学习率

运行方式：
  # 本地 dummy VLA（快速调试，不下载大模型）
  python train/4_smolvla_finetune.py \
    --repo_id your_org/your_dataset \
    --sidecar_root /path/to/dataset \
    --use_dummy_vla

  # 真实 SmolVLM
  python train/4_smolvla_finetune.py \
    --repo_id your_org/your_dataset \
    --sidecar_root /path/to/dataset \
    --vla_model_id HuggingFaceTB/SmolVLM-Instruct
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

# 项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataset.lerobot_tactile_dataset import LeRobotTactileDataset
from overfit.models import TactileVLAAdapter


# ============================================================================
# 1. 辅助函数
# ============================================================================

def tensor_to_pil(rgb_tensor: torch.Tensor) -> Image.Image:
    """
    [3, H, W] float [0,1] → PIL.Image
    """
    arr = (rgb_tensor.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    arr = arr.transpose(1, 2, 0)  # CHW → HWC
    return Image.fromarray(arr)


def batch_images_to_pil(
    rgb_batch: torch.Tensor,
    take_frame: int = -1,
) -> list:
    """
    rgb_batch: [B, T, 3, H, W] or [B, 3, H, W]
    取第 take_frame 帧（默认最后一帧），转 PIL list。
    """
    if rgb_batch.dim() == 5:
        rgb_batch = rgb_batch[:, take_frame]  # [B, 3, H, W]
    return [tensor_to_pil(rgb_batch[i]) for i in range(rgb_batch.shape[0])]


def batch_dual_images_to_pil(
    hand_batch: torch.Tensor,
    pano_batch: torch.Tensor,
    take_frame: int = -1,
) -> list:
    """
    hand_batch: [B, T, 3, H, W] or [B, 3, H, W]  (手眼视角)
    pano_batch: [B, T, 3, H, W] or [B, 3, H, W]  (全景视角)

    返回: List[List[PIL.Image]]，每个样本两个视角 [hand_eye, panoramic]
    """
    hand_list = batch_images_to_pil(hand_batch, take_frame=take_frame)
    pano_list = batch_images_to_pil(pano_batch, take_frame=take_frame)
    return [[hand_list[i], pano_list[i]] for i in range(len(hand_list))]


# ============================================================================
# 2. Stage 3 权重加载
# ============================================================================

def load_pretrained_tactile(model: TactileVLAAdapter, ckpt_path: str, device: str):
    """
    加载 Stage 3 预训练的触觉编码器权重到 TactileVLAAdapter.tactile_encoder。

    Stage 3 checkpoint 保存的是 DepthGuidedTLVModel 的 state_dict，
    其中 tactile_encoder.* 前缀对应我们需要的权重。

    DualTactileGridEncoder 内部的 encoder 就是 TactileEncoder，
    键名匹配: encoder.cnn.0.weight ↔ tactile_encoder.cnn.0.weight（加前缀映射）
    """
    if not os.path.exists(ckpt_path):
        print(f"[Stage4] 预训练权重不存在: {ckpt_path}，触觉编码器随机初始化。")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    s3_dict = ckpt.get("model", ckpt)

    # 收集 Stage 3 的 tactile_encoder.* 权重
    tac_weights = {}
    for k, v in s3_dict.items():
        if k.startswith("tactile_encoder."):
            # Stage 3 的 tactile_encoder 就是 TactileEncoder
            # DualTactileGridEncoder 里叫 self.encoder = TactileEncoder(...)
            new_k = k.replace("tactile_encoder.", "encoder.", 1)
            tac_weights[new_k] = v

    if not tac_weights:
        print("[Stage4] checkpoint 中无 tactile_encoder 权重，跳过。")
        return

    msg = model.tactile_encoder.load_state_dict(tac_weights, strict=False)
    print(f"[Stage4] 加载 {len(tac_weights)} 个触觉编码器参数层。")
    if msg.missing_keys:
        print(f"  missing keys（可能是右路 encoder_right）: {msg.missing_keys[:5]}...")


# ============================================================================
# 3. 配置
# ============================================================================

DEFAULT_CFG = """
seed: 42
window_T: 16
fps: 20.0
batch_size: 4
epochs: 50
lr_tactile: 1e-4
lr_adapter: 5e-4
weight_decay: 1e-4
action_dim: 7
n_tactile_tokens: 8
use_dummy_vla: true
amp: false
"""


def parse_args():
    p = argparse.ArgumentParser(description="Stage 4: SmolVLA Tactile Finetune")
    p.add_argument("--repo_id", type=str, required=True,
                   help="LeRobot 数据集 ID")
    p.add_argument("--root", type=str, default=None,
                   help="数据集本地根目录")
    p.add_argument("--sidecar_root", type=str, default=None,
                   help="sidecar 文件根目录（触觉 .npy）")
    p.add_argument("--config", type=str, default="configs/stage4_vla.yaml")
    p.add_argument("--stage3_ckpt", type=str, default="runs/ckpt_stage3_depth_final.pt",
                   help="Stage 3 预训练 checkpoint")
    p.add_argument("--use_dummy_vla", action="store_true", default=False,
                   help="使用 DummyVLA 跳过下载大模型")
    p.add_argument("--vla_model_id", type=str, default="HuggingFaceTB/SmolVLM-Instruct")
    p.add_argument("--has_right_tactile", action="store_true", default=True,
                   help="是否有右触觉传感器")
    return p.parse_args()


def load_cfg(path: str) -> dict:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(DEFAULT_CFG)
    cfg = yaml.safe_load(open(path))
    cfg["seed"]             = int(cfg.get("seed", 42))
    cfg["window_T"]         = int(cfg.get("window_T", 16))
    cfg["fps"]              = float(cfg.get("fps", 20.0))
    cfg["batch_size"]       = int(cfg.get("batch_size", 4))
    cfg["epochs"]           = int(cfg.get("epochs", 50))
    cfg["lr_tactile"]       = float(cfg.get("lr_tactile", 1e-4))
    cfg["lr_adapter"]       = float(cfg.get("lr_adapter", 5e-4))
    cfg["weight_decay"]     = float(cfg.get("weight_decay", 1e-4))
    cfg["action_dim"]       = int(cfg.get("action_dim", 7))
    cfg["n_tactile_tokens"] = int(cfg.get("n_tactile_tokens", 8))
    cfg["use_dummy_vla"]    = str(cfg.get("use_dummy_vla", "true")).lower() in ("1", "true", "yes")
    cfg["amp"]              = str(cfg.get("amp", "false")).lower() in ("1", "true", "yes")
    return cfg


# ============================================================================
# 4. 主训练循环
# ============================================================================

def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = cfg["amp"] and torch.cuda.is_available()
    use_dummy = args.use_dummy_vla or cfg["use_dummy_vla"]
    print(f"[Stage4] device={device}  AMP={use_amp}  dummy_vla={use_dummy}")

    # ------------------------------------------------------------------ #
    # 1. Dataset
    # ------------------------------------------------------------------ #
    T = cfg["window_T"]
    fps = cfg["fps"]
    dt = 1.0 / fps
    ts = [-(T - 1 - i) * dt for i in range(T)]  # [-0.75, -0.70, ..., 0.0]

    delta_timestamps = {
        "observation.images.realsense_rgb": ts,
        "observation.images.realsense_depth": ts,
        "observation.images.side": ts,
        "action": [0.0],
    }

    ds = LeRobotTactileDataset(
        repo_id=args.repo_id,
        root=args.root,
        delta_timestamps=delta_timestamps,
        sidecar_root=args.sidecar_root or args.root,
        window_T=T,
        fps=fps,
        has_right_tactile=args.has_right_tactile,
    )

    dl = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=(device == "cuda"),
    )
    print(f"[Stage4] 数据集: {len(ds)} 样本, {len(dl)} batches/epoch")

    # ------------------------------------------------------------------ #
    # 2. Model
    # ------------------------------------------------------------------ #
    model = TactileVLAAdapter(
        vla_model_id=args.vla_model_id,
        tactile_feat_dim=512,
        n_tactile_tokens=cfg["n_tactile_tokens"],
        action_dim=cfg["action_dim"],
        device=device,
        use_dummy_vla=use_dummy,
    ).to(device)

    # 加载 Stage 3 预训练触觉编码器
    load_pretrained_tactile(model, args.stage3_ckpt, device)

    # ------------------------------------------------------------------ #
    # 3. 分组优化器（差异学习率）
    # ------------------------------------------------------------------ #
    param_groups = [
        {
            "params": list(model.tactile_encoder.parameters()),
            "lr": cfg["lr_tactile"],
            "name": "tactile_encoder",
        },
        {
            "params": (
                list(model.projector.parameters())
                + list(model.action_head.parameters())
            ),
            "lr": cfg["lr_adapter"],
            "name": "adapter_head",
        },
    ]
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=cfg["weight_decay"],
    )
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # ------------------------------------------------------------------ #
    # 4. 参数统计
    # ------------------------------------------------------------------ #
    n_tac = sum(p.numel() for p in model.tactile_encoder.parameters() if p.requires_grad)
    n_proj = sum(p.numel() for p in model.projector.parameters())
    n_head = sum(p.numel() for p in model.action_head.parameters())
    print(f"[Stage4] 可训练参数: tactile={n_tac:,}  proj={n_proj:,}  head={n_head:,}")

    # ------------------------------------------------------------------ #
    # 5. 训练循环
    # ------------------------------------------------------------------ #
    best_loss = float("inf")

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        # VLA 始终冻结
        if model.use_dummy_vla:
            model.vla.eval()
        else:
            model.vla.eval()

        total_loss = 0.0
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{cfg['epochs']}", leave=False)

        for batch in pbar:
            # --- 触觉 ---
            # [B, T, 2, 16, 16]
            tactile = batch["tactile_grid"].to(device).float()

            # --- 双视角 RGB（手眼 + 全景）---
            # side: 你的手眼相机视角
            # realsense_rgb: 全景相机视角
            hand_rgb = batch["observation.images.side"].float()
            pano_rgb = batch["observation.images.realsense_rgb"].float()
            images_pil = batch_dual_images_to_pil(hand_rgb, pano_rgb, take_frame=-1)

            # --- 文本指令 ---
            texts = batch.get("language_instruction", ["grasp the cloth"] * tactile.shape[0])
            if isinstance(texts, torch.Tensor):
                texts = [str(t) for t in texts]

            # --- 动作标签 ---
            action_gt = batch["action"].to(device).float()
            # action 可能是 [B, 1, D] → squeeze
            if action_gt.dim() == 3:
                action_gt = action_gt[:, 0, :]

            with torch.amp.autocast(device_type=device, enabled=use_amp):
                action_pred = model(
                    images=images_pil,
                    texts=list(texts),
                    tactile_grids=tactile,
                )
                loss = criterion(action_pred, action_gt)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for g in param_groups for p in g["params"]], 1.0
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.5f}")

        avg_loss = total_loss / max(len(dl), 1)
        print(f"[Stage4] Epoch {epoch}/{cfg['epochs']}  avg_loss={avg_loss:.6f}")

        # 保存最优
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs("runs", exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss": best_loss,
                },
                "runs/ckpt_stage4_vla_best.pt",
            )
            print(f"  → 保存最优 checkpoint (loss={best_loss:.6f})")

    # 最终保存
    os.makedirs("runs", exist_ok=True)
    torch.save(
        {"epoch": cfg["epochs"], "model": model.state_dict()},
        "runs/ckpt_stage4_vla_final.pt",
    )
    print(f"[Stage4] 训练完成，最终 checkpoint 已保存。")


if __name__ == "__main__":
    main()
