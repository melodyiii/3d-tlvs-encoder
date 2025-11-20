import os
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from glob import glob
from torch.utils.data import Dataset, DataLoader


from models.tlv_student import TactileEncoder, improved_multi_pos_infonce

# 跨模态注意力池化 
class CrossModalAttentionPool(nn.Module):
    def __init__(self, dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm_ffn = nn.LayerNorm(dim)

    def forward(self, tactile_seq, anchor_emb):
       
        # 1.变成 Sequence 形式: [Batch, 1, Dim]
        query = anchor_emb.unsqueeze(1) 
        
        # 2. Cross Attention: Vision (Query) -> Tactile (Key/Value)
    
        attn_out, _ = self.multihead_attn(query, tactile_seq, tactile_seq)
        
        # 3. Residual + Norm
        x = self.norm(query + attn_out)
        
        # 4. FFN + Residual
        x = self.norm_ffn(x + self.ffn(x))
        
        return x.squeeze(1) # [Batch, Dim] 视觉引导后的触觉特征

#  Stage 2 整合模型 
class VisionGuidedTactileModel(nn.Module):
    def __init__(self, tau_init=0.07, feature_dim=512):
        super().__init__()
        # 基础触觉编码器 (Stage 1 训练好的)
        self.tactile_encoder = TactileEncoder(tau=tau_init)
        
        # 新增的跨模态注意力模块 
        self.cross_attn = CrossModalAttentionPool(dim=feature_dim)
        
        # 用于 Vision-Tactile 对齐的 Logit Scale (温度系数)
        self.logit_scale_tv = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x_tactile, z_vision_anchor):
        """
        x_tactile: [B, T, 16, 16]
        z_vision_anchor: [B, 512] (来自 Frozen CLIP)
        """
        # 1. 获取触觉的序列特征 [B, T, 512]
        
        z_t_seq, z_t_global, s_text = self.tactile_encoder(x_tactile, return_seq=True)
        
        # 2. 使用 Cross-Attention 进行视觉引导的特征融合
        z_t_aligned = self.cross_attn(z_t_seq, z_vision_anchor)
        
        return {
            "z_t_seq": z_t_seq,       # 原始序列
            "z_t_global": z_t_global, # 原始全局池化 (用于文本对齐)
            "z_t_aligned": z_t_aligned, # 视觉引导后的特征 (用于视觉对齐)
            "logit_scale_text": s_text,
            "logit_scale_tv": self.logit_scale_tv
        }

# InfoNCE Loss for Vision-Tactile ---
def contrastive_loss_tv(feat_t, feat_v, logit_scale):
    """
    标准的 CLIP-style Contrastive Loss
    feat_t: [B, D]
    feat_v: [B, D]
    """
    feat_t = F.normalize(feat_t, dim=-1)
    feat_v = F.normalize(feat_v, dim=-1)
    
    logit_scale = logit_scale.exp()
    logits = logit_scale * feat_t @ feat_v.t()
    
    labels = torch.arange(len(logits), device=logits.device)
    loss_t2v = F.cross_entropy(logits, labels)
    loss_v2t = F.cross_entropy(logits.t(), labels)
    
    return (loss_t2v + loss_v2t) / 2

# 数据集定义 
class TactileSetS2(Dataset):
    def __init__(self, root="dataset", T=16, phrases_mix_ratio=0.3, visual_embed=None, augment=True):
        self.items = sorted([d for d in glob(f"{root}/seq_*") if os.path.isdir(d)])
        self.T = T
        self.mix = phrases_mix_ratio
        self.text_repo = torch.load("teachers/text_embed.pt")
        self.visual_repo = visual_embed
        self.augment = augment

    def tactile_augmentation(self, tac):
        if np.random.rand() < 0.3: # Masking
            mask_frames = np.random.randint(1, self.T//4)
            mask_start = np.random.randint(0, self.T - mask_frames)
            tac[mask_start:mask_start+mask_frames] = 0
        if np.random.rand() < 0.4: # Noise
            noise = torch.randn_like(tac) * 0.03
            tac = tac + noise
        if np.random.rand() < 0.3: # Scale
            scale = 0.8 + 0.4 * np.random.rand()
            tac = tac * scale
        return tac

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        d = self.items[idx]
        sample_id = os.path.basename(d)
        tac = torch.from_numpy(np.load(f"{d}/tactile.npy")).float()
        TT = tac.shape[0]; t0 = 0 if TT<=self.T else np.random.randint(0, TT-self.T+1)
        tac = tac[t0:t0+self.T]
        
        if self.augment: tac = self.tactile_augmentation(tac)

        j = json.load(open(f"{d}/text.json"))
        sent = list(j.get("sentences", []))
        if np.random.rand() < self.mix: sent += list(j.get("phrases", []))
        
        z_list = []
        for s in sent:
            z = self.text_repo["sentences"].get(s, None)
            if z is None: z = self.text_repo["phrases"][s]
            z_list.append(z.unsqueeze(0))
        
        z_v = self.visual_repo[sample_id]
        return tac, z_list, z_v

def collate(batch):
    tacs, lists, z_vs = zip(*batch)
    tac = torch.stack(tacs,0)
    z_v = torch.stack(z_vs, 0)
    P = max(len(l) for l in lists)
    out=[]
    for k in range(P):
        vecs=[ (l[k] if k<len(l) else l[0]) for l in lists ]
        out.append(torch.cat(vecs,0))
    return {"tactile": tac, "z_x_list": out, "z_v": z_v}

# 配置解析
def parse_cfg():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/stage2_vision_attn.yaml")
    p.add_argument("--resume", type=str, default="runs/ckpt_stage1.pt")
    args = p.parse_args()
    
    if not os.path.exists(args.config):
        os.makedirs("configs", exist_ok=True)
        open(args.config,"w").write(
            "seed: 42\nwindow_T: 16\nbatch_size: 32\naccum_steps: 4\n"
            "lr: 0.0003\nweight_decay: 0.0001\ntau_init: 0.07\n"
            "epochs: 10\namp: true\nphrases_mix_ratio: 0.3\n"
            "lambda_tv: 1.0\naugment: true\n" 
        )
    cfg = yaml.safe_load(open(args.config))
    # Config type casting
    cfg["lr"]=float(cfg["lr"]); cfg["weight_decay"]=float(cfg["weight_decay"])
    cfg["tau_init"]=float(cfg["tau_init"]); cfg["batch_size"]=int(cfg["batch_size"])
    cfg["accum_steps"]=int(cfg["accum_steps"]); cfg["epochs"]=int(cfg["epochs"])
    cfg["amp"]=str(cfg.get("amp","false")).lower() in ["1","true","yes","y"]
    cfg["phrases_mix_ratio"]=float(cfg.get("phrases_mix_ratio",0.3))
    cfg["lambda_tv"]=float(cfg.get("lambda_tv", 1.0))
    cfg["augment"] = bool(cfg.get("augment", True))
    return args, cfg

# 主程序 
def main():
    args, cfg = parse_cfg()
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载离线视觉特征
    visual_embed_path = "teachers/visual_embed.pt"
    if not os.path.exists(visual_embed_path):
        raise RuntimeError(f"未找到 {visual_embed_path}。")
    visual_embed = torch.load(visual_embed_path)
    print(f"[Init] Loaded {len(visual_embed)} visual embeddings.")

    # 2. 数据集
    ds = TactileSetS2(T=cfg["window_T"], phrases_mix_ratio=cfg["phrases_mix_ratio"], 
                     visual_embed=visual_embed, augment=cfg["augment"])
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=4, collate_fn=collate, drop_last=True)

    # 3. 初始化整合模型 (Wrapper)
    model = VisionGuidedTactileModel(tau_init=cfg["tau_init"])
    
    # 4. 加载 Stage 1 权重
    if os.path.exists(args.resume):
        print(f"[Resume] Loading Stage 1 weights from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location="cpu")
        state_dict = checkpoint["model"]
        
    
        msg = model.tactile_encoder.load_state_dict(state_dict, strict=False)
        print(f"[Resume] Weights loaded. Missing keys (expected for attn): {len(msg.missing_keys)}")
    
    model.to(device)

    # 5. 优化器 & Scaler
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["amp"])

    # 6. 训练循环
    for ep in range(cfg["epochs"]):
        model.train()
        stats = {"loss": 0, "loss_text": 0, "loss_tv": 0}
        
        for it, batch in enumerate(dl):
            tac = batch["tactile"].to(device).float()
            z_v = batch["z_v"].to(device).float() # [B, 512] (CLIP Vision)
            
            with torch.cuda.amp.autocast(enabled=cfg["amp"]):
                # Forward: 传入视觉特征作为 Anchor，触发 Cross-Attention
                out = model(tac, z_v)
                
                z_t_global = out["z_t_global"]   # 原始 Pooling 特征
                z_t_aligned = out["z_t_aligned"] # 视觉引导后的特征
                s_text = out["logit_scale_text"]
                s_tv = out["logit_scale_tv"]

                # --- Loss 1: 触觉-文本对齐 (保持 Stage 1 的能力) ---
                # 使用原始 global 特征，因为文本描述通常是全局的
                z_x_list = [F.normalize(z.to(device).float(), dim=-1) for z in batch["z_x_list"]]
                all_text = torch.cat(z_x_list, dim=0)
                loss_text = improved_multi_pos_infonce(z_t_global, z_x_list, all_text, s_text, method="weighted")
                
                # --- Loss 2: 触觉-视觉对齐 (Stage 2 核心) ---
                # 使用 Attention 后的特征，计算 InfoNCE
                loss_tv = contrastive_loss_tv(z_t_aligned, z_v, s_tv)
                
                # 总损失
                total_loss = loss_text + cfg["lambda_tv"] * loss_tv
            
            # Backward
            scaler.scale(total_loss / cfg["accum_steps"]).backward()
            
            if (it+1) % cfg["accum_steps"] == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            
            # Logging
            stats["loss"] += total_loss.item()
            stats["loss_text"] += loss_text.item()
            stats["loss_tv"] += loss_tv.item()

        # Epoch Summary
        n = len(dl)
        print(f"[Epoch {ep+1}/{cfg['epochs']}] "
              f"Loss: {stats['loss']/n:.4f} | "
              f"Text-Loss: {stats['loss_text']/n:.4f} | "
              f"Vision-Loss: {stats['loss_tv']/n:.4f}")

    # Save
    os.makedirs("runs", exist_ok=True)
    save_path = "runs/ckpt_stage2_vision_attn.pt"
    torch.save({"model": model.state_dict()}, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()