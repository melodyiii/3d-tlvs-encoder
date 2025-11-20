import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from glob import glob
from tqdm import tqdm


from models.tlv_student import TactileEncoder
try:
    from models.pointnetpp_encoder import PointNetPPEncoder
except ImportError:
  
    raise ImportError("请确保 models/pointnetpp_encoder.py 存在！")


class CrossModalAttentionPool(nn.Module):
    def __init__(self, dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.norm_ffn = nn.LayerNorm(dim)

    def forward(self, query, key_value):
        # query: [B, 512] (Vision) -> [B, 1, 512]
        q = query.unsqueeze(1)
        # key_value: [B, T, 512] (Tactile)
        attn_out, _ = self.multihead_attn(q, key_value, key_value)
        x = self.norm(q + attn_out)
        x = self.norm_ffn(x + self.ffn(x))
        return x.squeeze(1)

class Hybrid3DTLVsModel(nn.Module):
    def __init__(self):
        super().__init__()
         
        self.tactile_encoder = TactileEncoder()
        
        # B. 3D 点云编码器 (PointNet++)
        # 假设输出维度是 512
        self.pointnet = PointNetPPEncoder(feature_dim=512)
        
        # C. 视觉引导注意力 (继承 Stage 2 能力)
        self.vision_guide_attn = CrossModalAttentionPool(dim=512)

    def forward(self, x_tac, x_pc, z_vision_frozen):
        # 1. 提取 3D 特征
        z_3d = self.pointnet(x_pc) # [B, 512]
        
        # 2. 提取触觉特征 (序列模式)
        z_t_seq, z_t_global, _ = self.tactile_encoder(x_tac, return_seq=True)
        
        # 3. 视觉引导触觉
        z_t_aligned_vision = self.vision_guide_attn(query=z_vision_frozen, key_value=z_t_seq)
        
        return z_t_aligned_vision, z_3d, z_t_global



class HybridDataset(Dataset):
    def __init__(self, root="dataset", T=16, n_points=2048, visual_embed=None):
        self.items = sorted([d for d in glob(f"{root}/seq_*") if os.path.isdir(d)])
        self.T = T
        self.n_points = n_points
        self.visual_repo = visual_embed

    def __getitem__(self, idx):
        d = self.items[idx]
        seq_name = os.path.basename(d)
        
        # 1. 读取触觉
        tac = torch.from_numpy(np.load(f"{d}/tactile.npy")).float()
        if tac.shape[0] > self.T: # 随机截取
            start = np.random.randint(0, tac.shape[0] - self.T)
            tac = tac[start:start+self.T]
        else: # 补齐
            pad = self.T - tac.shape[0]
            tac = torch.cat([tac, tac[:pad]], dim=0)

        # 2. 读取视觉特征
        z_v = self.visual_repo[seq_name]

        # 3. 读取点云 (如果没有文件，生成随机噪声防止报错，但训练会无效)
        pc_path = f"{d}/pointcloud.npy"
        if os.path.exists(pc_path):
            pc = np.load(pc_path) #[N, 3]
            # 简单采样到 2048 点
            if pc.shape[0] >= self.n_points:
                choice = np.random.choice(pc.shape[0], self.n_points, replace=False)
                pc = pc[choice]
            else:
                choice = np.random.choice(pc.shape[0], self.n_points, replace=True)
                pc = pc[choice]
        else:
          
            pc = np.random.rand(self.n_points, 3)
            
        pc = torch.from_numpy(pc).float()
        
        return tac, pc, z_v

    def __len__(self): return len(self.items)

def collate_fn(batch):
    tacs, pcs, zvs = zip(*batch)
    return {
        "tactile": torch.stack(tacs),
        "pointcloud": torch.stack(pcs),
        "z_vision": torch.stack(zvs)
    }



def contrastive_loss(x, y, temp=0.07):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    logits = (x @ y.t()) / temp
    labels = torch.arange(len(x), device=x.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2

def load_stage2_weights(model, stage2_path, device):
    print(f"[Init] 正在加载 Stage 2 权重: {stage2_path}")
    if not os.path.exists(stage2_path):
        print("[Warning] Stage 2 权重不存在！将从头开始训练。")
        return model
        
    ckpt = torch.load(stage2_path, map_location=device)
    s2_dict = ckpt["model"]
    model_dict = model.state_dict()
    
    loaded = []
    for k, v in s2_dict.items():
        # 映射规则1: tactile_encoder 直接复制
        if k.startswith("tactile_encoder.") and k in model_dict:
            model_dict[k] = v
            loaded.append(k)
        # 映射规则2: cross_attn -> vision_guide_attn
        elif k.startswith("cross_attn."):
            new_k = k.replace("cross_attn.", "vision_guide_attn.")
            if new_k in model_dict:
                model_dict[new_k] = v
                loaded.append(new_k)
                
    model.load_state_dict(model_dict, strict=False)
    print(f"[Init] 成功加载了 {len(loaded)} 个参数层。")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2_ckpt", type=str, default="runs/ckpt_stage2_vision_attn.pt")
    parser.add_argument("--save_path", type=str, default="runs/ckpt_stage3_3d_final.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # 1. 必须先有视觉特征
    if not os.path.exists("teachers/visual_embed.pt"):
        print(" 错误: 找不到 teachers/visual_embed.pt")
        print("请先运行之前的脚本生成离线视觉特征。")
        return
    visual_embed = torch.load("teachers/visual_embed.pt")

    # 2. 准备数据
    ds = HybridDataset(visual_embed=visual_embed)
    if len(ds) == 0:
        print("❌ 错误: dataset/ 目录下没有数据！")
        return
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn, drop_last=True)

    # 3. 初始化模型
    model = Hybrid3DTLVsModel().to(device)
    
    # 4. 加载 Stage 2 权重 (继承学习)
    model = load_stage2_weights(model, args.stage2_ckpt, device)

    # 5. 优化器 (PointNet用大学习率，触觉用小学习率微调)
    opt = torch.optim.AdamW([
        {"params": model.tactile_encoder.parameters(), "lr": args.lr * 0.1},
        {"params": model.vision_guide_attn.parameters(), "lr": args.lr * 0.1},
        {"params": model.pointnet.parameters(), "lr": args.lr}, # 3D部分主力训练
    ], weight_decay=1e-4)

    scaler = torch.cuda.amp.GradScaler()

    # 6. 训练循环
    print("开始 Stage 3: 3D 点云融合训练...")
    for ep in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(dl, desc=f"Epoch {ep+1}"):
            tac = batch["tactile"].to(device)
            pc = batch["pointcloud"].to(device)
            z_v = batch["z_vision"].to(device)

            with torch.cuda.amp.autocast():
                # 前向传播
                z_t_vis, z_3d, z_t_raw = model(tac, pc, z_v)
                
               
                # 1. 触觉 <-> 视觉
                l_tv = contrastive_loss(z_t_vis, z_v)
                # 2. 3D <-> 视觉 (PointNet学语义)
                l_3d2d = contrastive_loss(z_3d, z_v)
                # 3. 触觉 <-> 3D (几何对齐)
                l_t3d = contrastive_loss(z_t_raw, z_3d)
                
                loss = l_tv + 0.5 * l_3d2d + 0.5 * l_t3d
            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            total_loss += loss.item()

        print(f"Epoch {ep+1} Loss: {total_loss/len(dl):.4f}")

    # 保存
    os.makedirs("runs", exist_ok=True)
    torch.save({"model": model.tactile_encoder.state_dict()}, args.save_path)
    # 同时也保存 PointNet 以备后用
    torch.save({"model": model.pointnet.state_dict()}, args.save_path.replace("final", "pointnet"))
    print(f" 训练完成！最终触觉模型已保存至: {args.save_path}")

if __name__ == "__main__":
    main()