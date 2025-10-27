import torch, torch.nn as nn, torch.nn.functional as F

class TactileEncoder(nn.Module):
    def __init__(self, proj_dim=512, hid=128, d_model=256, tau=0.07):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,64,3,1,1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(64,128,3,1,1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128,hid,3,1,1), nn.BatchNorm2d(hid), nn.GELU(),
        )
        # 注意：GRU 输入尺寸与我们下面的展平保持一致（hid*16*16）
        self.gru  = nn.GRU(hid*16*16, d_model, batch_first=True)
        self.proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, proj_dim))
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0/tau)))  # = log(1/τ)

    def forward(self, x):                 # x: [B,T,16,16]
        B, T, H, W = x.shape
        # —— 关键修改：把时间维并到batch，逐帧过CNN ——
        x = x.view(B*T, 1, H, W)                 # [B*T,1,16,16]
        f = self.cnn(x).flatten(1)               # [B*T, hid*16*16]
        f = f.view(B, T, -1)                     # [B, T, hid*16*16]

        _, h = self.gru(f)                       # [1,B,d_model]
        z = F.normalize(self.proj(h[-1]), dim=-1)  # [B,512]
        s = self.logit_scale.exp().clamp(1e-3, 1e3)
        return z, s

def improved_multi_pos_infonce(z_q, pos_list, all_keys, logit_scale, method="weighted"):
    """
    改进的多正例对比学习损失
    method: "weighted" | "hard" | "simple"
    """
    if method == "simple":
        # 你原来的方法，保持兼容
        logits_all = (z_q @ all_keys.t()) * logit_scale
        pos_logits = torch.stack([(z_q * p).sum(-1)*logit_scale for p in pos_list], dim=1)
        return (torch.logsumexp(logits_all,1) - torch.logsumexp(pos_logits,1)).mean()
    
    elif method == "weighted":
        # 加权多正例损失 - 对不同的正例给予不同权重
        logits_all = (z_q @ all_keys.t()) * logit_scale
        
        losses = []
        for i, pos_emb in enumerate(pos_list):
            # 计算每个正例的相似度
            pos_sim = (z_q * pos_emb).sum(-1) * logit_scale
            
            # 创建目标：当前正例为正样本，其他为负样本
            exp_pos = torch.exp(pos_sim)
            exp_neg = torch.exp(logits_all).sum(1) - exp_pos
            
            # 加权：第一个正例（完整句子）权重更高
            weight = 1.0 if i == 0 else 0.7
            loss = -torch.log(exp_pos / (exp_pos + exp_neg)) * weight
            losses.append(loss)
        
        return torch.stack(losses).mean()
    
    elif method == "hard":
        # 困难负例挖掘版本
        logits_all = (z_q @ all_keys.t()) * logit_scale
        
        # 找到最难负例（相似度最高的负例）
        with torch.no_grad():
            pos_mask = torch.zeros_like(logits_all, dtype=torch.bool)
            for pos_emb in pos_list:
                # 找到正例在all_keys中的位置
                pos_indices = (all_keys.unsqueeze(0) == pos_emb.unsqueeze(1)).all(-1)
                pos_mask |= pos_indices
            
            # 负例的logits
            neg_logits = logits_all.clone()
            neg_logits[pos_mask] = -1e9  # 掩码正例
            hard_neg_idx = neg_logits.argmax(1)  # 最难负例
        
        losses = []
        for pos_emb in pos_list:
            pos_sim = (z_q * pos_emb).sum(-1) * logit_scale
            hard_neg_sim = logits_all[torch.arange(z_q.size(0)), hard_neg_idx]
            
            # 使用最难负例计算损失
            loss = F.cross_entropy(
                torch.stack([pos_sim, hard_neg_sim], dim=1), 
                torch.zeros(z_q.size(0), dtype=torch.long, device=z_q.device)
            )
            losses.append(loss)
        
        return torch.stack(losses).mean()