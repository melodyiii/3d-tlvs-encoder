import torch, torch.nn as nn, torch.nn.functional as F

class TactileEncoder(nn.Module):
    def __init__(self, proj_dim=512, hid=128, d_model=256, tau=0.07):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,64,3,1,1), nn.BatchNorm2d(64), nn.GELU(),
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

def multi_pos_infonce(z_q, pos_list, all_keys, logit_scale):
    # z_q: [B,512]; pos_list: list of [B,512]; all_keys: [N,512]
    logits_all = (z_q @ all_keys.t()) * logit_scale
    pos_logits = torch.stack([(z_q * p).sum(-1)*logit_scale for p in pos_list], dim=1)
    return (torch.logsumexp(logits_all,1) - torch.logsumexp(pos_logits,1)).mean()
