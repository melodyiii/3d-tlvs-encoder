import torch
import torch.nn as nn
from models.pointnetpp_encoder import PointNetPPEncoder
from models.tlv_student import TactileEncoder
from teachers.clip_teacher import OpenCLIPEncoder
import torch.nn.functional as F

def contrastive_loss(x, y, temperature=0.07):
   
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    logits = torch.matmul(x, y.t()) / temperature
    labels = torch.arange(x.size(0)).long().to(x.device)
    loss_x2y = F.cross_entropy(logits, labels)
    loss_y2x = F.cross_entropy(logits.t(), labels)
    return (loss_x2y + loss_y2x) / 2

class HybridAlignmentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_encoder = OpenCLIPEncoder(frozen=True)
        self.pointnet = PointNetPPEncoder(feature_dim=512)
        self.tactile_encoder = TactileEncoder()
        self.cross_modal_adapter = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 512)
        )

    def forward(self, rgb, pointcloud, tactile):
        z_2d = self.clip_encoder(rgb)
        z_3d = self.pointnet(pointcloud)
        z_t = self.tactile_encoder(tactile)
        z_t_proj = self.cross_modal_adapter(z_t)

        loss_t2d = contrastive_loss(z_t, z_2d)
        loss_t3d = contrastive_loss(z_t_proj, z_3d)
        loss_3d2d = contrastive_loss(z_3d, z_2d)
        total_loss = loss_t2d + 0.3 * loss_t3d + 0.3 * loss_3d2d
        return total_loss
