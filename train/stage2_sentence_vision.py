import os, json, yaml, numpy as np, torch, torch.nn.functional as F, argparse
from glob import glob
from torch.utils.data import Dataset, DataLoader
from models.tlv_student import TactileEncoder, improved_multi_pos_infonce  # 【修改】导入新损失函数


# 1. 数据集与数据加载器修改
class TactileSetS2(Dataset):
    def __init__(self, root="dataset", T=16, phrases_mix_ratio=0.3, visual_embed=None, augment=True):  # 新增augment参数
        self.items = sorted([d for d in glob(f"{root}/seq_*") if os.path.isdir(d)])
        self.T = T
        self.mix = phrases_mix_ratio
        self.text_repo = torch.load("teachers/text_embed.pt")
        self.visual_repo = visual_embed
        self.augment = augment  # 新增

    def tactile_augmentation(self, tac):
        """简单的在线数据增强"""
        # 随机时间掩码
        if np.random.rand() < 0.3:
            mask_frames = np.random.randint(1, self.T//4)
            mask_start = np.random.randint(0, self.T - mask_frames)
            tac[mask_start:mask_start+mask_frames] = 0
        
        # 添加高斯噪声
        if np.random.rand() < 0.4:
            noise = torch.randn_like(tac) * 0.03
            tac = tac + noise
        
        # 随机缩放
        if np.random.rand() < 0.3:
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
        
       
        if self.augment:
            tac = self.tactile_augmentation(tac)

        j = json.load(open(f"{d}/text.json"))
        sent = list(j.get("sentences", []))
        if np.random.rand() < self.mix:
            sent += list(j.get("phrases", []))
        
        z_list = []
        for s in sent:
            z = self.text_repo["sentences"].get(s, None)
            if z is None:
                z = self.text_repo["phrases"][s]
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


def parse_cfg():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/stage2_sentence_vision.yaml")
    p.add_argument("--resume", type=str, default="runs/ckpt_stage1.pt")
    args = p.parse_args()
    if not os.path.exists(args.config):
        os.makedirs("configs", exist_ok=True)
        open(args.config,"w").write(
            "seed: 0\nwindow_T: 16\nbatch_size: 32\naccum_steps: 4\n"
            "lr: 0.0003\nweight_decay: 0.0001\ntau_init: 0.07\n"
            "epochs: 5\namp: false\nphrases_mix_ratio: 0.3\n"
            "lambda_tv: 0.2\naugment: true\n"  # 新增augment配置
        )
    cfg = yaml.safe_load(open(args.config))
    # 强制类型
    cfg["lr"]=float(cfg["lr"]); cfg["weight_decay"]=float(cfg["weight_decay"]); cfg["tau_init"]=float(cfg["tau_init"])
    cfg["batch_size"]=int(cfg["batch_size"]); cfg["accum_steps"]=int(cfg["accum_steps"]); cfg["epochs"]=int(cfg["epochs"])
    cfg["amp"]=str(cfg.get("amp","false")).lower() in ["1","true","yes","y"]
    cfg["phrases_mix_ratio"]=float(cfg.get("phrases_mix_ratio",0.3))
    cfg["lambda_tv"]=float(cfg.get("lambda_tv", 0.2))
    cfg["augment"] = bool(cfg.get("augment", True))  # 新增：解析augment参数
    return args, cfg


# 3. 主训练循环修改

def main():
    args, cfg = parse_cfg()
    torch.manual_seed(cfg["seed"])
    
    visual_embed_path = "teachers/visual_embed.pt"
    if not os.path.exists(visual_embed_path):
        raise RuntimeError(f"未找到 {visual_embed_path}。请先运行脚本提取视觉嵌入向量。")
    visual_embed = torch.load(visual_embed_path)
    print(f"[信息] 已加载 {len(visual_embed)} 个视觉嵌入向量。")

    # 【修改】传入augment参数
    ds = TactileSetS2(T=cfg["window_T"], phrases_mix_ratio=cfg["phrases_mix_ratio"], 
                     visual_embed=visual_embed, augment=cfg["augment"])
    if len(ds)==0: raise RuntimeError("dataset/ 目录为空，请先运行数据生成脚本。")
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=2, collate_fn=collate, drop_last=True)

    model = TactileEncoder(tau=cfg["tau_init"])
    if os.path.exists(args.resume):
        sd = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(sd["model"], strict=False)
        print(f"[阶段2] 已从 {args.resume} 加载预训练权重。")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler('cuda', enabled=(cfg["amp"] and torch.cuda.is_available()))
    
    for ep in range(cfg["epochs"]):
        model.train()
        tot_loss, tot_t_text, tot_tv = 0, 0, 0
        
        for it, batch in enumerate(dl):
            tac = batch["tactile"].to(device).float()
            z_v = batch["z_v"].to(device).float()
            
            with torch.amp.autocast('cuda', enabled=(cfg["amp"] and torch.cuda.is_available())):
                z_t, s = model(tac)
                z_x_list = [F.normalize(z.to(device).float(), dim=-1) for z in batch["z_x_list"]]
                all_text = torch.cat(z_x_list, dim=0)
                
                # --- 损失计算 ---
                # 【修改】使用新的损失函数，method="weighted"用于句子+短语混合
                loss_t_text = improved_multi_pos_infonce(z_t, z_x_list, all_text, s, method="weighted")
                loss_tv = F.mse_loss(z_t, z_v)
                
                total_loss = loss_t_text + cfg["lambda_tv"] * loss_tv
            
            scaler.scale(total_loss / cfg["accum_steps"]).backward()
            
            if (it+1)%cfg["accum_steps"]==0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            
            tot_loss += total_loss.item()
            tot_t_text += loss_t_text.item()
            tot_tv += loss_tv.item()

        n_batches = len(dl)
        print(
            f"[阶段2-视觉] 周期 {ep+1}: 总损失={tot_loss/n_batches:.4f} | "
            f"文-触损失={tot_t_text/n_batches:.4f} | "
            f"视-触损失={tot_tv/n_batches:.4f}"
        )

    os.makedirs("runs", exist_ok=True)
    save_path = "runs/ckpt_stage2_vision.pt"
    torch.save({"model": model.state_dict()}, save_path)
    print(f"已保存模型到 {save_path}")

if __name__ == "__main__":
    main()