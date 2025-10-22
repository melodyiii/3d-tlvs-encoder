
import os, json, yaml, numpy as np, torch, torch.nn.functional as F, argparse
from glob import glob
from torch.utils.data import Dataset, DataLoader
from models.tlv_student import TactileEncoder, multi_pos_infonce

class TactileSetS2(Dataset):
    def __init__(self, root="dataset", T=16, phrases_mix_ratio=0.3):
        self.items = sorted([d for d in glob(f"{root}/seq_*") if os.path.isdir(d)])
        self.T = T
        self.mix = phrases_mix_ratio
        self.repo = torch.load("teachers/text_embed.pt")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        d = self.items[idx]
        tac = torch.from_numpy(np.load(f"{d}/tactile.npy")).float()  # [TT,16,16]
        TT = tac.shape[0]; t0 = 0 if TT<=self.T else np.random.randint(0, TT-self.T+1)
        tac = tac[t0:t0+self.T]                                      # [T,16,16]

        j = json.load(open(f"{d}/text.json"))
        sent = list(j.get("sentences", []))
        if np.random.rand() < self.mix:
            sent += list(j.get("phrases", []))  # 课程式：句子为主，混入部分短语
        # 句子优先，其次短语库兜底
        z_list = []
        for s in sent:
            z = self.repo["sentences"].get(s, None)
            if z is None:  # 没有对应句子向量时，回退到短语库
                z = self.repo["phrases"][s]
            z_list.append(z.unsqueeze(0))
        return tac, z_list

def collate(batch):
    tacs, lists = zip(*batch)
    tac = torch.stack(tacs,0)                                       # [B,T,16,16]
    P = max(len(l) for l in lists)
    out=[]
    for k in range(P):
        vecs=[ (l[k] if k<len(l) else l[0]) for l in lists ]
        out.append(torch.cat(vecs,0))                                # [B,512]
    return {"tactile":tac, "z_x_list":out}

def parse_cfg():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/stage2_sentence.yaml")
    p.add_argument("--resume", type=str, default="runs/ckpt_stage1.pt")
    args = p.parse_args()
    if not os.path.exists(args.config):
        os.makedirs("configs", exist_ok=True)
        open(args.config,"w").write(
            "seed: 0\nwindow_T: 16\nbatch_size: 32\naccum_steps: 4\n"
            "lr: 0.0003\nweight_decay: 0.0001\ntau_init: 0.07\n"
            "epochs: 5\namp: false\nphrases_mix_ratio: 0.3\n"
        )
    cfg = yaml.safe_load(open(args.config))
    # 强制类型
    cfg["lr"]=float(cfg["lr"]); cfg["weight_decay"]=float(cfg["weight_decay"]); cfg["tau_init"]=float(cfg["tau_init"])
    cfg["batch_size"]=int(cfg["batch_size"]); cfg["accum_steps"]=int(cfg["accum_steps"]); cfg["epochs"]=int(cfg["epochs"])
    cfg["amp"]=str(cfg.get("amp","false")).lower() in ["1","true","yes","y"]
    cfg["phrases_mix_ratio"]=float(cfg.get("phrases_mix_ratio",0.3))
    return args, cfg

def main():
    args, cfg = parse_cfg()
    torch.manual_seed(cfg["seed"])
    ds = TactileSetS2(T=cfg["window_T"], phrases_mix_ratio=cfg["phrases_mix_ratio"])
    if len(ds)==0: raise RuntimeError("dataset/ 为空，先运行 make_mock_dataset 与 build_text_embed_openclip")
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=2, collate_fn=collate, drop_last=True)

    model = TactileEncoder(tau=cfg["tau_init"])
    if os.path.exists(args.resume):
        sd = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(sd["model"], strict=False)
        print(f"[S2] loaded resume from {args.resume}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler('cuda', enabled=(cfg["amp"] and torch.cuda.is_available()))
    for ep in range(cfg["epochs"]):
        model.train(); tot=0
        for it, batch in enumerate(dl):
            tac = batch["tactile"].to(device).float()
            with torch.amp.autocast('cuda', enabled=(cfg["amp"] and torch.cuda.is_available())):
                z_t, s = model(tac)
                z_x_list = [F.normalize(z.to(device).float(), dim=-1) for z in batch["z_x_list"]]
                all_text = torch.cat(z_x_list, dim=0)
                loss = multi_pos_infonce(z_t, z_x_list, all_text, s) / cfg["accum_steps"]
            scaler.scale(loss).backward()
            if (it+1)%cfg["accum_steps"]==0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            tot += loss.item()*cfg["accum_steps"]
        print(f"[Stage2] epoch {ep+1}: loss={tot/len(dl):.4f}")

    os.makedirs("runs", exist_ok=True)
    torch.save({"model": model.state_dict()}, "runs/ckpt_stage2.pt")
    print("Saved runs/ckpt_stage2.pt")

if __name__ == "__main__":
    main()