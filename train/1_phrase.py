import os, json, yaml, numpy as np, torch, torch.nn.functional as F
from glob import glob
from torch.utils.data import Dataset, DataLoader
from models.tlv_student import TactileEncoder, improved_multi_pos_infonce

class TactileSet(Dataset):
    def __init__(self, root="dataset", T=16):
        self.items = sorted([d for d in glob(f"{root}/seq_*") if os.path.isdir(d)])
        self.T = T
        self.text_repo = torch.load("teachers/text_embed.pt")
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        d = self.items[idx]
        tac = torch.from_numpy(np.load(f"{d}/tactile.npy")).float()  # [TT,16,16]
        TT=tac.shape[0]; t0=0 if TT<=self.T else np.random.randint(0, TT-self.T+1)
        tac = tac[t0:t0+self.T]                                      # [T,16,16]
        phrases = json.load(open(f"{d}/text.json"))["phrases"]
        z_x_list = [self.text_repo["phrases"][p].unsqueeze(0) for p in phrases]
        return tac, z_x_list

def collate(batch):
    tacs, lists = zip(*batch)
    tac = torch.stack(tacs,0)                                       # [B,T,16,16]
    P = max(len(l) for l in lists)
    out=[]
    for k in range(P):
        vecs=[ (l[k] if k<len(l) else l[0]) for l in lists ]
        out.append(torch.cat(vecs,0))                                # [B,512]
    return {"tactile":tac, "z_x_list":out}

def main():
    os.makedirs("configs", exist_ok=True)
    if not os.path.exists("configs/stage1_phrase.yaml"):
        with open("configs/stage1_phrase.yaml","w") as f:
            f.write("seed: 0\nwindow_T: 16\nbatch_size: 32\naccum_steps: 4\nlr: 3e-4\nweight_decay: 1e-4\ntau_init: 0.07\nepochs: 3\namp: false\n")
    cfg = yaml.safe_load(open("configs/stage1_phrase.yaml"))
    cfg["lr"] = float(cfg["lr"])
    cfg["weight_decay"] = float(cfg["weight_decay"])
    cfg["tau_init"] = float(cfg["tau_init"])
    cfg["batch_size"] = int(cfg["batch_size"])
    cfg["accum_steps"] = int(cfg["accum_steps"])
    cfg["epochs"] = int(cfg["epochs"])
    cfg["amp"] = str(cfg.get("amp", "false")).lower() in ["1","true","yes","y"]
    torch.manual_seed(cfg["seed"])
    ds = TactileSet(T=cfg["window_T"])
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=2, collate_fn=collate, drop_last=True)
    model = TactileEncoder(tau=cfg["tau_init"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg["amp"] and torch.cuda.is_available()))
    for ep in range(cfg["epochs"]):
        model.train(); tot=0
        for it,batch in enumerate(dl):
            tac = batch["tactile"].to(device).float()
            with torch.cuda.amp.autocast(enabled=(cfg["amp"] and torch.cuda.is_available())):
                z_t, s = model(tac)
                z_x_list = [F.normalize(z.to(device).float(), dim=-1) for z in batch["z_x_list"]]
                all_text = torch.cat(z_x_list, dim=0)
                loss = improved_multi_pos_infonce(z_t, z_x_list, all_text, s, method="simple") / cfg["accum_steps"]
            scaler.scale(loss).backward()
            if (it+1) % cfg["accum_steps"] == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            tot += loss.item()*cfg["accum_steps"]
        print(f"[Stage1] epoch {ep+1}: loss={tot/len(dl):.4f}")
    os.makedirs("runs", exist_ok=True)
    torch.save({"model": model.state_dict()}, "runs/ckpt_stage1.pt")
    print("Saved runs/ckpt_stage1.pt")

if __name__ == "__main__":
    main()
