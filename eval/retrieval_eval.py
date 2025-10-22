
import os, json, argparse, csv, torch, numpy as np
import torch.nn.functional as F
from glob import glob
from models.tlv_student import TactileEncoder

def recall_at_k(sim, labels, k):
    topk = sim.topk(k, dim=1).indices  # [B,k]
    hit = sum([labels[i] in topk[i].tolist() for i in range(sim.size(0))])
    return hit / sim.size(0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="runs/ckpt_stage2.pt")
    ap.add_argument("--bank", type=str, default="phrases", choices=["phrases","sentences"])
    ap.add_argument("--csv", type=str, default="")
    ap.add_argument("--T", type=int, default=16)
    args = ap.parse_args()

    repo = torch.load("teachers/text_embed.pt", map_location="cpu")
    bank_texts = sorted(list(repo[args.bank].keys()))
    bank = F.normalize(torch.stack([repo[args.bank][t] for t in bank_texts]), dim=-1)  # [N,512]

    enc = TactileEncoder()
    enc.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    enc.eval()

    Zt, lbls, seq_names = [], [], []
    items = sorted([d for d in glob("dataset/seq_*") if os.path.isdir(d)])
    for d in items:
        tac = torch.from_numpy(np.load(f"{d}/tactile.npy")).float()[:args.T].unsqueeze(0)
        with torch.no_grad():
            z,_ = enc(tac); Zt.append(z)
        j = json.load(open(f"{d}/text.json"))
        # 标签定义：用该样本的第一条 phrase/sentence 在 bank 中的索引（简化版）
        pick = j["phrases"][0] if args.bank=="phrases" else j["sentences"][0]
        lbls.append(bank_texts.index(pick))
        seq_names.append(os.path.basename(d))

    Zt = torch.cat(Zt,0)                     # [B,512]
    sim = Zt @ bank.t()                      # [B,N]

    r1  = recall_at_k(sim, lbls, 1)
    r5  = recall_at_k(sim, lbls, 5)
    r10 = recall_at_k(sim, lbls, 10)
    print(f"[Eval:{args.bank}] R@1/5/10 = {r1:.3f} / {r5:.3f} / {r10:.3f}")

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["seq","label_text","R1_hit","top5_texts"])
            top5 = sim.topk(5, dim=1).indices.tolist()
            for i, idxs in enumerate(top5):
                label = bank_texts[lbls[i]]
                r1_hit = int(idxs[0]==lbls[i])
                w.writerow([seq_names[i], label, r1_hit, "|".join(bank_texts[j] for j in idxs)])
        print(f"CSV saved -> {args.csv}")

if __name__ == "__main__":
    main()
