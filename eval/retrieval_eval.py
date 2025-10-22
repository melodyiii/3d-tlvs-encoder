import os, json, argparse, csv, torch, numpy as np
import torch.nn.functional as F
from glob import glob
from tqdm import tqdm
from models.tlv_student import TactileEncoder

def recall_at_k(sim, labels, k):
    """计算 Recall@K 指标"""
    # sim: [B, N], labels: [B]
    topk = sim.topk(k, dim=1).indices  # [B, k]
    # 检查每个查询的正确标签是否存在于其top-k结果中
    hit = sum([labels[i] in topk[i].tolist() for i in range(sim.size(0))])
    return hit / sim.size(0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="runs/ckpt_stage2_vision.pt", help="模型检查点路径")
    # 【新增】评测模式参数
    ap.add_argument(
        "--mode", type=str, default="t2t", choices=["t2t", "t2v", "v2t"],
        help="评测模式: t2t (触觉->文本), t2v (触觉->视觉), v2t (视觉->文本)"
    )
    ap.add_argument("--bank", type=str, default="phrases", choices=["phrases", "sentences"], help="使用的文本库类型")
    ap.add_argument("--csv", type=str, default="", help="CSV输出路径，留空则不保存")
    ap.add_argument("--T", type=int, default=16, help="触觉序列长度")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[信息] 使用设备: {device}")

    # =========================================================
    # 1. 加载所有需要的模型和数据 (文本、视觉、触觉)
    # =========================================================
    # 加载文本库
    text_repo = torch.load("teachers/text_embed.pt", map_location="cpu")
    bank_texts = sorted(list(text_repo[args.bank].keys()))
    Zx_bank = F.normalize(torch.stack([text_repo[args.bank][t] for t in bank_texts]), dim=-1).to(device) # [N_text, D]

    # 加载视觉库
    visual_repo = torch.load("teachers/visual_embed.pt", map_location="cpu")

    # 加载触觉编码器
    enc = TactileEncoder()
    enc.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    enc.eval().to(device)

    # =========================================================
    # 2. 一次性为整个数据集提取所有特征和标签
    # =========================================================
    Zt_list, Zv_list, text_labels, seq_names = [], [], [], []
    items = sorted([d for d in glob("dataset/seq_*") if os.path.isdir(d)])
    
    print(f"[信息] 正在从 {len(items)} 个样本中提取特征...")
    for d in tqdm(items):
        seq_name = os.path.basename(d)
        seq_names.append(seq_name)

        # 提取触觉特征 Zt
        tac = torch.from_numpy(np.load(f"{d}/tactile.npy")).float()[:args.T].unsqueeze(0).to(device)
        with torch.no_grad():
            z_t, _ = enc(tac)
            Zt_list.append(z_t)

        # 提取视觉特征 Zv
        Zv_list.append(visual_repo[seq_name])

        # 提取文本标签 (在文本库中的索引)
        j = json.load(open(f"{d}/text.json"))
        pick = j["phrases"][0] if args.bank == "phrases" else j["sentences"][0]
        text_labels.append(bank_texts.index(pick))

    # 将列表转换为Tensor
    Zt = torch.cat(Zt_list, 0) # [B, D]
    Zv = F.normalize(torch.stack(Zv_list, 0), dim=-1).to(device) # [B, D]
    text_labels = torch.tensor(text_labels, device=device) # [B]
    
    # =========================================================
    # 3. 根据评测模式，选择 Query, Bank 和 Labels
    # =========================================================
    if args.mode == "t2t":
        print("[信息] 评测模式: 触觉 -> 文本 (Tactile -> Text)")
        Z_query, Z_bank, labels = Zt, Zx_bank, text_labels
        query_names, bank_names = seq_names, bank_texts
    elif args.mode == "t2v":
        print("[信息] 评测模式: 触觉 -> 视觉 (Tactile -> Vision)")
        Z_query, Z_bank = Zt, Zv
        # 在此模式下，第i个触觉样本的目标就是第i个视觉样本
        labels = torch.arange(len(items), device=device)
        query_names, bank_names = seq_names, seq_names
    elif args.mode == "v2t":
        print("[信息] 评测模式: 视觉 -> 文本 (Vision -> Text)")
        Z_query, Z_bank, labels = Zv, Zx_bank, text_labels
        query_names, bank_names = seq_names, bank_texts
    else:
        raise ValueError("无效的评测模式")

    # =========================================================
    # 4. 计算相似度并评估 Recall@K
    # =========================================================
    sim = Z_query @ Z_bank.t() # [B_query, N_bank]

    r1 = recall_at_k(sim, labels, 1)
    r5 = recall_at_k(sim, labels, 5)
    r10 = recall_at_k(sim, labels, 10)
    print(f"[{args.mode.upper()} Eval:{args.bank}] R@1/5/10 = {r1:.3f} / {r5:.3f} / {r10:.3f}")

    # =========================================================
    # 5. 如果需要，保存CSV结果文件
    # =========================================================
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="", encoding='utf-8') as f:
            w = csv.writer(f)
            # 【修改】动态设置CSV表头
            w.writerow(["query_id", "ground_truth", "R1_hit", "top5_retrieved_items"])
            
            top5_indices = sim.topk(5, dim=1).indices.tolist()
            labels_list = labels.tolist()

            for i, idxs in enumerate(top5_indices):
                query_id = query_names[i]
                gt_label_idx = labels_list[i]
                ground_truth = bank_names[gt_label_idx]
                r1_hit = int(idxs[0] == gt_label_idx)
                # 将top5的索引转换为可读的名称
                top5_names = "|".join(bank_names[j] for j in idxs)
                w.writerow([query_id, ground_truth, r1_hit, top5_names])
        print(f"CSV 结果已保存 -> {args.csv}")

if __name__ == "__main__":
    main()
    