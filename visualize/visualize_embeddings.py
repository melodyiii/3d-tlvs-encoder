import os
import json
import argparse
import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

# 降维和可视化库
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# 导入模型
import sys
sys.path.append('.') # 将项目根目录添加到Python路径
from models.tlv_student import TactileEncoder

def visualize_embeddings(
    ckpt_path="runs/ckpt_stage2_vision.pt",
    num_samples=20, # 选择可视化样本的数量，太多会很乱
    method='umap',  # 'umap' 或 'tsne'
    text_type='phrases', # 'phrases' 或 'sentences'
    seed=42
):
    """
    主函数：加载数据、降维并可视化
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # =========================================================
    # 1. 加载模型和特征库
    # =========================================================
    # 加载触觉编码器 (学生)
    model = TactileEncoder()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"])
    model.eval().to(device)
    print(f"已加载学生模型: {ckpt_path}")

    # 加载文本和视觉特征库 (老师)
    text_repo = torch.load("teachers/text_embed.pt", map_location="cpu")
    visual_repo = torch.load("teachers/visual_embed.pt", map_location="cpu")
    
    # =========================================================
    # 2. 为 N 个样本提取三模态特征
    # =========================================================
    items = sorted([d for d in glob("dataset/seq_*") if os.path.isdir(d)])
    if num_samples > len(items):
        num_samples = len(items)
    
    # 为了结果可复现，随机选择 N 个样本
    np.random.seed(seed)
    selected_items = np.random.choice(items, num_samples, replace=False)

    all_features = []
    all_modalities = []
    all_object_ids = []

    print(f"正在从 {num_samples} 个样本中提取特征...")
    for item_path in tqdm(selected_items):
        sample_id = os.path.basename(item_path)
        
        # 提取触觉特征 (z_t)
        tac_data = torch.from_numpy(np.load(f"{item_path}/tactile.npy")).float()[:16].unsqueeze(0).to(device)
        with torch.no_grad():
            z_t, _ = model(tac_data)
            z_t = z_t.cpu().numpy().flatten()

        # 提取视觉特征 (z_v)
        z_v = visual_repo[sample_id].numpy().flatten()

        # 提取文本特征 (z_text)
        j = json.load(open(f"{item_path}/text.json"))
        text = j[text_type][0] # 使用第一个短语/句子作为代表
        z_text = text_repo[text_type][text].numpy().flatten()

        # 存入列表
        all_features.extend([z_t, z_v, z_text])
        all_modalities.extend(["Tactile", "Visual", "Text"])
        all_object_ids.extend([sample_id] * 3)

    features_np = np.array(all_features)

    # =========================================================
    # 3. 执行降维 (t-SNE 或 UMAP)
    # =========================================================
    print(f"正在使用 {method.upper()} 进行降维...")
    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=15, random_state=seed, init='pca', learning_rate='auto')
    else: # 默认使用 UMAP
        # UMAP 通常更快且能更好地保留全局结构
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=seed)

    features_2d = reducer.fit_transform(features_np)

    # =========================================================
    # 4. 绘图
    # =========================================================
    df = pd.DataFrame({
        'x': features_2d[:, 0],
        'y': features_2d[:, 1],
        'object_id': all_object_ids,
        'modality': all_modalities
    })
    
    print("正在生成图像...")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 10))
    
    # 使用 seaborn 绘制精美的散点图
    # hue (颜色) 用于区分不同物体
    # style (形状) 用于区分不同模态
    sns.scatterplot(
        data=df,
        x='x',
        y='y',
        hue='object_id',
        style='modality',
        s=150, # 点的大小
        alpha=0.8
    )
    
    plt.title(f'{method.upper()} Visualization of Multimodal Embeddings', fontsize=16)
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0) # 图例放到图外面
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # 调整布局给图例留出空间
    
    # 保存图像
    output_path = f"viz/{method}_visualization.png"
    os.makedirs("viz", exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"图像已保存到: {output_path}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Multimodal Embeddings')
    parser.add_argument('--ckpt', type=str, default='runs/ckpt_stage2_vision.pt', help='Path to the model checkpoint')
    parser.add_argument('--samples', type=int, default=15, help='Number of samples to visualize')
    parser.add_argument('--method', type=str, default='umap', choices=['umap', 'tsne'], help='Dimensionality reduction method')
    args = parser.parse_args()
    
    visualize_embeddings(
        ckpt_path=args.ckpt,
        num_samples=args.samples,
        method=args.method
    )