import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import os

# 確保快取路徑一致
os.environ['HF_HOME'] = "C:/huggingface_cache"
def feature_reduction_pipeline(df):
    # 讀取原始資料
    model = SentenceTransformer('all-MiniLM-L6-v2')

    def cluster_and_label(column_name):
        print(f"\n--- 正在處理特徵: {column_name} ---")

        # 取得不重複的值，並確保轉為字串 list
        unique_vals = df[column_name].astype(str).unique().tolist()

        print(f"正在生成 {len(unique_vals)} 個不重複項目的嵌入向量...")
        # 修正點：確保輸入是 list
        embeddings = model.encode(unique_vals, batch_size=64, show_progress_bar=True)

        # 1. HDBSCAN 分群
        print("正在執行 HDBSCAN 分群...")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2, gen_min_span_tree=True)
        labels = clusterer.fit_predict(embeddings)

        # 2. UMAP 視覺化 (用於輔助命名決策)
        print("正在生成 UMAP 圖形...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        u_emb = reducer.fit_transform(embeddings)

        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(u_emb[:, 0], u_emb[:, 1], c=labels, cmap='Spectral', s=15, alpha=0.6)
        plt.colorbar(scatter, label='Cluster ID')
        plt.title(f"UMAP Clusters for {column_name}")
        plt.show()

        # 3. 顯示每群組成並進行互動式命名
        cluster_map = pd.DataFrame({'val': unique_vals, 'cluster': labels})
        new_labels_dict = {}

        print(f"\n>>> 開始為 {column_name} 的分群命名 <<<")
        for c in sorted(cluster_map['cluster'].unique()):
            members = cluster_map[cluster_map['cluster'] == c]['val'].tolist()
            if c == -1:
                print(f"\n群集 {c} (雜訊/未分類): 包含 {len(members)} 個項目")
                new_labels_dict[c] = f"Other_{column_name}"
                continue

            print(f"\n群集 {c} (包含 {len(members)} 個項目):")
            print(f"範例內容: {', '.join(members[:10])} ...")

            # 互動輸入名稱
            new_name = input(f"請為群集 {c} 定義一個大類名字 (直接按 Enter 則使用預設值): ").strip()
            if not new_name:
                new_name = f"{column_name}_group_{c}"
            new_labels_dict[c] = new_name

        # 4. 對應回原資料表
        # 先建立「原始值 -> 新標籤」的映射字典
        val_to_new_name = dict(zip(cluster_map['val'], cluster_map['cluster'].map(new_labels_dict)))
        return df[column_name].astype(str).map(val_to_new_name)

    # 執行轉化
    df['new_category_label'] = cluster_and_label('category')

    # 輸出檔案
    output_filename = "recipe_reduced_features.csv"
    df.to_csv(output_filename, index=False)
    print(f"\n任務完成！已儲存新特徵檔案至: {output_filename}")

if __name__ == "__main__":
    # 請確保 recipe.csv 在同目錄
    feature_reduction_pipeline('recipe.csv')
#主餐和湯 肉類 甜點 麵包 餅乾