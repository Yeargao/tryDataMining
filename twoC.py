import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import hdbscan
import turnToNum

def analyze_clusters(df):
    # 提取特徵（排除前兩欄文字）
    features = df.iloc[:, 2:].values
    info_df = df.iloc[:, :2]  # 保存 title 和 new_category_label

    # 2. HDBSCAN 分群
    print("正在進行 HDBSCAN 分群...")
    # min_cluster_size: 最小群集大小，可依資料量調整
    # min_samples: 控制群集保守程度，越大則雜訊越多
    clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=5, gen_min_span_tree=True)
    cluster_labels = clusterer.fit_predict(features)

    df['cluster'] = cluster_labels

    # 3. 繪製樹狀圖 (Condensed Tree)
    print("正在繪製樹狀圖...")
    plt.figure(figsize=(10, 6))
    clusterer.condensed_tree_.plot(select_clusters=True,
                                   selection_palette=sns.color_palette('deep', 20))
    plt.title("HDBSCAN Condensed Tree")
    plt.show()

    # 4. 分析每群包含的子類別 (subcategory)
    print("\n--- 各群集子類別組成分析 ---")
    # 排除雜訊群集 (-1) 的統計
    cluster_summary = df[df['cluster'] != -1].groupby('cluster')['new_category_label'].value_counts()

    for cluster_id in sorted(df['cluster'].unique()):
        if cluster_id == -1:
            print(f"\n群集 {cluster_id} (雜訊/未分類): 共 {len(df[df['cluster'] == -1])} 筆")
            continue

        print(f"\n群集 {cluster_id}:")
        top_subs = df[df['cluster'] == cluster_id]['new_category_label'].value_counts().head(5)
        for sub, count in top_subs.items():
            print(f"  - {sub}: {count} 筆")

    # 5. UMAP 降維並視覺化
    print("\n正在進行 UMAP 降維並儲存結果...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding_2d = reducer.fit_transform(features)

    plt.figure(figsize=(12, 8))
    # 繪製散點圖，顏色根據 HDBSCAN 分群結果
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1],
                          c=cluster_labels, cmap='Spectral', s=10, alpha=0.7)

    plt.colorbar(scatter, label='Cluster ID')
    plt.title('HDBSCAN Clustering Results (UMAP Projection)')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')

    # 儲存圖片
    plt.savefig("hdbscan_cluster_umap.png", dpi=300, bbox_inches='tight')
    print("圖片已儲存為 hdbscan_cluster_umap.png")
    plt.show()


if __name__ == "__main__":
    # 請確保你的檔案名稱為 recipe_vectors.csv
    # 1. 讀取資料
    print("正在讀取資料...")
    # 假設前兩欄是 title, new_category_label，其餘為數值特徵
    df = pd.read_csv('recipe_vectors.csv')
    df=turnToNum.recipe_vectorization_v2(df)
    analyze_clusters(df)