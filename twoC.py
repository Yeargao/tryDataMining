import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import hdbscan
import turnToNum

# 引入機器學習評估指標與隨機森林
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import adjusted_rand_score, silhouette_score


def evaluate_random_forest(features, labels, label_name):
    """
    功能 1: 使用 Random Forest 進行分類，並計算準確率/召回率/精確率/f1-score
    """
    print(f"\n>>> 正在執行 Random Forest 分類 (使用 Label: {label_name}) <<<")

    # 切分訓練集與測試集 (80% 訓練, 20% 測試)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 初始化並訓練隨機森林
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # 預測
    y_pred = rf.predict(X_test)

    # 計算指標 (使用 average='macro' 或 'weighted' 來處理多分類問題)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"[{label_name} - Random Forest 結果]")
    print(f"  - 準確率 (Accuracy):  {acc:.4f}")
    print(f"  - 精確率 (Precision): {prec:.4f} (weighted)")
    print(f"  - 召回率 (Recall):    {rec:.4f} (weighted)")
    print(f"  - F1-Score:           {f1:.4f} (weighted)")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def evaluate_hdbscan_metrics(features, cluster_labels, true_labels, label_name):
    """
    功能 2 & 3: 計算 HDBSCAN 的 分群準確率 / ARI / 輪廓係數
    """
    print(f"\n>>> 正在計算 HDBSCAN 分群指標 (對比 True Label: {label_name}) <<<")

    # 1. 調整蘭德係數 (ARI) - 最適合評估分群與真實標籤匹配度的指標
    ari = adjusted_rand_score(true_labels, cluster_labels)

    # 2. 輪廓係數 (Silhouette Score) - 評估分群本身的幾何緊湊度 (與 True Label 無關，僅算一次即可)
    # 注意：HDBSCAN 會產生雜訊 (-1)，計算輪廓係數時通常需排除雜訊，且資料量過大時會很慢
    non_noise_mask = cluster_labels != -1
    if np.sum(non_noise_mask) > 1 and len(np.unique(cluster_labels[non_noise_mask])) > 1:
        sil = silhouette_score(features[non_noise_mask], cluster_labels[non_noise_mask], metric='euclidean')
    else:
        sil = float('nan')  # 只有一群或全是雜訊時無法計算

    # 3. 分群準確率 (Cluster Accuracy / Purity)
    # 概念：找出每個 Cluster 裡數量最多的真實 Label 是什麼，以此當作該 Cluster 的預測值
    cluster_to_true_mapping = {}
    unique_clusters = np.unique(cluster_labels)

    # 排除雜訊點，只計算有分到群的準確率
    valid_mask = cluster_labels != -1
    if np.sum(valid_mask) > 0:
        for c in unique_clusters:
            if c == -1: continue
            # 找出這群中真實 Label 出現次數最多的
            majority_label = true_labels[cluster_labels == c].mode()[0]
            cluster_to_true_mapping[c] = majority_label

        # 對對應好的預測標籤計算準確率
        pred_labels_from_cluster = true_labels.copy()
        # 沒分到群的當作預測錯誤（或另外處理，這裡將其指派為一個不可能對的空值）
        pred_labels_from_cluster = cluster_labels.map(cluster_to_true_mapping).fillna("NOISE_OR_UNMATCHED")

        acc = accuracy_score(true_labels, pred_labels_from_cluster)
    else:
        acc = 0.0

    print(f"[{label_name} - HDBSCAN 結果]")
    print(f"  - 分群準確率 (Purity Accuracy): {acc:.4f}")
    print(f"  - 調整蘭德係數 (ARI):           {ari:.4f} (-1 到 1，越接近 1 越好)")
    print(f"  - 輪廓係數 (Silhouette Score):  {sil:.4f} (排除雜訊後計算)")

    return {"cluster_accuracy": acc, "ari": ari, "silhouette": sil}


def analyze_clusters(df):
    # 檢查欄位是否存在，若名稱不符請根據您的資料調整
    # 假設前兩欄是 title 與類別，此處確保我們要用的標籤欄位都在
    label_cols = ['new_label', 'new_category_label']
    for col in label_cols:
        if col not in df.columns:
            # 容錯處理：如果您的欄位名稱叫 'category' 而不是 'new_category_label'，自動幫您遞補
            if col == 'new_category_label' and 'category' in df.columns:
                df['new_category_label'] = df['category']
            elif col == 'new_label' and 'label' in df.columns:
                df['new_label'] = df['label']
            else:
                raise KeyError(f"在資料夾中找不到需要的標籤欄位: {col}，請檢查原始資料的 Column name")

    # 提取特徵（排除前兩欄文字，或是動態排除非數值欄位）
    # 這裡建議使用 select_dtypes 確保只拿數值特徵
    features_df = df.select_dtypes(include=[np.number])
    # 如果 'cluster' 欄位先前存在，先排除它
    if 'cluster' in features_df.columns:
        features_df = features_df.drop(columns=['cluster'])
    features = features_df.values

    # ----------------------------------------------------
    # 功能 1 & 3: Random Forest 分類評估 (分別針對兩種 Label)
    # ----------------------------------------------------
    evaluate_random_forest(features, df['new_label'], 'new_label')
    evaluate_random_forest(features, df['new_category_label'], 'new_category_label')

    # ----------------------------------------------------
    # 原本功能: HDBSCAN 分群
    # ----------------------------------------------------
    print("\n正在進行 HDBSCAN 分群...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=5, gen_min_span_tree=True)
    cluster_labels = clusterer.fit_predict(features)
    df['cluster'] = cluster_labels

    # ----------------------------------------------------
    # 功能 2 & 3: HDBSCAN 指標計算 (分別針對兩種 Label)
    # ----------------------------------------------------
    evaluate_hdbscan_metrics(features, df['cluster'], df['new_label'], 'new_label')
    evaluate_hdbscan_metrics(features, df['cluster'], df['new_category_label'], 'new_category_label')

    # 3. 繪製樹狀圖 (Condensed Tree)
    print("\n正在繪製樹狀圖...")
    plt.figure(figsize=(10, 6))
    clusterer.condensed_tree_.plot(select_clusters=True,
                                   selection_palette=sns.color_palette('deep', 20))
    plt.title("HDBSCAN Condensed Tree")
    plt.show()

    # 4. 分析每群包含的子類別 (subcategory)
    print("\n--- 各群集子類別組成分析 ---")
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
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1],
                          c=cluster_labels, cmap='Spectral', s=10, alpha=0.7)

    plt.colorbar(scatter, label='Cluster ID')
    plt.title('HDBSCAN Clustering Results (UMAP Projection)')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')

    plt.savefig("hdbscan_cluster_umap.png", dpi=300, bbox_inches='tight')
    print("圖片已儲存為 hdbscan_cluster_umap.png")
    plt.show()


if __name__ == "__main__":
    print("正在讀取資料...")
    df = pd.read_csv('recipe_vectors.csv')
    if input("直接分類/群打0"):
        df = turnToNum.turnWhat(df)
    analyze_clusters(df)