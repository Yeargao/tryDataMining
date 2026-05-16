import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import umap
import turnToGram
import revise_directions
import decreaseFeature
from sklearn.feature_extraction.text import TfidfVectorizer

def recipe_vectorization_v2(df):
    """Convert recipe data to vectors using embeddings and features."""
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("正在將新標籤轉為語義嵌入...")
    # 使用你自定義過的名字進行 Embedding，這會比原始亂糟糟的食材更有結構
    v1_input = (df['recipe_title'] + " " + df['new_category_label']).tolist()
    v1 = model.encode(v1_input, batch_size=64, show_progress_bar=True)

    v2_input = df['ingredients'].tolist()
    v2 = model.encode(v2_input, batch_size=64, show_progress_bar=True)

    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    v3 = tfidf.fit_transform(df['Description'].fillna(''))

    # 合併文字向量並降維
    text_features = np.hstack([v1, v2, v3])
    pca = PCA(n_components=128)  # 調整為主成分
    text_reduced = pca.fit_transform(text_features)

    # 數值型資料處理
    # 假設數值欄位為 'num_ingredients', 'num_steps'
    num_data = df[['num_ingredients', 'num_steps']]
    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(num_data)

    # --- 根據最終維度調權重 ---
    # 邏輯：文字特徵現在有 128 維，數值只有 2 維。
    # 為了不讓數值特徵被淹沒，我們將其加權，使其對 UMAP 的影響力增加
    numerical_weight = 10.0
    num_weighted = num_scaled * numerical_weight

    # 最終合併
    final_matrix = np.hstack([text_reduced, num_weighted])

    # 輸出向量檔
    # 保留 title 和 new_category_label 做為標記，後面接數值向量
    vector_df = pd.concat([
        df[['recipe_title', 'new_category_label']].reset_index(drop=True),
        pd.DataFrame(final_matrix)
    ], axis=1)

    vector_df.to_csv("recipe_vectors.csv", index=False)
    print("成功生成包含自定義特徵與權重調整後的向量檔：recipe_vectors.csv")
    return vector_df


def turnWhat(df):
    # list弄成陣列 map把東西換成bool split切資料
    doWhat = list(map(bool, input("要做1不做0 (轉重比 步驟 新類別): ").split()))
    
    # 定義函數與特徵清單
    func_list = [
        turnToGram.process_weight_ratio,
        revise_directions.clean_directions_nlp,
        decreaseFeature.feature_reduction_pipeline
    ]
    feature_list = ['ingredients', 'directions', 'new_category_label']
    
    # 逐一處理三個特徵
    for i in range(len(doWhat)):
        print(f"\n--- 處理特徵: {feature_list[i]} (doWhat={doWhat[i]}) ---")
        result = func_list[i](df[feature_list[i]], doWhat[i])
        df[feature_list[i]] = result
    
    # 儲存處理後的資料
    df.to_csv('recipe_reduced_features.csv', index=False)
    print("已儲存至 recipe_reduced_features.csv")
    
    # 生成最終向量
    recipe_vectorization_v2(df)
