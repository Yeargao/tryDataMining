import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import umap
import turnToGram
import revise_directions
import decreaseFeature

def recipe_vectorization_v2(file_path):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("正在將新標籤轉為語義嵌入...")
    # 使用你自定義過的名字進行 Embedding，這會比原始亂糟糟的食材更有結構
    v1_input = (df['recipe_title'] + " " + df['new_category_label']).tolist()
    v1 = model.encode(v1_input, batch_size=64, show_progress_bar=True)

    v2_input = df['ingredients'].tolist()
    v2 = model.encode(v2_input, batch_size=64, show_progress_bar=True)

    # 合併文字向量並降維
    text_features = np.hstack([v1, v2])
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


if __name__ == "__main__":
    df = pd.read_csv('recipe.csv')
    #list弄成陣列 map把東西換成bool split切資料
    doWhat=list(map(bool,input("要做1不做0: 轉重比 步驟 新類別").split()))
    func_list=[turnToGram.process_weight_ratio,revise_directions.clean_directions_nlp,decreaseFeature.feature_reduction_pipeline]
    feature_list=['ingredients','directions','new_category_label']
    for i in range(len(doWhat)):
        df[feature_list]=func_list[i](df,doWhat[i])
    recipe_vectorization_v2('recipe_reduced_features.csv')