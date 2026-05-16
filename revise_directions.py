import re
import spacy
from pandas import read_csv

# 1. 極致精簡模式：除了標記詞性(tagger)和還原原形(lemmatizer)所需的組件外，全部排除
# attribute_ruler 負責處理 token 的屬性映射，是 lemmatizer 的基礎
nlp = spacy.load("en_core_web_sm", exclude=["parser", "ner", "entity_linker", "textcat"])

def clean_directions_nlp(df_column):
    # --- 定義替換規則 (術語統一) ---
    term_map = {
        r"\btoss\b": "mix",
        r"\bfridge\b": "refrigerator",
        r"\bstep \d+\b": "",
    }

    # --- 定義要移除的順序詞 ---
    stop_sequence = {'first', 'then', 'finally', 'next', 'afterward', 'step'}

    def preprocess_text(text):
        if not isinstance(text, str): return ""
        text = text.lower()
        # 執行術語統一與基本清理
        for pattern, repl in term_map.items():
            text = re.sub(pattern, repl, text)
        # 移除計量單位與時間的常見模式 (例如 30 mins, 1 cup)
        text = re.sub(r'\d+\s*(mins?|minutes?|hours?|hrs?|cups?|oz|g|ml|tbsp|tsp)', '', text)
        # 移除純數字
        text = re.sub(r'\b\d+\b', '', text)
        return text

    processed_texts = []

    print("正在進行初步文本清洗...")
    cleaned_input = [preprocess_text(t) for t in df_column]

    # --- 2. 優化批次處理參數 ---
    # batch_size: 降至 200，減少記憶體瞬間壓力
    # n_process: 建議設為 2，避免所有 CPU 核心滿載導致電腦卡死
    print(f"正在透過 spaCy 提取原形動詞 (使用輕量化批次處理)...數據量: {len(cleaned_input)}")

    try:
        # 使用 nlp.pipe 的串流模式
        for doc in nlp.pipe(cleaned_input, batch_size=200, n_process=2):
            # 只保留原形動詞 (VERB) 且不在順序詞清單中
            verbs = [
                token.lemma_ for token in doc
                if token.pos_ == "VERB" and token.lemma_ not in stop_sequence
            ]
            processed_texts.append(" ".join(verbs))
    except Exception as e:
        print(f"處理過程中發生錯誤: {e}")

    return processed_texts