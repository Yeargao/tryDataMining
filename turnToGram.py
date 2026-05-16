import re
import pandas as pd

# 單位轉換率與物品重量字典
CONVERSION_FACTORS = {
    'gram': 1.0, 'g': 1.0, 'kg': 1000.0, 'ounce': 28.35, 'oz': 28.35,
    'lb': 453.59, 'pound': 453.59, 'cup': 236.0, 'tablespoon': 15.0,
    'tbsp': 15.0, 'tbs': 15.0, 'teaspoon': 5.0, 'tsp': 5.0, 'ml': 1.0, 'l': 1000.0
}
INGREDIENT_WEIGHTS = {'egg': 50.0, 'apple': 150.0, 'onion': 110.0, 'garlic': 5.0, 'lemon': 100.0}


def get_weight(ingredient_str):
    if not isinstance(ingredient_str, str): return 0.0
    text = ingredient_str.lower().strip()

    qty_match = re.search(r'(\d+\s*/\s*\d+|\d+\.?\d*)', text)
    if not qty_match: return 0.0

    raw_qty = qty_match.group(1)
    try:
        quantity = float(eval(raw_qty)) if '/' in raw_qty else float(raw_qty)
    except:
        return 0.0

    for unit, factor in CONVERSION_FACTORS.items():
        if re.search(rf'\b{unit}s?\b', text):  # 加上 s? 處理複數如 cups
            return quantity * factor
    for item, weight in INGREDIENT_WEIGHTS.items():
        if item in text: return quantity * weight

    return 0.0


def process_weight_ratio(df):
    print("正在計算材料重量百分比並清洗名稱...")

    def calculate_ratio(raw_list_str):
        # 1. 清洗字串並切分
        items = re.sub(r"[\[\]']", "", raw_list_str).split(',')

        weights = []
        names = []

        # 建立一個用來移除單位的正規表達式模式
        # 例如: \b(gram|g|kg|cup|...)\b
        unit_pattern = r'\b(' + '|'.join(CONVERSION_FACTORS.keys()) + r')s?\b'

        for i in items:
            # A. 取得重量
            w = get_weight(i)
            weights.append(w)

            # B. 取得乾淨名稱
            # 1. 轉小寫
            n = i.lower()
            # 2. 移除數字 (例如 1/2, 1.5, 10)
            n = re.sub(r'(\d+\s*/\s*\d+|\d+\.?\d*)', '', n)
            # 3. 移除單位字眼 (從 CONVERSION_FACTORS 來的)
            n = re.sub(unit_pattern, '', n)
            # 4. 移除多餘空白與標點
            n = n.strip().strip(',').strip()

            names.append(n if n else "unknown")

        # 2. 計算比例
        total_w = sum(weights)
        if total_w == 0: return "Unknown Ratio"

        ratio_strings = []
        for n, w in zip(names, weights):
            percentage = (w / total_w) * 100
            # 輸出格式： 只有材料名: 比例%
            ratio_strings.append(f"{n}: {percentage:.1f}%")

        return ", ".join(ratio_strings)

    df['ingredients'] = df['ingredients'].apply(calculate_ratio)
    return df