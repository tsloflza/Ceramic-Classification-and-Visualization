import json
import re
from collections import Counter

# ===== 手動設定 =====
input_file_path = "raw_data/ceramics.json"
output_file_path = "raw_data/ngrams.txt"
target_field = "desc"
# ===================

def generate_ngrams_from_token(token, n):
    """從單一 token（已無空白與標點）產生字元 n-gram"""
    return [token[i:i+n] for i in range(len(token) - n + 1)] if len(token) >= n else []

def clean_and_split(text):
    """
    清理文本：
    - 移除多餘空白
    - 以中英文標點與空白分割
    - 過濾掉空字串
    """
    # 標點與空白當作分隔符
    # 包含常見中英文標點符號
    tokens = re.split(r"[，。、．！？；：「」『』（）()【】［］《》〈〉“”‘’·—\-~‧,.!?;:\"'()\[\]{}<> \t\n\r]+", text.strip())
    return [t for t in tokens if t]  # 去除空字串

def main():
    with open(input_file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    # 移除「沒有圖片」的樣本
    NO_IMAGE_URL = "https://digitalarchive.npm.gov.tw/Image/GetImage?ImageId=0&randomCode=0"
    data = [item for item in raw_data if item.get("imageUrl_m") != NO_IMAGE_URL]

    texts = [item.get(target_field, "") for item in data if item.get(target_field)]

    results = {}
    for n in range(1, 5):
        counter = Counter()
        for t in texts:
            tokens = clean_and_split(t)
            for tok in tokens:
                ngrams = generate_ngrams_from_token(tok, n)
                counter.update(ngrams)
        results[f"{n}-gram"] = counter.most_common(100)

    with open(output_file_path, "w", encoding="utf-8") as f:
        for key, values in results.items():
            f.write(f"=== {key} Top 100 ===\n")
            for term, count in values:
                f.write(f"{term}\t{count}\n")
            f.write("\n")

    print(f"✅ 分析完成，結果輸出至 {output_file_path}")

if __name__ == "__main__":
    main()
