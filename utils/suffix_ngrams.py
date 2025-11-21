import json
import re
from collections import Counter

# ===== 手動設定 =====
input_file_path = "raw_data/ceramics.json"
target_field = "name"
output_file_path = "raw_data/decoration.txt"
target_char = "紋"    # 要查找結尾字
min_count = 50        # 次數門檻

'''
output_file_path = "raw_data/glaze.txt"
target_char = "釉"
min_count = 50

output_file_path = "raw_data/kiln.txt"
target_char = "窯"
min_count = 20
'''
# ===================

def generate_ngrams_from_token(token, n):
    """從單一 token（已無空白與標點）產生字元 n-gram"""
    return [token[i:i+n] for i in range(len(token) - n + 1)] if len(token) >= n else []

def clean_and_split(text):
    """清理文本並以標點與空白分割"""
    tokens = re.split(r"[，。、．！？；：「」『』（）()【】［］《》〈〉“”‘’·—\-~‧,.!?;:\"'()\[\]{}<> \t\n\r]+", text.strip())
    return [t for t in tokens if t]

def main():
    with open(input_file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    # 移除「沒有圖片」的樣本
    NO_IMAGE_URL = "https://digitalarchive.npm.gov.tw/Image/GetImage?ImageId=0&randomCode=0"
    data = [item for item in raw_data if item.get("imageUrl_m") != NO_IMAGE_URL]

    texts = [item.get(target_field, "") for item in data if item.get(target_field)]

    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(f"=== 以「{target_char}」結尾的 n-gram 統計 ===\n\n")

        # 逐層分析 n-gram
        n = 1
        while True:
            counter = Counter()
            for t in texts:
                tokens = clean_and_split(t)
                for tok in tokens:
                    ngrams = generate_ngrams_from_token(tok, n)
                    for ng in ngrams:
                        if ng.endswith(target_char):
                            counter[ng] += 1

            # 篩選次數 >= min_count
            filtered = [(term, cnt) for term, cnt in counter.items() if cnt >= min_count]
            filtered.sort(key=lambda x: x[1], reverse=True)

            if not filtered:
                break  # 沒有符合條件的，就停止往下

            # 輸出本層結果
            f.write(f"=== {n}-gram (count ≥ {min_count}) ===\n")
            for term, cnt in filtered:
                f.write(f"{term}\t{cnt}\n")
            f.write("\n")

            print(f"✅ 找到 {len(filtered)} 個以「{target_char}」結尾的 {n}-gram（次數 ≥ {min_count}）")

            # 若數量仍大於 50，則繼續往 n+1 gram
            if len(filtered) > 0:
                n += 1
            else:
                break

    print(f"✅ 分析完成，結果輸出至 {output_file_path}")

if __name__ == "__main__":
    main()
