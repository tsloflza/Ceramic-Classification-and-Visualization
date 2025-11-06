import json
from collections import Counter

# ===== 手動設定 =====
input_file_path = "raw_data/ceramics.json"
output_file_path = "raw_data/shape.txt"
target_field = "name"
# ===================

def main():
    with open(input_file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    # 移除「沒有圖片」的樣本
    NO_IMAGE_URL = "https://digitalarchive.npm.gov.tw/Image/GetImage?ImageId=0&randomCode=0"
    data = [item for item in raw_data if item.get("imageUrl_m") != NO_IMAGE_URL]

    last_chars = []
    for item in data:
        text = item.get(target_field, "")
        if text:
            last_chars.append(text.strip()[-1])  # 取最後一字

    counter = Counter(last_chars)
    most_common = counter.most_common(100)

    with open(output_file_path, "w", encoding="utf-8") as f:
        for char, count in most_common:
            f.write(f"{char}\t{count}\n")

    print(f"✅ 分析完成，結果輸出至 {output_file_path}")

if __name__ == "__main__":
    main()
