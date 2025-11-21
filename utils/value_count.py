import json
from collections import Counter

# ===== args =====
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="shape",
                    help="分類方法名稱，decoraction / dynasty / glaze / kiln / shape")
args = parser.parse_args()
CLASSIFICATION_METHOD = args.method
# =====================

# ===== 手動設定 =====
input_file_path = f"data/{CLASSIFICATION_METHOD}.json"
output_file_path = f"data/{CLASSIFICATION_METHOD}.txt"
target_field = "class"

'''
input_file_path = "raw_data/ceramics.json"
output_file_path = "raw_data/dynasty.txt"
target_field = "era"
'''
# ===================

def main():
    with open(input_file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    # 移除「沒有圖片」的樣本
    NO_IMAGE_URL = "https://digitalarchive.npm.gov.tw/Image/GetImage?ImageId=0&randomCode=0"
    data = [item for item in raw_data if item.get("imageUrl_m") != NO_IMAGE_URL]

    values = [item.get(target_field, "").strip() for item in data if item.get(target_field)]
    counter = Counter(values)
    most_common = counter.most_common()

    with open(output_file_path, "w", encoding="utf-8") as f:
        for value, count in most_common:
            f.write(f"{value}\t{count}\n")

    print(f"✅ 分析完成，結果輸出至 {output_file_path}")

if __name__ == "__main__":
    main()
