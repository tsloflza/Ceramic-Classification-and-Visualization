import os
import json
import requests
from tqdm import tqdm

# ===== args =====
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="shape",
                    help="分類方法名稱，decoraction / dynasty / glaze / kiln / shape")
args = parser.parse_args()
CLASSIFICATION_METHOD = args.method
# =====================

INPUT_FILE = f"./data/{CLASSIFICATION_METHOD}.json"
OUTPUT_DIR = "./picture"

def download_image(url, save_path):
    """下載圖片並儲存"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"❌ 無法下載 {url}: {e}")
        return False

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"📘 準備下載 {len(data)} 張圖片...")

    for item in tqdm(data, desc="Downloading", ncols=100):
        img_url = item.get("imageUrl_m")
        identifier = item.get("identifier", "unknown")

        if not img_url:
            continue

        save_path = os.path.join(OUTPUT_DIR, f"{identifier}.jpg")
        if os.path.exists(save_path):
            continue  # 若檔案已存在則略過

        download_image(img_url, save_path)

    print("✅ 下載完成，圖片已儲存至 ./picture/")

if __name__ == "__main__":
    main()
