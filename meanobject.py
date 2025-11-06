import os
import numpy as np
from PIL import Image, ImageEnhance
import torch
from diffusers import AutoencoderKL

# ===== args =====
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="shape",
                    help="分類方法名稱，decoraction / dynasty / glaze / kiln / shape")
args = parser.parse_args()
CLASSIFICATION_METHOD = args.method
# =====================

# ===== 手動設定 =====
FEATURE_FILE = f"./features/{CLASSIFICATION_METHOD}/features.npz"
OUTPUT_DIR = f"./visualize/{CLASSIFICATION_METHOD}/mean_object"
MODEL_NAME = "stabilityai/sd-vae-ft-mse"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ===================

def decode_latent(latent, vae):
    """將 latent vector decode 成圖片"""
    latent = torch.from_numpy(latent).float().to(DEVICE)
    latent = latent.view(1, 4, 64, 64)
    with torch.no_grad():
        img = vae.decode(latent).sample
        # 將 [-1,1] 轉回 [0,255]
        img = ((img / 2 + 0.5).clamp(0, 1) * 255).cpu().numpy()
        img = img[0].transpose(1, 2, 0).astype(np.uint8)
        pil_img = Image.fromarray(img)

        # 增加對比度 (Contrast)
        enhancer_contrast = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer_contrast.enhance(2.0)

        # 增加銳利度 (Sharpness)
        enhancer_sharpness = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer_sharpness.enhance(5.0)

    return pil_img


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 載入 VAE
    vae = AutoencoderKL.from_pretrained(MODEL_NAME)
    vae.to(DEVICE)
    vae.eval()

    # 載入 feature 檔案
    data = np.load(FEATURE_FILE, allow_pickle=True)
    features = data['features']       # shape: [N, D]
    labels = data['labels']           # array of strings
    ids = data['ids']                 # identifier list
    class_names = data['class_names'] # array of strings

    print(f"✅ 載入 features: {features.shape}, labels: {labels.shape}")

    for class_name in class_names:
        mask = labels == class_name
        if mask.sum() == 0:
            print(f"⚠️ 類別 {class_name} 沒有資料，跳過")
            continue

        class_features = features[mask]
        mean_object = class_features.mean(axis=0)

        # # 儲存 mean object .npy
        # out_npy = os.path.join(OUTPUT_DIR, f"{class_name}.npy")
        # np.save(out_npy, mean_object)
        # print(f"✅ 儲存 Mean Object NPY: {out_npy}, 樣本數: {class_features.shape[0]}")

        # Decode 成圖片並存 PNG
        try:
            img = decode_latent(mean_object, vae)
            out_png = os.path.join(OUTPUT_DIR, f"{class_name}.png")
            img.save(out_png)
            print(f"✅ 儲存 Mean Object PNG: {out_png}")
        except Exception as e:
            print(f"⚠️ 解碼類別 {class_name} 失敗: {e}")

    print("🎉 全部 Mean Object 已完成計算與儲存")

if __name__ == "__main__":
    main()
