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
OUTPUT_DIR = f"./visualize/{CLASSIFICATION_METHOD}/pca_grid"
FEATURE_FILE = f"./features/{CLASSIFICATION_METHOD}/features.npz"
PCA_FEATURE_FILE = f"./features/{CLASSIFICATION_METHOD}/pca_features.npz"
PCA_COMPONENT_FILE = f"./features/{CLASSIFICATION_METHOD}/pca_components.npy"
SCALER_MEAN_FILE = f"./features/{CLASSIFICATION_METHOD}/scaler_mean.npy"
SCALER_SCALE_FILE = f"./features/{CLASSIFICATION_METHOD}/scaler_scale.npy"

MODEL_NAME = "stabilityai/sd-vae-ft-mse"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRID_SIZE = 4
GRID_STEP = 20  # 調整 PCA 步進距離
# =====================


def decode_latent(latent, vae):
    """Decode latent tensor back to image."""
    latent = torch.from_numpy(latent).float().unsqueeze(0).to(DEVICE)
    latent = latent.view(1, 4, 64, 64)
    with torch.no_grad():
        img = vae.decode(latent).sample
        img = ((img / 2 + 0.5).clamp(0, 1) * 255).cpu().numpy()
        img = img[0].transpose(1, 2, 0).astype(np.uint8)
        pil_img = Image.fromarray(img)

        # 增加對比度 (Contrast)
        enhancer_contrast = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer_contrast.enhance(1.2)

        # 增加銳利度 (Sharpness)
        enhancer_sharpness = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer_sharpness.enhance(5.0)

    return pil_img


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"🧠 Loading VAE model from {MODEL_NAME} ...")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    # === Load data ===
    latent_data = np.load(FEATURE_FILE, allow_pickle=True)
    pca_data = np.load(PCA_FEATURE_FILE, allow_pickle=True)
    pca_components = np.load(PCA_COMPONENT_FILE)
    scaler_mean = np.load(SCALER_MEAN_FILE)
    scaler_scale = np.load(SCALER_SCALE_FILE)

    labels = latent_data["labels"].astype(str)
    latent_features = latent_data["features"]
    class_names = np.unique(labels)

    # === Convert PCA directions back to original latent space ===
    pca_components_orig = pca_components * scaler_scale[np.newaxis, :]

    # === Step size control ===
    latent_std = latent_features.std(axis=0).mean()
    scale_factor = latent_std * GRID_STEP

    print(f"📏 scale_factor = {scale_factor:.6f}")

    for class_name in class_names:
        mask = labels == class_name
        if mask.sum() == 0:
            continue

        mean_latent = latent_features[mask].mean(axis=0)
        imgs = []

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                # X 軸: PCA #1, Y 軸: PCA #2
                scale_i = (i - GRID_SIZE // 2) * scale_factor
                scale_j = (j - GRID_SIZE // 2) * scale_factor
                latent = mean_latent.copy()
                latent += scale_i * pca_components_orig[0] + scale_j * pca_components_orig[1]
                img = decode_latent(latent, vae)
                imgs.append(img)

        # === 合併成 grid ===
        w, h = imgs[0].size
        grid_img = Image.new("RGB", (w * GRID_SIZE, h * GRID_SIZE))
        for idx, img in enumerate(imgs):
            x = (idx % GRID_SIZE) * w
            y = (idx // GRID_SIZE) * h
            grid_img.paste(img, (x, y))

        out_path = os.path.join(OUTPUT_DIR, f"{class_name}.png")
        grid_img.save(out_path)
        print(f"✅ Saved PCA grid: {out_path}")


if __name__ == "__main__":
    main()
