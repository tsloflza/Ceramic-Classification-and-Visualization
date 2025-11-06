import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKL
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ===== args =====
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="shape",
                    help="分類方法名稱，decoraction / dynasty / glaze / kiln / shape")
args = parser.parse_args()
CLASSIFICATION_METHOD = args.method
# =====================

# ===== 手動設定 =====
DATA_FILE = f"./data/{CLASSIFICATION_METHOD}.json"
IMAGE_DIR = "./picture"
OUT_DIR = f"./features/{CLASSIFICATION_METHOD}"
OUT_FILE = os.path.join(OUT_DIR, "features.npz")
PCA_FILE = os.path.join(OUT_DIR, "pca_features.npz")
PCA_COMPONENTS = 50
MODEL_NAME = "stabilityai/sd-vae-ft-mse"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE = 42
# ===================

def load_vae(model_name=MODEL_NAME):
    print(f"🧠 載入 VAE: {model_name} 到 {DEVICE} ...")
    model = AutoencoderKL.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    return model

def get_transform():
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # to [-1,1]
    ])

def extract_feature(img_path, model, transform):
    try:
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            enc = model.encode(tensor)
            latent = enc.latent_dist.mean
            feat = latent.cpu().numpy().reshape(-1)
        return feat
    except Exception as e:
        print(f"⚠️ 無法處理圖片 {img_path}: {e}")
        return None

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    vae = load_vae()
    transform = get_transform()

    features, labels, ids, missing = [], [], [], []

    print(f"📦 開始從 {IMAGE_DIR} 抽取特徵，共 {len(data)} 項...")
    for item in tqdm(data):
        if "class" not in item:
            missing.append(item.get("identifier", "N/A"))
            continue

        img_path = os.path.join(IMAGE_DIR, f"{item['identifier']}.jpg")
        if not os.path.exists(img_path):
            missing.append(item.get("identifier", "N/A"))
            continue

        feat = extract_feature(img_path, vae, transform)
        if feat is not None:
            features.append(feat)
            labels.append(item["class"])
            ids.append(item["identifier"])
        else:
            missing.append(item.get("identifier", "N/A"))

    if not features:
        print("❌ 未抽取到任何特徵，請確認圖片存在並可讀取。")
        return

    features = np.stack(features, axis=0)  # (N, D)
    class_names = sorted(list(set(labels)))

    # === 儲存原始特徵 ===
    np.savez_compressed(
        OUT_FILE,
        features=features,
        labels=np.array(labels, dtype=object),
        ids=np.array(ids, dtype=object),
        class_names=np.array(class_names, dtype=object)
    )
    print(f"✅ 原始特徵已儲存：{OUT_FILE}")

    # === 標準化 + PCA 降維 ===
    print("⚙️ 執行標準化 (StandardScaler) ...")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(features)

    pca_components = min(PCA_COMPONENTS, Xs.shape[1])
    print(f"⚙️ 執行 PCA -> {pca_components} components ...")
    pca = PCA(n_components=pca_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(Xs)

    np.save(os.path.join(OUT_DIR, "pca_components.npy"), pca.components_)
    np.savez_compressed(
        PCA_FILE,
        features=X_pca,
        labels=np.array(labels, dtype=object),
        ids=np.array(ids, dtype=object),
        class_names=np.array(class_names, dtype=object)
    )
    np.save(os.path.join(OUT_DIR, "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(OUT_DIR, "scaler_scale.npy"), scaler.scale_)

    print(f"✅ PCA 特徵已儲存：{PCA_FILE}")

    if missing:
        print(f"⚠️ 有 {len(missing)} 張圖片缺失或無法處理：")
        print(missing[:50])

if __name__ == "__main__":
    main()
