import os
import json
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import LabelEncoder

# ===== args =====
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="shape",
                    help="分類方法名稱，decoraction / dynasty / glaze / kiln / shape")
args = parser.parse_args()
CLASSIFICATION_METHOD = args.method
# =====================

# ===== 手動設定 =====

FEATURE_FILE = f"./features/{CLASSIFICATION_METHOD}/pca_features.npz"  # 使用 PCA 特徵
OUTPUT_DIR = f"./visualize/{CLASSIFICATION_METHOD}"
UMAP_N_NEIGHBORS = 200
UMAP_MIN_DIST = 0.25
UMAP_SPREAD = 1.5
UMAP_METRIC = "cosine"
NUM_COLORS = 20
MARKERS = ['o', 's', '^', 'v', 'D', 'P', 'X']

# ===================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    SCATTER_FILE = os.path.join(OUTPUT_DIR, "umap_scatter.png")
    CENTROIDS_FILE = os.path.join(OUTPUT_DIR, "umap_centroids.png")
    CLASS_MAPPING_FILE = os.path.join(OUTPUT_DIR, "class_mapping.json")

    if not os.path.exists(FEATURE_FILE):
        print(f"❌ 找不到特徵檔：{FEATURE_FILE}")
        return

    # 讀取 PCA 特徵
    data = np.load(FEATURE_FILE, allow_pickle=True)
    X = data['features']
    labels = data['labels'].astype(str)
    class_names = data['class_names'].astype(str)

    print(f"📥 載入 PCA 特徵: {X.shape[0]} samples, {X.shape[1]} dims")

    # 標籤編碼
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    num_classes = len(encoder.classes_)

    # 儲存類別對應表
    class_mapping = {str(i): c for i, c in enumerate(encoder.classes_)}
    with open(CLASS_MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, ensure_ascii=False, indent=4)

    # UMAP 降維
    reducer = umap.UMAP(n_components=2,
                        n_neighbors=UMAP_N_NEIGHBORS,
                        min_dist=UMAP_MIN_DIST,
                        spread=UMAP_SPREAD,
                        metric=UMAP_METRIC,
                        random_state=42)
    X_umap = reducer.fit_transform(X)
    print("✅ UMAP 完成")

    # 計算群中心
    centroids = {i: (X_umap[y==i,0].mean(), X_umap[y==i,1].mean()) for i in range(num_classes)}

    # 繪製散點圖
    plt.figure(figsize=(10,8))
    cmap = plt.cm.get_cmap("tab20", NUM_COLORS)
    for i in range(num_classes):
        idx = y==i
        plt.scatter(X_umap[idx,0], X_umap[idx,1], c=[cmap(i%NUM_COLORS)], marker=MARKERS[i%len(MARKERS)],
                    s=30, alpha=0.7, label=i)
    plt.title(f"UMAP Scatter ({CLASSIFICATION_METHOD})")
    plt.xlabel("UMAP Dim 1")
    plt.ylabel("UMAP Dim 2")
    plt.grid(True, alpha=0.2)
    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(SCATTER_FILE, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 散點圖已儲存：{SCATTER_FILE}")

    # 繪製中心點圖
    plt.figure(figsize=(8,8))
    for i in range(num_classes):
        cx, cy = centroids[i]
        plt.scatter(cx, cy, c=[cmap(i%NUM_COLORS)], marker='o', s=200, alpha=0.9,
                    edgecolors='black', linewidth=1)
        plt.annotate(str(i), (cx, cy), fontsize=10, weight='bold', color='black',
                    ha='center', va='center')
    plt.xlim(X_umap[:,0].min()-0.5, X_umap[:,0].max()+0.5)
    plt.ylim(X_umap[:,1].min()-0.5, X_umap[:,1].max()+0.5)
    plt.title(f"UMAP Centroids ({CLASSIFICATION_METHOD})")
    plt.xlabel("UMAP Dim 1")
    plt.ylabel("UMAP Dim 2")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(CENTROIDS_FILE, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 中心點圖已儲存：{CENTROIDS_FILE}")

if __name__ == "__main__":
    main()
