# Ceramic Classification and Visualization

A comprehensive pipeline for analyzing and visualizing Chinese ceramics from the National Palace Museum using deep learning feature extraction and dimensionality reduction techniques.

## Overview

This project processes ceramic artifacts from the NPM digital collection, extracting visual features using a VAE (Variational Autoencoder) model, and generates multiple visualization types to explore patterns across different classification schemes: dynasty, shape, glaze, decoration, and kiln.

## Project Structure

```
.
├───build_dataset.py          # Dataset construction and sampling
├───clusters.py               # UMAP clustering visualization
├───download_picture.py       # Image downloading utility
├───extract_features.py       # VAE feature extraction
├───meanobject.py             # Mean object generation per class
├───run.sh                    # Complete pipeline execution script
├───visual_pca.py             # PCA-based visual grid generation
│
├───analyze_data/             # Data analysis utilities on specific field
│   ├───last_char.py          # Last character counting
│   ├───ngrams.py             # N-gram analysis
│   ├───suffix_ngrams.py      # N-gram analysis with specific suffix
│   └───value_count.py        # Value frequency counting
│
├───data/                     # Generated datasets (5 JSON files)
├───picture/                  # Downloaded ceramic images
├───features/                 # Extracted features and PCA results
└───visualize/                # Generated visualizations
```

## Classification Methods

The pipeline supports five classification schemes:

1. **Dynasty** (`dynasty`) - 18 classes: 漢, 北宋, 南宋, 金, 元, 明 (6 periods), 清 (6 periods)
2. **Shape** (`shape`) - 24 classes: 碗, 碟, 洗, 觚, 管, 盤, 壺, etc.
3. **Glaze** (`glaze`) - 20 classes: 茄皮紫釉, 孔雀綠釉, 松石綠釉, etc.
4. **Decoration** (`decoration`) - 36 classes: 花卉紋, 雲龍紋, 番蓮紋, etc.
5. **Kiln** (`kiln`) - 14 classes: 定窯, 官窯, 鈞窯, 哥窯, etc.

## Requirements

```bash
pip install torch torchvision diffusers pillow numpy scikit-learn umap-learn matplotlib requests tqdm
```

## Quick Start

### Run Complete Pipeline

```bash
bash run.sh
```

This script will:
1. Download raw data from NPM Open API
2. Build filtered datasets for all classification methods
3. Download images for each dataset
4. Extract VAE features and perform PCA
5. Generate UMAP clusters and visualizations
6. Create mean objects and PCA grids for each class

### Run Individual Steps

#### 1. Build Dataset

```bash
python build_dataset.py
```

- Downloads ceramics data from NPM API
- Filters out entries without images
- Creates 5 JSON files in `./data/` directory
- Samples up to 100 items per class using fixed-interval sampling

#### 2. Download Images

```bash
python download_picture.py --method shape
```

Downloads images for the specified classification method to `./picture/` directory.

#### 3. Extract Features

```bash
python extract_features.py --method shape
```

- Uses Stable Diffusion VAE (`stabilityai/sd-vae-ft-mse`) to extract latent features
- Applies StandardScaler normalization
- Performs PCA dimensionality reduction (default: 50 components)
- Saves raw and PCA features to `./features/{method}/`

#### 4. Generate UMAP Clusters

```bash
python clusters.py --method shape
```

- Creates 2D UMAP projections of PCA features
- Generates two visualizations:
  - `umap_scatter.png` - Scatter plot with all data points
  - `umap_centroids.png` - Class centroid positions
- Saves class mapping to `class_mapping.json`

#### 5. Generate Mean Objects

```bash
python meanobject.py --method shape
```

- Computes mean latent vector for each class
- Decodes mean vectors back to images using VAE
- Applies contrast (2.0x) and sharpness (5.0x) enhancement
- Saves to `./visualize/{method}/mean_object/`

#### 6. Generate PCA Visual Grids

```bash
python visual_pca.py --method shape
```

- Creates 4×4 grids exploring PCA dimensions 1 and 2
- Each grid shows variations around the class mean
- Applies contrast (1.2x) and sharpness (5.0x) enhancement
- Saves to `./visualize/{method}/pca_grid/`

## Configuration

### Key Parameters

**`extract_features.py`:**
- `PCA_COMPONENTS`: Number of PCA dimensions (default: 50)
- `MODEL_NAME`: VAE model (`stabilityai/sd-vae-ft-mse`)
- `DEVICE`: CPU or CUDA

**`clusters.py`:**
- `UMAP_N_NEIGHBORS`: 200
- `UMAP_MIN_DIST`: 0.25
- `UMAP_SPREAD`: 1.5
- `UMAP_METRIC`: "cosine"

**`visual_pca.py`:**
- `GRID_SIZE`: 4 (generates 4×4 grids)
- `GRID_STEP`: 20 (step size in PCA space)

**`build_dataset.py`:**
- Maximum samples per class: 100 (fixed-interval sampling)

## Output Structure

```
./data/
  ├── dynasty.json
  ├── shape.json
  ├── glaze.json
  ├── decoration.json
  └── kiln.json

./features/{method}/
  ├── features.npz              # Raw VAE features
  ├── pca_features.npz          # PCA-reduced features
  ├── pca_components.npy        # PCA transformation matrix
  ├── scaler_mean.npy          # Normalization parameters
  └── scaler_scale.npy

./visualize/{method}/
  ├── umap_scatter.png
  ├── umap_centroids.png
  ├── class_mapping.json
  ├── mean_object/
  │   └── {class_name}.png
  └── pca_grid/
      └── {class_name}.png
```

## Data Analysis Tools

Located in `./analyze_data/`:

- **`value_count.py`** - Counts unique values in specified fields
- **`last_char.py`** - Analyzes distribution of last characters
- **`ngrams.py`** - Performs n-gram analysis on text fields
- **`suffix_ngrams.py`** - N-gram analysis for specific suffix patterns

## Technical Details

### Feature Extraction

- **Model**: Stable Diffusion VAE encoder
- **Input**: 512×512 RGB images, normalized to [-1, 1]
- **Output**: 16,384-dimensional latent vectors (4×64×64)
- **Processing**: StandardScaler normalization + PCA reduction

### Visualization Methods

1. **UMAP Clustering**: Reveals class separability and inter-class relationships
2. **Mean Objects**: Represents average visual characteristics per class
3. **PCA Grids**: Explores latent space variations along principal components

## Notes

- Images without valid URLs are automatically filtered out
- Fixed-interval sampling ensures representative coverage across classes
- All visualizations use high-resolution output (300 DPI)
- The pipeline is GPU-accelerated when CUDA is available

## Citation

Data source: National Palace Museum Open Data API
- URL: https://odapi.npm.gov.tw/data/open/api/v1/digitalCollection/ceramics.json

## License

This project is for research and educational purposes. Please respect the National Palace Museum's data usage terms.