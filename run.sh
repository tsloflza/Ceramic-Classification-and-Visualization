mkdir raw_data
curl -L "https://odapi.npm.gov.tw/data/open/api/v1/digitalCollection/ceramics.json" -o "raw_data/ceramics.json"
python build_dataset.py

python download_picture.py --method decoration
python download_picture.py --method dynasty
python download_picture.py --method glaze
python download_picture.py --method kiln
python download_picture.py --method shape

python extract_features.py --method decoration
python clusters.py --method decoration
python meanobject.py --method decoration
python visual_pca.py --method decoration

python extract_features.py --method dynasty
python clusters.py --method dynasty
python meanobject.py --method dynasty
python visual_pca.py --method dynasty

python extract_features.py --method glaze
python clusters.py --method glaze
python meanobject.py --method glaze
python visual_pca.py --method glaze

python extract_features.py --method kiln
python clusters.py --method kiln
python meanobject.py --method kiln
python visual_pca.py --method kiln

python extract_features.py --method shape
python clusters.py --method shape
python meanobject.py --method shape
python visual_pca.py --method shape