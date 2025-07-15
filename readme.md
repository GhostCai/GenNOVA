# GenNOVA

This project implements sparse-view 3D Gaussian Splatting with GenNOVA-enhanced view synthesis for improved geometry reconstruction from minimal input views.

## Requirements
	•	Python 3.10
	•	PyTorch 2.7.1
	•	At least 20GB of GPU memory (e.g., NVIDIA A6000 recommended)

⸻

## Environment Setup

```
export HF_HOME=./cache
export HF_ENDPOINT=https://hf-mirror.com
```

## Install SVC (Stable Virtual Camera)
cd third_party/stable-virtual-camera
pip install .

## Install 3DGS custom operators

```
cd ../gaussian-splatting/submodules/diff-gaussian-rasterization
pip install .

cd ../simple-knn/
pip install .
```

## Dataset Setup (NeRF-Synthetic)

```
cd ../../../datasets/nerf
wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip
unzip nerf_example_data.zip
```


##Data Preprocessing

```
sudo apt update
sudo apt install colmap
bash 00.colmap2nerf.sh
```


(Optional) Speed Up PIL with SIMD

```
pip uninstall pillow
apt-get install libjpeg-dev
apt-get install zlib1g-dev
apt-get install libpng-dev
pip install pillow-simd
```


## Baseline: Original 3DGS (4 views)

```
python 03.train_original3dgs_4views.py
```


## Ours: Sparse View 3DGS + GenNOVA

```
cd third_party/FisherRF-tyk/
python active_train-chair.py
python active_train-lego.py
python active_train-hotdog.py
```
