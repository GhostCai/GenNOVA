python3.10 + pytorch 2.6.0
大概要20G的显存

### 环境
```bash

export HF_HOME=./cache
export HF_ENDPOINT=https://hf-mirror.com

# 安装svc
cd third_party/stable-virtual-camera
pip install .
# 安装3dgs的自定义算子
cd ../gaussian-splatting/submodules/diff-gaussian-rasterization
pip install .
cd ../simple-knn/
pip install .

# nerf数据集
cd ../../../datasets/nerf
wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip
unzip nerf_example_data.zip
```

### 数据集预处理
```bash
sudo apt update
sudo apt install colmap
bash 00.colmap2nerf.sh
```
