#!/bin/bash

# 设置基础目录路径
BASE_DIR="datasets/mipnerf360"

for dir in "$BASE_DIR"/*/ ; do
    # 获取文件夹名称
    dirname=$(basename "$dir")
    
    # 执行转换命令
    echo "Processing: $dirname"
    python 00.colmap2nerf.py --images "$BASE_DIR/$dirname/images" --text "$BASE_DIR/$dirname/sparse/0" --out "$BASE_DIR/$dirname/transforms.json"
    
    echo "Completed processing $dirname"
    echo "------------------------"
done
