#!/bin/bash

# 设置基础目录路径
BASE_DIR="datasets/mipnerf360"
OUTPUT_BASE_DIR="datasets/mipnerf360_nerf"

# 创建输出基础目录
mkdir -p "$OUTPUT_BASE_DIR"

for dir in "$BASE_DIR"/*/ ; do
    # 获取文件夹名称
    dirname=$(basename "$dir")
    
    # 跳过如果不是目录
    if [ ! -d "$dir" ]; then
        continue
    fi
    
    # 检查必要的输入文件是否存在
    if [ ! -d "$BASE_DIR/$dirname/images_4" ] || [ ! -d "$BASE_DIR/$dirname/sparse/0" ]; then
        echo "Skipping $dirname: missing images or sparse/0 directory"
        continue
    fi
    
    # 执行转换命令
    echo "Processing: $dirname"
    python 00.colmap2nerf.py \
        --images "$BASE_DIR/$dirname/images" \
        --text "$BASE_DIR/$dirname/sparse/0" \
        --out "$OUTPUT_BASE_DIR/$dirname" \
        --n_train 4 \
        --softlink
    
    echo "Completed processing $dirname"
    echo "Output saved to: $OUTPUT_BASE_DIR/$dirname"
    echo "------------------------"
done

echo "All processing completed!"
echo "Results saved in: $OUTPUT_BASE_DIR"