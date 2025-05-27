import os
import subprocess
import shutil
import argparse

# Define the base directories
# input_base = "datasets/mipnerf360_nerf"
# output_base = "datasets/mipnerf360_nerf_intp5"
# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process NeRF datasets with interpolation')
parser.add_argument('--interpolations', type=int, default=5,
                    help='Number of interpolations (default: 5)')
args = parser.parse_args()

input_base = "datasets/nerf_sparse/"
output_base = "datasets/nerf_sparse_intp" + str(args.interpolations)

# Get all scene directories
scene_dirs = os.listdir(input_base)

for scene_name in scene_dirs:
    # Construct paths
    json_path = os.path.join(input_base, scene_name, "transforms_train.json")
    output_dir = os.path.join(output_base, scene_name)
    
    # Skip if the input JSON doesn't exist
    if not os.path.exists(json_path):
        print(f"Skipping {scene_name}: transforms_train.json not found")
        continue
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Execute the interpolation command
    cmd = [
        "python",
        "02.naive_baseline_export.py",
        "--json_path", json_path,
        "--output_dir", output_dir,
        "--num_interpolations", str(args.interpolations)
    ]
    
    print(f"Processing scene: {scene_name}")
    try:
        subprocess.run(cmd, check=True)
        
        # Post-processing
        print(f"Post-processing scene: {scene_name}")
        
        # Remove everything except 'train' folder and transforms_train.json
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if item != 'train' and item != 'transforms_train.json':
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
        
        # Create symbolic links for test folder and transforms_test.json
        input_test_dir = os.path.join(input_base, scene_name, 'test')
        input_test_json = os.path.join(input_base, scene_name, 'transforms_test.json')
        # make it absolute
        input_test_dir = os.path.abspath(input_test_dir)
        input_test_json = os.path.abspath(input_test_json)
        # Define output paths
        output_test_dir = os.path.join(output_dir, 'test')
        output_test_json = os.path.join(output_dir, 'transforms_test.json')
        output_test_dir = os.path.abspath(output_test_dir)
        output_test_json = os.path.abspath(output_test_json)
        
        # Create symbolic links if source files/folders exist
        if os.path.exists(input_test_dir):
            if os.path.exists(output_test_dir):
                os.remove(output_test_dir) if os.path.islink(output_test_dir) else shutil.rmtree(output_test_dir)
            print(f"Creating symlink for test directory: {input_test_dir} -> {output_test_dir}")
            os.symlink(input_test_dir, output_test_dir)
            
        if os.path.exists(input_test_json):
            if os.path.exists(output_test_json):
                os.remove(output_test_json)
            os.symlink(input_test_json, output_test_json)
            
    except subprocess.CalledProcessError as e:
        print(f"Error processing {scene_name}: {e}")
    except Exception as e:
        print(f"Error in post-processing {scene_name}: {e}")

print("Processing and post-processing completed!")