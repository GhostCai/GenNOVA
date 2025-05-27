import os
import subprocess

# Define the base directories
input_base = "datasets/mipnerf360_nerf"
output_base = "datasets/mipnerf360_nerf_intp5"

# Get all scene directories
scene_dirs = os.listdir(input_base)

for scene_name in scene_dirs:
    # Construct the input JSON path
    json_path = os.path.join(input_base, scene_name, "transforms_train.json")
    
    # Construct the output directory path
    output_dir = os.path.join(output_base, scene_name)
    
    # Skip if the input JSON doesn't exist
    if not os.path.exists(json_path):
        print(f"Skipping {scene_name}: transforms_train.json not found")
        continue
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct the command
    cmd = [
        "python",
        "02.naive_baseline_export.py",
        "--json_path", json_path,
        "--output_dir", output_dir
    ]
    
    # Execute the command
    print(f"Processing scene: {scene_name}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error processing {scene_name}: {e}")

print("Processing completed!")