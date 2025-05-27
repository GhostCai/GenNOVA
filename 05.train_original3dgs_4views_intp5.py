import subprocess
from pathlib import Path
import argparse

# # Scenes to process
# scenes = ["bicycle","bonsai", "counter", "garden", "kitchen", "room", "stump"]

# # Resolve paths to absolute
# base_dataset_path = Path("datasets/mipnerf360_nerf_intp5").resolve()
# base_model_path = Path("/input0/experiments/original3dgs_4views_intp5").resolve()


# Scenes to process
scenes = ["lego_sparse"]

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train 3DGS with specified paths')
parser.add_argument('--dataset_path', type=str, default="datasets/nerf_sparse_intp5",
                    help='Base dataset path')
parser.add_argument('--model_path', type=str, default="/input0/experiments/original3dgs_nerf_sparse_intp5",
                    help='Saved 3dgs model path')
args = parser.parse_args()

# Resolve paths to absolute
base_dataset_path = Path(args.dataset_path).resolve()
base_model_path = Path(args.model_path).resolve()

project_dir = Path("third_party/gaussian-splatting").resolve()

#sanity check
for scene in scenes:
    dataset_path = base_dataset_path / scene
    model_path = base_model_path / scene
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

# Commands to run per scene
for scene in scenes:
    dataset_path = base_dataset_path / scene
    model_path = base_model_path / scene

    print(f"\n=== Processing Scene: {scene} ===")
    print(f"Dataset Path: {dataset_path}")
    print(f"Model Path:   {model_path}\n")

    commands = [
        ["python", "train.py", "-s", str(dataset_path), "-m", str(model_path)],
        ["python", "render.py", "-m", str(model_path)],
        ["python", "metrics.py", "-m", str(model_path)]
    ]

    for cmd in commands:
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=project_dir)