import subprocess
from pathlib import Path

# Scenes to process
scenes = ["lego_sparse"]

# Resolve paths to absolute
base_dataset_path = Path("datasets/nerf/nerf_synthetic").resolve()
base_model_path = Path("/input0/experiments/original3dgs_4views").resolve()
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