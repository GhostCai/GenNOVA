#!/usr/bin/env python3
"""
用SVC在视角之间暴力插值

Usage:
    python render_cli.py \
        --transforms_json path/to/transforms.json \
        --images_dir path/to/images \
        --out_dir path/to/output \
        --num_interp 5 \
        --chunk_strategy interp \
        --cfg 4.0 \
        --camera_scale 1.0 \
        --num_steps 50 \
        --fps 1.0 \
        --seed 23 \
        --device cuda:0
"""
import os
import json
import argparse

import sys
sys.path.append('third_party/stable-virtual-camera')

import numpy as np
import torch
import imageio.v3 as iio
from scipy.spatial.transform import Rotation as R, Slerp

from seva.eval import infer_prior_stats, run_one_scene
from seva.geometry import get_default_intrinsics
from seva.model import SGMWrapper
from seva.utils import load_model
from seva.modules.autoencoder import AutoEncoder
from seva.modules.conditioner import CLIPConditioner
from seva.sampling import DDPMDiscretization, DiscreteDenoiser

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate interpolated views using Stable Virtual Camera"
    )
    parser.add_argument("--transforms_json", type=str, required=True,
                        help="Path to transforms.json (NeRF synthetic format)")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Directory containing the input images")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory to save output frames/video")
    parser.add_argument("--num_interp", type=int, default=5,
                        help="Number of interpolated frames between each input pair")
    parser.add_argument("--chunk_strategy", type=str, default="interp",
                        choices=["interp", "interp-gt"],
                        help="Chunk strategy for sampling")
    parser.add_argument("--cfg", type=float, default=4.0,
                        help="CFG scale for guidance")
    parser.add_argument("--camera_scale", type=float, default=1.0,
                        help="Camera scale factor for generation")
    parser.add_argument("--num_steps", type=int, default=50,
                        help="Number of diffusion steps per frame")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="FPS for output sequence (used if saving video)")
    parser.add_argument("--seed", type=int, default=23,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Torch device to use (e.g., cpu, cuda:0)")
    return parser.parse_args()

def load_data(transforms_json, images_dir, device):
    with open(transforms_json) as f:
        data = json.load(f)

    angle_x = data["camera_angle_x"]
    imgs, c2ws = [], []
    H = W = None
    for frame in data["frames"]:
        # Resolve file path
        fp = frame["file_path"]
        if not os.path.isabs(fp):
            fp = os.path.join(images_dir, fp)
        # Read and normalize image
        img = iio.imread(fp)
        if H is None:
            H, W = img.shape[:2]
        imgs.append(torch.tensor(img / 255.0, dtype=torch.float32)
                    .permute(2, 0, 1)[None])  # 1xCxHxW
        c2ws.append(np.array(frame["transform_matrix"], dtype=np.float32))

    imgs = torch.cat(imgs, dim=0).to(device)  # NxCxHxW
    c2ws = torch.as_tensor(np.stack(c2ws, 0), dtype=torch.float32).to(device)  # Nx4x4
    # Compute intrinsics
    f = 0.5 * W / np.tan(0.5 * angle_x)
    K = np.array([[f, 0, W / 2],
                  [0, f, H / 2],
                  [0, 0, 1]], dtype=np.float32)
    Ks = torch.as_tensor(np.tile(K[None], (imgs.shape[0], 1, 1)), dtype=torch.float32)
    return imgs, Ks, c2ws, W, H

def interpolate_c2ws(c2ws, num_interp, device):
    """
    Linearly interpolate between each consecutive pair of camera-to-world matrices.
    Rotations are slerped, translations are linearly interpolated.
    """
    mats = c2ws.cpu().numpy()  # Nx4x4
    all_interp = []
    for i in range(len(mats) - 1):
        m1, m2 = mats[i], mats[i + 1]
        # Rotations as scipy Rotation
        times = [0, 1]
        R1 = R.from_matrix(m1[:3, :3])
        R2 = R.from_matrix(m2[:3, :3])
        slerp = Slerp(times, R.from_matrix([m1[:3, :3], m2[:3, :3]]))
        t1 = m1[:3, 3]
        t2 = m2[:3, 3]
        for j in range(1, num_interp + 1):
            t = j / (num_interp + 1)
            Rn = slerp([t]).as_matrix()[0]
            tn = (1 - t) * t1 + t * t2
            M = np.eye(4, dtype=np.float32)
            M[:3, :3] = Rn
            M[:3, 3] = tn
            all_interp.append(M)
    if not all_interp:
        return torch.empty((0, 4, 4), dtype=torch.float32, device=device)
    return torch.as_tensor(np.stack(all_interp, 0), dtype=torch.float32, device=device)

def main():
    args = parse_args()
    device = args.device

    # Load input images, intrinsics, and poses
    imgs, Ks, input_c2ws, W, H = load_data(
        args.transforms_json, args.images_dir, device
    )
    num_inputs = imgs.shape[0]

    # Generate interpolated target poses
    target_c2ws = interpolate_c2ws(input_c2ws, args.num_interp, device)
    num_targets = target_c2ws.shape[0]
    # Target intrinsics assume same as first
    target_Ks = Ks[:1].expand(num_targets, -1, -1)

    # Initialize model components
    model = SGMWrapper(load_model(device="cpu").eval()).to(device)
    ae = AutoEncoder(chunk_size=1).to(device)
    conditioner = CLIPConditioner().to(device)
    discret = DDPMDiscretization()
    denoiser = DiscreteDenoiser(discretization=discret,
                                 num_idx=1000,
                                 device=device)

    # Prepare all poses and intrinsics
    all_c2ws = torch.cat([input_c2ws, target_c2ws], dim=0)
    all_Ks = torch.cat([Ks, target_Ks], dim=0)

    # Prepare image conditioning: pad zeros for target frames
    imgs_np = (imgs.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
    zeros_np = np.zeros((num_targets, H, W, 3), dtype=np.uint8)
    all_imgs_np = np.concatenate([imgs_np, zeros_np], axis=0)
    image_cond = {
        "img": all_imgs_np,
        "input_indices": list(range(num_inputs)),
        "prior_indices": list(range(num_inputs)),
    }

    # Camera conditioning
    camera_cond = {
        "c2w": all_c2ws,
        "K": all_Ks,
        "input_indices": list(range(num_inputs + num_targets)),
    }

    # Version dict and options
    version_dict = {"H": H, "W": W, "T": 21, "C": 4, "f": 8, "options": {}}
    opts = version_dict["options"]
    opts.update({
        "chunk_strategy": args.chunk_strategy,
        "cfg": [float(args.cfg), 2.0],
        "camera_scale": args.camera_scale,
        "num_steps": args.num_steps,
        "video_save_fps": args.fps,
        "guider_types": [1, 2],
        "beta_linear_start": 5e-6,
        "log_snr_shift": 2.4,
        "cfg_min": 1.2,
        "encoding_t": 1,
        "decoding_t": 1,
    })

    # Infer number of anchors for prior
    # Note: this will modify version_dict["T"] in-place
    infer_prior_stats(
        version_dict["T"], num_inputs, num_targets, version_dict
    )
    # Determine anchor indices
    num_anchors = int(version_dict.get("num_anchors", 0))
    # Evenly sample anchors across targets
    anchor_ids = np.linspace(num_inputs,
                             num_inputs + num_targets - 1,
                             num_anchors).round().astype(int).tolist()
    anchor_c2ws = all_c2ws[anchor_ids]
    anchor_Ks = all_Ks[anchor_ids]

    # Run rendering
    os.makedirs(args.out_dir, exist_ok=True)
    gen = run_one_scene(
        task="img2trajvid",
        version_dict=version_dict,
        model=model,
        ae=ae,
        conditioner=conditioner,
        denoiser=denoiser,
        image_cond=image_cond,
        camera_cond=camera_cond,
        save_path=args.out_dir,
        use_traj_prior=True,
        traj_prior_c2ws=anchor_c2ws,
        traj_prior_Ks=anchor_Ks,
        seed=args.seed,
    )

    # Consume generator: prints progress and final video paths
    for out_path in gen:
        print(f"Output path: {out_path}")

if __name__ == "__main__":
    main()
