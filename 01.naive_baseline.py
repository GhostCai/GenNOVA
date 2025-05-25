import argparse
import copy
import json
import os
import os.path as osp
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))

# This is the key: add the parent directory of `seva` to sys.path
sys.path.insert(0, os.path.join(script_dir, 'third_party', 'stable_virtual_camera'))

from seva.eval import (
    IS_TORCH_NIGHTLY,
    chunk_input_and_test,
    create_transforms_simple,
    infer_prior_stats,
    run_one_scene,
    transform_img_and_K,
)
from seva.geometry import (
    DEFAULT_FOV_RAD,
    get_default_intrinsics,
    normalize_scene,
)
from seva.model import SGMWrapper
from seva.modules.autoencoder import AutoEncoder
from seva.modules.conditioner import CLIPConditioner
from seva.modules.preprocessor import Dust3rPipeline
from seva.sampling import DDPMDiscretization, DiscreteDenoiser
from seva.utils import load_model

device = "cuda:0"

# Model initialization flags
if IS_TORCH_NIGHTLY:
    COMPILE = True
    os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
else:
    COMPILE = False

VERSION_DICT = {
    "H": 576,
    "W": 576,
    "T": 21,
    "C": 4,
    "f": 8,
    "options": {},
}


def load_models():
    """加载所有必需的模型"""
    print("Loading models...")
    
    dust3r = Dust3rPipeline(device=device)
    model = SGMWrapper(load_model(device="cpu", verbose=True).eval()).to(device)
    ae = AutoEncoder(chunk_size=1).to(device)
    conditioner = CLIPConditioner().to(device)
    discretization = DDPMDiscretization()
    denoiser = DiscreteDenoiser(discretization=discretization, num_idx=1000, device=device)
    
    if COMPILE:
        model = torch.compile(model)
        conditioner = torch.compile(conditioner)
        ae = torch.compile(ae)
    
    print("Models loaded successfully!")
    return model, ae, conditioner, denoiser, dust3r


def parse_nerf_json(json_path: str) -> Tuple[List[str], np.ndarray, float]:
    """解析NeRF synthetic格式的JSON文件"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    camera_angle_x = data['camera_angle_x']
    frames = data['frames']
    
    img_paths = []
    transform_matrices = []
    
    base_dir = osp.dirname(json_path)
    
    for frame in frames:
        # 构建完整的图片路径
        file_path = frame['file_path']
        if file_path.startswith('./'):
            file_path = file_path[2:]
        
        # 尝试不同的图片扩展名
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = osp.join(base_dir, file_path + ext)
            if osp.exists(candidate):
                img_path = candidate
                break
        
        if img_path is None:
            # 如果没有扩展名，直接尝试原路径
            img_path = osp.join(base_dir, file_path)
            if not osp.exists(img_path):
                raise FileNotFoundError(f"Image not found: {file_path}")
        
        img_paths.append(img_path)
        transform_matrices.append(np.array(frame['transform_matrix']))
    
    return img_paths, np.array(transform_matrices), camera_angle_x


def nerf_to_opencv_pose(nerf_pose: np.ndarray) -> np.ndarray:
    """将NeRF格式的pose转换为OpenCV格式"""
    # NeRF使用右手坐标系 (x右, y上, z向后)
    # OpenCV使用右手坐标系 (x右, y下, z向前)
    # 转换矩阵: [1,0,0; 0,-1,0; 0,0,-1]
    opencv_pose = nerf_pose.copy()
    opencv_pose[:, 1:3] *= -1  # 翻转y和z轴
    return opencv_pose


def interpolate_poses(poses: np.ndarray, num_interpolations: int) -> np.ndarray:
    """在相邻poses之间插值生成新的poses"""
    from scipy.spatial.transform import Rotation, Slerp
    
    interpolated_poses = []
    
    for i in range(len(poses) - 1):
        pose1, pose2 = poses[i], poses[i + 1]
        
        # 提取旋转和平移
        R1, t1 = pose1[:3, :3], pose1[:3, 3]
        R2, t2 = pose2[:3, :3], pose2[:3, 3]
        
        # 创建旋转插值器
        rotations = Rotation.from_matrix([R1, R2])
        slerp = Slerp([0, 1], rotations)
        
        # 生成插值
        for j in range(num_interpolations):
            alpha = (j + 1) / (num_interpolations + 1)
            
            # 插值旋转
            interp_rot = slerp(alpha)
            
            # 线性插值平移
            interp_t = (1 - alpha) * t1 + alpha * t2
            
            # 构建插值pose
            interp_pose = np.eye(4)
            interp_pose[:3, :3] = interp_rot.as_matrix()
            interp_pose[:3, 3] = interp_t
            
            interpolated_poses.append(interp_pose)
    
    return np.array(interpolated_poses)


def preprocess_data(img_paths: List[str], poses: np.ndarray, camera_angle_x: float, shorter: int = 576):
    """预处理输入数据"""
    print("Preprocessing input data...")
    
    # 确保shorter是64的倍数
    shorter = round(shorter / 64) * 64
    
    # 读取图像
    input_imgs = []
    for img_path in img_paths:
        img = iio.imread(img_path)
        if img.shape[-1] == 4:  # 如果有alpha通道，去除
            img = img[..., :3]
        img = torch.as_tensor(img / 255.0, dtype=torch.float32)
        input_imgs.append(img)
    
    # 转换poses格式
    opencv_poses = np.array([nerf_to_opencv_pose(pose) for pose in poses])
    input_c2ws = torch.as_tensor(opencv_poses)
    
    # 计算内参矩阵
    H, W = input_imgs[0].shape[:2]
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
    input_Ks = torch.tensor([
        [focal, 0, W/2],
        [0, focal, H/2],
        [0, 0, 1]
    ]).float().unsqueeze(0).repeat(len(input_imgs), 1, 1)
    
    # 调整图像尺寸
    new_input_imgs, new_input_Ks = [], []
    for img, K in zip(input_imgs, input_Ks):
        img = rearrange(img, "h w c -> 1 c h w")
        img, K = transform_img_and_K(img, shorter, K=K[None], size_stride=64)
        K = K / K.new_tensor([img.shape[-1], img.shape[-2], 1])[:, None]
        new_input_imgs.append(img)
        new_input_Ks.append(K)
    
    input_imgs = torch.cat(new_input_imgs, 0)
    input_imgs = rearrange(input_imgs, "b c h w -> b h w c")[..., :3]
    input_Ks = torch.cat(new_input_Ks, 0)
    
    return {
        "input_imgs": input_imgs,
        "input_Ks": input_Ks,
        "input_c2ws": input_c2ws,
        "input_wh": (input_imgs.shape[2], input_imgs.shape[1]),
    }


def generate_interpolated_views(
    preprocessed: dict,
    target_poses: np.ndarray,
    target_camera_angle_x: float,
    models: tuple,
    output_dir: str,
    seed: int = 23,
    cfg: float = 3.0,
    camera_scale: float = 2.0,
    chunk_strategy: str = "interp-gt",
):
    """生成插值视角的图像"""
    model, ae, conditioner, denoiser, _ = models
    
    print("Generating interpolated views...")
    
    input_imgs, input_Ks, input_c2ws, input_wh = (
        preprocessed["input_imgs"],
        preprocessed["input_Ks"],
        preprocessed["input_c2ws"],
        preprocessed["input_wh"],
    )
    W, H = input_wh
    num_inputs = len(input_imgs)
    
    # 准备目标poses和内参
    target_c2ws = torch.as_tensor([nerf_to_opencv_pose(pose) for pose in target_poses])
    focal = 0.5 * W / np.tan(0.5 * target_camera_angle_x)
    target_Ks = torch.tensor([
        [focal, 0, W/2],
        [0, focal, H/2],
        [0, 0, 1]
    ]).float().unsqueeze(0).repeat(len(target_poses), 1, 1)
    target_Ks = target_Ks / target_Ks.new_tensor([W, H, 1])[:, None]
    
    # 合并所有poses和内参
    all_c2ws = torch.cat([input_c2ws, target_c2ws], 0)
    all_Ks = (
        torch.cat([input_Ks, target_Ks], 0)
        * input_Ks.new_tensor([W, H, 1])[:, None]
    )
    
    num_targets = len(target_c2ws)
    input_indices = list(range(num_inputs))
    target_indices = np.arange(num_inputs, num_inputs + num_targets).tolist()
    
    # 设置版本字典
    T = VERSION_DICT["T"]
    version_dict = copy.deepcopy(VERSION_DICT)
    num_anchors = infer_prior_stats(
        T,
        num_inputs,
        num_total_frames=num_targets,
        version_dict=version_dict,
    )
    T = version_dict["T"]
    
    # 获取anchor cameras
    anchor_indices = np.linspace(
        num_inputs,
        num_inputs + num_targets - 1,
        num_anchors,
    ).tolist()
    anchor_c2ws = all_c2ws[[round(ind) for ind in anchor_indices]]
    anchor_Ks = all_Ks[[round(ind) for ind in anchor_indices]]
    
    # 创建图像conditioning
    all_imgs_np = (
        F.pad(input_imgs, (0, 0, 0, 0, 0, 0, 0, num_targets), value=0.0).numpy()
        * 255.0
    ).astype(np.uint8)
    
    image_cond = {
        "img": all_imgs_np,
        "input_indices": input_indices,
        "prior_indices": anchor_indices,
    }
    
    # 创建相机conditioning
    camera_cond = {
        "c2w": all_c2ws,
        "K": all_Ks,
        "input_indices": list(range(num_inputs + num_targets)),
    }
    
    # 设置渲染选项
    options = copy.deepcopy(VERSION_DICT["options"])
    options.update({
        "chunk_strategy": chunk_strategy,
        "video_save_fps": 30.0,
        "beta_linear_start": 5e-6,
        "log_snr_shift": 2.4,
        "guider_types": [1, 2],
        "cfg": [float(cfg), 3.0 if num_inputs >= 9 else 2.0],
        "camera_scale": camera_scale,
        "num_steps": 50,
        "cfg_min": 1.2,
        "encoding_t": 1,
        "decoding_t": 1,
    })
    
    # 运行渲染
    task = "img2trajvid"
    video_path_generator = run_one_scene(
        task=task,
        version_dict={
            "H": H,
            "W": W,
            "T": T,
            "C": VERSION_DICT["C"],
            "f": VERSION_DICT["f"],
            "options": options,
        },
        model=model,
        ae=ae,
        conditioner=conditioner,
        denoiser=denoiser,
        image_cond=image_cond,
        camera_cond=camera_cond,
        save_path=output_dir,
        use_traj_prior=True,
        traj_prior_c2ws=anchor_c2ws,
        traj_prior_Ks=anchor_Ks,
        seed=seed,
        gradio=False,
    )
    
    # 处理生成的视频
    for i, video_path in enumerate(video_path_generator):
        print(f"Generated video {i+1}: {video_path}")
        if i >= 1:  # 只需要最终的视频
            break
    
    return video_path


def extract_frames_from_video(video_path: str, output_dir: str) -> List[str]:
    """从生成的视频中提取帧"""
    import cv2
    
    print(f"Extracting frames from {video_path}")
    
    frames_dir = osp.join(output_dir, "interpolated_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_path = osp.join(frames_dir, f"frame_{frame_idx:04d}.png")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        frame_idx += 1
    
    cap.release()
    print(f"Extracted {len(frame_paths)} frames to {frames_dir}")
    return frame_paths


def main():
    parser = argparse.ArgumentParser(description="NeRF视角插值工具")
    parser.add_argument("--json_path", type=str, required=True, help="NeRF synthetic格式的JSON文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--num_interpolations", type=int, default=5, help="每两个相邻视角之间插值的数量")
    parser.add_argument("--seed", type=int, default=23, help="随机种子")
    parser.add_argument("--cfg", type=float, default=3.0, help="CFG值")
    parser.add_argument("--camera_scale", type=float, default=2.0, help="相机缩放")
    parser.add_argument("--chunk_strategy", type=str, default="interp-gt", choices=["interp-gt", "interp"], help="分块策略")
    parser.add_argument("--extract_frames", action="store_true", help="是否从生成的视频中提取帧")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 解析NeRF JSON
    print("Parsing NeRF JSON file...")
    img_paths, poses, camera_angle_x = parse_nerf_json(args.json_path)
    print(f"Found {len(img_paths)} input images")
    
    # 生成插值poses
    print("Generating interpolated poses...")
    interpolated_poses = interpolate_poses(poses, args.num_interpolations)
    print(f"Generated {len(interpolated_poses)} interpolated poses")
    
    # 加载模型
    models = load_models()
    
    # 预处理数据
    preprocessed = preprocess_data(img_paths, poses, camera_angle_x)
    
    # 生成插值视角
    video_path = generate_interpolated_views(
        preprocessed=preprocessed,
        target_poses=interpolated_poses,
        target_camera_angle_x=camera_angle_x,
        models=models,
        output_dir=args.output_dir,
        seed=args.seed,
        cfg=args.cfg,
        camera_scale=args.camera_scale,
        chunk_strategy=args.chunk_strategy,
    )
    
    # 可选：提取视频帧
    if args.extract_frames:
        frame_paths = extract_frames_from_video(video_path, args.output_dir)
        
        # 保存帧路径信息
        with open(osp.join(args.output_dir, "interpolated_frames.txt"), "w") as f:
            for frame_path in frame_paths:
                f.write(f"{frame_path}\n")
    
    print(f"完成！输出保存到: {args.output_dir}")


if __name__ == "__main__":
    main()