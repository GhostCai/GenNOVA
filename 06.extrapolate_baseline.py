import argparse
import copy
import json
import os
import os.path as osp
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from scipy.spatial.transform import Rotation, Slerp
from scipy.spatial.distance import pdist, squareform
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


def parse_nerf_json(json_path: str) -> Tuple[List[str], np.ndarray, float, List[str]]:
    """解析NeRF synthetic格式的JSON文件"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    camera_angle_x = data['camera_angle_x']
    frames = data['frames']
    
    img_paths = []
    transform_matrices = []
    img_names = []  # 存储原始图像名称
    
    base_dir = osp.dirname(json_path)
    
    for frame in frames:
        # 构建完整的图片路径
        file_path = frame['file_path']
        if file_path.startswith('./'):
            file_path = file_path[2:]
        
        # 获取基础名称（不含扩展名）
        base_name = osp.basename(file_path)
        img_names.append(base_name)
        
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
    
    return img_paths, np.array(transform_matrices), camera_angle_x, img_names


def nerf_to_opencv_pose(nerf_pose: np.ndarray) -> np.ndarray:
    """将NeRF格式的pose转换为OpenCV格式"""
    # NeRF使用右手坐标系 (x右, y上, z向后)
    # OpenCV使用右手坐标系 (x右, y下, z向前)
    # 转换矩阵: [1,0,0; 0,-1,0; 0,0,-1]
    opencv_pose = nerf_pose.copy()
    opencv_pose[:, 1:3] *= -1  # 翻转y和z轴
    return opencv_pose


def interpolate_poses_with_metadata(poses: np.ndarray, img_names: List[str], num_interpolations: int) -> Tuple[np.ndarray, List[Dict]]:
    """在相邻poses之间插值生成新的poses，并记录插值元数据"""
    interpolated_poses = []
    interpolation_metadata = []  # 存储插值信息
    
    for i in range(len(poses) - 1):
        pose1, pose2 = poses[i], poses[i + 1]
        name1, name2 = img_names[i], img_names[i + 1]
        
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
            
            # 记录插值元数据
            metadata = {
                'source_img1': name1,
                'source_img2': name2,
                'alpha': alpha,
                'step': j,
                'total_steps': num_interpolations
            }
            interpolation_metadata.append(metadata)
    
    return np.array(interpolated_poses), interpolation_metadata


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


def create_3dgs_dataset(
    input_img_paths: List[str],
    input_poses: np.ndarray,
    input_img_names: List[str],
    interpolated_frame_paths: List[str],
    interpolated_poses: np.ndarray,
    interpolation_metadata: List[Dict],
    camera_angle_x: float,
    output_dir: str
):
    """创建3DGS格式的数据集"""
    print("Creating 3DGS format dataset...")
    
    # 创建train目录
    train_dir = osp.join(output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    
    # 准备transforms_train.json的数据
    transforms_data = {
        "camera_angle_x": camera_angle_x,
        "frames": []
    }
    
    # 复制输入图像到train目录并添加到transforms
    for i, (img_path, pose, img_name) in enumerate(zip(input_img_paths, input_poses, input_img_names)):
        # 复制图像
        import shutil
        dst_path = osp.join(train_dir, f"{img_name}.png")
        if img_path.endswith('.png'):
            shutil.copy2(img_path, dst_path)
        else:
            # 如果不是png，转换为png
            img = iio.imread(img_path)
            iio.imwrite(dst_path, img)
        
        # 添加到transforms
        frame_data = {
            "file_path": f"./train/{img_name}",
            "transform_matrix": pose.tolist()
        }
        transforms_data["frames"].append(frame_data)
    
    # 处理生成的图像（插值和外推）
    for i, (frame_path, pose, metadata) in enumerate(zip(interpolated_frame_paths, interpolated_poses, interpolation_metadata)):
        # 根据metadata类型生成不同的文件名
        strategy = metadata.get('strategy', 'interpolation')
        
        if strategy == 'interpolation' or 'alpha' in metadata:
            # 插值图像的处理
            alpha = metadata['alpha']
            source1 = metadata['source_img1']
            source2 = metadata['source_img2']
            step = metadata['step']
            
            # 创建插值图像名称
            img_name = f"int_{source1}_{source2}_{alpha:.3f}"
            
        else:
            # 外推图像的处理
            img_name = f"ext_{strategy}_{i:04d}"
            
        dst_path = osp.join(train_dir, f"{img_name}.png")
        
        # 复制生成的图像
        import shutil
        shutil.copy2(frame_path, dst_path)
        
        # 转换pose回NeRF格式（保持不变，因为我们的poses已经是正确的格式）
        nerf_pose = pose.copy()
        
        # 添加到transforms
        frame_data = {
            "file_path": f"./train/{img_name}",
            "transform_matrix": nerf_pose.tolist(),
            "generation_metadata": metadata  # 保存完整的生成元数据
        }
        transforms_data["frames"].append(frame_data)
    
    # 保存transforms_train.json
    transforms_path = osp.join(output_dir, "transforms_train.json")
    with open(transforms_path, 'w') as f:
        json.dump(transforms_data, f, indent=2)
    
    print(f"Created 3DGS dataset with {len(transforms_data['frames'])} total frames")
    print(f"- Input images: {len(input_img_paths)}")
    print(f"- Generated images: {len(interpolated_frame_paths)}")
    print(f"- Dataset saved to: {output_dir}")
    print(f"- Images directory: {train_dir}")
    print(f"- Transforms file: {transforms_path}")
    
    return transforms_path


def compute_camera_statistics(poses: np.ndarray) -> Dict:
    """计算相机姿态的统计信息"""
    positions = poses[:, :3, 3]
    rotations = Rotation.from_matrix(poses[:, :3, :3])
    
    # 计算位置统计
    center = np.mean(positions, axis=0)
    distances_to_center = np.linalg.norm(positions - center, axis=1)
    avg_distance = np.mean(distances_to_center)
    std_distance = np.std(distances_to_center)
    
    # 计算高度范围
    height_range = (np.min(positions[:, 1]), np.max(positions[:, 1]))
    
    # 计算相机朝向统计
    forward_vectors = rotations.apply([0, 0, -1])  # OpenCV: z轴向前为负
    avg_forward = np.mean(forward_vectors, axis=0)
    avg_forward = avg_forward / np.linalg.norm(avg_forward)
    
    # 计算角度变化范围
    euler_angles = rotations.as_euler('xyz', degrees=True)
    angle_ranges = {
        'x': (np.min(euler_angles[:, 0]), np.max(euler_angles[:, 0])),
        'y': (np.min(euler_angles[:, 1]), np.max(euler_angles[:, 1])),
        'z': (np.min(euler_angles[:, 2]), np.max(euler_angles[:, 2]))
    }
    
    return {
        'center': center,
        'avg_distance': avg_distance,
        'std_distance': std_distance,
        'height_range': height_range,
        'avg_forward': avg_forward,
        'angle_ranges': angle_ranges,
        'positions': positions,
        'rotations': rotations
    }


def generate_extrapolated_poses(poses: np.ndarray, img_names: List[str], 
                               num_extrapolations: int = 20, 
                               extrapolation_strategies: List[str] = None) -> Tuple[np.ndarray, List[Dict]]:
    """生成外推的相机姿态"""
    if extrapolation_strategies is None:
        extrapolation_strategies = [
            'radial_expansion',
            'height_variation', 
            'orbit_extension',
            'random_perturbation',
            'trajectory_extension'
        ]
    
    stats = compute_camera_statistics(poses)
    extrapolated_poses = []
    extrapolation_metadata = []
    
    poses_per_strategy = num_extrapolations // len(extrapolation_strategies)
    
    for strategy in extrapolation_strategies:
        if strategy == 'radial_expansion':
            # 径向扩展：基于现有相机位置，向外扩展
            new_poses, metadata = _radial_expansion(poses, stats, poses_per_strategy)
            
        elif strategy == 'height_variation':
            # 高度变化：改变相机高度，保持相似的朝向
            new_poses, metadata = _height_variation(poses, stats, poses_per_strategy)
            
        elif strategy == 'orbit_extension':
            # 轨道扩展：围绕场景中心创建新的轨道
            new_poses, metadata = _orbit_extension(poses, stats, poses_per_strategy)
            
        elif strategy == 'random_perturbation':
            # 随机扰动：在现有姿态基础上添加合理的随机变化
            new_poses, metadata = _random_perturbation(poses, stats, poses_per_strategy)
            
        elif strategy == 'trajectory_extension':
            # 轨迹延拓：基于现有轨迹的趋势延拓新的位置
            new_poses, metadata = _trajectory_extension(poses, img_names, stats, poses_per_strategy)
        
        extrapolated_poses.extend(new_poses)
        extrapolation_metadata.extend(metadata)
    
    return np.array(extrapolated_poses), extrapolation_metadata


def _radial_expansion(poses: np.ndarray, stats: Dict, num_poses: int) -> Tuple[List[np.ndarray], List[Dict]]:
    """径向扩展策略"""
    new_poses = []
    metadata = []
    
    center = stats['center']
    avg_distance = stats['avg_distance']
    positions = stats['positions']
    rotations = stats['rotations']
    
    for i in range(num_poses):
        # 选择一个参考姿态
        ref_idx = np.random.randint(len(poses))
        ref_pos = positions[ref_idx]
        ref_rot = rotations[ref_idx]
        
        # 计算从中心到参考位置的方向
        direction = ref_pos - center
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm > 0:
            direction = direction / direction_norm
            
            # 扩展距离：1.2x 到 2.0x 的范围
            expansion_factor = np.random.uniform(1.2, 2.0)
            new_distance = direction_norm * expansion_factor
            new_pos = center + direction * new_distance
            
            # 调整相机朝向：让相机看向场景中心附近
            look_at_target = center + np.random.normal(0, avg_distance * 0.1, 3)
            new_rot = _look_at_rotation(new_pos, look_at_target)
            
            new_pose = np.eye(4)
            new_pose[:3, :3] = new_rot.as_matrix()
            new_pose[:3, 3] = new_pos
            
            new_poses.append(new_pose)
            metadata.append({
                'strategy': 'radial_expansion',
                'reference_img': f"pose_{ref_idx}",
                'expansion_factor': expansion_factor,
                'original_distance': direction_norm,
                'new_distance': new_distance
            })
    
    return new_poses, metadata


def _height_variation(poses: np.ndarray, stats: Dict, num_poses: int) -> Tuple[List[np.ndarray], List[Dict]]:
    """高度变化策略"""
    new_poses = []
    metadata = []
    
    center = stats['center']
    height_range = stats['height_range']
    height_span = height_range[1] - height_range[0]
    positions = stats['positions']
    rotations = stats['rotations']
    
    for i in range(num_poses):
        # 选择参考姿态
        ref_idx = np.random.randint(len(poses))
        ref_pos = positions[ref_idx].copy()
        ref_rot = rotations[ref_idx]
        
        # 生成新的高度
        if np.random.random() < 0.5:
            # 向上扩展
            new_height = height_range[1] + np.random.uniform(0.1, 0.5) * height_span
        else:
            # 向下扩展
            new_height = height_range[0] - np.random.uniform(0.1, 0.5) * height_span
        
        new_pos = ref_pos.copy()
        new_pos[1] = new_height
        
        # 调整朝向，考虑高度变化
        look_at_target = center.copy()
        # 如果相机很高，略微向下看
        if new_height > height_range[1]:
            look_at_target[1] -= (new_height - height_range[1]) * 0.3
        # 如果相机很低，略微向上看
        elif new_height < height_range[0]:
            look_at_target[1] += (height_range[0] - new_height) * 0.3
            
        new_rot = _look_at_rotation(new_pos, look_at_target)
        
        new_pose = np.eye(4)
        new_pose[:3, :3] = new_rot.as_matrix()
        new_pose[:3, 3] = new_pos
        
        new_poses.append(new_pose)
        metadata.append({
            'strategy': 'height_variation',
            'reference_img': f"pose_{ref_idx}",
            'original_height': ref_pos[1],
            'new_height': new_height,
            'height_change': new_height - ref_pos[1]
        })
    
    return new_poses, metadata


def _orbit_extension(poses: np.ndarray, stats: Dict, num_poses: int) -> Tuple[List[np.ndarray], List[Dict]]:
    """轨道扩展策略"""
    new_poses = []
    metadata = []
    
    center = stats['center']
    avg_distance = stats['avg_distance']
    height_range = stats['height_range']
    
    for i in range(num_poses):
        # 随机选择轨道参数
        radius = avg_distance * np.random.uniform(0.8, 2.2)
        height = np.random.uniform(height_range[0] - 0.2 * (height_range[1] - height_range[0]),
                                 height_range[1] + 0.2 * (height_range[1] - height_range[0]))
        
        # 随机角度
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(-np.pi/6, np.pi/6)  # 仰角变化
        
        # 计算位置
        x = center[0] + radius * np.cos(theta) * np.cos(phi)
        z = center[2] + radius * np.sin(theta) * np.cos(phi)
        y = height
        
        new_pos = np.array([x, y, z])
        
        # 相机朝向场景中心
        look_at_target = center + np.random.normal(0, avg_distance * 0.05, 3)
        new_rot = _look_at_rotation(new_pos, look_at_target)
        
        new_pose = np.eye(4)
        new_pose[:3, :3] = new_rot.as_matrix()
        new_pose[:3, 3] = new_pos
        
        new_poses.append(new_pose)
        metadata.append({
            'strategy': 'orbit_extension',
            'radius': radius,
            'height': height,
            'theta': theta,
            'phi': phi
        })
    
    return new_poses, metadata


def _random_perturbation(poses: np.ndarray, stats: Dict, num_poses: int) -> Tuple[List[np.ndarray], List[Dict]]:
    """随机扰动策略"""
    new_poses = []
    metadata = []
    
    avg_distance = stats['avg_distance']
    std_distance = stats['std_distance']
    positions = stats['positions']
    rotations = stats['rotations']
    
    for i in range(num_poses):
        # 选择参考姿态
        ref_idx = np.random.randint(len(poses))
        ref_pos = positions[ref_idx]
        ref_rot = rotations[ref_idx]
        
        # 位置扰动
        pos_noise_scale = np.random.uniform(0.1, 0.3) * avg_distance
        pos_noise = np.random.normal(0, pos_noise_scale, 3)
        new_pos = ref_pos + pos_noise
        
        # 旋转扰动
        angle_noise_deg = np.random.uniform(5, 20)  # 5-20度的旋转扰动
        axis = np.random.normal(0, 1, 3)
        axis = axis / np.linalg.norm(axis)
        angle_noise_rad = np.radians(angle_noise_deg)
        
        noise_rot = Rotation.from_rotvec(axis * angle_noise_rad)
        new_rot = noise_rot * ref_rot
        
        new_pose = np.eye(4)
        new_pose[:3, :3] = new_rot.as_matrix()
        new_pose[:3, 3] = new_pos
        
        new_poses.append(new_pose)
        metadata.append({
            'strategy': 'random_perturbation',
            'reference_img': f"pose_{ref_idx}",
            'position_noise_scale': pos_noise_scale,
            'rotation_noise_deg': angle_noise_deg,
            'noise_axis': axis.tolist()
        })
    
    return new_poses, metadata


def _trajectory_extension(poses: np.ndarray, img_names: List[str], stats: Dict, num_poses: int) -> Tuple[List[np.ndarray], List[Dict]]:
    """轨迹延拓策略"""
    new_poses = []
    metadata = []
    
    positions = stats['positions']
    rotations = stats['rotations']
    
    if len(poses) < 3:
        # 如果姿态太少，回退到随机扰动
        return _random_perturbation(poses, stats, num_poses)
    
    for i in range(num_poses):
        # 选择连续的三个姿态来预测趋势
        start_idx = np.random.randint(0, len(poses) - 2)
        
        pos1 = positions[start_idx]
        pos2 = positions[start_idx + 1]
        pos3 = positions[start_idx + 2] if start_idx + 2 < len(poses) else positions[start_idx + 1]
        
        # 计算位置趋势
        if start_idx + 2 < len(poses):
            # 二次外推
            velocity1 = pos2 - pos1
            velocity2 = pos3 - pos2
            acceleration = velocity2 - velocity1
            
            # 外推到下一个位置
            extension_factor = np.random.uniform(1.0, 2.0)
            new_pos = pos3 + velocity2 * extension_factor + 0.5 * acceleration * extension_factor**2
        else:
            # 线性外推
            velocity = pos2 - pos1
            extension_factor = np.random.uniform(1.0, 2.5)
            new_pos = pos2 + velocity * extension_factor
        
        # 旋转也使用类似的外推
        rot1 = rotations[start_idx]
        rot2 = rotations[start_idx + 1]
        
        # 计算旋转差异并外推
        rot_diff = rot2 * rot1.inv()
        extension_factor = np.random.uniform(0.5, 1.5)
        
        # 减小旋转外推的幅度
        rot_diff_reduced = Rotation.from_rotvec(rot_diff.as_rotvec() * extension_factor)
        new_rot = rot_diff_reduced * rot2
        
        new_pose = np.eye(4)
        new_pose[:3, :3] = new_rot.as_matrix()
        new_pose[:3, 3] = new_pos
        
        new_poses.append(new_pose)
        metadata.append({
            'strategy': 'trajectory_extension',
            'source_poses': [start_idx, start_idx + 1, start_idx + 2 if start_idx + 2 < len(poses) else start_idx + 1],
            'extension_factor': extension_factor,
            'extrapolation_type': 'quadratic' if start_idx + 2 < len(poses) else 'linear'
        })
    
    return new_poses, metadata


def _look_at_rotation(camera_pos: np.ndarray, target_pos: np.ndarray, up: np.ndarray = None) -> Rotation:
    """计算从相机位置看向目标位置的旋转矩阵"""
    if up is None:
        up = np.array([0, 1, 0])  # 默认向上方向
    
    # 计算前向量（从相机指向目标）
    forward = target_pos - camera_pos
    forward = forward / np.linalg.norm(forward)
    
    # 计算右向量
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    # 重新计算向上向量
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    
    # OpenCV坐标系：x右，y下，z前
    # 所以我们需要调整
    rotation_matrix = np.column_stack([right, -up, forward])
    
    return Rotation.from_matrix(rotation_matrix)


def generate_diverse_poses(poses: np.ndarray, img_names: List[str], 
                          num_interpolations: int = 5, 
                          num_extrapolations: int = 20,
                          extrapolation_strategies: List[str] = None) -> Tuple[np.ndarray, List[Dict]]:
    """生成多样化的poses，包括插值和外推"""
    all_poses = []
    all_metadata = []
    
    # 1. 插值poses
    if num_interpolations > 0:
        interp_poses, interp_metadata = interpolate_poses_with_metadata(
            poses, img_names, num_interpolations
        )
        all_poses.extend(interp_poses)
        all_metadata.extend(interp_metadata)
        print(f"Generated {len(interp_poses)} interpolated poses")
    
    # 2. 外推poses
    if num_extrapolations > 0:
        extrap_poses, extrap_metadata = generate_extrapolated_poses(
            poses, img_names, num_extrapolations, extrapolation_strategies
        )
        all_poses.extend(extrap_poses)
        all_metadata.extend(extrap_metadata)
        print(f"Generated {len(extrap_poses)} extrapolated poses")
    
    return np.array(all_poses), all_metadata


def main():
    parser = argparse.ArgumentParser(description="NeRF视角插值和外推工具")
    parser.add_argument("--json_path", type=str, required=True, help="NeRF synthetic格式的JSON文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--num_interpolations", type=int, default=0, help="每两个相邻视角之间插值的数量")
    parser.add_argument("--num_extrapolations", type=int, default=20, help="外推生成的视角数量")
    parser.add_argument("--extrapolation_strategies", type=str, nargs='+', 
                       default=['radial_expansion', 'height_variation', 'orbit_extension', 'random_perturbation', 'trajectory_extension'],
                       help="外推策略列表")
    parser.add_argument("--seed", type=int, default=23, help="随机种子")
    parser.add_argument("--cfg", type=float, default=3.0, help="CFG值")
    parser.add_argument("--camera_scale", type=float, default=2.0, help="相机缩放")
    parser.add_argument("--chunk_strategy", type=str, default="interp-gt", choices=["interp-gt", "interp"], help="分块策略")
    parser.add_argument("--create_3dgs_dataset", type=bool, default=True, help="是否创建3DGS格式数据集")
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 解析NeRF JSON
    print("Parsing NeRF JSON file...")
    img_paths, poses, camera_angle_x, img_names = parse_nerf_json(args.json_path)
    print(f"Found {len(img_paths)} input images")
    
    # 生成多样化的poses（插值 + 外推）
    print("Generating diverse poses...")
    target_poses, pose_metadata = generate_diverse_poses(
        poses, img_names, 
        args.num_interpolations, 
        args.num_extrapolations,
        args.extrapolation_strategies
    )
    print(f"Generated {len(target_poses)} total target poses")
    
    # 统计不同策略的数量
    strategy_counts = {}
    for meta in pose_metadata:
        strategy = meta.get('strategy', 'interpolation')
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    print("Pose generation strategy breakdown:")
    for strategy, count in strategy_counts.items():
        print(f"  {strategy}: {count}")
    
    # 加载模型
    models = load_models()
    
    # 预处理数据
    preprocessed = preprocess_data(img_paths, poses, camera_angle_x)
    
    # 生成目标视角
    video_path = generate_interpolated_views(
        preprocessed=preprocessed,
        target_poses=target_poses,
        target_camera_angle_x=camera_angle_x,
        models=models,
        output_dir=args.output_dir,
        seed=args.seed,
        cfg=args.cfg,
        camera_scale=args.camera_scale,
        chunk_strategy=args.chunk_strategy,
    )
    
    # 提取视频帧
    frame_paths = extract_frames_from_video(video_path, args.output_dir)
    
    # 创建3DGS格式数据集
    if args.create_3dgs_dataset:
        transforms_path = create_3dgs_dataset(
            input_img_paths=img_paths,
            input_poses=poses,
            input_img_names=img_names,
            interpolated_frame_paths=frame_paths,
            interpolated_poses=target_poses,
            interpolation_metadata=pose_metadata,
            camera_angle_x=camera_angle_x,
            output_dir=args.output_dir
        )
        print(f"3DGS dataset created: {transforms_path}")
    
    # 保存详细的元数据
    metadata_path = osp.join(args.output_dir, "pose_generation_metadata.json")
    with open(metadata_path, "w") as f:
        # 将numpy数组转换为列表以便JSON序列化
        serializable_metadata = []
        for meta in pose_metadata:
            serializable_meta = copy.deepcopy(meta)
            for key, value in serializable_meta.items():
                if isinstance(value, np.ndarray):
                    serializable_meta[key] = value.tolist()
            serializable_metadata.append(serializable_meta)
        
        json.dump({
            'strategy_counts': strategy_counts,
            'total_poses': len(target_poses),
            'input_poses': len(poses),
            'pose_metadata': serializable_metadata
        }, f, indent=2)
    
    print(f"完成！输出保存到: {args.output_dir}")
    print(f"详细元数据保存到: {metadata_path}")


if __name__ == "__main__":
    main()
    
"""
1. 五种外推策略
径向扩展 (radial_expansion): 基于现有相机位置向外扩展，增加距离多样性
高度变化 (height_variation): 改变相机高度，探索不同的垂直视角
轨道扩展 (orbit_extension): 创建新的环绕轨道，增加角度覆盖
随机扰动 (random_perturbation): 在现有姿态基础上添加合理噪声
轨迹延拓 (trajectory_extension): 基于现有轨迹趋势预测新位置
2. 智能相机朝向
使用 _look_at_rotation() 函数确保新相机始终朝向场景中心
根据相机高度调整俯仰角
3. 统计驱动的参数
分析输入poses的统计特性（中心、距离、高度范围等）
基于这些统计信息生成合理的外推参数
"""