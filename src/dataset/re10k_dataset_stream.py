from __future__ import annotations

import json
import os
from io import BytesIO
from typing import Any

import numpy as np
import torch
import torchvision
import torchvision.transforms as tf
from PIL import Image
from torch.utils.data import Dataset

# 数据集路径
MY_DATA_ROOT = "/data0/xxy/data/re10k"


def get_uniform_indices(pool_size: int, num_pick: int) -> np.ndarray:
    """Pick `num_pick` indices from a pool of length `pool_size`.

    For the common RE10K streaming setup we use pool_size=15 (odd frames within a
    1-second window at 30fps). For that case we keep the original lookup-table
    behavior to match the author's intended sampling.
    """
    if num_pick <= 0:
        return np.array([], dtype=int)

    if pool_size == 15:
        if num_pick == 1:
            return np.array([7], dtype=int)
        if num_pick == 2:
            return np.array([5, 9], dtype=int)
        if num_pick == 3:
            return np.array([3, 7, 11], dtype=int)
        if num_pick == 4:
            return np.array([3, 6, 8, 11], dtype=int)
        if num_pick == 7:
            return np.array([1, 3, 5, 7, 9, 11, 13], dtype=int)
        if num_pick == 8:
            return np.array([0, 2, 4, 6, 8, 10, 12, 14], dtype=int)

    if num_pick >= pool_size:
        return np.arange(pool_size, dtype=int)
    return np.linspace(0, pool_size - 1, num_pick).astype(int)


def pick_k(pool_array: np.ndarray, k: int) -> np.ndarray:
    if len(pool_array) == 0 or k <= 0:
        return np.array([], dtype=int)
    indices = get_uniform_indices(len(pool_array), k)
    return pool_array[indices]


def build_re10k_stream_splits(
    *,
    num_frames: int,
    input_num: int = 4,
    num_seconds: int = 5,
    fps: int = 30,
    rng: np.random.Generator | None = None,
) -> dict[str, Any] | None:
    """Create per-second context indices and 5 streaming target sets.

    Returns a dict with:
      - input_indices_per_sec: list[np.ndarray] length=num_seconds
      - pool_indices_per_sec: list[np.ndarray] length=num_seconds (odd frames)
      - target_indices_per_step: list[np.ndarray] length=num_seconds (each len=15)

    If the clip is too short (< num_seconds*fps), returns None.
    """
    if num_seconds != 5:
        raise ValueError(
            "build_re10k_stream_splits currently implements the fixed 5-second "
            "(T1..T5) protocol; got num_seconds={num_seconds}."
        )

    if rng is None:
        rng = np.random.default_rng()

    min_len = num_seconds * fps
    valid_len = min(num_frames, min_len)
    if valid_len < min_len:
        return None

    inputs_per_sec: list[np.ndarray] = []
    pools_per_sec: list[np.ndarray] = []

    pool_pattern = np.arange(1, fps, 2, dtype=int)  # [1,3,...,fps-1] => 15 when fps=30

    for sec in range(num_seconds):
        start = sec * fps
        end = min((sec + 1) * fps, valid_len)

        local_pool = start + pool_pattern
        local_pool = local_pool[local_pool < valid_len]
        pools_per_sec.append(local_pool)

        frame_count = end - start
        if frame_count <= 0:
            inputs_per_sec.append(np.array([], dtype=int))
            continue

        count = min(input_num, frame_count)
        # Random within the second.
        local_input = rng.choice(frame_count, count, replace=False)
        local_input.sort()
        inputs_per_sec.append(start + local_input)

    # Targets follow the 1+2+4+8 aggregation logic.
    final_targets: list[np.ndarray] = []
    final_targets.append(pick_k(pools_per_sec[0], 15))
    final_targets.append(
        np.concatenate([pick_k(pools_per_sec[0], 7), pick_k(pools_per_sec[1], 8)])
    )
    final_targets.append(
        np.concatenate(
            [
                pick_k(pools_per_sec[0], 3),
                pick_k(pools_per_sec[1], 4),
                pick_k(pools_per_sec[2], 8),
            ]
        )
    )
    final_targets.append(
        np.concatenate(
            [
                pick_k(pools_per_sec[0], 1),
                pick_k(pools_per_sec[1], 2),
                pick_k(pools_per_sec[2], 4),
                pick_k(pools_per_sec[3], 8),
            ]
        )
    )
    final_targets.append(
        np.concatenate(
            [
                pick_k(pools_per_sec[1], 1),
                pick_k(pools_per_sec[2], 2),
                pick_k(pools_per_sec[3], 4),
                pick_k(pools_per_sec[4], 8),
            ]
        )
    )

    return {
        "input_indices_per_sec": inputs_per_sec,
        "pool_indices_per_sec": pools_per_sec,
        "target_indices_per_step": final_targets,
        "valid_len": valid_len,
        "fps": fps,
        "num_seconds": num_seconds,
    }

class RE10KDataset(Dataset):
    def __init__(self, data_root=None, input_num=4):
        #input_num: 每秒取几帧输入(默认4)，随机取
        # 优先使用传入的路径
        self.data_root = data_root if data_root else MY_DATA_ROOT
        self.input_num = input_num
        self.to_tensor = tf.ToTensor()
        
        # 寻找原数据集的 json 文件 
        json_file = os.path.join(self.data_root, 'train', 'index.json')

        if not os.path.exists(json_file):
             # 尝试根目录
             json_file = os.path.join(self.data_root, 'index.json')
        # 检查并输出信息
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                content = json.load(f)
            # 兼容格式的 JSON
            if isinstance(content, dict):
                self.file_list = list(content.values())
            else:
                self.file_list = content
            print(f"[RE10KDataset] Loaded {len(self.file_list)} scenes")
        else:
            raise FileNotFoundError(f"Index file not found at: {json_file}")

    def __len__(self):
        return len(self.file_list)
        
    # 最大间隔采样，尽量保持均匀分布，每s共30帧，取单数共15帧做target，后续即从1,3,5……29中再挑。
    # 15选1 -> [15]
    # 15选2 -> [11, 19] (近似均匀)
    # 15选3 -> [7, 15, 23] 
    # 15选4 -> [7, 13, 17, 23] (近似均匀)
    # 15选7 -> [3, 7, 11, 15, 19, 23, 27] 
    # 15选8 -> [1, 5, 9, 13, 17, 21, 25, 29] 
    def get_uniform_indices(self, pool_size, num_pick):
        return get_uniform_indices(pool_size, num_pick)
    
    def pick_k(self, pool_array, k):
        return pick_k(pool_array, k)

    # 解码函数
    def convert_images(self, raw_images_list):
        torch_images = []
        for img_bytes in raw_images_list:
            image = Image.open(BytesIO(img_bytes.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)
 
    def __getitem__(self, idx):
        # 读取.torch 文件
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_root, 'train', file_name)
        
        # 加载数据
        try:
            chunk = torch.load(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
        
        # 整理 Tensor 格式
        if isinstance(chunk, list):
            example = chunk[0]
        else:
            example = chunk

        # 获取原始数据 (二进制)
        raw_images = example.get('images')
        # 相机参数处理
        raw_cameras = example.get('cameras')

        # 截取前 5 秒    
        total_frames = len(raw_images)
        # 前 150 帧
        valid_len = min(total_frames, 150)
        # 过滤掉不够长的
        if valid_len < 150:
            return None 

        # 阶段一：生成每一秒的 Pool(备选池) 和 Input(随机输入)
        inputs_per_sec = [] 
        pools_per_sec = [] 
        
        # Pool 固定模式：每秒取单数帧 [1, 3, ..., 29]
        pool_pattern = np.arange(1, 30, 2) # [1, 3, 5, ..., 29] 共15个
        
        for i in range(5): # 遍历 5 秒
            start = i * 30
            end = min((i + 1) * 30, valid_len)
            # 1.生成 Pool (绝对帧号)
            local_pool = start + pool_pattern
            local_pool = local_pool[local_pool < valid_len]
            pools_per_sec.append(local_pool)
            
            # 2.生成 Input (默认随机4帧，允许与 Pool 重合)
            frame_count = end - start
            if frame_count > 0:
                # 取最小值
                count = min(self.input_num, frame_count)
                # 随机挑选
                if frame_count >= count:
                    local_input = np.random.choice(frame_count, count, replace=False)
                else:
                    local_input = np.arange(frame_count)
                    
                local_input.sort()
                inputs_per_sec.append(start + local_input)
            else:
                inputs_per_sec.append(np.array([], dtype=int))

        # 阶段二：Target 组合 (1+2+4+8)     
        final_targets = []
        # T1: P1 取全部 (15)
        final_targets.append(self.pick_k(pools_per_sec[0], 15))
        # T2: P1(7) + P2(8) 
        t2 = np.concatenate([self.pick_k(pools_per_sec[0], 7), self.pick_k(pools_per_sec[1], 8)])
        final_targets.append(t2)
        # T3: P1(3) + P2(4) + P3(8)
        t3 = np.concatenate([self.pick_k(pools_per_sec[0], 3), self.pick_k(pools_per_sec[1], 4), self.pick_k(pools_per_sec[2], 8)])
        final_targets.append(t3)
        # T4: P1(1) + P2(2) + P3(4) + P4(8)
        t4 = np.concatenate([self.pick_k(pools_per_sec[0], 1), self.pick_k(pools_per_sec[1], 2), self.pick_k(pools_per_sec[2], 4), self.pick_k(pools_per_sec[3], 8)])
        final_targets.append(t4)
        # T5: P2(1) + P3(2) + P4(4) + P5(8) (P1不再参与) 
        t5 = np.concatenate([self.pick_k(pools_per_sec[1], 1), self.pick_k(pools_per_sec[2], 2), self.pick_k(pools_per_sec[3], 4), self.pick_k(pools_per_sec[4], 8)])
        final_targets.append(t5)

        # 整理输出
        output_dict = {}
        
        # 1. 提取 Input
        all_inputs_indices = np.concatenate(inputs_per_sec)
        # 从 raw_images 中取出选中的帧 
        selected_inputs_raw = [raw_images[i] for i in all_inputs_indices]
        # 解码
        decoded_inputs = self.convert_images(selected_inputs_raw)
        # Reshape
        output_dict['input_images'] = decoded_inputs.view(5, -1, 3, 360, 640)
        
        # 2. 提取 Target
        all_targets_indices = np.concatenate(final_targets)
        selected_targets_raw = [raw_images[i] for i in all_targets_indices]
        # 解码
        decoded_targets = self.convert_images(selected_targets_raw)
        # Reshape
        output_dict['target_images'] = decoded_targets.view(5, 15, 3, 360, 640)
        
        # 相机参数 
        if raw_cameras is not None:
            pass

        return output_dict

if __name__ == "__main__":
    ds = RE10KDataset(input_num=4)
    
    for i, sample in enumerate(ds):
        if sample is not None:
            print(f"\n Found Valid Sample at Index {i}")
            print(f"Input : {sample['input_images'].shape}  (Exp: [5, 4, ...])")
            print(f"Target: {sample['target_images'].shape} (Exp: [5, 15, ...])")

            '''
            # 保存图片逻辑 
            scene_dir = f"output/scene_{i:03d}" 
            os.makedirs(scene_dir, exist_ok=True)
            # 遍历 5 个时间步
            for t in range(5):
                sec_name = f"sec_{t+1}"
                sec_dir = os.path.join(scene_dir, sec_name) # 子文件夹
                os.makedirs(sec_dir, exist_ok=True)
                
                # 保存 Input
                inputs = sample['input_images'][t] 
                for idx, img in enumerate(inputs):
                    torchvision.utils.save_image(img, f"{sec_dir}/input_{idx}.png")
                    
                # 保存 Target
                targets = sample['target_images'][t] 
                for idx, img in enumerate(targets):
                    torchvision.utils.save_image(img, f"{sec_dir}/target_{idx}.png")
            print(f"Saved Scene {i} to {scene_dir}")
            '''
            
    print("\n All Done! Check the 'output_vis' folder.")
    
