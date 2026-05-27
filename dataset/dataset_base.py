import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random

class SingleMotionDataset(Dataset):
    def __init__(self, data_dir, split_txt_path, stat_dir, window_size=64):
        self.data_dir = data_dir
        self.window_size = window_size
        
        # 1. 读取划分文件名单 (如 train.txt)，防止数据泄露
        if os.path.exists(split_txt_path):
            with open(split_txt_path, 'r', encoding='utf-8') as f:
                valid_ids = {line.strip() for line in f.readlines() if line.strip()}
            all_files = os.listdir(data_dir)
            self.file_list = [f for f in all_files if f.endswith('.npy') and f.replace('.npy', '') in valid_ids]
            print(f"[{os.path.basename(split_txt_path)}] 成功筛选读取了 {len(self.file_list)} 个动作文件。")
        else:
            raise FileNotFoundError(f"未找到数据集划分名单文件: {split_txt_path}，请检查路径。")

        # 2. 读取当前本体对应的归一化参数 mean 和 std
        mean_path = os.path.join(stat_dir, 'mean.npy')
        std_path = os.path.join(stat_dir, 'std.npy')
        if os.path.exists(mean_path) and os.path.exists(std_path):
            self.mean = np.load(mean_path)
            self.std = np.load(std_path)
            print(f"成功加载归一化统计量，特征维度大小: {self.mean.shape[-1]}")
        else:
            raise FileNotFoundError(f"归一化文件缺失！请检查 {stat_dir} 下是否存在 mean.npy 和 std.npy")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        motion = np.load(file_path)  # shape: (T, C)
        
        # 3. Z-score 归一化 (必须在 Padding 0 之前做！避免污染 0 特征)
        motion = (motion - self.mean) / (self.std + 1e-8)
        
        motion = torch.from_numpy(motion).float()
        motion_len = motion.shape[0]
        
        # 4. 定长采样裁剪与动态 Padding
        if motion_len >= self.window_size:
            start_idx = random.randint(0, motion_len - self.window_size)
            # 加上 .clone() 解除内存共享
            motion_window = motion[start_idx : start_idx + self.window_size].clone()
            mask = torch.ones(self.window_size)
        else:
            pad_len = self.window_size - motion_len
            motion_window = F.pad(motion, (0, 0, 0, pad_len), mode='constant', value=0.0)
            mask = torch.cat([torch.ones(motion_len), torch.zeros(pad_len)])
            
        # 安全起见，返回时全部 clone
        return motion_window.clone(), mask.clone()

def get_dataloader(data_dir, split_txt_path, stat_dir, batch_size, window_size, num_workers=8, shuffle=True):
    dataset = SingleMotionDataset(data_dir, split_txt_path, stat_dir, window_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=num_workers)