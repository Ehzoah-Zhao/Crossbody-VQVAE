import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random

class PairedMotionDataset(Dataset):
    def __init__(self, data_dir_A, data_dir_B, split_txt_path, stat_dir_A, stat_dir_B, window_size=64):
        self.data_dir_A = data_dir_A
        self.data_dir_B = data_dir_B
        self.window_size = window_size

        # 1. 读取划分名单过滤
        if os.path.exists(split_txt_path):
            with open(split_txt_path, 'r', encoding='utf-8') as f:
                valid_ids = {line.strip() for line in f.readlines() if line.strip()}
        else:
            raise FileNotFoundError(f"未找到配对数据集划分名单文件: {split_txt_path}")

        # 2. 获取双边共有的交集文件名，并经由名单二次过滤
        files_A = set([f for f in os.listdir(data_dir_A) if f.endswith('.npy')])
        files_B = set([f for f in os.listdir(data_dir_B) if f.endswith('.npy')])
        common_files = files_A.intersection(files_B)
        self.file_list = [f for f in common_files if f.replace('.npy', '') in valid_ids]
        print(f"[Paired Contrastive] 过滤后成功加载了 {len(self.file_list)} 对完全对齐的动作特征数据。")

        # 3. 读取各自本体的归一化参数
        self.mean_A = np.load(os.path.join(stat_dir_A, 'mean.npy'))
        self.std_A = np.load(os.path.join(stat_dir_A, 'std.npy'))
        self.mean_B = np.load(os.path.join(stat_dir_B, 'mean.npy'))
        self.std_B = np.load(os.path.join(stat_dir_B, 'std.npy'))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        
        motion_A = np.load(os.path.join(self.data_dir_A, filename))
        motion_B = np.load(os.path.join(self.data_dir_B, filename))
        
        # 各自本体 Z-score 归一化
        motion_A = (motion_A - self.mean_A) / (self.std_A + 1e-8)
        motion_B = (motion_B - self.mean_B) / (self.std_B + 1e-8)
        
        motion_A = torch.from_numpy(motion_A).float()
        motion_B = torch.from_numpy(motion_B).float()
        
        # ======== 修改这里：取两者长度的最小值，防止任一方越界截断 ========
        motion_len = min(motion_A.shape[0], motion_B.shape[0])
        
        if motion_len >= self.window_size:
            start_idx = random.randint(0, motion_len - self.window_size)
            window_A = motion_A[start_idx : start_idx + self.window_size].clone()
            window_B = motion_B[start_idx : start_idx + self.window_size].clone()
            mask = torch.ones(self.window_size)
        else:
            pad_len = self.window_size - motion_len
            # 注意：补齐时也要以最小的 motion_len 为准来截取，保证两者完全同步
            window_A = F.pad(motion_A[:motion_len], (0, 0, 0, pad_len), mode='constant', value=0.0).clone()
            window_B = F.pad(motion_B[:motion_len], (0, 0, 0, pad_len), mode='constant', value=0.0).clone()
            mask = torch.cat([torch.ones(motion_len), torch.zeros(pad_len)]).clone()
            
        return window_A, window_B, mask

def PairedDATALoader(data_dir_A, data_dir_B, split_txt_path, stat_dir_A, stat_dir_B, batch_size, window_size, num_workers=8):
    dataset = PairedMotionDataset(data_dir_A, data_dir_B, split_txt_path, stat_dir_A, stat_dir_B, window_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)