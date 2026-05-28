"""dataset/dataset_B_v4.py - 363-dim G1 dataset loader

363-dim format: r_velocity(1)+root_vel_xz(2)+root_y(1)+ric(87)+rot(180)+local_vel(90)+contacts(2)

Usage: from dataset.dataset_B_v4 import DATALoader as DataLoaderB
"""

from .dataset_base import get_dataloader


def DATALoader(data_dir, split_txt_path, stat_dir, batch_size, window_size, num_workers=8, shuffle=True):
    """
    Creates DataLoader for 363-dim G1 data.
    
    Args:
        data_dir: path to new_joint_vecs_v4 directory
        split_txt_path: path to train/val/test split file
        stat_dir: path to meta_v4 directory with Mean.npy and Std.npy
        batch_size: int
        window_size: int (number of frames per sample)
    """
    return get_dataloader(data_dir, split_txt_path, stat_dir, batch_size, window_size,
                          num_workers=num_workers, shuffle=shuffle)