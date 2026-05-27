from .dataset_base import get_dataloader

def DATALoader(data_dir, split_txt_path, stat_dir, batch_size, window_size):
    return get_dataloader(data_dir, split_txt_path, stat_dir, batch_size, window_size)