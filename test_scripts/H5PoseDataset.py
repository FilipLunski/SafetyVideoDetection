import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class H5PoseDataset(Dataset):
    def __init__(self, h5_path, transform=None):
        """
        Args:
            h5_path (str): Path to HDF5 file
            transform (callable, optional): Optional transform to apply to keypoints
        """
        self.h5_path = h5_path
        self.transform = transform
        self.index_map = []  # List of (video_name, frame_idx) tuples
        
        # Build index map and cache video lengths
        with h5py.File(h5_path, 'r') as f:
            for video_name in f:
                if 'dataset' in f[video_name]:
                    num_frames = f[video_name]['dataset']['keypoints'].shape[0]
                    self.index_map.extend([(video_name, i) for i in range(num_frames)])

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        video_name, frame_idx = self.index_map[idx]
        
        # Use a file handle cache to avoid reopening the file for each access
        with h5py.File(self.h5_path, 'r') as f:
            video_group = f[video_name]
            dataset_group = video_group['dataset']
            
            keypoints = torch.tensor(dataset_group['keypoints'][frame_idx], dtype=torch.float32)
            category = torch.tensor([dataset_group['categories'][frame_idx]], dtype=torch.long)
            # print(category)
            if self.transform:
                keypoints = self.transform(keypoints)
                
            return keypoints, category

    @staticmethod
    def collate_fn(batch):
        """Custom collate function to handle variable-length sequences if needed"""
        keypoints, categories = zip(*batch)
        return torch.stack(keypoints), torch.stack(categories)