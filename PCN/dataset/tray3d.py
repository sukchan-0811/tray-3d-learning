import sys
sys.path.append('.')

import os
import random

import torch
import torch.utils.data as data
import numpy as np
import open3d as o3d


class Tray3d(data.Dataset):
    """
    Custom Made Tray3d dataset
    """
    
    def __init__(self, dataroot, split):
        assert split in ['train', 'valid', 'test'], "split error value!"

        self.dataroot = dataroot
        self.split = split

        self.partial_paths, self.complete_paths = self._load_data()
    
    def __getitem__(self, index):
        partial_path = self.partial_paths[index]
        complete_path = self.complete_paths[index]

        partial_pc = self.read_point_cloud(partial_path)
        complete_pc = self.read_point_cloud(complete_path)

        return torch.from_numpy(partial_pc), torch.from_numpy(complete_pc)

    def __len__(self):
        return len(self.complete_paths)

    def _load_data(self):
        data_path = os.path.join(self.dataroot, self.split)

        partial_paths, complete_paths = list(), list()

        for line in os.listdir(os.path.join(data_path, 'partial')):
            partial_paths.append(os.path.join(data_path, 'partial', line))
            complete_paths.append(os.path.join(data_path, 'complete', line))
        
        return partial_paths, complete_paths
    
    def read_point_cloud(self, path):
        pc = o3d.io.read_point_cloud(path)
        return np.array(pc.points, np.float32)
