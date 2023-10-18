import os
import torch
import torch.distributed as dist
import numpy as np

class DatasetDDP():
    def __init__(self, rank : int = 0, world_size : int = 1,):
        self.rank = rank
        self.world_size = world_size
        self.data_x = None
        self.data_y = None
    
    def ddp_slice(self):
        if self.data_x is None or self.data_y is None:
            raise ValueError("Dataset classes inheriting from DatasetDDP must set self.data_x and self.data_y")
        slice_indices = np.arange(self.rank, len(self.data_x), self.world_size)
        self.data_x = self.data_x[slice_indices]
        self.data_y = self.data_y[slice_indices]
