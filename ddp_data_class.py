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


def get_dataloader(
            dataset,
            mode,
            batch_size=32,
            shuffle=True,
            num_workers=0,
            prefetch_factor=None,
        ):
    mode = mode.lower()
    if mode not in ["train", "training", "eval", "evaluate"]:
        raise ValueError("Dataloader mode must be 'train' or 'evalute'.")
    
    if mode in ["train", "training"]:
        return torch.utils.data.DataLoader(
            dataset,
            drop_last=True,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor
        )
    else:  
        return torch.utils.data.DataLoader(
            dataset,
            drop_last=False,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor
        )

