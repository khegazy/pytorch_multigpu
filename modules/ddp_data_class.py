import os
import torch
import torch.distributed as dist

class ddp_data_base():
    def __init__(self, rank : int = 0, world_size : int = 1):
        self.rank = rank
        self.world_size = world_size
        self.data_x = None
        self.data_y = None
    
    def ddp_slice(self):
        raise NotImplementedError