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


def evaluate_data(
        model,
        dataloader,
        metrics,
        is_ddp,
        device,
        output_rank=None,
        rank=None,
        names=None,
    ):
    if output_rank is not None and rank is None:
        raise ValueError(
            "If output_rank is specified then the rank of the calling"\
            + " process must be specified."
        )

    n_total_samples = 0
    output_sum = 0
    for batch_idx, batch in enumerate(dataloader):
        data_x, data_y = batch
        data_x = data_x.to(device)
        data_y = data_y.to(device)
        n_samples = data_x.shape[0]
        
        output = metrics(model(data_x), data_y)
        output_sum += torch.tensor(output)*n_samples
        n_total_samples += n_samples
    
    # Reduce values onto the master or all gpus via summation if using DDP
    reduced_values = torch.concat(
        [output_sum, torch.tensor([n_total_samples])]
    ).to(device)
    if is_ddp:
        if output_rank is not None:
            dist.reduce(reduced_values, output_rank, async_op=False)
        else:
            dist.all_reduce(reduced_values, async_op=False)
    
    # If names is specified then return a dictionary of the named results
    if names is None:
        results = reduced_values[:-1]/reduced_values[-1]
    else:
        if len(names) != len(reduced_values) - 1:
            raise ValueError("When specifying the metric names one"\
                " must give names for all metrics."
            )
        results = {
            names[idx] : reduced_values[idx]/reduced_values[-1]\
                for idx in range(len(names))
        }
    
    if output_rank is None or output_rank == rank:
        return results
    else:
        return None






