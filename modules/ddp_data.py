import os
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
import numpy as np

class DatasetDDP():
    def __init__(self, rank : int = 0, world_size : int = 1,):
        self.rank = rank
        self.world_size = world_size
        self.data_x = None
        self.data_y = None
        self.total_data_len = None
    
    def ddp_slice(self):
        if self.data_x is None or self.data_y is None:
            raise ValueError("Dataset classes inheriting from DatasetDDP must set self.data_x and self.data_y")
        self.total_data_len = len(self.data_x)
        slice_indices = np.arange(self.rank, len(self.data_x), self.world_size)
        self.data_x = self.data_x[slice_indices]
        self.data_y = self.data_y[slice_indices]


def get_dataloader(
            dataset,
            mode,
            process_config,
            batch_size=32,
            shuffle=True,
            num_workers=0,
            prefetch_factor=None,
            epoch=0,
        ):
    mode = mode.lower()
    if mode not in ["train", "training", "eval", "evaluate"]:
        raise ValueError("Dataloader mode must be 'train' or 'evalute'.")
    
    if mode in ["train", "training"]:
        train_data_sampler = DistributedTrainSampler(
            dataset, process_config, shuffle=shuffle
        )
        train_data_sampler.set_epoch(epoch)

        return torch.utils.data.DataLoader(
            dataset,
            sampler=train_data_sampler,
            drop_last=True,
            batch_size=batch_size,
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


class DistributedTrainSampler(Sampler):
    r"""
    DistributedTrainSampler is different from DistributedSampler.
    It does NOT add extra samples to make it evenly divisible. 
    Instead it drops extra samples so all the datasets across gpus have the same size.  
    DistributedTrainSampler should NOT be used for evaluating.
    The sampler may randomly drop extra samples.
    Shuffle is enabled by default.

    DistributedEvalSampler is for evaluation purpose where synchronization does not happen every epoch.
    Synchronization should be done outside the dataloader loop.

    Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        world_size (int, optional): Number of processes participating in
            distributed training. By default, :attr:`rank` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`world_size`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.

    .. warning::
        In distributed mode, calling the :meth`set_epoch(epoch) <set_epoch>` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset, process_config, shuffle=True, seed=0):
        self.dataset = dataset
        self.process_config = process_config
        self.epoch = 0
        self.num_samples = len(self.dataset)
        # Do not count extra samples in processes with extra data 
        if process_config['is_ddp'] and dataset.total_data_len is not None:
            self.num_samples =\
                self.dataset.total_data_len//self.process_config['world_size']

        self.shuffle = shuffle
        self.seed = seed + self.process_config['seed_offset']


    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
            indices = indices[:self.num_samples]
        else:
            indices = list(range(self.num_samples))

        assert(len(indices) == self.num_samples)
        return iter(indices)


    def __len__(self):
        return self.num_samples


    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): _epoch number.
        """
        self.epoch = epoch



