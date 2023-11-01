import os
import torch
import torch.distributed as dist

def init_process(
        backend : str = 'nccl',
        device_type : str = 'cuda',
        master_addr : str = None,
        master_port : str = None
    ):
    backend = backend.lower()

    if int(os.environ.get('RANK', -1)) == -1:
        config = {
            'is_ddp' : False,
            'is_master' : True,
            'rank' : 0,
            'world_size' : 1,
            'seed_offset' : 0,
            'device' : f"{device_type}:0" 
        }
    else:
        # Check if torch.distributed is available
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available") 
        if backend == 'nccl':
            assert dist.is_nccl_available()
        elif backend == 'mpi':
            assert dist.is_mpi_available()
        elif backend == 'gloo':
            assert dist.is_gloo_available()
        else:
            raise SyntaxWarning(f"Cannot check if {backend} is available.")
        
        # Creating process configuration file
        config = {
            'is_ddp' : True,
            'rank' : int(os.environ['RANK']),
            'local_rank' : int(os.environ['LOCAL_RANK']),
            'world_size' : int(os.environ['WORLD_SIZE']),
            'backend' : backend
        }
        config['device'] = f"{device_type}:{config['local_rank']}"
        config['is_master'] = config['rank'] == 0
        config['seed_offset'] = config['rank']

        # Set master IP address and port
        master_addr_port_none = (master_addr is None)\
            and (master_port is None)
        master_addr_port_not_none = (master_addr is not None)\
            and (master_port is not None)
        assert master_addr_port_none or master_addr_port_not_none
        
        if master_addr_port_not_none:
            os.environ['MASTER_ADDR'] = master_addr
            os.environ['MASTER_PORT'] = master_port
        
        # Initialize distributed process
        dist.init_process_group(
            backend=backend,
            world_size=config['world_size'],
            init_method='env://',
        )

        # Set device
        torch.cuda.set_device(config['device'])
 
    return config


def end_process():
    dist.destroy_process_group()