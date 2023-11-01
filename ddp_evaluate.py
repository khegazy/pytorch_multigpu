import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset

from ddp_modules import ddp
import ddp_modules.ddp_data_class as ddpData

if __name__ == "__main__":

    # Variables
    relative_error_scale = 2
    n_features = 10
    batch_size = 4

    # Initialize process
    process_config = ddp.init_process()
    title = f"#####  Process config (rank {process_config['rank']}) #####"
    print(title + "\n", process_config,"\n")

    # Model
    class linearModel(torch.nn.Module):
        def __init__(self, n_features, scale) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(n_features)*scale)
        
        def forward(self, input):
            return torch.einsum('ba,ac->bc', input, self.weight)

    model = linearModel(n_features, relative_error_scale + 1.).to(process_config['device'])
    model_device = model.weight.device
    if process_config["is_ddp"]:
        model = DDP(model, device_ids=[process_config["local_rank"]])
        model_device = model.device
    
    title = f"#####  Devices (rank {process_config['rank']}) #####"
    print(title + "\n" + f"N_gpus: {torch.cuda.device_count()}"\
        + f"\tInitialized GPU: {torch.cuda.current_device()}"\
        + f"\tModel GPU: {model_device}" + "\n"
    )


    ##################
    #####  Data  #####
    ##################

    class RandData(Dataset, ddpData.DatasetDDP):
        def __init__(
                self,
                n_features : int,
                n_samples : int = 1000,
                rank: int = 0,
                world_size: int = 1,
                
        ):
            super().__init__(rank, world_size)
            self.n_features = n_features

            # Create data
            self.data_x, self.data_y = self.create_data(
                self.n_features, n_samples
            )
            print(f"######  Original Dataset (rank {self.rank})  #####"\
                + "\nShape : " + str(self.data_x.shape) + "\n")
            self.n_total_samples = len(self.data_x)
            self.ddp_slice()

        def create_data(self, n_features, n_samples):
            data_x = torch.rand((n_samples, n_features))
            return data_x, data_x
        
        def __len__(self):
            return self.data_x.shape[0]
        
        def __getitem__(self, index):
            return self.data_x[index], self.data_y[index]
    
    # Dataset
    n_samples = 3*process_config["world_size"]*batch_size\
        + process_config["world_size"]//2 # Want uneven samples per process
    dataset = RandData(
        n_features,
        n_samples=n_samples,
        rank=process_config["rank"],
        world_size=process_config["world_size"]
    )
    print(f"######  Final Dataset (rank {process_config['rank']})  #####"\
        + "\nLength: " + str(len(dataset))
        + "\tShape: " + str(dataset.data_x.shape) + "\n"
    )
    
    # Dataloader
    eval_loader = ddpData.get_dataloader(
        dataset,
        "evaluate",
        process_config,
        batch_size=batch_size,
        shuffle=True,
    )


    ########################
    #####  Evaluation  #####
    ########################

    def MAE(pred : float, truth : float):
        return torch.mean(torch.abs(pred - truth))
    
    def relative_error(pred : float, truth : float):
        return torch.mean(torch.abs(pred - truth)/truth)
    
    def MSE(pred : float, truth : float):
        return torch.mean((pred - truth)**2)
    
    def calculate_metrics(pred : float, truth : float):
        mae = MAE(pred, truth)
        rel_error = relative_error(pred, truth)
        mse = MSE(pred, truth)

        return mse, mae, rel_error
    
    metrics = ddpData.evaluate_data(
        model,
        eval_loader,
        calculate_metrics,
        process_config['is_ddp'],
        process_config['device'],
        #output_rank=0, # Specifying the process rank calls dist.reduce to this process
        #rank=process_config['rank'], # Must specify the current process rank if output_rank
    )

    # Destroy process
    if process_config['is_ddp']:
        ddp.end_process()
    
    # Print results
    if process_config['is_master']:
        print("\n"\
            + "#####################\n"\
            + "#####  Results  #####\n"\
            + "#####################\n"\
            + f"MSE: {metrics[0]}\n"\
            + f"MAE: {metrics[1]}\n"\
            + f"relative error: {metrics[2]}"
        )