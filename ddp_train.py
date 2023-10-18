import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset

from ddp_modules import ddp
import ddp_modules.ddp_data_class as ddpData

if __name__ == "__main__":

    # Variables
    n_features = 10
    batch_size = 4

    # Initialize process
    process_config = ddp.init_process()
    title = f"#####  Process config (rank {process_config['rank']}) #####"
    print(title + "\n", process_config,"\n")

    # Model
    model = torch.nn.Linear(n_features, n_features, True).to(process_config['device'])
    if process_config["is_ddp"]:
        model = DDP(model, device_ids=[process_config["local_rank"]])
    
    title = f"#####  Devices (rank {process_config['rank']}) #####"
    print(title + "\n" + f"N_gpus: {torch.cuda.device_count()}"\
        + f"\tInitialized GPU: {torch.cuda.current_device()}"\
        + f"\tModel GPU: {model.device}" + "\n")


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
    n_samples = 2*process_config["world_size"]*batch_size\
        + process_config["world_size"]//2 # Want uneven samples per process
    dataset = RandData(
        n_features,
        n_samples=n_samples,
        rank=process_config["rank"],
        world_size=process_config["world_size"]
    )
    print(f"######  Final Dataset (rank {process_config['rank']})  #####"\
                + "\nLength: " + str(len(dataset))
                + "\tShape: " + str(dataset.data_x.shape) + "\n")
    
    # Dataloader
    train_loader = ddpData.get_dataloader(
        dataset,
        "train",
        batch_size=batch_size,
        shuffle=True
    )


    ######################
    #####  Training  #####
    ######################

    dist.barrier()
    loss_fxn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(1):
        for batch_idx, batch in enumerate(train_loader):
            data_x, data_y = batch
            data_x = data_x.to(process_config["device"])
            data_y = data_y.to(process_config["device"])
            optimizer.zero_grad(set_to_none=True)

            predict = model(data_x)
            loss = loss_fxn(predict, data_y)
            print(f"Train info (rank {process_config['rank']})"\
                + f"\tEpoch: {epoch}\tBatch: {batch_idx}"\
                + f"\tBatch Size: {data_x.shape}"
            )

            # Backward and optimize
            loss.backward()
            optimizer.step()


    # Destroy process
    ddp.end_process()

    print(process_config['rank'])
