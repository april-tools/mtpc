import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    """Initialize the distributed training environment."""
    assert torch.cuda.is_available()
    dist.init_process_group(backend='nccl')
    
    # Get distributed training details from environment
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{local_rank}'
    
    # Set up device
    torch.cuda.set_device(device)
    
    return rank, local_rank, world_size, device

def wrap_model_distributed(model, local_rank, compile):
    """Wrap model in DDP and prepare for training."""
    model = model.cuda()
    if compile:
        model = torch.compile(model)
    # NOTE: Set FIND_UNUSED_PARAMS if you want your model to run
    # despite some of the parameters not receiving gradients
    find_unused = os.getenv('FIND_UNUSED_PARAMS', 'False').lower() == 'true'
    return DDP(model, device_ids=[local_rank], find_unused_parameters=find_unused)
