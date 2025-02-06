import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def get_local_device():
    device = os.environ['DEVICE']
    if device == 'cuda':
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device = f'cuda:{local_rank}'
    return device


def setup_distributed():
    """Initialize the distributed training environment."""
    assert torch.cuda.is_available()
    dist.init_process_group(backend='nccl')
    
    # Get distributed training details from environment
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = get_local_device()
    
    if 'cuda' in device:
        # Set up device
        torch.cuda.set_device(device)
    
    return rank, local_rank, world_size, device


def wrap_model_distributed(model, local_rank, compile):
    """Wrap model in DDP and prepare for training."""
    device = get_local_device()
    if 'cuda' in device:
        model = model.cuda()
    if compile:
        model = torch.compile(model)
    if 'cuda' in device:
        return DDP(model, device_ids=[local_rank])
    else:
        return model
