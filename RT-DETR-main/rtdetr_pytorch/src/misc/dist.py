"""
reference
- https://github.com/pytorch/vision/blob/main/references/detection/utils.py
- https://github.com/facebookresearch/detr/blob/master/util/misc.py#L406

by lyuwenyu
"""

import random
import numpy as np

import torch
import torch.nn as nn
import torch.distributed
import torch.distributed as tdist

from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DistributedSampler
from torch.utils.data.dataloader import DataLoader


def init_distributed():
    '''
    distributed setup
    args:
        backend (str), ('nccl', 'gloo')
    '''
    try:
        # # https://pytorch.org/docs/stable/elastic/run.html
        # LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  
        # RANK = int(os.getenv('RANK', -1))
        # WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

        tdist.init_process_group(init_method='env://', )
        torch.distributed.barrier()

        rank = get_rank()
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(device)

        setup_print(rank == 0)
        print('Initialized distributed mode...')

        return True

    except:
        print('Not init distributed mode.')
        return False


def setup_print(is_main):
    '''This function disables printing when not in master process
    '''
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_main or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_available_and_initialized():
    if not tdist.is_available():
        return False
    if not tdist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_available_and_initialized():
        return 0
    return tdist.get_rank()


def get_world_size():
    if not is_dist_available_and_initialized():
        return 1
    return tdist.get_world_size()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


"""
检查分布式环境：
如果分布式环境可用且已初始化，继续后续操作；否则，直接返回原始模型。
获取当前进程的排名：
获取当前进程在分布式环境中的排名。
替换批量归一化层（可选）：
如果 sync_bn 为 True，将模型中的普通批量归一化层替换为同步批量归一化层。
使用 DDP 包装模型：
使用 torch.nn.parallel.DistributedDataParallel 对模型进行包装，使其支持分布式训练。
返回包装后的模型：
返回经过 DDP 包装的模型，或者在分布式环境不可用时返回原始模型。
"""

def warp_model(model, find_unused_parameters=False, sync_bn=False, ):
    if is_dist_available_and_initialized():
        rank = get_rank()
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) if sync_bn else model
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=find_unused_parameters)
    return model


def warp_loader(loader, shuffle=False):
    if is_dist_available_and_initialized():
        sampler = DistributedSampler(loader.dataset, shuffle=shuffle)
        loader = DataLoader(loader.dataset,
                            loader.batch_size,
                            sampler=sampler,
                            drop_last=loader.drop_last,
                            collate_fn=loader.collate_fn,
                            pin_memory=loader.pin_memory,
                            num_workers=loader.num_workers, )
    return loader


def is_parallel(model) -> bool:
    # Returns True if model is of type DP or DDP
    return type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)


def de_parallel(model) -> nn.Module:
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def reduce_dict(data, avg=True):
    '''
    Args 
        data dict: input, {k: v, ...}
        avg bool: true
    '''
    world_size = get_world_size()
    if world_size < 2:
        return data

    with torch.no_grad():
        keys, values = [], []
        for k in sorted(data.keys()):
            keys.append(k)
            values.append(data[k])

        values = torch.stack(values, dim=0)
        tdist.all_reduce(values)

        if avg is True:
            values /= world_size

        _data = {k: v for k, v in zip(keys, values)}

    return _data


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    tdist.all_gather_object(data_list, data)
    return data_list


import time


def sync_time():
    '''sync_time
    '''
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return time.time()


def set_seed(seed):
    # fix the seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
