"""
Distributed Training Helpers for OrchestraNet.

Provides PyTorch DDP (DistributedDataParallel) utilities for multi-GPU
training on AWS instances with multiple GPUs.

Usage:
    # Launch with torchrun (recommended):
    torchrun --nproc_per_node=4 training/train_joint.py --ddp

    # In training code:
    from orchestranet.utils.distributed import setup_ddp, cleanup_ddp, is_main_process
    setup_ddp()
    model = DDP(model, device_ids=[local_rank])
    ...
    if is_main_process():
        save_checkpoint(...)
    cleanup_ddp()
"""

import os
import torch
import torch.distributed as dist


def setup_ddp():
    """
    Initialize DDP process group.

    Reads from environment variables set by torchrun:
      - RANK, LOCAL_RANK, WORLD_SIZE
      - MASTER_ADDR, MASTER_PORT
    """
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")

    if dist.is_initialized():
        return

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size <= 1:
        print("   ℹ️  Single GPU mode, skipping DDP init")
        return

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )

    print(f"   DDP initialized: rank={rank}, local_rank={local_rank}, world_size={world_size}")


def cleanup_ddp():
    """Clean up DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main (rank 0) process."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """Get current process rank."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """Get local rank (GPU index on this node)."""
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    """Get total number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


@torch.no_grad()
def reduce_tensor(tensor: torch.Tensor, op=dist.ReduceOp.AVG) -> torch.Tensor:
    """
    All-reduce a tensor across all processes.

    Args:
        tensor: Tensor to reduce
        op: Reduction operation (AVG, SUM, etc.)

    Returns:
        Reduced tensor (same on all processes)
    """
    if not dist.is_initialized() or get_world_size() <= 1:
        return tensor

    cloned = tensor.clone()
    dist.all_reduce(cloned, op=op)
    return cloned


def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()
