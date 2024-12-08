import torch
import torch.backends.mps

import time

import threading
from dataclasses import dataclass
from enum import Enum 
from typing import Dict, List, Tuple, Optional, Union
from contextlib import nullcontext

class DeviceType(Enum):
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"

def get_device_type(device: torch.device) -> DeviceType:
    if device.type == "cuda":
        return DeviceType.CUDA
    elif device.type == "mps":
        return DeviceType.MPS 
    return DeviceType.CPU

def get_best_device() -> torch.device:
    if torch.cuda.is_availaible():
        return torch.device("cuda")
    if torch.backend.mps.is_availaible():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass 
class CacheBlock:
    key_block: torch.Tensor
    value_block: torch.Tensor
    block_id: int
    num_tokens: int
    start_pos: int
    last_access: float = 0.0

class BlockManager:
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.block_size = block_size
        self.dtype = dtype
        self.device = device
        self.device_type = get_device_type(device)

        self.block_size_bytes = (
            block_size * num_heads * head_size *
            torch.finfo(dtype).bits // 8 * 2
        )

        if self.device_type is DeviceType.CUDA:
            self.setup_cuda()
        elif self.device_type is DeviceType.MPS:
            self.setup_mps()
        else:
            self.setup_cpu()

        self.free_blocks: List[CacheBlock] = []
        " ""Compute attention scores"""
        self.used_blocks: Dict[int, CacheBlock] = {}
        self.block_counter = 0

    def setup_cuda(self):
        self.gpu_props = torch.cuda.get_device_properties(self.device)
        self.available_memory = self.gpu_props.total_memory
        self.stream = torch.cuda.Stream(device=self.device)

    def setup_mps(self):
        """
        MPS doesn't provide memory info or streams ;(
        """
        self.stream = None 
        self.available_memory = None

    def setup_cpu(self):
        self.stream = None
        self.available_memory = None
    
    def allocate_block(self) -> Optional[CacheBlock]:
        if self.device_type == DeviceType.CUDA:
            if torch.cuda_memory_allocated(self.device) + self.block_size_bytes > self.available_memory * 0.95:
                torch.cuda.empty_cache()
                if torch.cuda.memory_allocated(self.device) + self.block_size_bytes > self.available_memory * 0.95:
                    return None 
        
        if self.device_type == DeviceType.CUDA:
            ctx = torch.cuda.stream(self.stream)
        else:
            ctx = nullcontext()

        with ctx:
            try:
                key_block = torch.empty(
                        (self.block_size, self.num_heads, self.head_size),
                        dtype=self.dtype,
                        device=self.device
                )
                value_block = torch.empty(
                        (self.block_size, self.num_heads, self.head_size),
                        dtype=self.dtype,
                        device=self.device
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                return None
        
        block = CacheBlock(
            key_block=key_block,
            value_block=value_block,
            block_id=self.block_counter,
            num_tokens=0,
            start_pos=0
        )
        self.block_counter += 1
        return block

class PagedKVCache:
    """
    Multi-platform KV Cache impl
    Supports NVIDIA, M1, and CPU
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_size: int,
        block_size: int = 16,
        max_seq_len: int = 8192,
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None
    ):
            self.num_layers = num_layers
            self.num_heads = num_heads
            self.head_size = head_size
            self.block_size = block_size
            self.max_seq_len = max_seq_len
            self.dtype = dtype
            self.device = device if device is not None else get_best_device()
            self.device_type = get_device_type(self.device)

            self.block_managers = [
                    BlockManager(
                        num_heads=num_heads,
                        head_size=head_size,
                        block_size=block_size,
                        dtype=dtype,
                        device=self.device
                    )
                    for _ in range(num_layers)
            ]

            self.sequence_blocks: Dict[int, List[List[CacheBlock]]] = {}

            self.stats = {"hits": 0, "misses": 0, "allocs": 0}
            self.lock = threading.Lock()
    
    """
    Allocate blocks according to device-specific optimization
    """
    def allocate_blocks(
        self,
        seq_id: int,
        num_tokens: int,
        layer_ids: Optional[List[int]] = None
    ) -> bool:
        if layer_ids is None:
            layer_ids = list(range(self.num_layers))

        num_blocks = (num_tokens + self.block_size - 1) // self.block_size

        with self.lock:
            if seq_id not in self.sequence_blocks:
                 self.sequence_blocks[seq_id] = [[] for _ in range(self.num_layers)]
                
            if self.device_type == DeviceType.CUDA:
                 ctx = torch.cuda.stream(self.block_managers[0].stream)
            else:
                 ctx = nullcontext()

            with ctx:
                 for layer_id in layer_ids:
                    manager = self.block_managers[layer_id]
                    blocks = []
                    
                    for _ in range(num_blocks):
                        block = manager.allocate_block()
                        if block is None:
                            for b in blocks:
                                self._free_block(b)
                            return False
                        blocks.append(block)
                    
                    self.sequence_blocks[seq_id][layer_id].extend(blocks)
            
            if self.device_type == DeviceType.CUDA:
                 self.block_managers[0].stream.synchronize()

            self.stats["allocs"] += num_blocks * len(layer_ids)
            return True
    """
    Get key-value cache for a given sequence and layer
    """
    def get_kv_cache(
        self,
        seq_id: int,
        layer_id: int,
        start_idx: int,
        end_idx: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        with self.lock:
            if seq_id not in self.sequence_blocks:
                self.stats["misses"] += 1
                return None

            blocks = self.sequence_blocks[seq_id][layer_id]
            start_block = start_idx // self.block_size 
            end_block = (end_idx + self.block_size  -1) 

            if start_block >= len(blocks) or end_block > len(blocks):
                self.stats["misses"] += 1 
                return None
            
            if self.device_type == DeviceType.CUDA:
                ctx = torch.cuda.stream(self.block_managers[layer_id].stream)
            else:
                ctx = nullcontext()
            
            with ctx:
                total_tokens = end_idx - start_idx
                key_cache = torch.empty(
                    (total_tokens, self.num_heads, self.head_size),
                    dtype=self.dtype,
                    device=self.device,
                )
                value_cache = torch.empty(
                    (total_tokens, self.num_heads, self.head_size),
                    dtype=self.dtype,
                    device=self.device
                )
                offset = 0
                for block in blocks[start_block:end_block]:
                    block.last_access = time.time()
                    tokens_to_copy = min(
                        self.block_size,
                        total_tokens - offset
                    )
                    key_cache[offset:offset + tokens_to_copy].copy_(
                        block.key_block[:tokens_to_copy],
                        non_blocking=self.device_type == DeviceType.CUDA
                    )
                    value_cache[offset:offset + tokens_to_copy].copy_(
                        block.value_block[:tokens_to_copy],
                        non_blocking=self.device_type == DeviceType.CUDA
                    )
                    offset += tokens_to_copy
            
            if self.device_type == DeviceType.CUDA:
                self.block_managers[layer_id].stream.synchronize()

            self.stats["hits"] += 1
            return key_cache, value_cache              
    """
    Update cache for a given sequence and layer
    """
    def update_kv_cache(
            self,
            seq_id: int,
            layer_id: int,
            position: int,
            key: torch.Tensor,
            value: torch.Tensor
    ):
        with self.lock:
            if seq_id not in self.sequence_blocks:
                return
            
            block_idx = position // self.block_size
            blocks = self.sequence_blocks[seq_id][layer_id]
            
            if block_idx >= len(blocks):
                return
            
            block = blocks[block_idx]
            pos_in_block = position % self.block_size

            if self.device_type == DeviceType.CUDA:
                ctx = torch.cuda.stream(self.block_managers[0].stream)
            else:
                ctx = nullcontext()

        
            with ctx:
                block.key_block[pos_in_block].copy_(
                    key,
                    non_blocking=self.device_type == DeviceType.CUDA
                )
                block.value_block[pos_in_block].copy_(
                    value,
                    non_blocking=self.device_type == DeviceType.CUDA
                )
                block.num_tokens = max(block.num_tokens, pos_in_block + 1)
                block.last_access = time.time()
            
            if self.device_type == DeviceType.CUDA:
                self.block_managers[0].stream.syncrhonize()
    
    def _free_block(self, block: CacheBlock):
        if self.device_type == DeviceType.CUDA:
            with torch.cuda.stream(self.block_managers[0].stream):
                block.key_block = None
                block.value_block = None
                torch.cuda.empty_cache()
        else:
            block.key_block = None
            block.value_block = None
    """
    Get memory stats
    """
    def get_memory_stats(self) -> Dict:
        stats = {}
        if self.device_type == DeviceType.CUDA:
            stats.update({
                "allocated": torch.cuda.memory_allocated(self.device),
                "reserved": torch.cuda.memory_reserved(self.device),
                "max_allocated": torch.cuda.max_memory_allocated(self.device),
            })
        elif self.device_type == DeviceType.MPS:
            stats["device"] = "mps"
        else:
            stats["device"] = "cpu"
        return {**stats, **self.stats}

    """
    Clear all cache 
    """
    def clear(self):
        with self.lock:
            for seq_blocks in self.sequence_blocks.values():
                for layer_blocks in seq_blocks:
                    for block in layer_blocks:
                        self._free_block(block)
            self.sequence_blocks.clear()
