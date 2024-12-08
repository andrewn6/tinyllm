import time 
import os
import mmap
import numpy as np
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

class BlockType(Enum):
    KV_CACHE = "kv_cache"
    ATTENTION = "attention"
    INTERMEDIATE = "intermediate"
    TEMPORARY = "temporary"

@dataclass
class MemoryBlock:
    address: int
    size: int
    block_type: BlockType 
    is_free: bool = True 
    last_access: float = 0.0
    access_count: int = 0
    is_pinned: bool = False
    alginment: int = 64

class MemoryAllocator:
    """
    - KV-Cache prioritization
    - Block coaelescing and splitting
    - Dynamic defragmentation
    """

    def __init__(
            self,
            total_size: int = 4 * 1024 * 1024 * 1024,
            enable_mmap: bool = True,
            defrag_threshold: float = 0.7,
            kv_cache_reserve: float = 0.3
    ):
        self.total_size = total_size
        self.defrag_threshold = defrag_threshold
        self.kv_cache_reserve = int(total_size * kv_cache_reserve)
        self.enable_mmap = enable_mmap
        
        """
        Initialize a memory pool
        """
        if enable_mmap:
            self.memory = mmap.mmap(
                    -1,
                    total_size,
                    flags=mmap.PRIVATE | mmap.MAP_ANONYMOUS,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE
                )
        else:
            self.memory = bytearray(total_size)

        self.blocks: List[MemoryBlock] = [
                MemoryBlock(
                    address=0,
                    size=total_size,
                    block_type=BlockType.TEMPORARY
                )
        ]

        self.allocated_blocks: Dict[int, MemoryBlock] = {}
        self.block_type_map: Dict[BlockType, Set[int]] = {
                block_type: set() for block_type in BlockType
        }
        self.fragmentation_level: float = 0.0

    """
    Allocate memory block with given size and type
    """
    def allocate(
        self,
        size: int,
        block_type: BlockType,
        alignment: int = 64
    ) -> Optional[int]:
        aligned_size = (size + alignment - 1) & ~(alignment - 1)

        if block_type != BlockType.KV_CACHE:
            used_by_kv = sum(
                block.size for block in self.blocks
                if block.block_type == BlockType.KV_CACHE
            )
            if self.total_size - used_by_kv < self.kv_cache_reserve:
                self._defragment()
                return None

        best_block = None
        best_block_idx = -1

        for idx, block in enumerate(self.blocks):
            if block.is_free and block.size >= aligned_size:
                if best_block is None or block.size < best_block.size:
                    best_block = block
                    best_block_idx = idx

        if best_block is None:
            self._defragment()
            return self.allocate(size, block_type, alignment)

        if best_block.size > aligned_size * 1.5:
            self._split_block(best_block_idx, aligned_size)

        block = self.blocks[best_block_idx]
        block.is_free = False
        block.block_type = block_type
        block.last_access = time.time()
        block.access_count = 1
        block.alignment = alignment

        """
        Update tracking
        """
        self.allocated_blocks[block.address] = block
        self.block_type_map[block_type].add(block.address)

        self._update_fragmentation_level()
        return block.address
    
    def free(self, address: int):
        if address not in self.allocated_blocks:
            raise ValueError(f"Invalid address: {address}")

        block = self.allocated_blocks[address]
        block.is_free = True 
        block.access_count = 0

        self.block_type_map[block.block_type].remove(address)
        del self.allocated_blocks[address]

        self._coalesce_blocks()
        self._update_fragmentation_level()
    
    def _split_block(self, block_idx: int, size: int):
        original = self.blocks[block_idx]

        new_block = MemoryBlock(
                address=original.address + size,
                size=original.size - size,
                block_type=BlockType.TEMPORARY
        )

        original.size = size

        self.blocks.insert(block_idx + 1, new_block)
    
    def _coalesce_blocks(self):
        idx = 0
        while idx < len(self.blocks) - 1:
            current = self.blocks[idx]
            next_block = self.blocks[idx + 1]

            if current.is_free and next_block.is_free:
                current.size += next_block.size
                self.blocks.pop(idx + 1)
            else:
                idx += 1
    
    def _defragment(self):
        if self.fragmentation_level < self.defrag_threshold:
            return

        self.blocks.sort(key=lambda b: (b.is_pinned, -b.size))

        current_address = 0
        new_blocks = []

        for block in self.blocks:
            if not block.is_free:
                if block.address != current_address:
                    self._move_block(block, current_address)
                new_blocks.append(block)
                current_address += block.size

        if current_address < self.total_size:
            new_blocks.append(MemoryBlock(
                address=current_address,
                size=self.total_size - current_address,
                block_type=BlockType.TEMPORARY
            ))

        self.blocks = new_blocks
        self._update_fragmentation_level()

    def _move_block(self, block: MemoryBlock, new_address: int):
        if self.enable_mmap:
            self.memory.move(new_address, block.address, block.size)
        else:
            data = self.memory[block.address:block.address + block.size]
            self.memory[new_address:new_address + block.size] = data

        del self.allocated_blocks[block.address]
        block.address = new_address
        self.allocated_blocks[new_address] = block

    def _update_fragmentation_level(self):
        total_free = sum(b.size for b in self.blocks if b.is_free)
        largest_free = max((b.size for b in self.blocks if b.is_free), default=0)

        if total_free == 0:
            self.fragmentation_level = 0.0
        else:
            self.fragmentation_level = 1.0 - (largest_free / total_free)

    def get_stats(self) -> Dict:
        return {
            "total_size": self.total_size,
            "used_size": sum(b.size for b in self.blocks if not b.is_free),
            "free_size": sum(b.size for b in self.blocks if b.is_free),
            "fragmentation_level": self.fragmentation_level,
            "block_count": len(self.blocks),
            "allocated_count": len(self.allocated_blocks),
            "kv_cache_used": sum(
                b.size for b in self.blocks
                if b.block_type == BlockType.KV_CACHE
            ),
        }
