"""
Block Allocator: Manages the global pool of physical token blocks and Copy-on-Write semantics.
"""

from typing import List, Dict, Optional, Tuple, Callable
from collections import deque
import torch
from .block import PhysicalTokenBlock
from .block_table import BlockTable


class BlockAllocator:
    """
    Manages the global memory pool of physical KV blocks.

    Provides constant-time O(1) allocation and deallocation, reference tracking,
    and Copy-on-Write (CoW) split execution.
    """

    def __init__(self, num_blocks: int, block_size: int = 16):
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        self.num_blocks = num_blocks
        self.block_size = block_size

        # Create all physical blocks
        self.blocks: List[PhysicalTokenBlock] = [
            PhysicalTokenBlock(block_id=i, block_size=block_size)
            for i in range(num_blocks)
        ]

        # Free block pool (FIFO queue for uniform wear/allocation)
        self.free_blocks: deque[int] = deque(range(num_blocks))

        # Block tables for active sequences: sequence_id -> BlockTable
        self.block_tables: Dict[int, BlockTable] = {}

    @property
    def num_free_blocks(self) -> int:
        """Count of unallocated blocks currently in the free pool."""
        return len(self.free_blocks)

    @property
    def num_allocated_blocks(self) -> int:
        """Count of physical blocks currently referenced by at least one sequence."""
        return self.num_blocks - self.num_free_blocks

    @property
    def memory_utilization(self) -> float:
        """Fraction of physical memory pool actively allocated."""
        return self.num_allocated_blocks / float(self.num_blocks)

    def can_allocate(self, num_required: int = 1) -> bool:
        """Checks if the allocator has sufficient free blocks available."""
        return self.num_free_blocks >= num_required

    def allocate(self) -> PhysicalTokenBlock:
        """
        Allocates a single physical block from the free list in O(1) time.
        Raises OutOfMemoryError if the pool is exhausted.
        """
        if not self.free_blocks:
            raise MemoryError(
                f"Paged KV memory exhausted! 0/{self.num_blocks} free blocks available. "
                "Trigger sequence preemption or eviction."
            )
        block_id = self.free_blocks.popleft()
        block = self.blocks[block_id]
        block.reset()
        block.ref_count = 1
        return block

    def free(self, block_id: int):
        """
        Decrements a block's reference count. If count reaches zero, recycles it into the free pool.
        """
        if block_id < 0 or block_id >= self.num_blocks:
            raise IndexError(f"Invalid physical block ID {block_id}")

        block = self.blocks[block_id]
        if block.ref_count <= 0:
            raise RuntimeError(f"Double free detected on physical block {block_id}")

        block.ref_count -= 1
        if block.ref_count == 0:
            block.reset()
            self.free_blocks.append(block_id)

    def allocate_sequence(self, sequence_id: int) -> BlockTable:
        """Initializes and registers an empty block table for a new sequence."""
        if sequence_id in self.block_tables:
            raise ValueError(f"Sequence {sequence_id} is already registered in allocator.")
        table = BlockTable(sequence_id=sequence_id)
        self.block_tables[sequence_id] = table
        return table

    def free_sequence(self, sequence_id: int):
        """Frees all physical blocks mapped to the sequence and removes its table."""
        if sequence_id not in self.block_tables:
            return
        table = self.block_tables[sequence_id]
        for block_id in table.physical_block_ids:
            self.free(block_id)
        del self.block_tables[sequence_id]

    def fork_sequence(self, parent_seq_id: int, child_seq_id: int) -> BlockTable:
        """
        Executes zero-copy branching (CoW):
        Increments the reference count of all physical blocks in the parent's table,
        allowing the child sequence to share the prompt cache without data duplication.
        """
        if parent_seq_id not in self.block_tables:
            raise KeyError(f"Parent sequence {parent_seq_id} not found.")
        if child_seq_id in self.block_tables:
            raise ValueError(f"Child sequence {child_seq_id} already exists.")

        parent_table = self.block_tables[parent_seq_id]
        child_table = parent_table.fork(new_sequence_id=child_seq_id)

        # Increment reference count for all shared blocks
        for block_id in child_table.physical_block_ids:
            self.blocks[block_id].ref_count += 1

        self.block_tables[child_seq_id] = child_table
        return child_table

    def copy_on_write(
        self,
        sequence_id: int,
        logical_block_index: int,
        copy_fn: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """
        Ensures the physical block at logical_block_index is exclusively owned before writing.

        If the block is shared (ref_count > 1):
        1. Allocates a new physical block.
        2. Copies contents from old block to new block via copy_fn(src_id, dst_id).
        3. Copies metadata (num_tokens).
        4. Decrements ref_count on old block.
        5. Updates sequence's BlockTable with new block.
        6. Returns new block ID.

        If exclusively owned (ref_count == 1):
        Returns existing block ID without allocating.
        """
        table = self.block_tables[sequence_id]
        old_block_id = table.get_physical_block(logical_block_index)
        old_block = self.blocks[old_block_id]

        if old_block.ref_count == 1:
            return old_block_id

        # CoW split required
        new_block = self.allocate()
        new_block.num_tokens = old_block.num_tokens

        if copy_fn is not None:
            copy_fn(old_block_id, new_block.block_id)

        self.free(old_block_id)
        table.set_physical_block(logical_block_index, new_block.block_id)
        return new_block.block_id

    def get_stats(self) -> Dict[str, float]:
        """Returns diagnostic telemetry on memory allocation and fragmentation."""
        total_tokens_capacity = self.num_blocks * self.block_size
        active_tokens = sum(b.num_tokens for b in self.blocks if not b.is_free)
        shared_blocks = sum(1 for b in self.blocks if b.is_shared)

        return {
            "total_blocks": self.num_blocks,
            "allocated_blocks": self.num_allocated_blocks,
            "free_blocks": self.num_free_blocks,
            "utilization_pct": self.memory_utilization * 100.0,
            "active_tokens": active_tokens,
            "token_capacity": total_tokens_capacity,
            "shared_blocks_cow": shared_blocks,
        }
