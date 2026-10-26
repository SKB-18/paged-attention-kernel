"""
Paged KV Cache Manager: Coordinates physical memory buffers, block tables, and CoW data replication.
"""

from typing import List, Dict, Tuple, Optional
import math
import torch
try:
    from ..memory.allocator import BlockAllocator
    from ..memory.block_table import BlockTable
    from ..memory.block import PhysicalTokenBlock
except (ImportError, ValueError):
    from memory.allocator import BlockAllocator
    from memory.block_table import BlockTable
    from memory.block import PhysicalTokenBlock


class PagedKVCacheManager:
    """
    Coordinates multi-layer/multi-head physical KV cache tensors and block allocator.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int = 16,
        num_kv_heads: int = 4,
        head_dim: int = 64,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # Allocate contiguous physical KV tensor storage pools
        # Shape: [num_blocks, block_size, num_kv_heads, head_dim]
        self.key_cache = torch.zeros(
            (num_blocks, block_size, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self.value_cache = torch.zeros(
            (num_blocks, block_size, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )

        # Virtual memory allocator
        self.allocator = BlockAllocator(num_blocks=num_blocks, block_size=block_size)

        # Sequence lengths tracker: seq_id -> int
        self.seq_lengths: Dict[int, int] = {}

    def _copy_block_data(self, src_block_id: int, dst_block_id: int):
        """Copies tensor slices from src physical block to dst physical block during CoW split."""
        self.key_cache[dst_block_id].copy_(self.key_cache[src_block_id])
        self.value_cache[dst_block_id].copy_(self.value_cache[src_block_id])

    def allocate_prompt(
        self,
        seq_id: int,
        prompt_keys: torch.Tensor,
        prompt_values: torch.Tensor,
    ) -> BlockTable:
        """
        Allocates blocks and writes prompt keys/values into physical cache.

        Args:
            seq_id: Unique sequence identifier.
            prompt_keys: [seq_len, num_kv_heads, head_dim]
            prompt_values: [seq_len, num_kv_heads, head_dim]
        """
        seq_len = prompt_keys.shape[0]
        needed_blocks = math.ceil(seq_len / self.block_size)

        if not self.allocator.can_allocate(needed_blocks):
            raise MemoryError(
                f"Cannot allocate {needed_blocks} blocks for sequence {seq_id}. "
                f"Available: {self.allocator.num_free_blocks}"
            )

        table = self.allocator.allocate_sequence(seq_id)
        self.seq_lengths[seq_id] = seq_len

        # Fill physical blocks
        written = 0
        while written < seq_len:
            block = self.allocator.allocate()
            table.append(block.block_id)

            chunk_len = min(self.block_size, seq_len - written)
            # Copy chunk into physical block
            self.key_cache[block.block_id, :chunk_len].copy_(prompt_keys[written : written + chunk_len])
            self.value_cache[block.block_id, :chunk_len].copy_(prompt_values[written : written + chunk_len])

            block.num_tokens = chunk_len
            written += chunk_len

        return table

    def append_token(
        self,
        seq_id: int,
        key_token: torch.Tensor,
        value_token: torch.Tensor,
    ):
        """
        Appends a newly generated token's KV vector to the sequence cache.
        Handles block boundary allocation and Copy-on-Write splits automatically.

        Args:
            seq_id: Sequence ID.
            key_token: [num_kv_heads, head_dim]
            value_token: [num_kv_heads, head_dim]
        """
        table = self.allocator.block_tables[seq_id]
        current_len = self.seq_lengths.get(seq_id, 0)

        # Check if we need to allocate a brand new block
        if table.num_blocks == 0 or (current_len % self.block_size == 0):
            new_block = self.allocator.allocate()
            table.append(new_block.block_id)
            active_block_id = new_block.block_id
            active_block = new_block
            slot_idx = 0
        else:
            # Check if active block needs Copy-on-Write
            logical_idx = table.num_blocks - 1
            active_block_id = self.allocator.copy_on_write(
                sequence_id=seq_id,
                logical_block_index=logical_idx,
                copy_fn=self._copy_block_data,
            )
            active_block = self.allocator.blocks[active_block_id]
            slot_idx = current_len % self.block_size

        # Write vectors into physical slot
        self.key_cache[active_block_id, slot_idx].copy_(key_token)
        self.value_cache[active_block_id, slot_idx].copy_(value_token)
        active_block.num_tokens = slot_idx + 1
        self.seq_lengths[seq_id] = current_len + 1

    def fork_sequence(self, parent_seq_id: int, child_seq_id: int) -> BlockTable:
        """Forks sequence using zero-copy Copy-on-Write."""
        table = self.allocator.fork_sequence(parent_seq_id, child_seq_id)
        self.seq_lengths[child_seq_id] = self.seq_lengths[parent_seq_id]
        return table

    def free_sequence(self, seq_id: int):
        """Frees all blocks associated with the sequence."""
        self.allocator.free_sequence(seq_id)
        if seq_id in self.seq_lengths:
            del self.seq_lengths[seq_id]

    def get_paged_attention_inputs(
        self,
        seq_ids: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Constructs batched `block_tables` and `context_lens` tensors for kernel invocation.

        Returns:
            (block_tables: [batch_size, max_blocks], context_lens: [batch_size])
        """
        batch_size = len(seq_ids)
        if batch_size == 0:
            return (
                torch.empty((0, 0), dtype=torch.long, device=self.device),
                torch.empty((0,), dtype=torch.long, device=self.device),
            )

        context_lens = torch.tensor(
            [self.seq_lengths[sid] for sid in seq_ids],
            dtype=torch.long,
            device=self.device,
        )

        max_blocks = max(
            len(self.allocator.block_tables[sid]) for sid in seq_ids
        )

        block_tables = torch.zeros(
            (batch_size, max_blocks),
            dtype=torch.long,
            device=self.device,
        )

        for b, sid in enumerate(seq_ids):
            tbl = self.allocator.block_tables[sid]
            for logical_idx, p_id in enumerate(tbl.physical_block_ids):
                block_tables[b, logical_idx] = p_id

        return block_tables, context_lens
