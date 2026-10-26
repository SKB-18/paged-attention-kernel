"""
Physical Token Block: The atomic unit of memory allocation in PagedAttention.
"""

from typing import Optional


class PhysicalTokenBlock:
    """
    Represents a fixed-capacity contiguous physical memory block holding KV cache slots.
    
    Analogous to a physical page frame in an operating system.
    """

    def __init__(self, block_id: int, block_size: int = 16):
        self.block_id = block_id
        self.block_size = block_size
        self.ref_count = 0
        self.num_tokens = 0  # Number of valid tokens written in this block

    @property
    def is_free(self) -> bool:
        """Returns True if the block is not referenced by any logical sequence."""
        return self.ref_count == 0

    @property
    def is_full(self) -> bool:
        """Returns True if the block has reached its maximum token capacity."""
        return self.num_tokens >= self.block_size

    @property
    def is_shared(self) -> bool:
        """Returns True if multiple sequences share this physical block (CoW active)."""
        return self.ref_count > 1

    def append_token(self) -> int:
        """Increments token count and returns the slot index within this block."""
        if self.is_full:
            raise RuntimeError(f"Cannot append to full block {self.block_id} (capacity {self.block_size})")
        slot = self.num_tokens
        self.num_tokens += 1
        return slot

    def reset(self):
        """Resets the block state to unallocated."""
        self.ref_count = 0
        self.num_tokens = 0

    def __repr__(self) -> str:
        return (
            f"PhysicalTokenBlock(id={self.block_id}, tokens={self.num_tokens}/{self.block_size}, "
            f"ref_count={self.ref_count})"
        )
