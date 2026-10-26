"""
Block Table: Virtual-to-physical address translation table for sequence KV caches.
"""

from typing import List, Dict, Optional
import copy


class BlockTable:
    """
    Translates sequence logical block indices to physical memory block IDs.
    
    Analogous to a process page table in OS virtual memory architectures.
    """

    def __init__(self, sequence_id: int):
        self.sequence_id = sequence_id
        self.physical_block_ids: List[int] = []

    def append(self, physical_block_id: int):
        """Appends a new physical block to the sequence's logical table."""
        self.physical_block_ids.append(physical_block_id)

    def get_physical_block(self, logical_index: int) -> int:
        """Returns physical block ID for the given logical block index."""
        if logical_index < 0 or logical_index >= len(self.physical_block_ids):
            raise IndexError(
                f"Logical block index {logical_index} out of range [0, {len(self.physical_block_ids)}) "
                f"for sequence {self.sequence_id}"
            )
        return self.physical_block_ids[logical_index]

    def set_physical_block(self, logical_index: int, new_physical_block_id: int):
        """Replaces a physical block mapping (used during Copy-on-Write split)."""
        if logical_index < 0 or logical_index >= len(self.physical_block_ids):
            raise IndexError(f"Logical block index {logical_index} out of range.")
        self.physical_block_ids[logical_index] = new_physical_block_id

    @property
    def last_physical_block(self) -> Optional[int]:
        """Returns the active (most recently appended) physical block ID."""
        return self.physical_block_ids[-1] if self.physical_block_ids else None

    @property
    def num_blocks(self) -> int:
        """Number of allocated blocks in this table."""
        return len(self.physical_block_ids)

    def fork(self, new_sequence_id: int) -> "BlockTable":
        """Creates a shallow copy for parallel sampling or beam search branching."""
        child_table = BlockTable(sequence_id=new_sequence_id)
        child_table.physical_block_ids = list(self.physical_block_ids)
        return child_table

    def clear(self):
        """Clears all mapped block references."""
        self.physical_block_ids.clear()

    def __len__(self) -> int:
        return len(self.physical_block_ids)

    def __getitem__(self, index: int) -> int:
        return self.get_physical_block(index)

    def __repr__(self) -> str:
        return f"BlockTable(seq={self.sequence_id}, blocks={self.physical_block_ids})"
