"""
Unit tests for Copy-on-Write (CoW) memory sharing and fork splits.
"""

import pytest
from memory.allocator import BlockAllocator


def test_cow_fork_sequence_reference_counts():
    """Verify forking increments reference count on parent blocks without allocating new blocks."""
    allocator = BlockAllocator(num_blocks=10, block_size=16)

    # Sequence 1 owns 2 blocks
    t1 = allocator.allocate_sequence(sequence_id=1)
    b0 = allocator.allocate()
    b1 = allocator.allocate()
    t1.append(b0.block_id)
    t1.append(b1.block_id)

    assert allocator.num_free_blocks == 8
    assert b0.ref_count == 1
    assert b1.ref_count == 1

    # Fork to Sequence 2
    t2 = allocator.fork_sequence(parent_seq_id=1, child_seq_id=2)

    # Blocks are shared, no new physical blocks consumed
    assert allocator.num_free_blocks == 8
    assert t2.physical_block_ids == t1.physical_block_ids
    assert b0.ref_count == 2
    assert b1.ref_count == 2


def test_cow_split_on_write():
    """Verify writing to a shared block triggers an allocation and decrements old block ref_count."""
    allocator = BlockAllocator(num_blocks=10, block_size=16)

    t1 = allocator.allocate_sequence(sequence_id=1)
    b0 = allocator.allocate()
    t1.append(b0.block_id)

    # Fork
    t2 = allocator.fork_sequence(parent_seq_id=1, child_seq_id=2)
    assert b0.ref_count == 2

    # Trigger CoW on Sequence 2
    copied_pairs = []
    def record_copy(src, dst):
        copied_pairs.append((src, dst))

    new_bid = allocator.copy_on_write(sequence_id=2, logical_block_index=0, copy_fn=record_copy)

    # Must allocate a new block for Seq 2
    assert new_bid != b0.block_id
    assert t2[0] == new_bid
    assert t1[0] == b0.block_id  # Parent still has original block

    # Reference counts updated
    assert allocator.blocks[new_bid].ref_count == 1
    assert b0.ref_count == 1  # Decremented from 2 to 1
    assert copied_pairs == [(b0.block_id, new_bid)]


def test_cow_exclusive_block_no_split():
    """Verify calling copy_on_write on a block with ref_count 1 does not allocate a new block."""
    allocator = BlockAllocator(num_blocks=5, block_size=16)
    t1 = allocator.allocate_sequence(sequence_id=1)
    b0 = allocator.allocate()
    t1.append(b0.block_id)

    free_before = allocator.num_free_blocks
    bid = allocator.copy_on_write(sequence_id=1, logical_block_index=0)

    assert bid == b0.block_id
    assert allocator.num_free_blocks == free_before


def test_cow_free_parent_leaves_child_intact():
    """Verify freeing the parent sequence decrements ref_count but preserves the child's block."""
    allocator = BlockAllocator(num_blocks=5, block_size=16)
    t1 = allocator.allocate_sequence(sequence_id=1)
    b0 = allocator.allocate()
    t1.append(b0.block_id)

    t2 = allocator.fork_sequence(1, 2)
    assert b0.ref_count == 2

    # Free parent
    allocator.free_sequence(1)

    # Block 0 is still alive for child!
    assert b0.ref_count == 1
    assert not b0.is_free
    assert t2[0] == b0.block_id

    # Free child
    allocator.free_sequence(2)
    assert b0.ref_count == 0
    assert b0.is_free
