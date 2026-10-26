"""
Unit tests for BlockAllocator memory pool management.
"""

import pytest
from memory.allocator import BlockAllocator


def test_allocator_initialization():
    """Verify pool initialization with specified block counts."""
    allocator = BlockAllocator(num_blocks=16, block_size=8)
    assert allocator.num_blocks == 16
    assert allocator.block_size == 8
    assert allocator.num_free_blocks == 16
    assert allocator.num_allocated_blocks == 0
    assert allocator.memory_utilization == 0.0


def test_allocator_allocate_and_free():
    """Verify O(1) block allocation and deallocation lifecycle."""
    allocator = BlockAllocator(num_blocks=4, block_size=16)

    b0 = allocator.allocate()
    assert b0.block_id == 0
    assert b0.ref_count == 1
    assert allocator.num_free_blocks == 3
    assert allocator.num_allocated_blocks == 1

    b1 = allocator.allocate()
    assert b1.block_id == 1
    assert allocator.num_free_blocks == 2

    # Free b0
    allocator.free(b0.block_id)
    assert allocator.num_free_blocks == 3
    assert allocator.blocks[0].is_free


def test_allocator_out_of_memory():
    """Verify MemoryError when pool is completely exhausted."""
    allocator = BlockAllocator(num_blocks=2, block_size=16)
    _ = allocator.allocate()
    _ = allocator.allocate()

    assert not allocator.can_allocate(1)
    with pytest.raises(MemoryError, match="Paged KV memory exhausted"):
        allocator.allocate()


def test_allocator_double_free_protection():
    """Verify RuntimeError when freeing a block that has ref_count 0."""
    allocator = BlockAllocator(num_blocks=4, block_size=16)
    b0 = allocator.allocate()
    allocator.free(b0.block_id)

    with pytest.raises(RuntimeError, match="Double free detected"):
        allocator.free(b0.block_id)


def test_allocator_invalid_block_id():
    """Verify IndexError on invalid physical block id."""
    allocator = BlockAllocator(num_blocks=4, block_size=16)
    with pytest.raises(IndexError, match="Invalid physical block ID"):
        allocator.free(99)


def test_allocator_sequence_lifecycle():
    """Verify allocating and freeing an entire sequence's blocks."""
    allocator = BlockAllocator(num_blocks=10, block_size=16)
    table = allocator.allocate_sequence(sequence_id=1)

    b0 = allocator.allocate()
    b1 = allocator.allocate()
    table.append(b0.block_id)
    table.append(b1.block_id)

    assert allocator.num_free_blocks == 8
    allocator.free_sequence(sequence_id=1)
    assert allocator.num_free_blocks == 10
    assert 1 not in allocator.block_tables


def test_allocator_duplicate_sequence_raises():
    """Verify ValueError when registering an existing sequence id."""
    allocator = BlockAllocator(num_blocks=4, block_size=16)
    allocator.allocate_sequence(1)
    with pytest.raises(ValueError, match="already registered"):
        allocator.allocate_sequence(1)


def test_allocator_can_allocate_checks():
    """Verify can_allocate handles edge boundaries."""
    allocator = BlockAllocator(num_blocks=3, block_size=16)
    assert allocator.can_allocate(3)
    assert not allocator.can_allocate(4)

    _ = allocator.allocate()
    assert allocator.can_allocate(2)
    assert not allocator.can_allocate(3)


def test_allocator_telemetry_stats():
    """Verify get_stats returns comprehensive memory metrics."""
    allocator = BlockAllocator(num_blocks=10, block_size=16)
    b0 = allocator.allocate()
    b0.append_token()
    b0.append_token()

    stats = allocator.get_stats()
    assert stats["total_blocks"] == 10
    assert stats["allocated_blocks"] == 1
    assert stats["free_blocks"] == 9
    assert stats["utilization_pct"] == 10.0
    assert stats["active_tokens"] == 2
    assert stats["token_capacity"] == 160
