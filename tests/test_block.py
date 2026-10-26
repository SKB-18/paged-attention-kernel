"""
Unit tests for PhysicalTokenBlock atomic page frame.
"""

import pytest
from memory.block import PhysicalTokenBlock


def test_block_initialization():
    """Verify block is initialized with correct capacity and zero ref_count."""
    block = PhysicalTokenBlock(block_id=5, block_size=16)
    assert block.block_id == 5
    assert block.block_size == 16
    assert block.ref_count == 0
    assert block.num_tokens == 0
    assert block.is_free
    assert not block.is_full
    assert not block.is_shared


def test_block_append_token():
    """Verify token appending advances slot index and detects full state."""
    block = PhysicalTokenBlock(block_id=0, block_size=4)
    slots = [block.append_token() for _ in range(4)]

    assert slots == [0, 1, 2, 3]
    assert block.is_full
    assert block.num_tokens == 4

    # Exceeding capacity must raise RuntimeError
    with pytest.raises(RuntimeError, match="Cannot append to full block"):
        block.append_token()


def test_block_shared_state():
    """Verify is_shared property based on reference count."""
    block = PhysicalTokenBlock(block_id=1, block_size=16)
    block.ref_count = 1
    assert not block.is_shared
    assert not block.is_free

    block.ref_count = 2
    assert block.is_shared


def test_block_reset():
    """Verify reset clears reference count and token occupancy."""
    block = PhysicalTokenBlock(block_id=2, block_size=16)
    block.ref_count = 3
    block.append_token()
    block.append_token()

    block.reset()
    assert block.ref_count == 0
    assert block.num_tokens == 0
    assert block.is_free
