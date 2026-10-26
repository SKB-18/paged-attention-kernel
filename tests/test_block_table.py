"""
Unit tests for BlockTable virtual address translation.
"""

import pytest
from memory.block_table import BlockTable


def test_block_table_append_and_indexing():
    """Verify appending and indexing physical block mappings."""
    table = BlockTable(sequence_id=42)
    assert len(table) == 0
    assert table.last_physical_block is None

    table.append(10)
    table.append(25)
    table.append(3)

    assert len(table) == 3
    assert table[0] == 10
    assert table[1] == 25
    assert table[2] == 3
    assert table.last_physical_block == 3


def test_block_table_out_of_bounds():
    """Verify IndexError is raised when indexing non-existent logical blocks."""
    table = BlockTable(sequence_id=1)
    table.append(5)

    with pytest.raises(IndexError, match="out of range"):
        _ = table[1]

    with pytest.raises(IndexError, match="out of range"):
        _ = table[-1]


def test_block_table_set_physical_block():
    """Verify in-place physical block replacement for CoW splits."""
    table = BlockTable(sequence_id=1)
    table.append(5)
    table.append(6)

    table.set_physical_block(logical_index=1, new_physical_block_id=99)
    assert table[1] == 99


def test_block_table_fork():
    """Verify forking creates an independent child table with identical block mapping."""
    parent = BlockTable(sequence_id=10)
    parent.append(2)
    parent.append(4)

    child = parent.fork(new_sequence_id=20)
    assert child.sequence_id == 20
    assert child.physical_block_ids == [2, 4]

    # Mutating child does not mutate parent
    child.append(8)
    assert len(child) == 3
    assert len(parent) == 2


def test_block_table_clear():
    """Verify clearing the block table resets its size."""
    table = BlockTable(sequence_id=5)
    table.append(1)
    table.append(2)
    assert len(table) == 2
    table.clear()
    assert len(table) == 0
    assert table.last_physical_block is None
