"""
Unit tests for PagedKVCacheManager lifecycle, token appending, and boundary crossing.
"""

import pytest
import torch
from serving.cache_manager import PagedKVCacheManager


def test_cache_manager_prompt_allocation():
    """Verify prompt allocation divides tokens into correct number of physical blocks."""
    manager = PagedKVCacheManager(num_blocks=10, block_size=16, num_kv_heads=2, head_dim=32)

    k = torch.randn(35, 2, 32)
    v = torch.randn(35, 2, 32)
    table = manager.allocate_prompt(seq_id=1, prompt_keys=k, prompt_values=v)

    # 35 tokens with block_size=16 requires 3 blocks (16 + 16 + 3)
    assert len(table) == 3
    assert manager.seq_lengths[1] == 35
    assert manager.allocator.blocks[table[0]].num_tokens == 16
    assert manager.allocator.blocks[table[1]].num_tokens == 16
    assert manager.allocator.blocks[table[2]].num_tokens == 3


def test_cache_manager_append_token_boundary_crossing():
    """Verify appending tokens across block boundaries automatically allocates a new block."""
    manager = PagedKVCacheManager(num_blocks=10, block_size=4, num_kv_heads=2, head_dim=16)

    # Allocate prompt of 4 tokens (fills block 0 completely)
    k_prompt = torch.randn(4, 2, 16)
    v_prompt = torch.randn(4, 2, 16)
    table = manager.allocate_prompt(seq_id=1, prompt_keys=k_prompt, prompt_values=v_prompt)

    assert len(table) == 1
    assert manager.seq_lengths[1] == 4

    # Append 5th token -> must allocate second block
    k_token = torch.randn(2, 16)
    v_token = torch.randn(2, 16)
    manager.append_token(seq_id=1, key_token=k_token, value_token=v_token)

    assert len(table) == 2
    assert manager.seq_lengths[1] == 5
    assert manager.allocator.blocks[table[1]].num_tokens == 1


def test_cache_manager_free_sequence():
    """Verify freeing sequence returns physical blocks to the allocator free pool."""
    manager = PagedKVCacheManager(num_blocks=10, block_size=16)

    k = torch.randn(32, 4, 64)
    v = torch.randn(32, 4, 64)
    manager.allocate_prompt(seq_id=1, prompt_keys=k, prompt_values=v)

    assert manager.allocator.num_free_blocks == 8
    manager.free_sequence(seq_id=1)
    assert manager.allocator.num_free_blocks == 10
    assert 1 not in manager.seq_lengths


def test_cache_manager_fork_and_cow_token_append():
    """Verify appending a token to a forked sequence isolates data and creates new physical block."""
    manager = PagedKVCacheManager(num_blocks=10, block_size=4, num_kv_heads=2, head_dim=16)

    k_p = torch.randn(3, 2, 16)
    v_p = torch.randn(3, 2, 16)
    manager.allocate_prompt(seq_id=1, prompt_keys=k_p, prompt_values=v_p)

    # Fork to child 2
    manager.fork_sequence(parent_seq_id=1, child_seq_id=2)
    t1 = manager.allocator.block_tables[1]
    t2 = manager.allocator.block_tables[2]
    assert t1.physical_block_ids == t2.physical_block_ids

    # Append token to child 2
    manager.append_token(seq_id=2, key_token=torch.randn(2, 16), value_token=torch.randn(2, 16))

    # Child 2 must have split its active block
    assert t2[0] != t1[0]
    assert manager.seq_lengths[2] == 4
    assert manager.seq_lengths[1] == 3
