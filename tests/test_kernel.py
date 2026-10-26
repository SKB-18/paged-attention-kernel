"""
Unit tests for PagedAttention kernel numerical equivalence, masking, and GQA.
"""

import pytest
import math
import torch
from kernel.paged_attn import paged_attention, reconstruct_contiguous_kv
from kernel.naive_attn import naive_attention
from serving.cache_manager import PagedKVCacheManager


def test_paged_attention_single_sequence_parity():
    """Verify exact numerical parity for a single sequence across block boundaries."""
    torch.manual_seed(42)
    seq_len = 35  # Spans 3 blocks with block_size=16
    block_size = 16
    num_heads = 4
    num_kv_heads = 4
    head_dim = 32

    manager = PagedKVCacheManager(
        num_blocks=16,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    k = torch.randn(seq_len, num_kv_heads, head_dim)
    v = torch.randn(seq_len, num_kv_heads, head_dim)
    manager.allocate_prompt(seq_id=1, prompt_keys=k, prompt_values=v)

    query = torch.randn(1, num_heads, head_dim)
    b_tables, c_lens = manager.get_paged_attention_inputs([1])

    # 1. PagedAttention
    out_paged = paged_attention(
        query=query,
        key_cache=manager.key_cache,
        value_cache=manager.value_cache,
        block_tables=b_tables,
        context_lens=c_lens,
        block_size=block_size,
    )

    # 2. Naive Attention Ground Truth
    # Reshape k, v to [1, num_heads, seq_len, head_dim]
    k_contig = k.unsqueeze(0).permute(0, 2, 1, 3)
    v_contig = v.unsqueeze(0).permute(0, 2, 1, 3)
    out_naive = naive_attention(query=query, key=k_contig, value=v_contig)

    torch.testing.assert_close(out_paged, out_naive, rtol=1e-5, atol=1e-6)


def test_paged_attention_batched_ragged_sequences():
    """Verify batched execution with heterogeneous sequence lengths."""
    torch.manual_seed(42)
    batch_size = 4
    lengths = [10, 48, 25, 64]
    block_size = 16
    num_heads = 4
    num_kv_heads = 4
    head_dim = 32

    manager = PagedKVCacheManager(num_blocks=32, block_size=block_size, num_kv_heads=num_kv_heads, head_dim=head_dim)

    for b, length in enumerate(lengths):
        k = torch.randn(length, num_kv_heads, head_dim)
        v = torch.randn(length, num_kv_heads, head_dim)
        manager.allocate_prompt(seq_id=b, prompt_keys=k, prompt_values=v)

    query = torch.randn(batch_size, num_heads, head_dim)
    b_tables, c_lens = manager.get_paged_attention_inputs(list(range(batch_size)))

    out_paged = paged_attention(
        query=query,
        key_cache=manager.key_cache,
        value_cache=manager.value_cache,
        block_tables=b_tables,
        context_lens=c_lens,
        block_size=block_size,
    )

    k_contig, v_contig, mask = reconstruct_contiguous_kv(
        manager.key_cache, manager.value_cache, b_tables, c_lens, block_size=block_size
    )
    out_naive = naive_attention(query=query, key=k_contig, value=v_contig, mask=mask)

    torch.testing.assert_close(out_paged, out_naive, rtol=1e-5, atol=1e-6)


def test_paged_attention_gqa_grouped_heads():
    """Verify Grouped Query Attention (GQA) with num_heads > num_kv_heads."""
    torch.manual_seed(42)
    num_heads = 8
    num_kv_heads = 2  # 4 query heads per KV head
    head_dim = 32
    block_size = 8
    seq_len = 20

    manager = PagedKVCacheManager(num_blocks=10, block_size=block_size, num_kv_heads=num_kv_heads, head_dim=head_dim)

    k = torch.randn(seq_len, num_kv_heads, head_dim)
    v = torch.randn(seq_len, num_kv_heads, head_dim)
    manager.allocate_prompt(seq_id=1, prompt_keys=k, prompt_values=v)

    query = torch.randn(1, num_heads, head_dim)
    b_tables, c_lens = manager.get_paged_attention_inputs([1])

    out_paged = paged_attention(
        query=query,
        key_cache=manager.key_cache,
        value_cache=manager.value_cache,
        block_tables=b_tables,
        context_lens=c_lens,
        block_size=block_size,
    )

    # Reference using repeated KV heads
    k_rep = k.repeat_interleave(4, dim=1).unsqueeze(0).permute(0, 2, 1, 3)
    v_rep = v.repeat_interleave(4, dim=1).unsqueeze(0).permute(0, 2, 1, 3)
    out_naive = naive_attention(query=query, key=k_rep, value=v_rep)

    torch.testing.assert_close(out_paged, out_naive, rtol=1e-5, atol=1e-6)


def test_paged_attention_dim_validation():
    """Verify ValueError is raised when query or cache dimensions mismatch."""
    manager = PagedKVCacheManager(num_blocks=4, block_size=16, num_kv_heads=2, head_dim=32)
    b_tables = torch.zeros(1, 1, dtype=torch.long)
    c_lens = torch.tensor([10], dtype=torch.long)

    # Query head_dim mismatch
    bad_query = torch.randn(1, 2, 64)
    with pytest.raises(ValueError, match="head_dim mismatch"):
        paged_attention(bad_query, manager.key_cache, manager.value_cache, b_tables, c_lens, block_size=16)

    # Block size mismatch
    good_query = torch.randn(1, 2, 32)
    with pytest.raises(ValueError, match="block_size mismatch"):
        paged_attention(good_query, manager.key_cache, manager.value_cache, b_tables, c_lens, block_size=8)


def test_paged_attention_empty_sequence():
    """Verify sequences with 0 context length return zeros without error."""
    manager = PagedKVCacheManager(num_blocks=4, block_size=16, num_kv_heads=2, head_dim=16)
    query = torch.randn(2, 2, 16)
    b_tables = torch.zeros(2, 1, dtype=torch.long)
    c_lens = torch.tensor([0, 0], dtype=torch.long)

    out = paged_attention(query, manager.key_cache, manager.value_cache, b_tables, c_lens, block_size=16)
    assert out.shape == (2, 2, 16)
    assert torch.all(out == 0.0)


def test_paged_attention_4d_query_input():
    """Verify query of shape [batch, 1, heads, dim] is accepted."""
    manager = PagedKVCacheManager(num_blocks=4, block_size=16, num_kv_heads=2, head_dim=16)
    k = torch.randn(8, 2, 16)
    v = torch.randn(8, 2, 16)
    manager.allocate_prompt(1, k, v)

    query_3d = torch.randn(1, 2, 16)
    query_4d = query_3d.unsqueeze(1)  # [1, 1, 2, 16]

    b_tables, c_lens = manager.get_paged_attention_inputs([1])
    out_3d = paged_attention(query_3d, manager.key_cache, manager.value_cache, b_tables, c_lens, block_size=16)
    out_4d = paged_attention(query_4d, manager.key_cache, manager.value_cache, b_tables, c_lens, block_size=16)

    torch.testing.assert_close(out_3d, out_4d)


def test_paged_attention_custom_scale():
    """Verify custom scale parameter influences attention weights."""
    manager = PagedKVCacheManager(num_blocks=4, block_size=16, num_kv_heads=2, head_dim=16)
    k = torch.randn(8, 2, 16)
    v = torch.randn(8, 2, 16)
    manager.allocate_prompt(1, k, v)

    query = torch.randn(1, 2, 16)
    b_tables, c_lens = manager.get_paged_attention_inputs([1])

    out_default = paged_attention(query, manager.key_cache, manager.value_cache, b_tables, c_lens, block_size=16)
    out_scaled = paged_attention(query, manager.key_cache, manager.value_cache, b_tables, c_lens, block_size=16, scale=10.0)

    assert not torch.allclose(out_default, out_scaled)
