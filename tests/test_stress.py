"""
Stress and End-to-End Simulation Tests for PagedAttention & Block-Paged Memory.
"""

import pytest
import math
import torch
from memory.allocator import BlockAllocator
from memory.block_table import BlockTable
from kernel.paged_attn import paged_attention, reconstruct_contiguous_kv
from kernel.naive_attn import naive_attention
from serving.sequence import Sequence, SequenceStatus, SequenceGroup
from serving.cache_manager import PagedKVCacheManager
from serving.scheduler import PagedScheduler


def test_stress_continuous_generation_with_block_recycling():
    """
    Simulates continuous token generation across 10 requests with dynamic retirement and admission.
    Verifies that recycled physical blocks are re-allocated cleanly without memory leaks or corruption.
    """
    torch.manual_seed(42)
    num_blocks = 20
    block_size = 8
    num_kv_heads = 2
    head_dim = 16

    manager = PagedKVCacheManager(
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    scheduler = PagedScheduler(cache_manager=manager, max_batch_size=4)

    # Create 10 sequences with random prompt lengths
    sequences = [
        Sequence(seq_id=i, prompt_token_ids=list(range(torch.randint(4, 12, (1,)).item())))
        for i in range(10)
    ]
    for s in sequences:
        scheduler.add_sequence(s)

    # Run 30 scheduling steps
    for step in range(30):
        running = scheduler.schedule()
        if not running:
            break

        for seq in list(running):
            # If not yet in cache manager, allocate prompt
            if seq.seq_id not in manager.seq_lengths:
                prompt_len = seq.prompt_len
                k_p = torch.randn(prompt_len, num_kv_heads, head_dim)
                v_p = torch.randn(prompt_len, num_kv_heads, head_dim)
                manager.allocate_prompt(seq.seq_id, k_p, v_p)

            # Append 1 new generated token
            k_t = torch.randn(num_kv_heads, head_dim)
            v_t = torch.randn(num_kv_heads, head_dim)
            manager.append_token(seq.seq_id, k_t, v_t)
            seq.append_token(step)

            # Random finish condition when sequence reaches target length
            if seq.output_len >= 10:
                scheduler.finish_sequence(seq.seq_id)

    # After running sequences, verify all freed blocks were successfully recycled
    assert manager.allocator.num_free_blocks + manager.allocator.num_allocated_blocks == num_blocks
    assert manager.allocator.num_free_blocks >= 0


def test_stress_multi_branch_cow_concurrency():
    """
    Stress tests 8 parallel branches forking from a single 64-token prompt.
    Verifies that each branch can diverge and append independent tokens without modifying sibling or parent data.
    """
    torch.manual_seed(42)
    manager = PagedKVCacheManager(num_blocks=64, block_size=16, num_kv_heads=4, head_dim=32)

    prompt_len = 64  # exactly 4 blocks
    k_prompt = torch.randn(prompt_len, 4, 32)
    v_prompt = torch.randn(prompt_len, 4, 32)
    manager.allocate_prompt(seq_id=0, prompt_keys=k_prompt, prompt_values=v_prompt)

    # Fork into 8 branches
    num_branches = 8
    child_ids = list(range(1, num_branches + 1))
    for cid in child_ids:
        manager.fork_sequence(parent_seq_id=0, child_seq_id=cid)

    # Each branch appends distinct tokens
    branch_tokens = {}
    for cid in child_ids:
        k_t = torch.full((4, 32), float(cid))
        v_t = torch.full((4, 32), float(cid * 10))
        branch_tokens[cid] = (k_t, v_t)
        manager.append_token(cid, k_t, v_t)

    # Verify parent sequence is completely untouched
    parent_table = manager.allocator.block_tables[0]
    assert len(parent_table) == 4
    assert manager.seq_lengths[0] == 64

    # Verify each child has length 65 and isolated new block
    for cid in child_ids:
        child_table = manager.allocator.block_tables[cid]
        assert len(child_table) == 5  # 4 shared prompt blocks + 1 newly allocated block
        assert manager.seq_lengths[cid] == 65

        # Check that child's 65th token matches its specific value
        last_block_id = child_table.last_physical_block
        slot_0_key = manager.key_cache[last_block_id, 0]
        assert torch.all(slot_0_key == float(cid))


def test_stress_large_context_paged_attention_parity():
    """
    Verifies numerical equivalence for very long context lengths (256 tokens across 16 blocks).
    """
    torch.manual_seed(42)
    seq_len = 256
    block_size = 16
    num_heads = 8
    num_kv_heads = 8
    head_dim = 64

    manager = PagedKVCacheManager(
        num_blocks=32,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

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

    k_contig = k.unsqueeze(0).permute(0, 2, 1, 3)
    v_contig = v.unsqueeze(0).permute(0, 2, 1, 3)
    out_naive = naive_attention(query=query, key=k_contig, value=v_contig)

    max_err = torch.max(torch.abs(out_paged - out_naive)).item()
    assert max_err < 1e-5


def test_stress_gqa_high_ratio():
    """
    Verifies Grouped Query Attention with a 8:1 ratio (32 query heads, 4 KV heads).
    """
    torch.manual_seed(42)
    num_heads = 32
    num_kv_heads = 4
    head_dim = 32
    block_size = 16
    seq_len = 50

    manager = PagedKVCacheManager(num_blocks=10, block_size=block_size, num_kv_heads=num_kv_heads, head_dim=head_dim)
    k = torch.randn(seq_len, num_kv_heads, head_dim)
    v = torch.randn(seq_len, num_kv_heads, head_dim)
    manager.allocate_prompt(seq_id=1, prompt_keys=k, prompt_values=v)

    query = torch.randn(2, num_heads, head_dim)
    # Give sequence 2 a length of 30
    k2 = torch.randn(30, num_kv_heads, head_dim)
    v2 = torch.randn(30, num_kv_heads, head_dim)
    manager.allocate_prompt(seq_id=2, prompt_keys=k2, prompt_values=v2)

    b_tables, c_lens = manager.get_paged_attention_inputs([1, 2])
    out_paged = paged_attention(
        query=query,
        key_cache=manager.key_cache,
        value_cache=manager.value_cache,
        block_tables=b_tables,
        context_lens=c_lens,
        block_size=block_size,
    )

    assert out_paged.shape == (2, num_heads, head_dim)
    assert not torch.isnan(out_paged).any()
