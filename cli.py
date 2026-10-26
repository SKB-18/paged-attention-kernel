"""
Interactive and Scriptable CLI for PagedAttention & Block-Paged KV Cache Engine.
"""

import argparse
import sys
import os
import math
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from memory.allocator import BlockAllocator
from memory.block_table import BlockTable
from kernel.paged_attn import paged_attention, reconstruct_contiguous_kv
from kernel.naive_attn import naive_attention
from serving.cache_manager import PagedKVCacheManager


def cmd_inspect(args):
    """Displays virtual-to-physical block table mapping and memory pool telemetry."""
    print("=" * 65)
    print(f" PagedAttention Memory Pool Inspector (Pool: {args.blocks} Blocks, Block Size: {args.block_size})")
    print("=" * 65)

    manager = PagedKVCacheManager(num_blocks=args.blocks, block_size=args.block_size)

    # Allocate a few synthetic sequences
    lens = [28, 45, 12, 60]
    for i, length in enumerate(lens):
        k = torch.randn(length, 4, 64)
        v = torch.randn(length, 4, 64)
        manager.allocate_prompt(seq_id=100 + i, prompt_keys=k, prompt_values=v)

    stats = manager.allocator.get_stats()
    print(f"  • Total Physical Blocks:       {stats['total_blocks']}")
    print(f"  • Allocated Blocks:            {stats['allocated_blocks']}")
    print(f"  • Free Pool Blocks:            {stats['free_blocks']}")
    print(f"  • Pool Utilization:            {stats['utilization_pct']:.1f}%")
    print(f"  • Active Stored Tokens:        {stats['active_tokens']}")
    print("-" * 65)
    print(" Active Sequence Page Tables:")
    for seq_id, table in manager.allocator.block_tables.items():
        total_tokens = manager.seq_lengths[seq_id]
        print(f"   Seq ID {seq_id:3d} ({total_tokens:2d} tokens) -> Physical Blocks: {table.physical_block_ids}")
    print("=" * 65)


def cmd_cow(args):
    """Demonstrates Copy-on-Write (CoW) memory sharing and mutation isolation."""
    print("=" * 65)
    print(" Copy-on-Write (CoW) Forking & Mutation Isolation Demo")
    print("=" * 65)

    manager = PagedKVCacheManager(num_blocks=64, block_size=16)

    # 1. Allocate parent sequence prompt (32 tokens = exactly 2 blocks)
    prompt_len = 32
    k_prompt = torch.randn(prompt_len, 4, 64)
    v_prompt = torch.randn(prompt_len, 4, 64)
    manager.allocate_prompt(seq_id=1, prompt_keys=k_prompt, prompt_values=v_prompt)

    parent_table = manager.allocator.block_tables[1]
    print(f"  [Step 1] Parent (Seq 1) Allocated:  Blocks = {parent_table.physical_block_ids}")
    for bid in parent_table.physical_block_ids:
        print(f"           Physical Block {bid:2d} ref_count = {manager.allocator.blocks[bid].ref_count}")

    # 2. Fork into 2 child sequences (parallel sampling)
    manager.fork_sequence(parent_seq_id=1, child_seq_id=2)
    manager.fork_sequence(parent_seq_id=1, child_seq_id=3)

    print("\n  [Step 2] Forked Child 2 & Child 3 (Zero Allocation Copy):")
    for bid in parent_table.physical_block_ids:
        print(f"           Physical Block {bid:2d} ref_count = {manager.allocator.blocks[bid].ref_count} (Shared by 3 seqs)")

    # 3. Append token to Child 2 (Triggers Copy-on-Write split on active block)
    print("\n  [Step 3] Append token to Child 2 (Writing to shared block):")
    k_token = torch.randn(4, 64)
    v_token = torch.randn(4, 64)
    manager.append_token(seq_id=2, key_token=k_token, value_token=v_token)

    table2 = manager.allocator.block_tables[2]
    print(f"           Child 2 Page Table:       {table2.physical_block_ids} (Split to new physical block!)")
    print(f"           Parent (Seq 1) Table:     {parent_table.physical_block_ids} (Untouched)")
    print("=" * 65)


def cmd_parity(args):
    """Verifies numerical equivalence between PagedAttention and standard attention."""
    from benchmarks.run_benchmark import benchmark_numerical_parity
    res = benchmark_numerical_parity()
    print(f" Result: Parity Passed = {res['parity_passed']} (Max Err: {res['max_difference']:.8e})")


def cmd_benchmark(args):
    """Runs full benchmark suite."""
    from benchmarks.run_benchmark import main as run_bench
    run_bench()


def main():
    parser = argparse.ArgumentParser(description="paged-attention-kernel CLI: Block-Paged KV Cache Engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # inspect
    p_insp = subparsers.add_parser("inspect", help="Inspect physical memory pool and page tables")
    p_insp.add_argument("--blocks", type=int, default=64, help="Total physical blocks in pool")
    p_insp.add_argument("--block-size", type=int, default=16, help="Tokens per physical block")
    p_insp.set_defaults(func=cmd_inspect)

    # cow
    p_cow = subparsers.add_parser("cow", help="Demonstrate Copy-on-Write (CoW) memory sharing")
    p_cow.set_defaults(func=cmd_cow)

    # parity
    p_parity = subparsers.add_parser("parity", help="Verify numerical parity with standard attention")
    p_parity.set_defaults(func=cmd_parity)

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run full empirical benchmark suite")
    p_bench.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
