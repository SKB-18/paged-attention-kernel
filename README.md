# PagedAttention & Block-Paged KV Cache Engine (`paged-attention-kernel`)

[![Tests](https://img.shields.io/badge/tests-37%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

Production-grade, tensor-level implementation of **vLLM's PagedAttention algorithm** and block-paged KV cache memory subsystem (Kwon et al., SOSP 2023). Models operating system virtual memory paging, page tables, dynamic allocation, and **Copy-on-Write (CoW)** to eliminate the 60%–80% memory fragmentation waste of large language model serving.

---

## 🏛️ System Architecture

```
                        ┌──────────────────────────────────────────────┐
                        │              Sequence Request                │
                        │     Logical Tokens: [t_0, t_1, ... t_35]     │
                        └──────────────────────┬───────────────────────┘
                                               │
                        ┌──────────────────────▼───────────────────────┐
                        │             Logical Block View               │
                        │  Block 0 [t0..t15] | Block 1 [t16..t31] | ...│
                        └──────────────────────┬───────────────────────┘
                                               │
                        ┌──────────────────────▼───────────────────────┐
                        │          Block Table (Page Table)            │
                        │    Logical 0 -> Physical Block 4             │
                        │    Logical 1 -> Physical Block 9             │
                        │    Logical 2 -> Physical Block 1             │
                        └──────────────────────┬───────────────────────┘
                                               │
                        ┌──────────────────────▼───────────────────────┐
                        │         Physical Memory Pool (RAM)           │
                        │  Block 0 [Unallocated]                       │
                        │  Block 1 [t32..t35] (ref_count = 1)          │
                        │  Block 4 [t0..t15]  (ref_count = 2) [Shared] │
                        │  Block 9 [t16..t31] (ref_count = 1)          │
                        └──────────────────────────────────────────────┘
```

---

## 🔬 Core Engineering Concepts

### 1. The KV Cache Memory Crisis
In standard LLM serving, KV caches are pre-allocated contiguously for the theoretical maximum sequence length (e.g. 2048 or 4096 tokens). Because actual completions are variable-length and cannot be known in advance, this causes:
- **Internal Fragmentation**: Reserved slots for tokens that are never generated.
- **External Fragmentation**: Memory allocators cannot satisfy requests because free space is split into non-contiguous fragments.
- **Over-Reservation**: Up to **85.7%** of memory is idle waste.

### 2. Operating System Paging Analogy
PagedAttention treats the KV cache as virtual memory:
- **Logical Blocks**: A sequence's tokens are partitioned into logical blocks of size $B$ (default 16 tokens).
- **Physical Page Frames**: Memory is pre-allocated as a pool of fixed-size physical blocks `[num_blocks, block_size, num_kv_heads, head_dim]`.
- **Page Table (`BlockTable`)**: Maintains the dynamic mapping from logical sequence blocks to non-contiguous physical blocks.
- **Dynamic Allocation on Demand**: Physical blocks are allocated only when tokens are generated, keeping internal fragmentation bounded to at most $B-1$ tokens per sequence.

### 3. Copy-on-Write (CoW) Parallel Sampling
During parallel sampling, speculative decoding verification, or beam search:
- Child sequences fork from the parent sequence by creating a new `BlockTable` pointing to the exact same physical blocks.
- Reference counts on shared blocks are incremented (`ref_count > 1`).
- Physical blocks are duplicated **only when a sequence mutates the block** (writing new tokens).
- Achieves **44.4% to 83.3% memory savings** across parallel branches.

---

## 📊 Empirical Benchmark Results

Measured via `benchmarks/run_benchmark.py`:

### 1. Memory Fragmentation Savings (32 Concurrent Requests)

| Metric | Contiguous Pre-allocation | PagedAttention | Waste Eliminated | Savings |
|:-------|:-------------------------:|:--------------:|:----------------:|:-------:|
| **KV Cache RAM** | 256.00 MB | 36.56 MB | **219.44 MB** | **85.7%** |

### 2. Copy-on-Write (CoW) Parallel Sampling Efficiency (512-Token Prompt)

| Parallel Branches ($K$) | Without CoW (Blocks) | With CoW (Blocks) | Memory Savings |
|:-----------------------:|:--------------------:|:-----------------:|:--------------:|
| **$K = 2$**             | 72 blocks            | 40 blocks         | **44.4%**      |
| **$K = 4$**             | 144 blocks           | 48 blocks         | **66.7%**      |
| **$K = 8$**             | 288 blocks           | 64 blocks         | **77.8%**      |
| **$K = 16$**            | 576 blocks           | 96 blocks         | **83.3%**      |

### 3. Numerical Parity vs. Contiguous Attention Ground Truth

| Metric | Measured Value | Threshold | Status |
|:-------|:--------------:|:---------:|:------:|
| **Maximum Absolute Error** | `0.00000000e+00` | $< 10^{-5}$ | **PASSED (Exact)** |
| **Mean Absolute Error** | `0.00000000e+00` | $< 10^{-6}$ | **PASSED (Exact)** |

---

## 🚀 Quickstart & CLI

```bash
# Clone and enter directory
git clone https://github.com/SKB-18/paged-attention-kernel.git
cd paged-attention-kernel

# Install dependencies
pip install -r requirements.txt
```

### Inspect Physical Memory Pool & Page Tables
```bash
python cli.py inspect --blocks 64 --block-size 16
```

### Run Copy-on-Write Demonstration
```bash
python cli.py cow
```

### Verify Numerical Parity
```bash
python cli.py parity
```

### Run Full Benchmark Suite
```bash
python cli.py benchmark
```

---

## 🧪 Automated Test Suite (37 Tests)

Run comprehensive verification:
```bash
pytest tests/ -v
```

```
tests/test_allocator.py::test_allocator_initialization PASSED            [  2%]
tests/test_allocator.py::test_allocator_allocate_and_free PASSED         [  5%]
tests/test_allocator.py::test_allocator_out_of_memory PASSED             [  8%]
tests/test_allocator.py::test_allocator_double_free_protection PASSED    [ 10%]
tests/test_allocator.py::test_allocator_invalid_block_id PASSED          [ 13%]
tests/test_allocator.py::test_allocator_sequence_lifecycle PASSED        [ 16%]
tests/test_allocator.py::test_allocator_duplicate_sequence_raises PASSED [ 18%]
tests/test_allocator.py::test_allocator_can_allocate_checks PASSED       [ 21%]
tests/test_allocator.py::test_allocator_telemetry_stats PASSED           [ 24%]
tests/test_block.py::test_block_initialization PASSED                    [ 27%]
tests/test_block.py::test_block_append_token PASSED                      [ 29%]
tests/test_block.py::test_block_shared_state PASSED                      [ 32%]
tests/test_block.py::test_block_reset PASSED                             [ 35%]
tests/test_block_table.py::test_block_table_append_and_indexing PASSED   [ 37%]
tests/test_block_table.py::test_block_table_out_of_bounds PASSED         [ 40%]
tests/test_block_table.py::test_block_table_set_physical_block PASSED    [ 43%]
tests/test_block_table.py::test_block_table_fork PASSED                  [ 45%]
tests/test_block_table.py::test_block_table_clear PASSED                 [ 48%]
tests/test_cache_manager.py::test_cache_manager_prompt_allocation PASSED [ 51%]
tests/test_cache_manager.py::test_cache_manager_append_token_boundary_crossing PASSED [ 54%]
tests/test_cache_manager.py::test_cache_manager_free_sequence PASSED     [ 56%]
tests/test_cache_manager.py::test_cache_manager_fork_and_cow_token_append PASSED [ 59%]
tests/test_cow.py::test_cow_fork_sequence_reference_counts PASSED        [ 62%]
tests/test_cow.py::test_cow_split_on_write PASSED                        [ 64%]
tests/test_cow.py::test_cow_exclusive_block_no_split PASSED              [ 67%]
tests/test_cow.py::test_cow_free_parent_leaves_child_intact PASSED       [ 70%]
tests/test_kernel.py::test_paged_attention_single_sequence_parity PASSED [ 72%]
tests/test_kernel.py::test_paged_attention_batched_ragged_sequences PASSED [ 75%]
tests/test_kernel.py::test_paged_attention_gqa_grouped_heads PASSED      [ 78%]
tests/test_kernel.py::test_paged_attention_dim_validation PASSED         [ 81%]
tests/test_kernel.py::test_paged_attention_empty_sequence PASSED          [ 83%]
tests/test_kernel.py::test_paged_attention_4d_query_input PASSED         [ 86%]
tests/test_kernel.py::test_paged_attention_custom_scale PASSED           [ 89%]
tests/test_scheduler.py::test_scheduler_add_and_schedule PASSED          [ 91%]
tests/test_scheduler.py::test_scheduler_finish_and_reclamation PASSED    [ 94%]
tests/test_scheduler.py::test_sequence_group_branching PASSED            [ 97%]
tests/test_scheduler.py::test_scheduler_preempt_sequence PASSED          [100%]

============================= 37 passed in 4.15s ==============================
```

---

## 📄 License
MIT License. Created by [SKB-18](https://github.com/SKB-18).
