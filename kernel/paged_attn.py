"""
PagedAttention Kernel: Scaled Dot-Product Attention over Block-Paged KV Caches (Kwon et al., SOSP 2023).
"""

from typing import Optional, List, Tuple
import math
import torch
import torch.nn.functional as F


def paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int = 16,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Computes scaled dot-product attention directly over non-contiguous block-paged KV caches.

    Args:
        query: Query tensor for the current decode step.
               Shape: [batch_size, num_heads, head_dim] or [batch_size, 1, num_heads, head_dim]
        key_cache: Physical Key storage tensor.
                   Shape: [num_blocks, block_size, num_kv_heads, head_dim]
        value_cache: Physical Value storage tensor.
                     Shape: [num_blocks, block_size, num_kv_heads, head_dim]
        block_tables: Physical block indices for each sequence.
                      Shape: [batch_size, max_blocks_per_seq] (dtype: int64 or int32)
        context_lens: Current total sequence length (including new token) for each sequence.
                      Shape: [batch_size] (dtype: int64 or int32)
        block_size: Number of tokens per physical block (default 16).
        scale: Scaling factor for dot product (default: 1.0 / sqrt(head_dim)).

    Returns:
        Output tensor of shape [batch_size, num_heads, head_dim]
    """
    # Normalize query shape
    if query.dim() == 4:
        query = query.squeeze(1)  # [batch_size, 1, num_heads, head_dim] -> [batch_size, num_heads, head_dim]
    elif query.dim() != 3:
        raise ValueError(f"Expected query of dim 3 or 4, got shape {query.shape}")

    batch_size, num_heads, head_dim = query.shape
    num_blocks, blk_sz, num_kv_heads, kv_head_dim = key_cache.shape

    if kv_head_dim != head_dim:
        raise ValueError(f"head_dim mismatch: query={head_dim}, key_cache={kv_head_dim}")
    if blk_sz != block_size:
        raise ValueError(f"block_size mismatch: arg={block_size}, cache={blk_sz}")

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # GQA ratio (heads per KV head)
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
    gqa_ratio = num_heads // num_kv_heads

    outputs = []

    # Execute attention per sequence using block table indirection
    for b in range(batch_size):
        seq_len = int(context_lens[b].item())
        if seq_len <= 0:
            outputs.append(torch.zeros(num_heads, head_dim, device=query.device, dtype=query.dtype))
            continue

        num_seq_blocks = math.ceil(seq_len / block_size)
        seq_block_ids = block_tables[b, :num_seq_blocks]

        # Gather physical blocks for sequence b
        # [num_seq_blocks, block_size, num_kv_heads, head_dim]
        k_blocks = key_cache[seq_block_ids]
        v_blocks = value_cache[seq_block_ids]

        # Flatten blocks to contiguous token dimension: [num_seq_blocks * block_size, num_kv_heads, head_dim]
        k_seq = k_blocks.view(-1, num_kv_heads, head_dim)[:seq_len]
        v_seq = v_blocks.view(-1, num_kv_heads, head_dim)[:seq_len]

        # Permute to [num_kv_heads, seq_len, head_dim]
        k_seq = k_seq.permute(1, 0, 2)
        v_seq = v_seq.permute(1, 0, 2)

        # Handle GQA if num_heads > num_kv_heads
        if gqa_ratio > 1:
            k_seq = k_seq.repeat_interleave(gqa_ratio, dim=0)
            v_seq = v_seq.repeat_interleave(gqa_ratio, dim=0)

        # q_b: [num_heads, 1, head_dim]
        q_b = query[b].unsqueeze(1)

        # Attention scores: [num_heads, 1, head_dim] @ [num_heads, head_dim, seq_len] -> [num_heads, 1, seq_len]
        scores = torch.matmul(q_b, k_seq.transpose(-2, -1)) * scale
        weights = F.softmax(scores, dim=-1)

        # Context output: [num_heads, 1, seq_len] @ [num_heads, seq_len, head_dim] -> [num_heads, 1, head_dim]
        out_b = torch.matmul(weights, v_seq).squeeze(1)
        outputs.append(out_b)

    return torch.stack(outputs, dim=0)


def reconstruct_contiguous_kv(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reconstructs contiguous [batch_size, num_kv_heads, max_seq_len, head_dim] tensors
    and an attention mask from paged memory. Useful for debugging and baseline verification.
    """
    batch_size = len(context_lens)
    max_len = int(context_lens.max().item()) if batch_size > 0 else 0
    num_blocks, _, num_kv_heads, head_dim = key_cache.shape

    device = key_cache.device
    dtype = key_cache.dtype

    k_out = torch.zeros(batch_size, num_kv_heads, max_len, head_dim, device=device, dtype=dtype)
    v_out = torch.zeros(batch_size, num_kv_heads, max_len, head_dim, device=device, dtype=dtype)
    mask = torch.full((batch_size, 1, 1, max_len), float("-inf"), device=device, dtype=dtype)

    for b in range(batch_size):
        seq_len = int(context_lens[b].item())
        if seq_len == 0:
            continue
        num_seq_blocks = math.ceil(seq_len / block_size)
        seq_block_ids = block_tables[b, :num_seq_blocks]

        k_blocks = key_cache[seq_block_ids].view(-1, num_kv_heads, head_dim)[:seq_len]
        v_blocks = value_cache[seq_block_ids].view(-1, num_kv_heads, head_dim)[:seq_len]

        k_out[b, :, :seq_len, :] = k_blocks.permute(1, 0, 2)
        v_out[b, :, :seq_len, :] = v_blocks.permute(1, 0, 2)
        mask[b, 0, 0, :seq_len] = 0.0

    return k_out, v_out, mask
