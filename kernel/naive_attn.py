"""
Baseline Standard Contiguous Attention for numerical parity verification.
"""

from typing import Optional
import math
import torch
import torch.nn.functional as F


def naive_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Standard scaled dot-product attention with contiguous tensors.

    Args:
        query: [batch_size, num_heads, q_len, head_dim] or [batch_size, num_heads, head_dim]
        key:   [batch_size, num_heads, kv_len, head_dim]
        value: [batch_size, num_heads, kv_len, head_dim]
        mask:  Optional attention mask [batch_size, 1, q_len, kv_len]
        scale: Optional scaling factor (defaults to 1 / sqrt(head_dim))

    Returns:
        Tensor of shape [batch_size, num_heads, q_len, head_dim]
    """
    is_single_token = (query.dim() == 3)
    if is_single_token:
        # [batch_size, num_heads, head_dim] -> [batch_size, num_heads, 1, head_dim]
        query = query.unsqueeze(2)

    head_dim = query.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Q * K^T: [B, H, 1, D] @ [B, H, D, S] -> [B, H, 1, S]
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    if mask is not None:
        scores = scores + mask

    weights = F.softmax(scores, dim=-1)
    out = torch.matmul(weights, value)

    if is_single_token:
        out = out.squeeze(2)

    return out
