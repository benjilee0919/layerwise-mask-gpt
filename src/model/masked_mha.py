"""
Masked Multi-Head Attention implementation with layer-wise window scheduling.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMultiHeadAttention(nn.Module):
    """Multi-head attention with configurable masking patterns."""

    def __init__(self, config, layer_idx: int = None, mask_config: dict = None):
        super().__init__()
        self.num_heads = config.n_head
        self.hidden_size = config.n_embd
        self.head_dim = self.hidden_size // self.num_heads
        self.layer_idx = layer_idx
        self.mask_config = mask_config or {}

        assert (
            self.hidden_size % self.num_heads == 0
        ), f"Hidden size {self.hidden_size} not divisible by num_heads {self.num_heads}"

        # Linear projections (Q, K, V in a single Linear)
        self.c_attn = nn.Linear(self.hidden_size, 3 * self.hidden_size)
        self.c_proj = nn.Linear(self.hidden_size, self.hidden_size)

        # Dropout
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Register buffer for causal mask (standard GPT-2)
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones(
                    (config.n_positions, config.n_positions),
                    dtype=torch.bool,
                )
            ).view(1, 1, config.n_positions, config.n_positions),
            persistent=False,
        )

    def _apply_window_mask(
        self,
        attention_mask: torch.Tensor,
        seq_length: int,
        window_size: int,
        group_idx: int = 0,
        num_groups: int = 1,
    ) -> torch.Tensor:
        """
        Apply sliding window mask with optional group offset.

        - attention_mask: base causal mask [1, 1, T, T]
        - window_size: causal window size
        - group_idx / num_groups: group-based offset for layer-wise scheduling
        """
        if window_size is None or window_size >= seq_length:
            return attention_mask

        device = attention_mask.device
        window_mask = torch.zeros(
            seq_length, seq_length, device=device, dtype=torch.bool
        )

        # Simple group-based shift inside the window
        step = max(1, window_size // max(1, num_groups))
        shift = group_idx * step

        for i in range(seq_length):
            base_start = i - window_size + 1
            base_end = i + 1

            start = max(0, base_start + shift)
            end = min(base_end + shift, i + 1)  # enforce causality (j <= i)

            if start >= end:
                # Fallback: at least attend to self
                start = i
                end = i + 1

            window_mask[i, start:end] = True

        window_mask = window_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
        if attention_mask is not None:
            combined_mask = attention_mask & window_mask
        else:
            combined_mask = window_mask

        return combined_mask

    def _apply_block_sparse_mask(
        self,
        attention_mask: torch.Tensor,
        seq_length: int,
        block_size: int,
        num_blocks: int,
    ) -> torch.Tensor:
        """Apply block sparse attention mask."""
        if block_size is None:
            return attention_mask

        device = attention_mask.device
        block_mask = torch.zeros(
            seq_length, seq_length, device=device, dtype=torch.bool
        )

        for i in range(0, seq_length, block_size):
            end_i = min(i + block_size, seq_length)
            # Attend to previous num_blocks blocks and current block
            for j in range(max(0, i - num_blocks * block_size), end_i, block_size):
                end_j = min(j + block_size, seq_length)
                if j <= i:
                    block_mask[i:end_i, j:end_j] = True

        block_mask = block_mask.unsqueeze(0).unsqueeze(0)
        if attention_mask is not None:
            combined_mask = attention_mask & block_mask
        else:
            combined_mask = block_mask

        return combined_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for masked multi-head attention.

        Signature is intentionally compatible with HuggingFace GPT-2:
        GPT2Block may pass arguments like `past_key_value`, `past_key_values`,
        or `head_mask` as keyword arguments.
        """

        # Support both "layer_past" and "past_key_value(s)" naming used by HF
        if layer_past is None:
            if "past_key_value" in kwargs and kwargs["past_key_value"] is not None:
                layer_past = kwargs["past_key_value"]
            elif "past_key_values" in kwargs and kwargs["past_key_values"] is not None:
                layer_past = kwargs["past_key_values"]

        batch_size, seq_length, _ = hidden_states.size()

        # QKV projection
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.hidden_size, dim=2)

        # Reshape for multi-head attention
        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Handle past key/value states for generation
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        # Store present key/value states for generation
        present = (key, value) if use_cache else None

        # Attention scores
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        key_length = key.size(-2)

        # Base causal mask from GPT-2 (bias buffer)
        causal_mask = self.bias[:, :, key_length - seq_length : key_length, :key_length]

        # Layer-specific masking based on configuration
        if self.mask_config:
            mask_type = self.mask_config.get("mask_type", "none")

            if mask_type == "sliding_window":
                window_size = self.mask_config.get("window_size")
                num_groups = self.mask_config.get("num_groups", 1)
                group_idx = self.mask_config.get("group_idx", 0)
                causal_mask = self._apply_window_mask(
                    causal_mask,
                    key_length,
                    window_size,
                    group_idx=group_idx,
                    num_groups=num_groups,
                )
            elif mask_type == "block_sparse":
                block_size = self.mask_config.get("block_size", 64)
                num_blocks = self.mask_config.get("num_blocks", 4)
                causal_mask = self._apply_block_sparse_mask(
                    causal_mask, key_length, block_size, num_blocks
                )

        # Apply causal + layer-wise mask
        mask_value = torch.finfo(attn_weights.dtype).min
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        # Apply input attention_mask (padding mask)
        if attention_mask is not None:
            # attention_mask: [batch_size, 1, 1, seq_length]
            attention_mask = attention_mask[:, :, :, :key_length]
            attn_weights = attn_weights + (attention_mask * mask_value)

        # (Optional) head_mask is ignored here, but we keep it in the signature
        # for compatibility with HuggingFace. If you want, you can apply it
        # to attn_weights per head.

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum
        attn_output = torch.matmul(attn_weights, value)

        # Reshape + projection
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_length, self.hidden_size)
        )
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class MaskedGPT2Attention(MaskedMultiHeadAttention):
    """GPT-2 style masked multi-head attention."""

    def __init__(self, config, layer_idx: int = None, mask_config: dict = None):
        super().__init__(config, layer_idx, mask_config)

        # GPT-2 specific initialization
        self.split_size = self.hidden_size
        self._init_weights()

    def _init_weights(self):
        """Initialize weights similar to GPT-2."""
        nn.init.normal_(self.c_attn.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.02)

        nn.init.zeros_(self.c_attn.bias)
        nn.init.zeros_(self.c_proj.bias)
