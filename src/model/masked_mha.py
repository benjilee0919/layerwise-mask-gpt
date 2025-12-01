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
        # ===========================
        # ORIGINAL IMPLEMENTATION (COMMENTED OUT FOR REFERENCE)
        # ===========================
        #
        # if window_size is None or window_size >= seq_length:
        #     return attention_mask
        #
        # device = attention_mask.device
        # window_mask = torch.zeros(
        #     seq_length, seq_length, device=device, dtype=torch.bool
        # )
        #
        # # Simple group-based shift inside the window
        # step = max(1, window_size // max(1, num_groups))
        # shift = group_idx * step
        #
        # for i in range(seq_length):
        #     base_start = i - window_size + 1
        #     base_end = i + 1
        #
        #     start = max(0, base_start + shift)
        #     end = min(base_end + shift, i + 1)  # enforce causality (j <= i)
        #
        #     if start >= end:
        #         # Fallback: at least attend to self
        #         start = i
        #         end = i + 1
        #
        #     window_mask[i, start:end] = True
        #
        # window_mask = window_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
        # if attention_mask is not None:
        #     combined_mask = attention_mask & window_mask
        # else:
        #     combined_mask = window_mask
        #
        # return combined_mask
        #
        # ===========================
        # END ORIGINAL IMPLEMENTATION
        # ===========================

        # === LMS-style Sliding Window + Group Shift (Paper-Style Implementation) ===
        #
        # device = attention_mask.device
        #
        # # Compute offset based on group index (LMS scheme)
        # # offset is a fraction of the window assigned to this group
        # base_step = max(1, window_size // max(1, num_groups))
        # offset = group_idx * base_step
        #
        # window_mask = torch.zeros(seq_length, seq_length, device=device, dtype=torch.bool)
        #
        # for i in range(seq_length):
        #     # Base causal window: tokens in [i - window_size + 1, i]
        #     base_start = i - window_size + 1
        #     base_end = i + 1  # exclusive
        #
        #     # Apply vertical shift to the left boundary
        #     shifted_start = base_start + offset
        #
        #     # Enforce causality: cannot attend to future tokens
        #     final_end = base_end
        #
        #     # Clip to valid range
        #     final_start = max(0, shifted_start)
        #     final_end = min(final_end, i + 1)
        #
        #     # Safety: if shifting pushes us past the causal boundary,
        #     # fall back to the original causal window
        #     if final_start >= final_end:
        #         final_start = max(0, base_start)
        #         final_end = base_end
        #
        #     # Final safety: if still invalid, attend only to self
        #     if final_start >= final_end:
        #         final_start = i
        #         final_end = i + 1
        #
        #     window_mask[i, final_start:final_end] = True

        # window_mask = window_mask.unsqueeze(0).unsqueeze(0)
        #
        # if attention_mask is not None:
        #     combined_mask = attention_mask & window_mask
        # else:
        #     combined_mask = window_mask
        #
        # return combined_mask

        # === Stable Sliding Window Mask (no group shift; layer-wise window only) ===
        if window_size is None or window_size >= seq_length:
            return attention_mask

        device = attention_mask.device
        window_mask = torch.zeros(seq_length, seq_length, device=device, dtype=torch.bool)

        for i in range(seq_length):
            # Standard causal sliding window: tokens in [i - window_size + 1, i]
            start = i - window_size + 1
            end = i + 1  # exclusive

            # Clip to valid range and enforce causality
            start = max(0, start)
            end = min(end, i + 1)

            if start >= end:
                # Fallback: attend to self at least
                start = i
                end = i + 1

            window_mask[i, start:end] = True

        window_mask = window_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            combined_mask = attention_mask & window_mask
        else:
            combined_mask = window_mask

        return combined_mask

        # === Robust Sliding Window + Group Shift Mask (Stable LMS Version) ===
        # if window_size is None or window_size >= seq_length:
        #     return attention_mask
        #
        # device = attention_mask.device
        #
        # step = max(1, window_size // max(1, num_groups))
        # shift = group_idx * step
        #
        # window_mask = torch.zeros(seq_length, seq_length, device=device, dtype=torch.bool)
        #
        # for i in range(seq_length):
        #     base_start = i - window_size + 1
        #     base_end = i + 1
        #
        #     shifted_start = base_start + shift
        #     shifted_end = base_end + shift
        #
        #     shifted_end = min(shifted_end, i + 1)
        #
        #     if shifted_start >= shifted_end:
        #         final_start = max(0, base_start)
        #         final_end = base_end
        #     else:
        #         final_start = max(0, shifted_start)
        #         final_end = shifted_end
        #
        #     if final_start >= final_end:
        #         final_start = i
        #         final_end = i + 1
        #
        #     window_mask[i, final_start:final_end] = True
        #
        # window_mask = window_mask.unsqueeze(0).unsqueeze(0)
        #
        # if attention_mask is not None:
        #     combined_mask = attention_mask & window_mask
        # else:
        #     combined_mask = window_mask
        #
        # return combined_mask

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
        mask_value = -1e4
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        # Apply input attention_mask (padding mask)
        if attention_mask is not None:
            # attention_mask: [batch_size, 1, 1, seq_length]
            attention_mask = attention_mask[:, :, :, :key_length]
            attn_weights = attn_weights + (attention_mask * mask_value)

        # (Optional) head_mask is ignored here, but we keep it in the signature
        # for compatibility with HuggingFace. If you want, you can apply it
        # to attn_weights per head.

        # Debug-safety guard: prevent rows with all -inf which would lead to NaNs after softmax
        # If an entire row is -inf (i.e., no valid attention positions), set that row to zeros so
        # the softmax becomes a uniform distribution instead of producing NaNs.
        with torch.no_grad():
            row_max = attn_weights.max(dim=-1, keepdim=True).values  # [batch, heads, seq, 1]
            invalid_rows = torch.isinf(row_max) & (row_max < 0)
        if invalid_rows.any():
            attn_weights = torch.where(
                invalid_rows,
                torch.zeros_like(attn_weights),
                attn_weights,
            )

        # Additional NaN/Inf sanitize to stabilize softmax on MPS
        if torch.isnan(attn_weights).any() or torch.isinf(attn_weights).any():
            print("DEBUG: NaN/Inf detected in attn_weights; sanitizing (LMS)")
            attn_weights = torch.nan_to_num(
                attn_weights,
                nan=0.0,
                posinf=1e4,
                neginf=-1e4,
            )
        attn_weights = torch.clamp(attn_weights, -1e4, 1e4)

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum
        attn_output = torch.matmul(attn_weights, value)

        # For debug to find Nan
        if torch.isnan(attn_output).any():
            print("DEBUG: NaN in attn_output at layer", self.layer_idx)

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
