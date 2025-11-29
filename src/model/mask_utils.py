"""
Attention mask generation utilities for different masking patterns.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Union, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns


def create_causal_mask(seq_length: int, device: torch.device = None) -> torch.Tensor:
    """Create standard causal (lower triangular) mask."""
    mask = torch.tril(torch.ones(seq_length, seq_length, dtype=torch.bool, device=device))
    return mask


def create_sliding_window_mask(
    seq_length: int,
    window_size: int,
    device: torch.device = None,
    stride: int = 1,
    group_idx: int = 0,
    num_groups: int = 1,
) -> torch.Tensor:
    """
    Create sliding window attention mask, optionally shifted by group.

    - Causal: each query i only attends to j <= i
    - window_size: max number of tokens to look back
    - group_idx / num_groups:
        Used for Layer-wise Mask Scheduling (Half A/B, Quarter A-D).
        Different groups see slightly different sub-ranges within the same
        overall window, to diversify coverage across layers.
    """
    if window_size >= seq_length:
        return create_causal_mask(seq_length, device)

    mask = torch.zeros(seq_length, seq_length, dtype=torch.bool, device=device)

    # Compute shift step so different groups cover different parts of the window
    step = max(1, window_size // max(1, num_groups))
    shift = group_idx * step

    for i in range(seq_length):
        # Base causal window [i - window_size + 1, i]
        base_start = i - window_size + 1
        base_end = i + 1

        # Apply group-dependent left shift; still clamp to causal and seq bounds
        start = max(0, base_start + shift)
        end = min(base_end + shift, i + 1)  # never allow j > i (causality)

        if start < 0:
            start = 0
        if start >= end:
            # Fallback: at least attend to self
            start = i
            end = i + 1

        mask[i, start:end:stride] = True

    return mask


def create_block_sparse_mask(
    seq_length: int,
    block_size: int,
    num_blocks: int = 4,
    device: torch.device = None,
) -> torch.Tensor:
    """Create block sparse attention mask."""
    mask = torch.zeros(seq_length, seq_length, dtype=torch.bool, device=device)

    # Number of blocks per dimension
    num_blocks_seq = (seq_length + block_size - 1) // block_size

    for i in range(num_blocks_seq):
        row_start = i * block_size
        row_end = min((i + 1) * block_size, seq_length)

        # Attend to previous num_blocks blocks and current block
        for j in range(max(0, i - num_blocks), i + 1):
            col_start = j * block_size
            col_end = min((j + 1) * block_size, seq_length)
            mask[row_start:row_end, col_start:col_end] = True

    return mask


def create_strided_mask(
    seq_length: int,
    stride: int = 2,
    device: torch.device = None,
) -> torch.Tensor:
    """Create strided attention mask."""
    mask = torch.zeros(seq_length, seq_length, dtype=torch.bool, device=device)

    for i in range(seq_length):
        # Attend to positions at stride intervals
        positions = list(range(0, i + 1, stride))
        # Always include the current position
        if i not in positions:
            positions.append(i)
        mask[i, positions] = True

    return mask


def create_local_global_mask(
    seq_length: int,
    local_window: int,
    global_positions: list = None,
    device: torch.device = None,
) -> torch.Tensor:
    """Create local + global attention mask."""
    # Start with local window
    mask = create_sliding_window_mask(seq_length, local_window, device=device)

    # Add global positions
    if global_positions is None:
        # Default: attend to every 64th position
        global_positions = list(range(0, seq_length, 64))

    for pos in global_positions:
        if pos < seq_length:
            # All positions can attend to global positions
            mask[:, pos] = True
            # Global positions can attend to all previous positions
            mask[pos, : pos + 1] = True

    return mask


def create_random_sparse_mask(
    seq_length: int,
    sparsity: float = 0.1,
    device: torch.device = None,
    seed: int = None,
) -> torch.Tensor:
    """Create random sparse attention mask while maintaining causality."""
    if seed is not None:
        torch.manual_seed(seed)

    # Start with causal mask
    mask = create_causal_mask(seq_length, device)

    # Randomly zero out some attention weights
    random_mask = torch.rand(seq_length, seq_length, device=device) > sparsity

    # Combine with causal constraint
    mask = mask & random_mask

    # Ensure each position can attend to at least itself
    for i in range(seq_length):
        mask[i, i] = True

    return mask


def create_progressive_mask(
    seq_length: int,
    layer_idx: int,
    total_layers: int,
    max_window: int = None,
    device: torch.device = None,
) -> torch.Tensor:
    """Create progressively smaller window masks based on layer depth."""
    if max_window is None:
        max_window = seq_length

    progress = layer_idx / total_layers
    window_size = int(max_window * (1 - progress) + 1)
    window_size = max(1, window_size)  # Ensure at least window size of 1

    return create_sliding_window_mask(seq_length, window_size, device=device)


def apply_mask_to_attention_scores(
    attention_scores: torch.Tensor,
    mask: torch.Tensor,
    mask_value: float = None,
) -> torch.Tensor:
    """Apply mask to attention scores."""
    if mask_value is None:
        mask_value = torch.finfo(attention_scores.dtype).min

    # Expand mask to match attention scores shape if needed
    if mask.dim() == 2 and attention_scores.dim() == 4:
        # mask: [seq_length, seq_length]
        # attention_scores: [batch_size, num_heads, seq_length, seq_length]
        mask = mask.unsqueeze(0).unsqueeze(0)

    return torch.where(mask, attention_scores, mask_value)


def get_attention_pattern_efficiency(mask: torch.Tensor) -> dict:
    """Calculate efficiency metrics for an attention pattern."""
    seq_length = mask.size(-1)
    total_positions = seq_length * seq_length

    # Count attended positions
    attended_positions = mask.sum().item()

    # Calculate sparsity (fraction of positions NOT attended to)
    sparsity = 1 - (attended_positions / total_positions)

    # Calculate average attention span per position
    attention_spans = mask.sum(dim=-1).float()
    avg_attention_span = attention_spans.mean().item()

    # Calculate memory reduction factor
    memory_reduction = 1 / (attended_positions / total_positions)

    return {
        "sparsity": sparsity,
        "attended_positions": attended_positions,
        "total_positions": total_positions,
        "avg_attention_span": avg_attention_span,
        "memory_reduction_factor": memory_reduction,
    }


def visualize_attention_pattern(
    mask: torch.Tensor,
    title: str = "Attention Pattern",
    save_path: str = None,
) -> None:
    """Visualize attention pattern."""
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        mask_np.astype(int),
        cmap="Blues",
        cbar=True,
        square=True,
        xticklabels=False,
        yticklabels=False,
    )
    plt.title(title)
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Attention pattern saved to {save_path}")

    plt.show()


def compare_attention_patterns(patterns: dict, seq_length: int = 512):
    """Compare different attention patterns."""
    device = torch.device("cpu")

    results = {}
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    pattern_configs = {
        "Full": {"type": "causal"},
        "Window-128": {"type": "sliding_window", "window_size": 128},
        "Window-64": {"type": "sliding_window", "window_size": 64},
        "Block-64": {
            "type": "block_sparse",
            "block_size": 64,
            "num_blocks": 4,
        },
        "Stride-4": {"type": "strided", "stride": 4},
        "Local+Global": {"type": "local_global", "local_window": 32},
    }

    for idx, (name, config) in enumerate(pattern_configs.items()):
        if idx >= len(axes):
            break

        # Create mask
        if config["type"] == "causal":
            mask = create_causal_mask(seq_length, device)
        elif config["type"] == "sliding_window":
            mask = create_sliding_window_mask(
                seq_length, config["window_size"], device=device
            )
        elif config["type"] == "block_sparse":
            mask = create_block_sparse_mask(
                seq_length,
                config["block_size"],
                config["num_blocks"],
                device,
            )
        elif config["type"] == "strided":
            mask = create_strided_mask(seq_length, config["stride"], device)
        elif config["type"] == "local_global":
            mask = create_local_global_mask(
                seq_length,
                config["local_window"],
                device=device,
            )

        # Get efficiency metrics
        efficiency = get_attention_pattern_efficiency(mask)
        results[name] = efficiency

        # Visualize (subsample for better visualization)
        subsample_size = min(128, seq_length)
        mask_sub = mask[:subsample_size, :subsample_size]

        axes[idx].imshow(mask_sub.numpy(), cmap="Blues")
        axes[idx].set_title(f"{name}\nSparsity: {efficiency['sparsity']:.2f}")
        axes[idx].set_xlabel("Key Position")
        axes[idx].set_ylabel("Query Position")

    plt.tight_layout()
    plt.show()

    # Print comparison table
    print("\nAttention Pattern Efficiency Comparison:")
    print("-" * 80)
    print(f"{'Pattern':<15} {'Sparsity':<10} {'Avg Span':<10} {'Memory Reduction':<15}")
    print("-" * 80)

    for name, metrics in results.items():
        print(
            f"{name:<15} {metrics['sparsity']:<10.3f} "
            f"{metrics['avg_attention_span']:<10.1f} "
            f"{metrics['memory_reduction_factor']:<15.2f}x"
        )

    return results


def create_layer_wise_masks(
    seq_length: int, num_layers: int, schedule: Dict[str, Dict[str, Any]]
) -> dict:
    """
    Create masks for all layers according to schedule.

    schedule 형식 예시 (schedule_config.json의 'layers' 부분):

    {
        "0": {"mask_type": "none"},
        "1": {"mask_type": "sliding_window", "window_size": 256, "num_groups": 2, "group_idx": 0},
        ...
    }
    """
    masks = {}
    device = torch.device("cpu")

    for layer_idx in range(num_layers):
        layer_config = schedule.get(str(layer_idx), {"mask_type": "none"})
        mask_type = layer_config.get("mask_type", "none")

        if mask_type == "none":
            mask = create_causal_mask(seq_length, device)
        elif mask_type == "sliding_window":
            window_size = layer_config.get("window_size", seq_length)
            num_groups = layer_config.get("num_groups", 1)
            group_idx = layer_config.get("group_idx", 0)
            mask = create_sliding_window_mask(
                seq_length,
                window_size,
                device=device,
                group_idx=group_idx,
                num_groups=num_groups,
            )
        elif mask_type == "block_sparse":
            block_size = layer_config.get("block_size", 64)
            num_blocks = layer_config.get("num_blocks", 4)
            mask = create_block_sparse_mask(
                seq_length,
                block_size,
                num_blocks,
                device,
            )
        elif mask_type == "progressive":
            max_window = layer_config.get("max_window", seq_length)
            mask = create_progressive_mask(
                seq_length,
                layer_idx,
                num_layers,
                max_window,
                device=device,
            )
        else:
            mask = create_causal_mask(seq_length, device)

        masks[layer_idx] = mask

    return masks
