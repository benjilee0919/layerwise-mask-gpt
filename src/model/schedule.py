"""
Layer-wise masking schedule management and configuration.
"""

import json
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class LayerMaskConfig:
    """Configuration for a single layer's masking."""
    layer_idx: int
    window_size: Optional[int] = None
    mask_type: str = "none"
    block_size: Optional[int] = None
    num_blocks: Optional[int] = None
    stride: Optional[int] = None
    sparsity: Optional[float] = None

    # NEW: group-based scheduling (Half A/B, Quarter A–D 등)
    group_idx: int = 0
    num_groups: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "layer_idx": self.layer_idx,
            "window_size": self.window_size,
            "mask_type": self.mask_type,
            "block_size": self.block_size,
            "num_blocks": self.num_blocks,
            "stride": self.stride,
            "sparsity": self.sparsity,
            "group_idx": self.group_idx,
            "num_groups": self.num_groups,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LayerMaskConfig":
        """Create from dictionary."""
        return cls(**data)


class MaskSchedule:
    """Manages layer-wise attention masking schedules."""

    def __init__(self, schedule_name: str, description: str = ""):
        self.schedule_name = schedule_name
        self.description = description
        self.layer_configs: Dict[int, LayerMaskConfig] = {}
        self.mask_patterns: Dict[str, Dict[str, Any]] = {}

    def add_layer_config(self, layer_idx: int, mask_config: LayerMaskConfig):
        """Add configuration for a specific layer."""
        self.layer_configs[layer_idx] = mask_config

    def get_layer_config(self, layer_idx: int) -> Optional[LayerMaskConfig]:
        """Get configuration for a specific layer."""
        return self.layer_configs.get(layer_idx)

    def add_mask_pattern(self, pattern_name: str, pattern_config: Dict[str, Any]):
        """Add a mask pattern configuration."""
        self.mask_patterns[pattern_name] = pattern_config

    def get_mask_pattern(self, pattern_name: str) -> Optional[Dict[str, Any]]:
        """Get a mask pattern configuration."""
        return self.mask_patterns.get(pattern_name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert schedule to dictionary."""
        return {
            "schedule_name": self.schedule_name,
            "description": self.description,
            "layers": {
                str(layer_idx): config.to_dict()
                for layer_idx, config in self.layer_configs.items()
            },
            "mask_patterns": self.mask_patterns,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MaskSchedule":
        """Create schedule from dictionary."""
        schedule = cls(data["schedule_name"], data.get("description", ""))

        # Load layer configurations
        for layer_idx_str, config_dict in data.get("layers", {}).items():
            layer_idx = int(layer_idx_str)
            config_dict["layer_idx"] = layer_idx
            layer_config = LayerMaskConfig.from_dict(config_dict)
            schedule.add_layer_config(layer_idx, layer_config)

        # Load mask patterns
        schedule.mask_patterns = data.get("mask_patterns", {})

        return schedule

    def save(self, filepath: str):
        """Save schedule to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "MaskSchedule":
        """Load schedule from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_efficiency_summary(self, seq_length: int = 1024) -> Dict[str, Any]:
        """
        Get efficiency summary for this schedule using mask_utils.
        Note: This is primarily for analysis/visualization, not training.
        """
        from src.model.mask_utils import (
            create_layer_wise_masks,
            get_attention_pattern_efficiency,
        )

        # Create masks for all layers
        schedule_dict = {
            str(layer_idx): config.to_dict()
            for layer_idx, config in self.layer_configs.items()
        }

        masks = create_layer_wise_masks(
            seq_length, len(self.layer_configs), schedule_dict
        )

        # Calculate efficiency for each layer
        layer_efficiency = {}
        total_attended = 0
        total_possible = 0

        for layer_idx, mask in masks.items():
            efficiency = get_attention_pattern_efficiency(mask)
            layer_efficiency[layer_idx] = efficiency
            total_attended += efficiency["attended_positions"]
            total_possible += efficiency["total_positions"]

        overall_sparsity = 1 - (total_attended / total_possible)
        overall_reduction = total_possible / total_attended

        return {
            "schedule_name": self.schedule_name,
            "overall_sparsity": overall_sparsity,
            "overall_memory_reduction": overall_reduction,
            "layer_efficiency": layer_efficiency,
            "total_layers": len(self.layer_configs),
            "seq_length": seq_length,
        }


# -----------------------------
# Predefined Schedules
# -----------------------------


def create_lms_main_schedule() -> MaskSchedule:
    """
    Ben's main Layer-wise Mask Scheduling (LMS) schedule for GPT-2 small (12 layers).

    Layout (by layer index):
        L0  : Full
        L1  : Half A
        L2  : Half B
        L3  : Full
        L4  : Quarter A
        L5  : Quarter B
        L6  : Quarter C
        L7  : Quarter D
        L8  : Full
        L9  : Half A
        L10 : Half B
        L11 : Full
    """
    sched = MaskSchedule(
        "lms_main",
        "Layer-wise Mask Scheduling: Full / Half / Quarter pattern with shifted groups.",
    )

    # NOTE: window_size는 예시 값이므로, 실제 실험에서 튜닝 가능
    # 여기서는 seq_length=256, 512 기준으로 reasonable한 값으로 설정

    # L0: Full
    sched.add_layer_config(
        0,
        LayerMaskConfig(
            layer_idx=0,
            mask_type="none",
        ),
    )

    # Half layers: use larger window (e.g., 256)
    half_window = 256

    # L1: Half A
    sched.add_layer_config(
        1,
        LayerMaskConfig(
            layer_idx=1,
            mask_type="sliding_window",
            window_size=half_window,
            num_groups=2,
            group_idx=0,
        ),
    )

    # L2: Half B
    sched.add_layer_config(
        2,
        LayerMaskConfig(
            layer_idx=2,
            mask_type="sliding_window",
            window_size=half_window,
            num_groups=2,
            group_idx=1,
        ),
    )

    # L3: Full
    sched.add_layer_config(
        3,
        LayerMaskConfig(
            layer_idx=3,
            mask_type="none",
        ),
    )

    # Quarter layers: smaller window (e.g., 128)
    quarter_window = 128

    for l, g in zip(range(4, 8), range(4)):  # group_idx 0..3
        sched.add_layer_config(
            l,
            LayerMaskConfig(
                layer_idx=l,
                mask_type="sliding_window",
                window_size=quarter_window,
                num_groups=4,
                group_idx=g,
            ),
        )

    # L8: Full
    sched.add_layer_config(
        8,
        LayerMaskConfig(
            layer_idx=8,
            mask_type="none",
        ),
    )

    # L9: Half A (again)
    sched.add_layer_config(
        9,
        LayerMaskConfig(
            layer_idx=9,
            mask_type="sliding_window",
            window_size=half_window,
            num_groups=2,
            group_idx=0,
        ),
    )

    # L10: Half B (again)
    sched.add_layer_config(
        10,
        LayerMaskConfig(
            layer_idx=10,
            mask_type="sliding_window",
            window_size=half_window,
            num_groups=2,
            group_idx=1,
        ),
    )

    # L11: Full
    sched.add_layer_config(
        11,
        LayerMaskConfig(
            layer_idx=11,
            mask_type="none",
        ),
    )

    return sched


def create_predefined_schedules() -> Dict[str, MaskSchedule]:
    """Create predefined masking schedules."""
    schedules = {}

    # Full attention schedule (no masking)
    full_schedule = MaskSchedule("full", "Full attention without masking")
    for layer_idx in range(12):  # GPT-2 small has 12 layers
        config = LayerMaskConfig(layer_idx=layer_idx, mask_type="none")
        full_schedule.add_layer_config(layer_idx, config)
    schedules["full"] = full_schedule

    # Half schedule (masking in later 6 layers)
    half_schedule = MaskSchedule("half", "Progressive masking in later 6 layers")
    for layer_idx in range(12):
        if layer_idx < 6:
            config = LayerMaskConfig(layer_idx=layer_idx, mask_type="none")
        else:
            # Progressive window reduction
            window_size = max(16, 512 // (2 ** (layer_idx - 5)))
            config = LayerMaskConfig(
                layer_idx=layer_idx,
                mask_type="sliding_window",
                window_size=window_size,
            )
        half_schedule.add_layer_config(layer_idx, config)
    schedules["half"] = half_schedule

    # Quarter schedule (masking from layer 3 onwards)
    quarter_schedule = MaskSchedule("quarter", "Progressive masking from layer 3 onwards")
    for layer_idx in range(12):
        if layer_idx < 3:
            config = LayerMaskConfig(layer_idx=layer_idx, mask_type="none")
        else:
            # Progressive window reduction
            window_size = max(4, 1024 // (2 ** (layer_idx - 2)))
            config = LayerMaskConfig(
                layer_idx=layer_idx,
                mask_type="sliding_window",
                window_size=window_size,
            )
        quarter_schedule.add_layer_config(layer_idx, config)
    schedules["quarter"] = quarter_schedule

    # Aggressive schedule (early and aggressive masking)
    aggressive_schedule = MaskSchedule(
        "aggressive", "Early and aggressive masking from layer 1"
    )
    for layer_idx in range(12):
        if layer_idx == 0:
            config = LayerMaskConfig(layer_idx=layer_idx, mask_type="none")
        else:
            # Very aggressive window reduction
            window_size = max(1, 1024 // (2 ** layer_idx))
            config = LayerMaskConfig(
                layer_idx=layer_idx,
                mask_type="sliding_window",
                window_size=window_size,
            )
        aggressive_schedule.add_layer_config(layer_idx, config)
    schedules["aggressive"] = aggressive_schedule

    # Block sparse schedule
    block_schedule = MaskSchedule("block_sparse", "Block sparse attention pattern")
    for layer_idx in range(12):
        if layer_idx < 4:
            config = LayerMaskConfig(layer_idx=layer_idx, mask_type="none")
        else:
            # Block sparse with decreasing block size
            block_size = max(32, 128 // (layer_idx - 3))
            config = LayerMaskConfig(
                layer_idx=layer_idx,
                mask_type="block_sparse",
                block_size=block_size,
                num_blocks=4,
            )
        block_schedule.add_layer_config(layer_idx, config)
    schedules["block_sparse"] = block_schedule

    # 🔥 Ben's LMS main schedule
    schedules["lms_main"] = create_lms_main_schedule()

    return schedules


def load_mask_schedule(schedule_config_path: str, schedule_name: str) -> Dict[str, Any]:
    """Load mask schedule from configuration file."""
    try:
        with open(schedule_config_path, "r") as f:
            config = json.load(f)

        if schedule_name in config["schedules"]:
            schedule_data = config["schedules"][schedule_name]
            return schedule_data
        else:
            raise ValueError(f"Schedule '{schedule_name}' not found in config file")

    except FileNotFoundError:
        print(f"Config file not found: {schedule_config_path}")
        print("Creating predefined schedules...")

        schedules = create_predefined_schedules()
        if schedule_name in schedules:
            return schedules[schedule_name].to_dict()
        else:
            raise ValueError(f"Schedule '{schedule_name}' not available")


def save_predefined_schedules(output_path: str = "./config/schedule_config.json"):
    """Save predefined schedules to configuration file."""
    schedules = create_predefined_schedules()

    config = {
        "schedules": {
            name: schedule.to_dict() for name, schedule in schedules.items()
        },
        "default_schedule": "lms_main",
        "mask_patterns": {
            "sliding_window": {
                "description": "Sliding window attention pattern",
                "parameters": {
                    "stride": 1,
                    "overlap": 0.1,
                },
            },
            "block_sparse": {
                "description": "Block sparse attention pattern",
                "parameters": {
                    "block_size": 64,
                    "num_blocks": 4,
                },
            },
            "strided": {
                "description": "Strided attention pattern",
                "parameters": {
                    "stride": 2,
                },
            },
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Predefined schedules saved to {output_path}")


def compare_schedules(schedules: List[str], seq_length: int = 1024) -> Dict[str, Any]:
    """Compare efficiency of different masking schedules."""
    predefined = create_predefined_schedules()

    comparison = {}
    for schedule_name in schedules:
        if schedule_name in predefined:
            schedule = predefined[schedule_name]
            efficiency = schedule.get_efficiency_summary(seq_length)
            comparison[schedule_name] = efficiency
        else:
            print(f"Warning: Schedule '{schedule_name}' not found")

    return comparison


def visualize_schedule(schedule: MaskSchedule, seq_length: int = 512, save_path: str = None):
    """Visualize a masking schedule across layers."""
    import matplotlib.pyplot as plt
    from src.model.mask_utils import create_layer_wise_masks

    # Create masks for all layers
    schedule_dict = {
        str(layer_idx): config.to_dict()
        for layer_idx, config in schedule.layer_configs.items()
    }

    masks = create_layer_wise_masks(seq_length, len(schedule.layer_configs), schedule_dict)

    # Create visualization
    num_layers = len(masks)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    # Subsample for visualization
    vis_size = min(128, seq_length)

    for layer_idx, mask in masks.items():
        if layer_idx >= len(axes):
            break

        mask_sub = mask[:vis_size, :vis_size]
        axes[layer_idx].imshow(mask_sub.cpu().numpy(), cmap="Blues")

        # Get layer config for title
        layer_config = schedule.get_layer_config(layer_idx)
        title = f"Layer {layer_idx}"
        if layer_config and layer_config.mask_type != "none":
            title += f"\n{layer_config.mask_type}"
            if layer_config.window_size:
                title += f" (w={layer_config.window_size})"

        axes[layer_idx].set_title(title, fontsize=10)
        axes[layer_idx].set_xticks([])
        axes[layer_idx].set_yticks([])

    # Hide unused subplots
    for idx in range(num_layers, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        f"Masking Schedule: {schedule.schedule_name}\n{schedule.description}",
        fontsize=14,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Schedule visualization saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    # Example usage
    schedules = create_predefined_schedules()

    # Save to config file
    save_predefined_schedules()

    # Compare schedules
    comparison = compare_schedules(["full", "half", "quarter", "aggressive", "lms_main"])

    print("Schedule Efficiency Comparison:")
    print("-" * 60)
    print(f"{'Schedule':<15} {'Sparsity':<10} {'Memory Reduction':<15}")
    print("-" * 60)

    for name, metrics in comparison.items():
        print(
            f"{name:<15} {metrics['overall_sparsity']:<10.3f} "
            f"{metrics['overall_memory_reduction']:<15.2f}x"
        )

    # Visualize LMS schedule
    visualize_schedule(schedules["lms_main"])
