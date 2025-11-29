# # Layer-wise Mask GPT

# A research project exploring layer-wise attention masking in GPT-2 models for improved efficiency.

# ## Package Structure
# - `dataset.py` - Dataset loading and preprocessing utilities
# - `train_baseline.py` - Baseline GPT-2 training script
# - `train_masked.py` - Masked GPT-2 training script
# - `evaluate.py` - Model evaluation utilities
# - `model/` - Custom model implementations

# ## Model Components
# - `model/masked_mha.py` - Masked multi-head attention module
# - `model/mask_utils.py` - Attention mask generation utilities
# - `model/schedule.py` - Layer-wise masking schedule management
# - `model/gpt2_custom.py` - Custom GPT-2 with attention masking support