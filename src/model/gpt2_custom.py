"""
Custom GPT-2 implementation with layer-wise attention masking support.
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from src.model.masked_mha import MaskedGPT2Attention
from typing import Optional, Dict, Any, Union, Tuple
import json


class MaskedGPT2Block(GPT2Block):
    """GPT-2 block with masked attention."""

    def __init__(self, config: GPT2Config, layer_idx: int = None, mask_config: dict = None):
        # Initialize parent class first (sets up ln_1, ln_2, mlp, attn, etc.)
        super().__init__(config, layer_idx)

        # Replace standard attention with masked attention
        self.attn = MaskedGPT2Attention(config, layer_idx, mask_config)
        self.layer_idx = layer_idx
        self.mask_config = mask_config or {}


class MaskedGPT2Model(nn.Module):
    """GPT-2 model with layer-wise attention masking."""

    def __init__(self, config: GPT2Config, mask_schedule: dict = None):
        super().__init__()
        self.config = config
        self.mask_schedule = mask_schedule or {}

        # Load the base GPT-2 model
        base_model = GPT2LMHeadModel.from_pretrained("gpt2")

        # Copy embeddings and other components
        self.wte = base_model.transformer.wte  # Token embeddings
        self.wpe = base_model.transformer.wpe  # Position embeddings
        self.drop = base_model.transformer.drop
        self.ln_f = base_model.transformer.ln_f

        # Create masked transformer blocks
        self.h = nn.ModuleList()
        for layer_idx in range(config.n_layer):
            # Get mask config for this layer (from schedule JSON or dict)
            layer_mask_config = self._get_layer_mask_config(layer_idx)

            # Create masked block
            block = MaskedGPT2Block(config, layer_idx, layer_mask_config)

            # Copy weights from the original block
            original_block = base_model.transformer.h[layer_idx]
            self._copy_block_weights(original_block, block)

            self.h.append(block)

        # Other attributes
        self.gradient_checkpointing = False

    def _get_layer_mask_config(self, layer_idx: int) -> dict:
        """Get mask configuration for a specific layer."""
        if not self.mask_schedule:
            return {}

        # New format: {"layers": {"0": {...}, "1": {...}, ...}}
        if "layers" in self.mask_schedule:
            layer_config = self.mask_schedule["layers"].get(str(layer_idx), {})
        else:
            # Old format – direct "0", "1", ... keys
            layer_config = self.mask_schedule.get(str(layer_idx), {})

        # Ensure types are sane
        layer_config = dict(layer_config)  # shallow copy
        layer_config.setdefault("mask_type", "none")
        layer_config.setdefault("group_idx", 0)
        layer_config.setdefault("num_groups", 1)

        return layer_config

    def _copy_block_weights(self, original_block: GPT2Block, masked_block: MaskedGPT2Block):
        """Copy weights from original block to masked block."""
        # Layer norms
        masked_block.ln_1.load_state_dict(original_block.ln_1.state_dict())
        masked_block.ln_2.load_state_dict(original_block.ln_2.state_dict())

        # MLP
        masked_block.mlp.load_state_dict(original_block.mlp.state_dict())

        # ----- Copy attention projections -----
        orig_attn = original_block.attn
        new_attn = masked_block.attn

        # c_attn: Conv1D (out=768, in=2304)  → Linear(in=768, out=2304)
        orig_w = orig_attn.c_attn.weight    # e.g. [768, 2304]
        orig_b = orig_attn.c_attn.bias

        if new_attn.c_attn.weight.shape == orig_w.shape:
            # If transformers already uses Linear with same shape
            new_attn.c_attn.load_state_dict(orig_attn.c_attn.state_dict())
        else:
            # Conv1D → Linear: transpose
            with torch.no_grad():
                new_attn.c_attn.weight.copy_(orig_w.t())  # [2304, 768]
                new_attn.c_attn.bias.copy_(orig_b)

        # c_proj: Conv1D(n_embd, n_embd) → Linear(n_embd, n_embd)
        new_attn.c_proj.weight.data.copy_(orig_attn.c_proj.weight.data)
        new_attn.c_proj.bias.data.copy_(orig_attn.c_proj.bias.data)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass through the model.

        Note:
        For training we explicitly disable KV cache (use_cache=False) to avoid
        mismatches in outputs length from GPT2Block across different
        transformers versions. Cache is not needed for training.
        """

        # Config flags
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        # Force no cache use during training
        use_cache = False

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Input shape & device
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        # Past
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        # Position IDs
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                input_shape[-1] + past_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask (padding)
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # Head mask
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        # Embeddings
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        # We do not use cache during training
        presents = None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # Transformer blocks
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # GPT2Block always returns at least (hidden_states,)
            hidden_states = outputs[0]

            # If attentions are returned, they are always the last element
            if output_attentions and len(outputs) > 1:
                all_self_attentions = all_self_attentions + (outputs[-1],)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions]
                if v is not None
            )

        from transformers.modeling_outputs import (
            BaseModelOutputWithPastAndCrossAttentions,
        )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def get_head_mask(self, head_mask, num_hidden_layers):
        """Get head mask."""
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """Convert head mask to 5D tensor."""
        if head_mask.dim() == 1:
            head_mask = (
                head_mask.unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)
        return head_mask

    @property
    def dtype(self):
        """Get model dtype."""
        return next(self.parameters()).dtype


class MaskedGPT2LMHeadModel(nn.Module):
    """GPT-2 language model with masked attention."""

    def __init__(self, config: GPT2Config, mask_schedule: dict = None):
        super().__init__()
        self.config = config
        self.transformer = MaskedGPT2Model(config, mask_schedule)

        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize / tie weights
        self.post_init()

    def post_init(self):
        """Post initialization setup."""
        if hasattr(self.config, "tie_word_embeddings") and self.config.tie_word_embeddings:
            self.tie_weights()

    def tie_weights(self):
        """Tie input and output embeddings."""
        self.lm_head.weight = self.transformer.wte.weight

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """Forward pass with language modeling loss."""

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]

        # LM head
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        from transformers.modeling_outputs import (
            CausalLMOutputWithPastAndCrossAttentions,
        )

        return CausalLMOutputWithPastAndCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, mask_schedule: dict = None, **kwargs
    ):
        """Load pretrained model with optional mask schedule."""
        base_model = GPT2LMHeadModel.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        config = base_model.config

        model = cls(config, mask_schedule)
        model.lm_head.load_state_dict(base_model.lm_head.state_dict())

        return model

    def save_pretrained(self, save_directory: str):
        """Save model to directory."""
        import os

        os.makedirs(save_directory, exist_ok=True)

        # Save model weights
        torch.save(
            self.state_dict(),
            os.path.join(save_directory, "pytorch_model.bin"),
        )

        # Save config
        self.config.save_pretrained(save_directory)

        # Save mask schedule if available
        if hasattr(self.transformer, "mask_schedule") and self.transformer.mask_schedule:
            with open(
                os.path.join(save_directory, "mask_schedule.json"), "w"
            ) as f:
                json.dump(self.transformer.mask_schedule, f, indent=2)

    def resize_token_embeddings(self, new_num_tokens: int):
        """Resize token embeddings."""
        old_embeddings = self.transformer.wte
        new_embeddings = nn.Embedding(
            new_num_tokens, old_embeddings.embedding_dim
        )

        num_tokens_to_copy = min(
            old_embeddings.num_embeddings, new_num_tokens
        )
        new_embeddings.weight.data[:num_tokens_to_copy, :] = (
            old_embeddings.weight.data[:num_tokens_to_copy, :]
        )

        self.transformer.wte = new_embeddings

        # Resize lm_head if not tied
        if not (
            hasattr(self.config, "tie_word_embeddings")
            and self.config.tie_word_embeddings
        ):
            old_lm_head = self.lm_head
            new_lm_head = nn.Linear(
                old_lm_head.in_features, new_num_tokens, bias=False
            )
            new_lm_head.weight.data[:num_tokens_to_copy, :] = (
                old_lm_head.weight.data[:num_tokens_to_copy, :]
            )
            self.lm_head = new_lm_head
        else:
            self.tie_weights()

        self.config.vocab_size = new_num_tokens
        return self.transformer.wte

    def get_mask_schedule_summary(self) -> Dict[str, Any]:
        """Get summary of the current mask schedule."""
        if (
            not hasattr(self.transformer, "mask_schedule")
            or not self.transformer.mask_schedule
        ):
            return {"message": "No mask schedule applied"}

        summary: Dict[str, Any] = {"layers": {}}

        for layer_idx, block in enumerate(self.transformer.h):
            layer_config = (
                block.mask_config if hasattr(block, "mask_config") else {}
            )
            summary["layers"][layer_idx] = {
                "mask_type": layer_config.get("mask_type", "none"),
                "window_size": layer_config.get("window_size"),
                "block_size": layer_config.get("block_size"),
                "num_blocks": layer_config.get("num_blocks"),
                "group_idx": layer_config.get("group_idx", 0),
                "num_groups": layer_config.get("num_groups", 1),
            }

        return summary
