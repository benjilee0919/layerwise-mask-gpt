"""
Training script for a GPT-2 model with layer-wise attention masking (LMS).

This script trains a masked GPT-2 variant using a specified layer-wise
mask schedule and logs both training loss and timing statistics.
"""

import os
import json
import torch
import argparse
import time
import csv
from transformers import TrainerCallback
from transformers import (
    GPT2Tokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from src.dataset import prepare_dataset
from src.model.gpt2_custom import MaskedGPT2LMHeadModel
from src.model.schedule import load_mask_schedule
import logging

class TimeLoggingCallback(TrainerCallback):
    """
    Trainer callback that logs per-step timing and loss values to a CSV file.
    """
    def __init__(self, log_path):
        self.log_path = log_path
        self.start_time = None
        self.epoch_start = None
        self._prepare_file()

    def _prepare_file(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step",
                "epoch",
                "step_time_sec",
                "total_time_sec",
                "loss",
            ])

    def on_step_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, logs=None, **kwargs):
        step_time = time.time() - self.start_time
        total_time = state.global_step * step_time

        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                state.global_step,
                state.epoch,
                round(step_time, 4),
                round(total_time, 2),
                logs.get("loss") if logs else None,
            ])

def safe_data_collator(features):
    """
    Collate function that drops examples with empty tensors.

    This is a defensive wrapper around the default data collator to
    avoid crashes when a batch contains zero-length sequences.
    """
    cleaned = []
    for f in features:
        ids = f.get("input_ids", None)
        labels = f.get("labels", None)
        if isinstance(ids, torch.Tensor) and isinstance(labels, torch.Tensor):
            if ids.numel() == 0 or labels.numel() == 0:
                continue
            cleaned.append(f)
    # Fallback: if everything was filtered out, use the original batch
    if not cleaned:
        cleaned = features
    return default_data_collator(cleaned)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "./config/train_config.json") -> dict:
    """Load training configuration from JSON file."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def setup_masked_model_and_tokenizer(
    model_name: str = "gpt2", mask_schedule: dict = None
):
    """Initialize masked GPT-2 model and tokenizer."""
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize custom model with masking capability
    model = MaskedGPT2LMHeadModel.from_pretrained(
        model_name,
        mask_schedule=mask_schedule,
    )

    # Resize embeddings if needed
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def main():
    """
    Entry point for training a masked GPT-2 model with a given schedule.

    Parses command-line arguments, loads configuration and mask schedule,
    initializes the model and tokenizer, prepares the dataset, and then
    launches training and evaluation via HuggingFace Trainer.
    """
    parser = argparse.ArgumentParser(description="Train masked GPT-2 model")
    parser.add_argument(
        "--dataset",
        type=str,
        default="tiny_stories",
        choices=["wikitext-103", "tiny_stories"],
        help="Dataset to use for training",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config/train_config.json",
        help="Path to training config file",
    )
    parser.add_argument(
        "--schedule_config",
        type=str,
        default="./config/schedule_config.json",
        help="Path to mask schedule config file",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="lms_main",
        choices=["full", "half", "quarter", "aggressive", "lms_main"],
        help="Mask schedule to use",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the model",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="Base model name",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    args = parser.parse_args()

    # Set output directory if not provided
    if args.output_dir is None:
        args.output_dir = f"./models/mask_gpt2/{args.schedule}"

    # Load configurations
    config = load_config(args.config)
    mask_schedule = load_mask_schedule(args.schedule_config, args.schedule)

    logger.info(f"Loaded training config: {config}")
    logger.info(f"Using mask schedule: {args.schedule}")
    logger.info(f"Mask schedule config: {mask_schedule}")

    # Setup model and tokenizer
    logger.info("Initializing masked model and tokenizer...")
    model, tokenizer = setup_masked_model_and_tokenizer(args.model_name, mask_schedule)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model total parameters: {total_params:,}")
    logger.info(f"Model trainable parameters: {trainable_params:,}")

    # Prepare dataset
    logger.info(f"Preparing {args.dataset} dataset...")
    datasets, _ = prepare_dataset(args.dataset, config["max_length"])

    # Data collator for causal language modeling
    data_collator = safe_data_collator

    # NOTE:
    # This project targets a transformers version that does not support newer
    # TrainingArguments fields such as `evaluation_strategy`, `save_strategy`,
    # `report_to`, etc. To maintain compatibility, only a minimal, widely
    # supported subset of arguments is passed here.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_steps=config["warmup_steps"],
        max_grad_norm=config["max_grad_norm"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        save_safetensors=False,
        # All newer/optional fields are omitted for compatibility:
        # - evaluation_strategy
        # - save_strategy
        # - load_best_model_at_end
        # - metric_for_best_model
        # - greater_is_better
        # - save_total_limit
        # - dataloader_num_workers
        # - fp16
        # - report_to
        # - run_name
        # - data_seed
        # - remove_unused_columns
        seed=42,
    )

    # Initialize Trainer
    trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=datasets["train"],
    eval_dataset=datasets["validation"],
    tokenizer=tokenizer,
    callbacks=[
        TimeLoggingCallback(
            log_path=f"./results/{args.schedule}_training_time.csv"
            ),
        ],
    )

    # Train model
    logger.info(f"Starting training with {args.schedule} masking schedule...")
    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    # Save final model and tokenizer
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    # Save configurations
    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(args.output_dir, "mask_schedule.json"), "w") as f:
        json.dump(mask_schedule, f, indent=2)

    # Evaluate on test set if available
    if "test" in datasets:
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(datasets["test"])
        logger.info(f"Test results: {test_results}")

        with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
            json.dump(test_results, f, indent=2)

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
