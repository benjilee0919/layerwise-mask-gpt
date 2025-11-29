"""
Training script for baseline GPT-2 model without attention masking.
"""

import os
import json
import torch
import argparse
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from src.dataset import prepare_dataset, get_dataloader
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "./config/train_config.json") -> dict:
    """Load training configuration."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def setup_model_and_tokenizer(model_name: str = 'gpt2'):
    """Initialize model and tokenizer."""
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Resize embeddings if needed
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description='Train baseline GPT-2 model')
    parser.add_argument('--dataset', type=str, default='tiny_stories',
                       choices=['wikitext-103', 'tiny_stories'],
                       help='Dataset to use for training')
    parser.add_argument('--config', type=str, default='./config/train_config.json',
                       help='Path to training config file')
    parser.add_argument('--output_dir', type=str, default='./models/baseline_gpt2',
                       help='Directory to save the model')
    parser.add_argument('--model_name', type=str, default='gpt2',
                       help='Base model name')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded config: {config}")
    
    # Setup model and tokenizer
    logger.info("Initializing model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(args.model_name)
    
    # Prepare dataset
    logger.info(f"Preparing {args.dataset} dataset...")
    datasets, _ = prepare_dataset(args.dataset, config['max_length'])
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # GPT-2 is causal LM, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=config['num_epochs'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        warmup_steps=config['warmup_steps'],
        max_grad_norm=config['max_grad_norm'],
        logging_steps=config['logging_steps'],
        eval_steps=config['eval_steps'],
        save_steps=config['save_steps'],
        evaluation_strategy=config['evaluation_strategy'],
        save_strategy=config['save_strategy'],
        load_best_model_at_end=config['load_best_model_at_end'],
        metric_for_best_model=config['metric_for_best_model'],
        greater_is_better=config['greater_is_better'],
        save_total_limit=config['save_total_limit'],
        dataloader_num_workers=config['dataloader_num_workers'],
        fp16=config['fp16'],
        report_to=config['report_to'],
        run_name=f"baseline_gpt2_{args.dataset}",
        seed=42,
        data_seed=42,
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=datasets['train'],
        eval_dataset=datasets['validation'],
        tokenizer=tokenizer
    )
    
    # Train model
    logger.info("Starting training...")
    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()
    
    # Save final model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training config
    with open(os.path.join(args.output_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Evaluate on test set
    if 'test' in datasets:
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(datasets['test'])
        logger.info(f"Test results: {test_results}")
        
        # Save test results
        with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()