"""
Dataset loading and preprocessing utilities for WikiText-103 and TinyStories.

Note:
Although WikiText-103 loading is supported, all experiments in this project
use TinyStories exclusively. The WikiText loader is retained for completeness.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset
import argparse
from typing import Optional, Dict, Any


class TextDataset(Dataset):
    """Custom dataset for text data with GPT-2 tokenizer."""
    
    def __init__(self, texts, tokenizer, max_length=1024):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize with padding and truncation
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        # Labels are the same as input_ids for language modeling
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_wikitext103(cache_dir: str = "./data/processed") -> Dict[str, Any]:
    """Load WikiText-103 dataset."""
    print("Loading WikiText-103 dataset...")
    
    dataset = load_dataset("wikitext", "wikitext-103-v1", cache_dir=cache_dir)
    
    # Filter out empty texts
    train_texts = [text for text in dataset["train"]["text"] if text.strip()]
    val_texts = [text for text in dataset["validation"]["text"] if text.strip()]
    test_texts = [text for text in dataset["test"]["text"] if text.strip()]
    
    return {
        "train": train_texts,
        "validation": val_texts,
        "test": test_texts,
    }


def load_tinystories(cache_dir: str = "./data/processed") -> Dict[str, Any]:
    """
    Load TinyStories dataset.

    Note:
    On some systems (e.g., Windows with OneDrive or very long file paths),
    using a local cache_dir may produce path-length issues when HuggingFace
    attempts to create .arrow files. To avoid this, the default HuggingFace
    dataset cache (~/.cache/huggingface/datasets) is used.
    """
    print("Loading TinyStories dataset...")
    
    # cache_dir is accepted for API consistency but not used.
    # TinyStories is always loaded using the default HuggingFace cache.
    dataset = load_dataset("roneneldan/TinyStories")
    
    # Use a subset for faster experiments
    train_texts = dataset["train"]["text"][:50000]   # First 50k examples
    val_texts = dataset["validation"]["text"][:5000] # First 5k examples
    
    # Create test split from validation
    test_texts = val_texts[2500:]   # Last 2.5k for test
    val_texts = val_texts[:2500]    # First 2.5k for validation
    
    return {
        "train": train_texts,
        "validation": val_texts,
        "test": test_texts,
    }


def prepare_dataset(
    dataset_name: str,
    max_length: int = 1024,
    cache_dir: str = "./data/processed",
) -> Dict[str, TextDataset]:
    """Prepare dataset with tokenization."""
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load raw dataset
    if dataset_name.lower() == "wikitext-103":
        data = load_wikitext103(cache_dir)
    elif dataset_name.lower() == "tiny_stories":
        # TinyStories uses the default HuggingFace cache (cache_dir is unused).
        data = load_tinystories()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create tokenized datasets
    datasets = {}
    for split, texts in data.items():
        print(f"Preparing {split} split with {len(texts)} examples...")
        datasets[split] = TextDataset(texts, tokenizer, max_length)
    
    return datasets, tokenizer


def get_dataloader(
    dataset: TextDataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Create DataLoader for dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def save_dataset_info(
    dataset_name: str,
    datasets: Dict[str, TextDataset],
    tokenizer: GPT2Tokenizer,
    save_dir: str = "./data/processed",
):
    """Save dataset information and tokenizer."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save dataset info
    info = {
        "dataset_name": dataset_name,
        "num_examples": {split: len(ds) for split, ds in datasets.items()},
        "max_length": datasets["train"].max_length,
        "vocab_size": tokenizer.vocab_size,
    }
    
    with open(f"{save_dir}/{dataset_name}_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    # Save tokenizer
    tokenizer.save_pretrained(f"{save_dir}/{dataset_name}_tokenizer")
    
    print(f"Dataset info saved to {save_dir}/{dataset_name}_info.json")
    print(f"Tokenizer saved to {save_dir}/{dataset_name}_tokenizer/")


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for training")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["wikitext-103", "tiny_stories"],
        help="Dataset to prepare",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./data/processed",
        help="Directory to cache processed data (only used for wikitext)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for testing",
    )
    
    args = parser.parse_args()
    
    # Prepare dataset
    datasets, tokenizer = prepare_dataset(
        args.dataset,
        args.max_length,
        args.cache_dir,
    )
    
    # Save dataset info
    save_dataset_info(args.dataset, datasets, tokenizer, args.cache_dir)
    
    # Test dataloader
    train_loader = get_dataloader(
        datasets["train"],
        batch_size=args.batch_size,
        num_workers=2,
    )
    
    print(f"\nTesting DataLoader...")
    batch = next(iter(train_loader))
    print(f"Batch shape: {batch['input_ids'].shape}")
    print(f"Input IDs dtype: {batch['input_ids'].dtype}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    
    # Display a sample decoded text snippet for verification
    sample_ids = batch["input_ids"][0][:50]  # First 50 tokens
    sample_text = tokenizer.decode(sample_ids, skip_special_tokens=True)
    print(f"\nSample text: {sample_text}")


if __name__ == "__main__":
    main()
