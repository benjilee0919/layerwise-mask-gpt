"""
Evaluation utilities for measuring perplexity, latency, and throughput.
"""

import time
import json
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
from src.dataset import prepare_dataset, get_dataloader
from src.model.gpt2_custom import MaskedGPT2LMHeadModel
import argparse
import logging
from typing import Dict, Any, Tuple, List
import psutil
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation utility."""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def calculate_perplexity(self, dataloader: DataLoader) -> float:
        """Calculate perplexity on a dataset."""
        total_loss = 0
        total_tokens = 0
        
        logger.info("Calculating perplexity...")
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Calculate number of valid tokens (excluding padding)
            valid_tokens = attention_mask.sum().item()
            
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
            
            if batch_idx % 100 == 0:
                logger.info(f"Processed {batch_idx} batches")
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return perplexity
    
    @torch.no_grad()
    def measure_throughput(self, dataloader: DataLoader, num_batches: int = 100) -> Dict[str, float]:
        """Measure model throughput (tokens/second, batches/second)."""
        logger.info(f"Measuring throughput over {num_batches} batches...")
        
        # Warmup
        for i, batch in enumerate(dataloader):
            if i >= 10:  # 10 warmup batches
                break
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Actual measurement
        start_time = time.time()
        total_tokens = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
                
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Count actual tokens (excluding padding)
            batch_tokens = attention_mask.sum().item()
            total_tokens += batch_tokens
            
            _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        throughput = {
            'tokens_per_second': total_tokens / elapsed_time,
            'batches_per_second': num_batches / elapsed_time,
            'elapsed_time': elapsed_time,
            'total_tokens': total_tokens,
            'total_batches': num_batches
        }
        
        return throughput
    
    @torch.no_grad()
    def measure_latency(self, input_length: int = 512, num_trials: int = 100) -> Dict[str, float]:
        """Measure inference latency for single sequences."""
        logger.info(f"Measuring latency over {num_trials} trials...")
        
        # Create dummy input
        input_ids = torch.randint(0, self.tokenizer.vocab_size, (1, input_length)).to(self.device)
        attention_mask = torch.ones_like(input_ids).to(self.device)
        
        # Warmup
        for _ in range(10):
            _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Measure latency
        latencies = []
        for _ in range(num_trials):
            start_time = time.time()
            _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        latency_stats = {
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'median_latency_ms': np.median(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'input_length': input_length,
            'num_trials': num_trials
        }
        
        return latency_stats
    
    def measure_memory_usage(self) -> Dict[str, float]:
        """Measure memory usage."""
        memory_stats = {}
        
        if torch.cuda.is_available():
            memory_stats['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024**2
            memory_stats['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024**2
            memory_stats['gpu_max_memory_allocated_mb'] = torch.cuda.max_memory_allocated() / 1024**2
        
        # CPU memory
        process = psutil.Process(os.getpid())
        memory_stats['cpu_memory_mb'] = process.memory_info().rss / 1024**2
        
        return memory_stats
    
    def full_evaluation(self, dataloader: DataLoader, output_file: str = None) -> Dict[str, Any]:
        """Run comprehensive evaluation."""
        logger.info("Starting full evaluation...")
        
        results = {}
        
        # Perplexity
        results['perplexity'] = self.calculate_perplexity(dataloader)
        
        # Throughput
        results['throughput'] = self.measure_throughput(dataloader)
        
        # Latency for different input lengths
        results['latency'] = {}
        for length in [128, 256, 512, 1024]:
            results['latency'][f'length_{length}'] = self.measure_latency(length)
        
        # Memory usage
        results['memory'] = self.measure_memory_usage()
        
        # Model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        results['model_info'] = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / 1024**2  # Assuming float32
        }
        
        # Save results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_file}")
        
        return results


def load_model(model_path: str, model_type: str = 'baseline') -> Tuple[torch.nn.Module, GPT2Tokenizer]:
    """Load model and tokenizer."""
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    
    if model_type == 'baseline':
        model = GPT2LMHeadModel.from_pretrained(model_path)
    elif model_type == 'masked':
        # Load mask schedule if available
        mask_schedule_path = os.path.join(model_path, 'mask_schedule.json')
        mask_schedule = None
        if os.path.exists(mask_schedule_path):
            with open(mask_schedule_path, 'r') as f:
                mask_schedule = json.load(f)
        
        model = MaskedGPT2LMHeadModel.from_pretrained(
            model_path, 
            mask_schedule=mask_schedule
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, tokenizer


def compare_models(baseline_path: str, masked_paths: List[str], 
                  dataset_name: str = 'tiny_stories', 
                  output_dir: str = './results') -> Dict[str, Any]:
    """Compare baseline and masked models."""
    logger.info("Starting model comparison...")
    
    # Prepare dataset
    datasets, _ = prepare_dataset(dataset_name, max_length=1024)
    test_loader = get_dataloader(datasets['test'], batch_size=8, shuffle=False)
    
    results = {}
    
    # Evaluate baseline model
    logger.info("Evaluating baseline model...")
    baseline_model, baseline_tokenizer = load_model(baseline_path, 'baseline')
    baseline_evaluator = ModelEvaluator(baseline_model, baseline_tokenizer)
    
    results['baseline'] = baseline_evaluator.full_evaluation(
        test_loader, 
        os.path.join(output_dir, 'baseline', 'evaluation_results.json')
    )
    
    # Evaluate masked models
    results['masked'] = {}
    for masked_path in masked_paths:
        schedule_name = os.path.basename(masked_path)
        logger.info(f"Evaluating masked model: {schedule_name}")
        
        masked_model, masked_tokenizer = load_model(masked_path, 'masked')
        masked_evaluator = ModelEvaluator(masked_model, masked_tokenizer)
        
        results['masked'][schedule_name] = masked_evaluator.full_evaluation(
            test_loader,
            os.path.join(output_dir, 'mask_scheduled', f'{schedule_name}_evaluation_results.json')
        )
    
    # Save comparison results
    comparison_file = os.path.join(output_dir, 'model_comparison.json')
    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Comparison results saved to {comparison_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate GPT-2 models')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model to evaluate')
    parser.add_argument('--model_type', type=str, default='baseline',
                       choices=['baseline', 'masked'],
                       help='Type of model to evaluate')
    parser.add_argument('--dataset', type=str, default='tiny_stories',
                       choices=['wikitext-103', 'tiny_stories'],
                       help='Dataset to evaluate on')
    parser.add_argument('--output_file', type=str, default='evaluation_results.json',
                       help='Output file for results')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple models')
    parser.add_argument('--masked_models', nargs='+', default=[],
                       help='Paths to masked models for comparison')
    
    args = parser.parse_args()
    
    if args.compare:
        # Model comparison mode
        compare_models(
            baseline_path=args.model_path,
            masked_paths=args.masked_models,
            dataset_name=args.dataset,
            output_dir='./results'
        )
    else:
        # Single model evaluation
        model, tokenizer = load_model(args.model_path, args.model_type)
        
        # Prepare dataset
        datasets, _ = prepare_dataset(args.dataset, max_length=1024)
        test_loader = get_dataloader(
            datasets['test'], 
            batch_size=args.batch_size, 
            shuffle=False
        )
        
        # Run evaluation
        evaluator = ModelEvaluator(model, tokenizer)
        results = evaluator.full_evaluation(test_loader, args.output_file)
        
        # Print summary
        logger.info("Evaluation Summary:")
        logger.info(f"Perplexity: {results['perplexity']:.2f}")
        logger.info(f"Throughput: {results['throughput']['tokens_per_second']:.2f} tokens/sec")
        logger.info(f"Latency (512 tokens): {results['latency']['length_512']['mean_latency_ms']:.2f} ms")
        logger.info(f"GPU Memory: {results['memory'].get('gpu_memory_allocated_mb', 0):.2f} MB")


if __name__ == "__main__":
    main()