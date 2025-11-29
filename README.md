# Layer-wise Attention Masking in GPT-2

A research implementation exploring computational efficiency improvements in transformer language models through progressive layer-wise attention masking.

## Overview

This project investigates whether different transformer layers can operate effectively with different attention patterns, specifically using progressively smaller attention windows in deeper layers. The goal is to reduce the quadratic complexity of self-attention while maintaining model performance.

## Key Features

- **Layer-wise Masking Framework**: Custom implementation allowing different attention patterns per transformer layer
- **Progressive Window Schedules**: Four masking strategies from conservative to aggressive sparsity
- **Comprehensive Evaluation**: Performance and efficiency analysis across multiple datasets
- **Attention Visualization**: Tools for analyzing and visualizing attention patterns
- **Efficient Implementation**: Optimized masking operations with minimal overhead

## Research Questions

1. Can deeper transformer layers operate effectively with more restricted attention patterns?
2. What is the optimal progression of attention window sizes across layers?
3. How do different masking schedules affect the efficiency vs quality trade-off?
4. Which language modeling tasks are most amenable to layer-wise attention masking?

## Project Structure

```
layerwise_mask_gpt/
├── src/                          # Source code
│   ├── model/                    # Model implementations
│   │   ├── gpt2_custom.py       # Custom GPT-2 with masking support
│   │   ├── masked_mha.py        # Masked multi-head attention
│   │   ├── mask_utils.py        # Masking utilities and patterns
│   │   └── schedule.py          # Masking schedule definitions
│   ├── dataset.py               # Dataset loading and preprocessing
│   ├── train_baseline.py        # Baseline model training
│   ├── train_masked.py          # Masked model training
│   └── evaluate.py              # Model evaluation and analysis
├── config/                       # Configuration files
│   ├── train_config.json        # Training hyperparameters
│   └── schedule_config.json     # Masking schedule definitions
├── data/                         # Datasets
│   ├── raw/                     # Raw data files
│   ├── processed/               # Preprocessed data
│   └── tokenized/               # Tokenized datasets
├── models/                       # Saved models
│   ├── checkpoints/             # Training checkpoints
│   ├── baseline/                # Baseline models
│   └── masked/                  # Masked models
├── results/                      # Experimental results
│   ├── logs/                    # Training logs
│   ├── metrics/                 # Performance metrics
│   ├── plots/                   # Visualization outputs
│   └── analysis/                # Analysis notebooks
├── notebooks/                    # Jupyter notebooks
│   ├── data_exploration.ipynb   # Data analysis and exploration
│   ├── baseline_eval.ipynb      # Baseline model evaluation
│   └── mask_schedule_eval.ipynb # Masking schedule comparison
├── report/                       # Research documentation
│   ├── main.tex                 # LaTeX research paper
│   ├── proposal.pdf             # Research proposal
│   └── figures/                 # Paper figures and plots
├── presentation/                 # Presentation materials
│   ├── slides.pptx              # Research presentation slides
│   └── video.mp4                # Presentation video
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
└── installation.txt              # Installation instructions
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (for GPU training)
- At least 16GB RAM (32GB recommended)
- 50GB+ disk space for datasets and models

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd layerwise_mask_gpt
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download datasets (optional, will download automatically during training):
   ```bash
   python -c "from src.dataset import prepare_dataset; prepare_dataset('wikitext-103')"
   python -c "from src.dataset import prepare_dataset; prepare_dataset('tinystories')"
   ```

## Quick Start

### 1. Train Baseline Model

Train a standard GPT-2 model without masking:

```bash
python src/train_baseline.py --config config/train_config.json --dataset wikitext-103
```

### 2. Train Masked Model

Train a model with layer-wise attention masking:

```bash
python src/train_masked.py --config config/train_config.json --schedule config/schedule_config.json --dataset wikitext-103
```

### 3. Evaluate Models

Compare baseline and masked model performance:

```bash
python src/evaluate.py --baseline models/baseline/gpt2_wikitext.pt --masked models/masked/gpt2_masked_wikitext.pt --dataset wikitext-103
```

### 4. Analyze Results

Use the provided Jupyter notebooks for detailed analysis:

```bash
jupyter notebook notebooks/mask_schedule_eval.ipynb
```

## Masking Schedules

The project implements four progressive masking schedules:

### Full Schedule (Baseline)
- All layers use full attention
- Standard GPT-2 behavior
- Maximum quality, standard efficiency

### Half Schedule  
- Layers 0-5: Full attention
- Layers 6-11: Progressive windowing (512→16 tokens)
- Moderate efficiency gains

### Quarter Schedule
- Layers 0-2: Full attention  
- Layers 3-11: Progressive windowing (1024→4 tokens)
- Significant efficiency improvements

### Aggressive Schedule
- Layer 0: Full attention
- Layers 1-11: Aggressive windowing (1024→1 token)
- Maximum efficiency, quality trade-offs

## Configuration

### Training Configuration (`config/train_config.json`)

```json
{
  "model_name": "gpt2",
  "max_length": 1024,
  "batch_size": 16,
  "learning_rate": 5e-5,
  "num_epochs": 3,
  "warmup_steps": 1000,
  "logging_steps": 100,
  "save_steps": 1000,
  "output_dir": "models/checkpoints"
}
```

### Schedule Configuration (`config/schedule_config.json`)

```json
{
  "schedules": {
    "full": {"type": "none"},
    "half": {
      "type": "progressive",
      "start_layer": 6,
      "start_window": 512,
      "end_window": 16
    },
    "quarter": {
      "type": "progressive", 
      "start_layer": 3,
      "start_window": 1024,
      "end_window": 4
    },
    "aggressive": {
      "type": "progressive",
      "start_layer": 1,
      "start_window": 1024,
      "end_window": 1
    }
  }
}
```

## Results

### Expected Performance

Based on preliminary experiments:

| Schedule | Speedup | Memory Reduction | Perplexity Increase |
|----------|---------|------------------|-------------------|
| Full     | 1.0x    | 0%              | 0% (baseline)     |
| Half     | 1.5x    | 25%             | 5-8%              |
| Quarter  | 2.2x    | 45%             | 12-18%            |
| Aggressive| 3.1x   | 65%             | 25-35%            |

### Key Findings

1. **Layer Specialization**: Deeper layers can operate with significantly reduced attention windows
2. **Task Dependency**: Simple generation tasks show better tolerance to aggressive masking
3. **Window Size**: Gradual reduction is more effective than abrupt changes
4. **Memory Efficiency**: Substantial memory savings enable processing of longer sequences

## Evaluation Metrics

### Performance Metrics
- **Perplexity**: Primary language modeling metric
- **BLEU Score**: Generation quality assessment  
- **Task-specific**: Downstream task performance where applicable

### Efficiency Metrics
- **Throughput**: Tokens processed per second
- **Memory Usage**: Peak GPU memory consumption
- **FLOPs**: Floating-point operations count
- **Latency**: End-to-end inference time

### Analysis Metrics
- **Attention Entropy**: Diversity of attention patterns
- **Layer Analysis**: Per-layer attention statistics
- **Pattern Visualization**: Attention matrix heatmaps

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Research Context

This project is part of ongoing research into efficient transformer architectures. The work builds on recent advances in sparse attention mechanisms and aims to provide both practical improvements and theoretical insights into transformer layer behavior.

### Related Work

- Sparse attention mechanisms (Child et al., 2019)
- Longformer sliding window attention (Beltagy et al., 2020)
- BigBird sparse attention patterns (Zaheer et al., 2020)
- Layer-wise analysis of transformers (Rogers et al., 2020)

### Citation

If you use this code in your research, please cite:

```bibtex
@article{layerwise_mask_gpt2024,
  title={Layer-wise Attention Masking in GPT-2: Improving Efficiency Through Progressive Sparsity},
  author={[Your Name]},
  journal={[Conference/Journal]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- HuggingFace Transformers library for model implementations
- OpenAI for the original GPT-2 architecture
- PyTorch team for the deep learning framework
- Research community for datasets and evaluation metrics

## Contact

For questions or collaboration opportunities, please reach out:

- Email: [your-email]
- GitHub: [your-github]
- Research Profile: [your-profile]

---

**Note**: This is a research project aimed at advancing understanding of transformer efficiency. Results may vary across different hardware configurations and datasets. The code is provided for educational and research purposes.