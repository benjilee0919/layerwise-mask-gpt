# Layer-wise Mask Scheduling in GPT-2

This repository contains the code for my final project in **EN.705.743 — ChatGPT from Scratch**.  
The goal of the project was to investigate whether **Layer-wise Mask Scheduling (LMS)**  
— applying different attention window sizes at different transformer layers —  
can improve the efficiency of GPT-2 in plain PyTorch.

The key takeaway is a **negative but very informative result**:
mask-based sparsity in PyTorch provides only *logical* sparsity, not *compute* sparsity.
As a result, the LMS model is both **slower** and **less accurate** than the baseline GPT-2.

---

## 1. Overview

### Research Question

> Can we make GPT-2 more efficient by giving deeper layers smaller attention windows,  
> without using any custom CUDA kernels or specialized sparse-attention libraries?

### Short Answer

> **No.** In standard PyTorch, masks do **not** reduce FLOPs.  
> PyTorch still performs a full dense QK matmul, so Layer-wise Mask Scheduling
> only reduces visible context and model capacity, not computation.

---

## 2. Key Ideas

- **Layer-wise Mask Scheduling (LMS)**  
  Different transformer layers use different attention window sizes:
  - Some layers use full causal attention.
  - Others use medium (e.g., 256-token) windows.
  - Others use narrow (e.g., 128-token) windows.

- **Logical vs. Compute Sparsity**  
  - *Logical sparsity*: masks tell the model which positions to ignore,  
    but the dense attention matrix is still fully computed.
  - *Compute sparsity*: the underlying kernel avoids computing masked positions.  
    This requires CUDA/Triton/FlashAttention-style kernels and is **not** provided by
    vanilla PyTorch attention.

- **Outcome**  
  - LMS **increased** eval loss (worse perplexity).  
  - LMS **decreased** throughput (slower training/evaluation).  
  - This demonstrates the limitations of mask-only approaches in practice.

---

## 3. Project Structure

The actual project layout is:

```text
layerwise-mask-gpt/
├── config/
│   ├── train_config.json        # Training hyperparameters
│   └── schedule_config.json     # Layer-wise mask schedule definitions (incl. lms_main)
│
├── src/
│   ├── dataset.py               # TinyStories dataset loading
│   ├── evaluate.py              # Evaluation utilities
│   ├── train_baseline.py        # Baseline GPT-2 training script
│   ├── train_masked.py          # LMS model training script
│   ├── __init__.py
│   └── model/
│       ├── gpt2_custom.py       # GPT-2 model wired with masked attention
│       ├── masked_mha.py        # Masked multi-head attention implementation
│       ├── mask_utils.py        # Helper functions for mask manipulation/visualization
│       ├── schedule.py          # Schedule loading / parsing
│       └── __init__.py
│
├── config/
│   ├── train_config.json        # Training hyperparameters for both models
│   └── schedule_config.json     # Mask schedule definitions (full, half, quarter, lms_main, ...)
├── models/                      # (Local only) model checkpoints — ignored by .gitignore
├── results/                     # (Local only) logs and metrics — ignored by .gitignore
├── notebooks/                   # (Local only) analysis notebooks — ignored by .gitignore
├── presentation/                # (Local only) slides and video — ignored by .gitignore
├── report/                      # (Local only) proposal / LaTeX files — ignored by .gitignore
├── requirements.txt             # Python dependencies
└── installation.txt             # Installation instructions
```

> Note: The folders (`models/`, `results/`, `notebooks/`, `presentation/`, `report/`)
> are *local-only* and intentionally ignored by `.gitignore`.  
> They are used during development, experimentation, and course submission,
> but are not uploaded to GitHub to keep the repository lightweight.

---

## 4. Installation

See **`installation.txt`** for a detailed setup guide.  
A minimal workflow is:

```bash
# 1. Create environment
conda create -n lms-gpt python=3.10
conda activate lms-gpt

# 2. Install dependencies
pip install -r requirements.txt
```

PyTorch with CUDA can be installed via the official wheels if you want GPU support.

---

## 5. Usage

All experiments in this project are conducted on **TinyStories**.

### 5.1 Train Baseline GPT-2 (full attention)

```bash
python src/train_baseline.py \
  --dataset tiny_stories \
  --config config/train_config.json \
  --output_dir models/baseline_gpt2
```

### 5.2 Train LMS Model (Layer-wise Mask Scheduling)

```bash
python src/train_masked.py \
  --dataset tiny_stories \
  --config config/train_config.json \
  --schedule config/schedule_config.json \
  --output_dir models/mask_gpt2
```

The `schedule_config.json` file defines several possible schedules
(`full`, `half`, `quarter`, `aggressive`, `block_sparse`, `lms_main`),
but the **final results in the report and slides use only `lms_main`.**

### 5.3 Evaluation and Analysis

After training:

- The training scripts write evaluation metrics (e.g., `eval_loss`,
  `eval_samples_per_second`) into `test_results.json` inside the model output dirs.
- Additional analysis (loss curves, speed plots) is done in local notebooks
  under `notebooks/` using those JSON logs.

---

## 6. Results (Summary)

All results are reported on TinyStories with GPT-2 Small, 2 epochs, batch size 8, A100 GPU.

- **Evaluation Loss**
  - Baseline: ~**0.30**
  - LMS: ~**0.66**  
  → LMS has more than **2× higher** eval loss.

- **Perplexity**
  - Baseline: **1.35**
  - LMS: **1.94**  
  → About **43% worse** perplexity.

- **Evaluation Throughput**
  - Baseline: **45.8 samples/sec**
  - LMS: **26.0 samples/sec**  
  → About **40% slower**.

- **Training Runtime**
  - Baseline: ~**6,247 s**
  - LMS: ~**9,378 s**  
  → About **50% longer** training time.

These numbers clearly show that LMS, as implemented with PyTorch masking,
does **not** improve efficiency and instead makes things worse.

---

## 7. Lessons Learned

The main lessons from this project are:

1. **Mask scheduling provides logical sparsity, not compute sparsity.**  
   PyTorch still executes dense QK matmul, so FLOPs remain $O(n^2)$.

2. **LMS underperformed in both speed and quality.**  
   Smaller attention windows without kernel changes simply reduce model capacity.

3. **Real speedups require kernel-level sparse attention.**  
   To exploit structured sparsity, the attention kernels themselves must be
   designed to skip masked positions (e.g., FlashAttention/Triton-style kernels).

4. **Negative results are informative.**  
   This project clarifies why mask-only approaches are insufficient and where
   future work must focus to make sparse transformers practical.

---

## 8. Acknowledgments

- This project was completed as part of  
  **EN.705.743 — ChatGPT from Scratch** at Johns Hopkins University.
- Many components (GPT-2, training utilities) are built on top of  
  the excellent **HuggingFace Transformers** and **PyTorch** ecosystems.