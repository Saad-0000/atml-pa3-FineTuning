# Advanced Topics in Machine Learning (Fall 2026) – Programming Assignment 3

This repository contains the implementation and experimental framework for **Programming Assignment 3: Large Language Model Fine-Tuning, Vision-Language Grounding, and Test-Time Compute** (Course: EE-5102 / CS-6304).

## Repository Architecture

Per the assignment specification and Revision 5 prompt instructions:
- **`task1/`**: Fully implemented and self-contained codebase for **Task 1: Large Language Model Fine-Tuning** (Full FT, Full FT + L2-SP, LoRA-4, LoRA-16, and OPLoRA-16 on `Qwen/Qwen3-0.6B`). Includes tests, configs, data pipelines, training routines, evaluation protocols, diagnostics, results generation, Colab runbook, and evidence audit.
- **`task2/`**: **Intentionally empty placeholder** (contains only `.gitkeep`). Dedicated to future implementation of Vision-Language Models and Visual Grounding (SigLIP + Qwen3-0.6B).
- **`task3/`**: **Intentionally empty placeholder** (contains only `.gitkeep`). Dedicated to future implementation of Test-Time Compute (Sampling, Selection, and Adaptive Inference).
- **`manifests/`** and **`shared_assets/`**: Cross-task asset inventories and future-task readiness manifests (`FUTURE_TASK_READINESS.md`).

```text
repository-root/
├── README.md
├── .gitignore
├── manifests/
│   └── future_task_readiness.json
├── FUTURE_TASK_READINESS.md
├── shared_assets/                 # inventory/manifests; large raw assets stored persistently on Drive
│   └── README.md
├── task1/                         # Full Task 1 implementation & evidence
│   ├── README.md
│   ├── COLAB_RUNBOOK.md
│   ├── pyproject.toml
│   ├── configs/
│   │   ├── task1.yaml
│   │   ├── methods.yaml
│   │   └── evaluation.yaml
│   ├── notebooks/
│   │   └── task1_colab.ipynb
│   ├── scripts/
│   ├── src/task1/
│   ├── tests/
│   ├── artifacts/                 # persistent artifacts cache
│   ├── checkpoints/               # persistent recovery checkpoints
│   ├── logs/                      # persistent training logs
│   └── results/                   # tables, plots, raw metrics, audit
├── task2/
│   └── .gitkeep                   # strictly placeholder
└── task3/
    └── .gitkeep                   # strictly placeholder
```

## Quick Navigation

- **Task 1 Codebase & Documentation**: [`task1/README.md`](task1/README.md)
- **Google Colab Tesla T4 Runbook**: [`task1/COLAB_RUNBOOK.md`](task1/COLAB_RUNBOOK.md)
- **Interactive Colab Notebook**: [`task1/notebooks/task1_colab.ipynb`](task1/notebooks/task1_colab.ipynb)
- **Future Task Readiness**: [`FUTURE_TASK_READINESS.md`](FUTURE_TASK_READINESS.md)

## Reproducibility & Attribution

- **Base Model**: `Qwen/Qwen3-0.6B` (`enable_thinking=False`).
- **Seed**: Fixed seed `6304` throughout data filtering, probe sampling, adapter initialization, and evaluation.
- **Evaluation**: `lm-evaluation-harness` with pinned splits and zero-shot configurations (`gsm8k_cot` test 1,319; `hellaswag` val 10,042; `boolq` val 3,270; `arc_easy` test 2,376).
- **External Code Attribution**:
  - LoRA parameterization based on Hu et al. (2021).
  - L2-SP penalty based on Li et al. (2018).
  - OPLoRA orthogonal projection based on Xiong & Xie (2024).
  - Evaluation harnesses via EleutherAI `lm-evaluation-harness`.
