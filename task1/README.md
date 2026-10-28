# Task 1: Large Language Model Fine-Tuning and Trade-Offs

This directory contains the complete implementation, verification suite, configuration, and experimental workflow for **Task 1: Large Language Model Fine-Tuning** of EE-5102 / CS-6304: Advanced Topics in Machine Learning (Fall 2026).

---

## 1. Experimental Overview

Task 1 explores how adapting a language model to mathematical reasoning affects capabilities on other domains, investigating the fundamental trade-off between **adaptation** (specialization in mathematics) and **retention** (preservation of general reasoning, commonsense, and reading comprehension).

We compare five fine-tuning conditions starting independently from the same post-trained checkpoint **`Qwen/Qwen3-0.6B`** using fixed seed **`6304`**:

1. **Full Fine-Tuning (Full FT)**: All original parameters are updated ($\lambda = 0$). Peak learning rate $2 \times 10^{-5}$.
2. **Full Fine-Tuning with L2-SP (Full FT + L2-SP)**: Regularizes departures from starting parameters $\theta_0$:
   $$\Omega(\theta, \theta_0) = \frac{1}{2} \sum_{j=1}^P (\theta_j - \theta_{0,j})^2, \quad \lambda = 10^{-3}$$
   The penalty is applied once per optimizer update. Master parameters and reference weights are kept in FP32.
3. **LoRA Rank 4**: Freezes backbone; adds trainable low-rank adapters to all projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) in all 28 layers:
   $$W = W_0 + \frac{\alpha}{r} B A, \quad (r, \alpha) = (4, 8), \quad \text{peak LR } 2 \times 10^{-4}$$
4. **LoRA Rank 16**: Same adapter architecture and targets with quadrupled capacity:
   $$(r, \alpha) = (16, 32), \quad \text{peak LR } 2 \times 10^{-4}$$
5. **OPLoRA Rank 16 (Protection Rank $k=128$)**: Restricts updates to lie outside dominant singular subspaces of $W_0 = U \Sigma V^T$:
   $$\tilde{B} = B - U_k(U_k^T B), \quad \tilde{A} = A - (A V_k)V_k^T, \quad \Delta W = \frac{\alpha}{r} \tilde{B} \tilde{A}$$
   Bases $U_k, V_k$ are computed once in FP32. Projections are computed via effective factors without forming dense projection matrices.

---

## 2. Training Data and SFT Recipe

- **Source Dataset**: `meta-math/MetaMathQA`.
- **Filtering Pipeline**:
  - Scanning in preserved source order.
  - Exclude empty questions or assistant responses.
  - Exclude examples with formatted length $> 2,048$ tokens. **No truncation is permitted.**
  - Exclude contamination: any example whose `query` or `original_question` matches a GSM8K test question after deterministic Unicode NFKC, lowercasing, and whitespace normalization is removed.
  - Select exactly the **first 40,000 eligible examples**.
- **Batching & Updates**:
  - 40,000 examples $\div$ effective batch size $32$ = exactly **1,250 optimizer updates**.
  - Microbatch size: 2; Gradient accumulation steps: 16.
  - Token-weighted loss: Causal cross-entropy normalized by total valid response tokens in the effective batch of 32 examples.
  - User prompts and padding tokens are masked (`-100`). Assistant response and exactly one assistant end-of-turn token are supervised.
  - Thinking is disabled (`enable_thinking=False`).

---

## 3. Checkpoint & Evaluation Schedule

### Checkpoint Milestones
Full recovery checkpoints are saved after complete optimizer updates:
- **Milestone 1**: Update 312 (9,984 examples)
- **Milestone 2**: Update 625 (20,000 examples) – Midpoint evaluation
- **Milestone 3**: Update 937 (29,984 examples)
- **Milestone 4**: Update 1,250 (40,000 examples) – Final evaluation
- **Emergency Checkpoints**: Saved every 50 updates for Colab resiliency (keeping latest and 1 prior known-good).

### Evaluation Benchmarks (Zero-Shot via `lm-evaluation-harness`)
| Benchmark | Split / Examples | Scoring Method | Primary Metric | Chat Template Setting |
|---|---|---|---|---|
| **GSM8K** | `test` / 1,319 | `gsm8k_cot`, generation | Flexible exact match | `apply_chat_template=True` (non-thinking) |
| **HellaSwag** | `validation` / 10,042 | Candidate likelihood | Normalized accuracy (`acc_norm`) | `apply_chat_template=False` |
| **BoolQ** | `validation` / 3,270 | Candidate likelihood | Accuracy (`acc`) | `apply_chat_template=False` |
| **ARC-Easy** | `test` / 2,376 | Candidate likelihood | Normalized accuracy (`acc_norm`) | `apply_chat_template=False` |

---

## 4. Execution Commands (Colab & CLI)

All scripts run from the `task1/` directory or with `PYTHONPATH=task1/src`:

```bash
# 1. Run Preflight Verification (VRAM, T4 smoke test, persistent storage)
python scripts/preflight.py

# 2. Prepare Data (40k MetaMathQA filter + 64 activation probes)
python scripts/prepare_data.py

# 3. Evaluate Unchanged Baseline & Cache Reference theta0 and SVD Bases
python scripts/evaluate_baseline.py

# 4. Train Methods (Independent runs from starting checkpoint)
python scripts/train_method.py --method full_ft --resume auto
python scripts/train_method.py --method l2sp --resume auto
python scripts/train_method.py --method lora4 --resume auto
python scripts/train_method.py --method lora16 --resume auto
python scripts/train_method.py --method oplora16 --resume auto

# 5. Evaluate Required Conditions Across 4 Benchmarks
python scripts/evaluate_required.py --condition all

# 6. Compute Weight-Space and Activation Diagnostics (Layers 0, 13, 27)
python scripts/run_diagnostics.py

# 7. Generate Qualitative Responses & 42-Annotation Sheet
python scripts/run_qualitative.py

# 8. Build All Tables, Plots, and Evidence Manifests
python scripts/build_results.py

# 9. Audit Evidence Completeness and Repository Hygiene
python scripts/audit_evidence.py
```

---

## 5. Directory Structure

```text
task1/
├── README.md                  # This documentation
├── COLAB_RUNBOOK.md           # Step-by-step Colab execution instructions
├── pyproject.toml             # Pinned package specifications
├── configs/
│   ├── task1.yaml             # Global task parameters
│   ├── methods.yaml           # Method-specific hyperparameters
│   └── evaluation.yaml        # Evaluation and benchmark settings
├── notebooks/
│   └── task1_colab.ipynb      # Interactive Colab notebook
├── scripts/                   # Autonomous CLI entrypoints
├── src/task1/                 # Modular Python implementation
├── tests/                     # 25 verification unit tests
├── artifacts/                 # Data cache, theta0, SVD bases, probe cache
├── checkpoints/               # Resumable training milestones
└── results/                   # Tables, plots, raw metrics, and audit manifests
    ├── run_manifest.json
    ├── environment.json
    ├── dictionary_manifest.json
    ├── evidence_manifest.json
    ├── STATUS.md
    ├── raw/
    ├── tables/
    ├── plots/
    └── qualitative/
```
