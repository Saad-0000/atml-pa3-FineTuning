# ATML PA3 — Task 2: Vision-Language Models and Visual Grounding

This directory contains the complete, reproducible implementation for **Task 2 only** of Assignment 3 for **EE-5102 / CS-6304: Advanced Topics in Machine Learning (Fall 2026)**.

---

## 1. System Architecture & Components

| Component | Architecture / Model ID | Precision / Training Policy |
|---|---|---|
| **Vision Encoder** | `google/siglip-base-patch16-224` | Frozen; 196 patch features in 14×14 spatial order; final LayerNorm applied exactly once; features cached in FP16. |
| **Connector** | Two-layer MLP (`768 → 1024 → GELU → 1024`) with scale calibration buffer | Trainable throughout all three stages. |
| **Language Model** | `Qwen/Qwen3-0.6B` | Frozen original weights (`torch.float16`). LoRA enabled in Stages 2 & 3 (`r=16, \alpha=32, \text{dropout}=0.05` on `q_proj`, `k_proj`, `v_proj`, `o_proj`). `enable_thinking=False`. |

### Multimodal Input Construction & Label Masking
- Visual tokens (196 embeddings) are mapped into Qwen's embedding space via the connector.
- Visual tokens are inserted into the user turn **before the question text**:
  `[BOS] <|im_start|>user\n[196 visual tokens] {question}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>`
- Labels: Visual tokens, prompt tokens, and padding tokens are masked with label `-100`. Only target response tokens and the end-of-turn token (`<|im_end|>`) are supervised.
- **Example-Weighted Loss**:
  $$L = \frac{1}{|B|} \sum_{i \in B} \left[ \frac{1}{|R_i|} \sum_{t \in R_i} -\log p(y_{i,t} \mid I_i, q_i, y_{i,<t}) \right]$$

---

## 2. Directory Layout

```text
task2/
├── README.md                      # Complete system documentation (this file)
├── requirements.txt               # Locked dependencies
├── configs/
│   ├── task2.yaml                 # Master configuration (paths, hyperparameters, content caps)
│   ├── stages.yaml                # Three-stage schedule and sampling distributions
│   └── evaluation.yaml            # Dev schedule (12 checkpoints) & final test protocol
├── notebooks/
│   └── task2_colab.ipynb          # End-to-end Colab T4 interactive notebook
├── scripts/
│   ├── preflight.py               # Hardware inspection, autograd smoke test, magnitude calibration
│   ├── prepare_data.py            # Deterministic data selection, disjointness audit, split manifests
│   ├── cache_features.py          # Resumable SigLIP patch feature extraction in FP16
│   ├── train_stage.py             # 3-stage training loop & crash-safe checkpointing
│   ├── evaluate_dev.py            # Development evaluation across 12 scheduled checkpoints
│   ├── evaluate_final.py          # Final greedy evaluation on 4 conditions + 4,000 bootstrap CIs
│   ├── run_qualitative.py         # Categorized qualitative pair selection and review sheets
│   ├── build_results.py           # Compiles all tables (T01-T12) and plots (P01-P12)
│   ├── audit_evidence.py          # Validates every Required Evidence item
│   └── optional_compression.py    # Optional 49-token visual compression experiment
├── src/
│   ├── data/                      # Normalization, COCO captions, grounding pairs, VQAv2, samplers
│   ├── models/                    # SigLIP extractor, MLP connector, LoRA linear, VLM model
│   ├── training/                  # Example-weighted loss, scheduler, checkpoint manager, trainer
│   ├── evaluation/                # Dev and final evaluators
│   ├── metrics/                   # Soft VQA consensus, exact match, paired complete-pair bootstrap
│   ├── diagnostics/               # Calibration, gradient flow, memory smoke tests
│   ├── visualization/             # Table builders, plotters, qualitative HTML viewers
│   └── utils/                     # Logging, Google Drive persistence, manifests
├── tests/                         # Unit tests (loss, metrics, models, data, checkpointing)
└── results/                       # Target directory for tables, plots, predictions, logs, manifests
```

---

## 3. Prescribed Dataset & Split Counts

All data selection uses random seed **`6304`**. Training images come from COCO `train2014`; development and final test images come from COCO `val2014`. All splits are strictly mutually image-disjoint.

| Component | Split | Prescribed Counts | Implementation Details |
|---|---|---|---|
| **COCO Captions** | Train | 6,000 images × 3 captions = 18,000 examples | Stage 1 (2 passes = 36,000 presentations) & replay |
| **COCO Captions** | Dev | 96 images × 1 caption = 96 examples | Reference caption loss evaluation |
| **Grounding Pairs** | Dev | 24 pairs / 48 questions | Presence & counting balanced |
| **Grounding Pairs** | Final Test | 64 pairs / 128 questions | Presence & counting balanced |
| **Natural VQAv2** | Train | 1,536 pairs / 3,072 questions | 640 yes/no, 256 number, 640 other |
| **Natural VQAv2** | Dev | 32 pairs / 64 questions | 12 yes/no, 8 number, 12 other |
| **Natural VQAv2** | Final Test | 96 pairs / 192 questions | 36 yes/no, 24 number, 36 other |

---

## 4. Training Schedule & Mixture Protocol

The main run executes **4,500 optimizer updates** (72,000 example presentations) at effective batch size 16 (microbatch 8, gradient accumulation 2) using AdamW (weight decay 0.01, gradient clipping norm 1.0 with unscaling before clipping, FP16 autocast):

- **Stage 1: Caption Alignment** (Updates 1 – 2,250):
  - Trainable: Connector only. LoRA disabled.
  - Connector peak LR: `1e-3` (5% warmup, cosine decay to 10% peak LR).
  - Data: 2 passes over 18,000 COCO captions.
- **Stage 2: Visual Instruction Tuning** (Updates 2,251 – 3,750):
  - Trainable: Connector + new LoRA adapters.
  - Connector peak LR: `1e-4`, LoRA peak LR: `2e-5`.
  - Mixture sampling per draw: `0.50` presence grounding, `0.25` counting grounding, `0.25` caption replay.
- **Stage 3: Natural VQA Specialization** (Updates 3,751 – 4,500):
  - Trainable: Same connector + same LoRA adapters continued from Stage 2.
  - Connector peak LR: `5e-5`, LoRA peak LR: `1e-5`.
  - Mixture sampling per draw: `0.70` natural VQAv2, `0.10` presence replay, `0.10` counting replay, `0.10` caption replay.

### Checkpointing & Resumption
- Resumable checkpoints are saved every **250 updates** (250, 500, ..., 4,500) and at stage endpoints.
- Checkpoints use an atomic write pattern (`.tmp` → SHA-256 verification → `.pt`) and update `checkpoint_manifest.json`.
- Resumption restores: connector state, LoRA state, optimizer state, scheduler state, GradScaler state, sampler RNG states, and mixture counters.

---

## 5. Execution Guide (Google Colab T4)

### Step 1: Preflight Verification
```bash
python -m task2.scripts.preflight --config task2/configs/task2.yaml
```
Verifies GPU VRAM, autograd gradient flow through frozen Qwen, and outputs Table `T03`.

### Step 2: Data Preparation & Audit
```bash
python -m task2.scripts.prepare_data --config task2/configs/task2.yaml
```
Filters splits, validates image disjointness, and outputs Tables `T01` and `T02`.

### Step 3: Feature Extraction & Caching
```bash
python -m task2.scripts.cache_features --config task2/configs/task2.yaml
```
Caches SigLIP FP16 tokens (`[196, 768]`) on persistent storage and unloads the vision encoder from GPU memory.

### Step 4: Training Orchestration
```bash
# Train all 3 stages sequentially:
python -m task2.scripts.train_stage --all --config task2/configs/task2.yaml

# To resume after a disconnect:
python -m task2.scripts.train_stage --all --resume --config task2/configs/task2.yaml
```

### Step 5: Development & Final Evaluation
```bash
# Evaluate scheduled checkpoints (T06):
python -m task2.scripts.evaluate_dev --all-scheduled --config task2/configs/task2.yaml

# Evaluate final test model across 4 image conditions with 4,000 bootstrap resamples (T07, T08, T09, P05-P10):
python -m task2.scripts.evaluate_final --config task2/configs/task2.yaml --n-bootstrap 4000
```

### Step 6: Qualitative Findings
```bash
python -m task2.scripts.run_qualitative --config task2/configs/task2.yaml
```
Generates `qualitative_example_manifest.csv`, `human_review_sheet.md`, and `aligned_complementary_examples.html`.

### Step 7: Build Results & Audit Evidence
```bash
python -m task2.scripts.build_results --config task2/configs/task2.yaml
python -m task2.scripts.audit_evidence --config task2/configs/task2.yaml
```
Validates all Required Evidence items and catalogs artifacts in `evidence_manifest.json` and `dictionary_manifest.json`.

---

## 6. Output Tables & Plots Index

### Tables (`results/tables/`)
- `T01_dataset_split_inventory.csv` / `.md`: Dataset components, split counts, image counts, and audit status.
- `T02_filtering_pair_selection_audit.csv` / `.md`: Candidate filtering counts, agreement checks, and disjointness verification.
- `T03_model_interface_verification.csv` / `.md`: Model revisions, feature shapes, scale calibration, and autograd gradient flow test.
- `T04_training_configuration.csv` / `.md`: Training parameters, learning rates, batch sizes, and schedule endpoints.
- `T05_stage_sampling_mixture.csv` / `.md`: Target vs. realized draw counts and mixture proportions.
- `T06_development_evaluation_by_checkpoint.csv` / `.md`: Caption loss, reference NLLs, and $\Delta\text{NLL}$ across 12 checkpoints.
- `T07_final_grounding_results.csv` / `.md`: Presence, counting, combined exact match, pair accuracy, and 95% CIs.
- `T08_final_vqav2_results.csv` / `.md`: Yes/no, number, other, overall soft consensus accuracy, pair accuracy, and 95% CIs.
- `T09_correct_image_advantage_bootstrap.csv` / `.md`: Paired correct-image advantages over complementary swaps.
- `T10_resource_comparison.csv` / `.md`: Phase runtimes, peak GPU memory, throughput, and persistent storage sizes.
- `T11_checkpoint_recovery_audit.csv` / `.md`: 250-update checkpoints, SHA-256 checksums, and recovery status.
- `T12_evidence_completeness.csv` / `.md`: Traceable status for every Required Evidence requirement.

### Plots (`results/plots/`)
- `P01_training_loss.png`: Main causal cross-entropy loss curve (updates 1–4,500) with Stage 1/2/3 boundaries.
- `P02_development_caption_loss.png`: Caption reference loss across 12 scheduled checkpoints.
- `P03_development_answer_nll.png`: Grounding and VQAv2 answer NLL across Correct, Swap, and No-Image conditions.
- `P04_development_image_swap_penalty.png`: Image-swap penalty $\Delta\text{NLL}$ trajectory over training.
- `P05_final_grounding_accuracy_by_condition.png`: Grounding accuracy by condition with 95% bootstrap error bars.
- `P06_final_vqav2_accuracy_by_condition.png`: VQAv2 soft consensus accuracy by condition with 95% bootstrap error bars.
- `P07_grounding_pair_diagnostics.png`: Pair accuracy, identical-answer rate, empty rate, truncation rate.
- `P08_vqav2_pair_diagnostics.png`: Pair diagnostics for natural VQAv2.
- `P09_reference_nll_and_generation_time.png`: Reference NLL and inference latency across conditions.
- `P10_correct_image_advantages.png`: Paired correct-image advantages with 95% complete-pair bootstrap intervals.
- `P11_training_mixture_composition.png`: Realized vs. target sampling proportions.
- `P12_resource_usage.png`: GPU memory, phase runtimes, and trainable parameter counts.

---

## 7. Attributions & External Libraries
- **PyTorch & torchvision**: Deep learning primitives and tensor autograd operations.
- **Hugging Face Transformers**: Base implementations for `google/siglip-base-patch16-224` and `Qwen/Qwen3-0.6B`.
- **LoRA**: Parameter-efficient fine-tuning linear layers (`r=16, \alpha=32`).
- **MS COCO & VQAv2**: Annotations and images according to assignment license specifications.
