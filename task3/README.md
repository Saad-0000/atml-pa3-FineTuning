# Task 3: Test-Time Compute: Sampling, Selection, and Adaptive Inference

This repository directory contains the complete, reproducible implementation of **Task 3** for ATML PA3 (Fall 2026). It investigates test-time compute scaling across sampling budgets, consensus and quality-aware selection rules, multimodal fusion, parallel vs. sequential refinement, and adaptive inference stopping policies.

---

## 1. Directory Structure

```text
task3/
├── README.md                         # Task 3 documentation & CLI manual
├── requirements.txt                  # Pinned dependencies
├── configs/
│   ├── offline_analysis.yaml         # Dataset metadata, budgets N in {1,2,4,8,16,32}, 10 orders
│   ├── live_generation.yaml          # Qwen2.5-3B-Instruct live generation parameters
│   └── adaptive_inference.yaml       # Development stopping grids (tau, gamma)
├── notebooks/
│   └── task3_colab_runbook.ipynb     # Step-by-step reproducible Colab runbook
├── scripts/
│   ├── preflight.py                  # Environment & GPU verification
│   ├── inspect_data.py               # Schema, alignment, and row count checks (Table T00)
│   ├── build_splits.py               # Deterministic 200 dev / 800 held-out splits (Table T01)
│   ├── prepare_data.py               # Rollout scoring & answer normalization (Table T02)
│   ├── score_rollouts.py             # Standalone rollout scoring
│   ├── run_offline_sampling.py       # Offline Experiments A & B (Tables T03, T04, T05)
│   ├── run_signal_analysis.py        # Signal Analysis Experiment C (Tables T06, T07, T08)
│   ├── run_live_generation.py        # Live Generation on 50 MATH questions (Tables T09, T10, T11)
│   ├── run_adaptive_tuning.py        # Dev split policy threshold selection (Table T12)
│   ├── run_adaptive_test.py          # Held-out frozen policy evaluation (Tables T13, T14)
│   ├── build_qualitative_cases.py    # Illustrative case extraction (Table T15)
│   ├── make_tables.py                # Master table compiler (Tables T16, T17)
│   ├── make_plots.py                 # Master figure renderer (Figures P01 through P16)
│   ├── audit_evidence.py             # Deliverable completeness audit & manifest generator
│   └── resume.py                     # Crash-safe orchestrator & pipeline resumer
├── src/
│   ├── data/                         # Parquet loading, splits, synthetic fallbacks
│   ├── scoring/                      # Balanced-brace boxed extractor & normalization
│   ├── sampling/                     # PermutationManager (10 orders, nested prefixes)
│   ├── selection/                    # FirstAns, SC, Highest Score, RWC
│   ├── signals/                      # Quartile analysis, c-v correlation, covariance fusion
│   ├── live_generation/              # Single-long, parallel, sequential refinement
│   ├── adaptive/                     # Agreement & Quality-Aware controllers, grid search
│   ├── metrics/                      # Oracle coverage, failure decomposition
│   ├── visualization/                # Tables & publication-quality plot renderers
│   └── utils/                        # Logging, atomic JSON, bootstrap CIs
├── tests/                            # Automated test suite (21 unit tests)
├── manifests/                        # Split manifests & frozen adaptive policies
├── checkpoints/                      # Model & resume checkpoints
├── artifacts/                        # Intermediate artifacts
├── logs/                             # Execution logs & preflight reports
└── results/
    ├── tables/                       # Mandatory tables T00 through T17 (CSV & JSON)
    ├── plots/                        # Mandatory publication figures P01 through P16 (PNG & PDF)
    ├── per_question/                 # Question-level offline outputs
    ├── per_rollout/                  # Canonical parquet rollouts
    ├── live_generation/              # Raw prompts, responses, timings, memory
    ├── adaptive/                     # Adaptive evaluation outputs
    ├── qualitative/                  # Qualitative case gallery
    ├── evidence_manifest.json        # Machine-readable deliverables manifest
    └── EVIDENCE_AUDIT.md             # Human-readable evidence audit report
```

---

## 2. Experimental Protocol & Non-Negotiable Rules

1. **Seed Anchoring:** All random orderings, permutations, bootstrap resamplings, and generations are anchored to `seed=6304`.
2. **Deterministic Splits:**
   - 1,000 questions per dataset sorted lexicographically as strings.
   - First 200 permuted questions $\rightarrow$ **Development split**.
   - Remaining 800 permuted questions $\rightarrow$ **Held-out evaluation split**.
   - The first 50 held-out MATH questions in this exact order form the live generation subset.
3. **Strict No-Data-Leakage:**
   - Gold answers are NEVER consulted during answer-group formation, candidate scoring, tie-breaking, or stopping decisions.
   - Development data is used strictly for threshold tuning; held-out data is used only for frozen evaluation.
   - At stopping checks $n \in \{4, 8, 16\}$, controllers and VisConf percentile scalings ($Q_{.05}, Q_{.95}$) only inspect candidates revealed up to $n$.
4. **Mandatory Live Reuse:**
   - In live generation, the first 1,024-token parallel response is generated once and reused as the initial response for sequential refinement.
5. **Cost Accounting:**
   - FirstAns consumed token cost counts ONLY the first candidate.
   - Parallel and refinement both charge the shared initial response.

---

## 3. CLI Commands & Execution Flow

Run the full end-to-end pipeline with one command:
```bash
python task3/scripts/resume.py --data_dir data/ATML-PA3-Data
```

Or execute individual stages:

### Step 1: Preflight Verification
```bash
python task3/scripts/preflight.py --data_dir data/ATML-PA3-Data
```

### Step 2: Data Inspection (Table T00)
```bash
python task3/scripts/inspect_data.py --data_dir data/ATML-PA3-Data
```

### Step 3: Split Generation (Table T01)
```bash
python task3/scripts/build_splits.py --data_dir data/ATML-PA3-Data
```

### Step 4: Rollout Scoring & Normalization (Table T02)
```bash
python task3/scripts/prepare_data.py --data_dir data/ATML-PA3-Data
```

### Step 5: Offline Sampling Experiments A & B (Tables T03, T04, T05)
```bash
python task3/scripts/run_offline_sampling.py
```

### Step 6: Signal Analysis Experiment C (Tables T06, T07, T08)
```bash
python task3/scripts/run_signal_analysis.py
```

### Step 7: Live Generation (50 MATH questions, Tables T09, T10, T11)
```bash
python task3/scripts/run_live_generation.py --data_dir data/ATML-PA3-Data
```
*(Use `--dry_run` for testing without a GPU)*

### Step 8: Adaptive Threshold Tuning on Dev Split (Table T12)
```bash
python task3/scripts/run_adaptive_tuning.py
```

### Step 9: Adaptive Evaluation on Held-Out Split (Tables T13, T14)
```bash
python task3/scripts/run_adaptive_test.py
```

### Step 10: Qualitative Cases Gallery (Table T15)
```bash
python task3/scripts/build_qualitative_cases.py
```

### Step 11: Tables & Plots Compilation (Tables T16, T17; Figures P01–P16)
```bash
python task3/scripts/make_tables.py
python task3/scripts/make_plots.py
```

### Step 12: Deliverables & Evidence Audit
```bash
python task3/scripts/audit_evidence.py
```

---

## 4. Running Unit Tests

Run the test suite using `unittest` or `pytest`:
```bash
python -m unittest discover -s task3/tests
```
All 21 tests pass, validating split disjointness, nested prefixes, tie-breaking, population covariance identity, VisConf isolation, prompt templating, response reuse, and adaptive controllers.
