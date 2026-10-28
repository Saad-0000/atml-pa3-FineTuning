# Google Colab Tesla T4 Execution Runbook

This runbook provides exact, step-by-step instructions for running **Task 1: Large Language Model Fine-Tuning** on **Google Colab** with a **Tesla T4 GPU (16 GiB VRAM)** and **Google Drive persistent storage**.

---

## 1. Prerequisites & Environment Setup

### Colab Runtime Configuration
1. Open Google Colab.
2. Select **Runtime** > **Change runtime type**.
3. Choose **T4 GPU** as the hardware accelerator.
4. Set **Python 3** as the runtime.

### Mount Google Drive
All canonical checkpoints, logs, cached datasets, and generated evidence must reside on Google Drive so they persist across session disconnects:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Configure Persistent Directories
Create your project root on Google Drive (default: `/content/drive/MyDrive/ATML_PA3_Task1`):

```bash
mkdir -p /content/drive/MyDrive/ATML_PA3_Task1/checkpoints
mkdir -p /content/drive/MyDrive/ATML_PA3_Task1/artifacts
mkdir -p /content/drive/MyDrive/ATML_PA3_Task1/results
mkdir -p /content/drive/MyDrive/ATML_PA3_Task1/logs
```

---

## 2. Clone Repository & Setup Symlinks

Clone your GitHub repository into Colab's local disk `/content`:

```bash
cd /content
git clone <YOUR_GITHUB_REPO_URL> atml-pa3-FineTuning
cd atml-pa3-FineTuning/task1

# Link runtime directories to Google Drive for automatic persistence
rm -rf checkpoints artifacts results logs
ln -s /content/drive/MyDrive/ATML_PA3_Task1/checkpoints checkpoints
ln -s /content/drive/MyDrive/ATML_PA3_Task1/artifacts artifacts
ln -s /content/drive/MyDrive/ATML_PA3_Task1/results results
ln -s /content/drive/MyDrive/ATML_PA3_Task1/logs logs
```

### Install Pinned Dependencies
```bash
pip install -r requirements-lock.txt
export PYTHONPATH="/content/atml-pa3-FineTuning/task1/src:$PYTHONPATH"
```

---

## 3. Preflight & Verification

Run the preflight check to verify GPU VRAM, PyTorch mixed precision, and persistent storage:

```bash
python scripts/preflight.py --persistent-root /content/drive/MyDrive/ATML_PA3_Task1
```

Confirm that:
- The GPU is verified as a **Tesla T4 (16 GiB)**.
- The GPU memory smoke test passes.
- Persistent Drive storage has at least 15 GiB free.

---

## 4. Data Preparation & Baseline Evaluation

```bash
# 1. Filter MetaMathQA (first 40,000 eligible examples) & sample 64 probes
python scripts/prepare_data.py

# 2. Evaluate unchanged starting model, save theta0 reference, and cache SVD bases
python scripts/evaluate_baseline.py
```

---

## 5. Sequential Training of Five Independent Conditions

Train the five variants sequentially. Do not run concurrent training jobs.

```bash
# Method 1: Full Fine-Tuning (Peak LR 2e-5)
python scripts/train_method.py --method full_ft --resume auto

# Method 2: Full Fine-Tuning + L2-SP (Peak LR 2e-5, lambda 1e-3)
python scripts/train_method.py --method l2sp --resume auto

# Method 3: LoRA Rank 4 (r=4, alpha=8, Peak LR 2e-4)
python scripts/train_method.py --method lora4 --resume auto

# Method 4: LoRA Rank 16 (r=16, alpha=32, Peak LR 2e-4)
python scripts/train_method.py --method lora16 --resume auto

# Method 5: OPLoRA Rank 16, Protection Rank 128 (r=16, alpha=32, k=128, Peak LR 2e-4)
python scripts/train_method.py --method oplora16 --resume auto
```

### Handling Colab Disconnects
- Checkpoints are saved automatically at official milestones (312, 625, 937, 1250) and emergency checkpoints are saved every 50 updates.
- If Colab disconnects, simply re-run the `train_method.py` command with `--resume auto`. It will discover the newest verified complete checkpoint from Google Drive and resume without loss of completed updates.
- Maximum possible work lost on disconnect: **at most 50 optimizer updates**.

---

## 6. Evaluation, Diagnostics, and Results Generation

```bash
# 1. Evaluate required model checkpoints across 4 benchmarks
python scripts/evaluate_required.py --condition all

# 2. Compute weight-space (rho_k, eta) and activation diagnostics (R(z))
python scripts/run_diagnostics.py

# 3. Generate qualitative responses & 42-annotation sheet
# (Requires first_plot_questions.yaml in shared_assets/inputs/ or task1/configs/)
python scripts/run_qualitative.py

# 4. Build all authoritative summary tables and publication plots
python scripts/build_results.py

# 5. Run full evidence audit
python scripts/audit_evidence.py
```

