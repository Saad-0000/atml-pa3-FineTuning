# Future Task Readiness & Asset Preservation Manifest

This document records the asset inventory, provenance, and readiness for future tasks in **EE-5102/CS-6304: Advanced Topics in Machine Learning (Fall 2026), Programming Assignment 3**.

As mandated by the assignment specification and Revision 5 prompt guardrails:
1. **`task2/` and `task3/` are strictly empty placeholders** (containing only `.gitkeep`). No code, configs, READMEs, generated results, or data may be placed inside them during the Task 1 implementation.
2. Shared persistent assets and input manifests live outside `task2/` and `task3/` under `shared_assets/` and `manifests/`.
3. Source assets must never be deleted, overwritten, or stored solely in ephemeral scratch locations.

---

## 1. Assignment Specification

- **Document**: `ATML PA3.pdf`
- **SHA-256**: `66766f7aafb6046f617d990399dad687ab79b37b8d963cb60bdb4cb0d8871f62`
- **Size**: 505,742 bytes
- **Fixed Assignment Seed**: `6304`

---

## 2. Model Provenance & Cross-Task Boundaries

- **Task 1 Starting Checkpoint**: `Qwen/Qwen3-0.6B` (post-trained checkpoint).
  - All five Task 1 fine-tuning variants (Full FT, Full FT + L2-SP, LoRA-4, LoRA-16, OPLoRA-16) start from this untouched reference.
  - The immutable base weights $\theta_0$ must be preserved on persistent storage for L2-SP regularization and weight-space diagnostics ($||\theta - \theta_0||_2$).
- **Task 2 Starting Checkpoints**:
  - Language Backbone: `Qwen/Qwen3-0.6B`.
  - Vision Encoder: `google/siglip-base-patch16-224` (always frozen).
  - **Critical Boundary**: Task 2 starts from the **original untouched** `Qwen/Qwen3-0.6B` checkpoint, **not** from any Task 1 fine-tuned checkpoint.
- **Task 3 Models**:
  - Offline Rollouts: `Qwen2.5-3B-Instruct` (LLM, MATH subset) and `Qwen2.5-VL-3B-Instruct` (VLM, MathVista testmini).
  - Live Generation: `Qwen/Qwen2.5-3B-Instruct` (revision pinned in the supplied `README.MD`).
  - **Critical Boundary**: Task 3 uses `Qwen2.5-3B-Instruct`, **not** `Qwen3-0.6B`.

---

## 3. Input Inventory & Status

| Task | Asset Name | Target Persistent Path | Current Status | Description / Requirements |
|---|---|---|---|---|
| **Task 1** | MetaMathQA Dataset | `shared_assets/inputs/task1_metamathqa/` | Available on HF Hub | First 40k eligible, <=2048 tokens, GSM8K leak-filtered |
| **Task 1** | GSM8K Test Split | `shared_assets/inputs/gsm8k/` | Available on HF Hub | 1,319 test questions for leak filter and evaluation |
| **Task 1** | `first_plot_questions.yaml` | `shared_assets/inputs/first_plot_questions.yaml` | Missing | 24 qualitative prompts (8 questions x 3 formats) |
| **Task 2** | COCO 2014 Images & Annotations | `shared_assets/inputs/task2_coco/` | Missing | train2014 & val2014 splits, captions, instances |
| **Task 2** | VQAv2 Complementary Pairs | `shared_assets/inputs/task2_vqav2/` | Missing | Consensus questions and paired images |
| **Task 3** | `ATML-PA3-Data` Archive | `shared_assets/inputs/task3_rollouts/` | Missing | Parquet rollouts for LLM and VLM |

---

## 4. Preserved Requirements for Future Implementation

### Task 2 (When Implemented)
- Extract 196 patch tokens ($14 \times 14$) from `google/siglip-base-patch16-224`, applying final layer normalization exactly once.
- Cache patch features in FP16 to persistent storage and release the vision encoder from GPU memory before training Qwen.
- Connector: Linear $768 \to 1024$, GELU, Linear $1024 \to 1024$.
- 3 sequential training stages:
  1. Caption Alignment (2,250 updates, connector only).
  2. Visual Instruction Tuning (1,500 updates, connector + LoRA-16 on `q_proj, k_proj, v_proj, o_proj`).
  3. Natural VQA Specialization (750 updates, connector + LoRA-16).
- Image-disjoint splits between train2014 and val2014; complementary pair controls (correct, swap, mismatched, no-image).

### Task 3 (When Implemented)
- Lexicographically sort 1,000 question IDs per dataset and permute with `np.random.default_rng(6304)`: first 200 dev, remaining 800 held-out.
- First 50 MATH evaluation questions for live refinement comparison.
- Answer scoring: `math-verify` for MATH and MathVista precision rules.
- Quality scoring: Self-certainty ($K_i$), Log-probability ($L_i$), Visual engagement ($V_i$), and VisConf ($Z_i$).
- RWC consensus weighting and adaptive sampling stopping policies.
