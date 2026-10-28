# Shared Assets Directory

This directory is part of the repository structure for cross-task asset tracking and metadata manifests.

## Storage Policy

- **Git Tracking**: Only lightweight manifests, inventory tables, and placeholder `.gitkeep` files are tracked in Git.
- **Large Inputs & Weights**: Raw datasets (MetaMathQA, COCO 2014, VQAv2, ATML-PA3-Data) and base model checkpoints (`Qwen/Qwen3-0.6B`) must reside on persistent storage (e.g., Google Drive at `/content/drive/MyDrive/ATML_PA3/shared_assets/` or local persistent disk), **never** committed to Git.

## Subdirectories

```text
shared_assets/
├── base_models/             # Reference starting checkpoints (Qwen3-0.6B)
├── inputs/
│   ├── assignment/          # ATML PA3.pdf and official assignment guides
│   ├── task1_metamathqa/    # Filtered 40k MetaMathQA dataset cache
│   ├── task2_coco/          # COCO 2014 images and annotations (Task 2)
│   ├── task2_vqav2/         # VQAv2 complementary pairs (Task 2)
│   └── task3_rollouts/      # ATML-PA3-Data parquet files (Task 3)
└── README.md
```

