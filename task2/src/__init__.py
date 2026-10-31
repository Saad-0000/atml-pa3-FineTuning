"""Task 2 package for ATML PA3: Vision-Language Models and Visual Grounding."""

import os
# Configure PyTorch memory allocator to avoid virtual memory fragmentation on Tesla T4 GPUs
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

__version__ = "0.1.0"
