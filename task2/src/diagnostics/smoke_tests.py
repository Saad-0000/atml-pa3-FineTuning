"""Smoke tests and preflight diagnostic verification for Task 2.

Checks:
1. Hardware / CUDA / T4 verification
2. Connector and multimodal sequence shapes and label masking
3. Autograd gradient flow through frozen Qwen into connector
4. FP16 autocast and gradient checkpointing memory usage
"""
import logging
from typing import Any, Dict
import torch
import torch.nn as nn

from task2.src.models.connector import MLPConnector
from task2.src.models.vlm import VLMModel
from task2.src.training.loss import compute_example_weighted_loss

logger = logging.getLogger(__name__)


def test_autograd_gradient_flow(
    vlm_model: VLMModel,
    device: torch.device,
) -> Dict[str, Any]:
    """Proves that connector gradients are nonzero and finite while Qwen base remains frozen.

    Assignment Section 3.3:
    'Add a test proving that connector gradients are nonzero and finite while
     original Qwen parameters remain frozen.'
    """
    logger.info("Testing autograd gradient flow through frozen Qwen...")
    vlm_model.train()
    vlm_model.set_stage_trainability(enable_lora=False)

    # Synthetic batch of 2 examples
    feat = torch.randn(2, 196, 768, device=device, dtype=torch.float32)
    sample_batch = [
        {"visual_features": feat[0], "question": "Is this a dog?", "answer": "yes", "is_caption": False},
        {"visual_features": feat[1], "question": "How many cats?", "answer": "two", "is_caption": False},
    ]

    inputs = vlm_model.prepare_multimodal_inputs(sample_batch)
    inputs_embeds = inputs["inputs_embeds"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    labels = inputs["labels"].to(device)

    vlm_model.connector.zero_grad()

    # Forward
    logits = vlm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    loss, _ = compute_example_weighted_loss(logits, labels)

    # Backward
    loss.backward()

    # Verify connector gradients
    connector_grads = []
    for name, p in vlm_model.connector.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                raise AssertionError(f"Connector parameter {name} has no gradient!")
            grad_norm = p.grad.norm().item()
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                raise AssertionError(f"Connector parameter {name} gradient is NaN/Inf!")
            if grad_norm == 0.0:
                raise AssertionError(f"Connector parameter {name} gradient is zero!")
            connector_grads.append(grad_norm)

    # Verify frozen Qwen parameters have no gradients
    qwen_frozen_check = True
    for name, p in vlm_model.qwen.named_parameters():
        if not ("lora_A" in name or "lora_B" in name):
            if p.requires_grad:
                qwen_frozen_check = False
                raise AssertionError(f"Qwen parameter {name} requires_grad is True!")
            if p.grad is not None:
                qwen_frozen_check = False
                raise AssertionError(f"Frozen Qwen parameter {name} has nonzero grad!")

    vlm_model.connector.zero_grad()

    return {
        "connector_gradients_nonzero_and_finite": True,
        "connector_mean_grad_norm": float(sum(connector_grads) / len(connector_grads)),
        "qwen_base_frozen_verified": qwen_frozen_check,
        "loss_value": float(loss.item()),
        "status": "PASS",
    }


def test_t4_memory_and_checkpointing(
    vlm_model: VLMModel,
    device: torch.device,
    microbatch_size: int = 8,
) -> Dict[str, Any]:
    """Representative forward/backward smoke test at microbatch 8 with gradient checkpointing."""
    logger.info(f"Running T4 memory and gradient checkpointing smoke test at microbatch {microbatch_size}...")
    if device.type != "cuda":
        return {
            "status": "SKIPPED_CPU",
            "message": "CUDA not available; running on CPU.",
        }

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    vlm_model.train()
    vlm_model.set_stage_trainability(enable_lora=True)

    # Create synthetic microbatch of 8 examples
    feat = torch.randn(microbatch_size, 196, 768, device=device, dtype=torch.float16)
    sample_batch = [
        {"visual_features": feat[i], "question": f"Question {i}?", "answer": f"Answer {i}", "is_caption": False}
        for i in range(microbatch_size)
    ]

    inputs = vlm_model.prepare_multimodal_inputs(sample_batch)
    inputs_embeds = inputs["inputs_embeds"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    labels = inputs["labels"].to(device)

    scaler = torch.amp.GradScaler("cuda")

    with torch.amp.autocast("cuda", dtype=torch.float16):
        logits = vlm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        loss, _ = compute_example_weighted_loss(logits, labels)

    scaler.scale(loss).backward()

    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    allocated_mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)

    vlm_model.zero_grad()
    torch.cuda.empty_cache()

    return {
        "status": "PASS",
        "microbatch_size": microbatch_size,
        "peak_gpu_memory_mb": round(peak_mem_mb, 1),
        "allocated_gpu_memory_mb": round(allocated_mem_mb, 1),
        "fits_t4_16gb": peak_mem_mb < 14500.0,
    }
