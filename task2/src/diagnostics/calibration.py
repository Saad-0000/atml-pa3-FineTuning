"""Magnitude calibration for connector output scale using training data only.

Assignment Section 3.3:
Calibrate connector output magnitude against Qwen text-embedding magnitude using
training data only. Measure mean L2 norm of Qwen token embeddings across training
captions/questions, and compute the calibration scale factor so that connector output
tokens have matching initial expected norm.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch

from task2.src.models.connector import MLPConnector, calibrate_connector_scale
from task2.src.models.vlm import VLMModel

logger = logging.getLogger(__name__)


def run_connector_magnitude_calibration(
    vlm_model: VLMModel,
    train_caption_texts: List[str],
    sample_visual_features: torch.Tensor,
    output_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Measures Qwen text embedding norm vs connector output norm on training data.

    Calculates:
    - Text embedding mean L2 norm over tokens
    - Raw connector output mean L2 norm over patch tokens
    - Scale factor = (text_mean_norm / raw_connector_mean_norm)
    - Updates connector.scale buffer
    - Saves verification metrics
    """
    logger.info("Running connector magnitude calibration on training data...")
    device = vlm_model.device
    vlm_model.eval()

    # 1. Measure text embedding norm
    text_norms = []
    with torch.no_grad():
        for text in train_caption_texts[:500]:
            token_ids = vlm_model.tokenizer(text, return_tensors="pt").input_ids.to(device)
            embeds = vlm_model.text_embed_tokens(token_ids)  # [1, seq_len, 1024]
            norms = torch.norm(embeds.squeeze(0), dim=-1)  # [seq_len]
            text_norms.append(norms.mean().item())

    mean_text_norm = float(torch.tensor(text_norms).mean().item())

    # 2. Measure raw connector output norm (with scale=1.0)
    feat_sample = sample_visual_features.to(device)
    if feat_sample.ndim == 2:
        feat_sample = feat_sample.unsqueeze(0)  # [1, 196, 768]

    # Temporarily set scale to 1.0 to measure raw output
    orig_scale = vlm_model.connector.scale.item()
    vlm_model.connector.scale.fill_(1.0)

    with torch.no_grad():
        raw_connector_out = vlm_model.connector(feat_sample)
        raw_norms = torch.norm(raw_connector_out.squeeze(0), dim=-1)
        mean_raw_connector_norm = float(raw_norms.mean().item())

    # 3. Compute scale factor
    scale_factor = mean_text_norm / max(mean_raw_connector_norm, 1e-6)
    vlm_model.connector.set_scale(scale_factor)

    # 4. Verify calibrated output norm
    with torch.no_grad():
        calibrated_out = vlm_model.connector(feat_sample)
        calibrated_norm = float(torch.norm(calibrated_out.squeeze(0), dim=-1).mean().item())

    calibration_stats = {
        "mean_text_embedding_l2_norm": round(mean_text_norm, 4),
        "mean_raw_connector_l2_norm": round(mean_raw_connector_norm, 4),
        "calculated_scale_factor": round(scale_factor, 6),
        "verified_calibrated_norm": round(calibrated_norm, 4),
        "norm_ratio_calibrated_to_text": round(calibrated_norm / mean_text_norm, 4),
        "status": "PASS" if abs(calibrated_norm - mean_text_norm) < 0.1 else "WARN",
    }

    logger.info(
        f"Calibration completed: Text Norm={mean_text_norm:.3f}, "
        f"Raw Conn Norm={mean_raw_connector_norm:.3f} -> Scale={scale_factor:.4f} "
        f"(Calibrated Norm={calibrated_norm:.3f})"
    )

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(calibration_stats, f, indent=2)

    return calibration_stats
