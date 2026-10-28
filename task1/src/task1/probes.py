"""Activation probe sampling (32 GSM8K + 32 HellaSwag) and base model activation caching."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset

from .formatting import format_chat_prompt

logger = logging.getLogger("task1.probes")


def sample_probe_indices(seed: int = 6304, gsm8k_count: int = 32, hellaswag_count: int = 32) -> Dict[str, List[int]]:
    """Deterministically samples probe indices using seed 6304."""
    rng = np.random.default_rng(seed)
    # GSM8K test has 1,319 examples
    gsm8k_indices = sorted(rng.choice(1319, size=gsm8k_count, replace=False).tolist())
    # HellaSwag validation has 10,042 examples
    hellaswag_indices = sorted(rng.choice(10042, size=hellaswag_count, replace=False).tolist())

    return {
        "gsm8k_indices": gsm8k_indices,
        "hellaswag_indices": hellaswag_indices,
    }


def extract_probe_inputs(
    tokenizer: Any,
    seed: int = 6304,
    cache_path: Optional[Path | str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extracts and saves the fixed 64 probe prompts (32 math from GSM8K, 32 commonsense from HellaSwag).
    Returns list of probe dictionaries with prompt text, tokens, and metadata.
    """
    if cache_path and Path(cache_path).exists():
        logger.info(f"Loading cached probe manifest from {cache_path}...")
        with open(cache_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        return manifest["probes"], manifest

    indices = sample_probe_indices(seed=seed)

    logger.info("Loading GSM8K test split for probe extraction...")
    gsm8k_ds = load_dataset("gsm8k", "main", split="test")

    logger.info("Loading HellaSwag validation split for probe extraction...")
    hellaswag_ds = load_dataset("hellaswag", split="validation")

    probes: List[Dict[str, Any]] = []

    # 1. 32 Math Probes from GSM8K (using Qwen chat template with enable_thinking=False)
    for idx in indices["gsm8k_indices"]:
        row = gsm8k_ds[int(idx)]
        question = row.get("question", "")
        messages = format_chat_prompt(question, enable_thinking=False)
        try:
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, enable_thinking=False
            )
        except TypeError:
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            )

        probes.append({
            "probe_id": f"gsm8k_{idx}",
            "group": "math",
            "source_split": "gsm8k_test",
            "dataset_index": int(idx),
            "raw_text": question,
            "input_ids": list(input_ids),
            "seq_len": len(input_ids),
            "prompt_hash": hashlib.sha256(question.encode("utf-8")).hexdigest(),
        })

    # 2. 32 Non-Math Probes from HellaSwag (stock completion prompt without chat template)
    for idx in indices["hellaswag_indices"]:
        row = hellaswag_ds[int(idx)]
        context = row.get("ctx", "")
        # Completion prompt: tokenize directly without chat template
        input_ids = tokenizer.encode(context, add_special_tokens=True)

        probes.append({
            "probe_id": f"hellaswag_{idx}",
            "group": "non_math",
            "source_split": "hellaswag_val",
            "dataset_index": int(idx),
            "raw_text": context,
            "input_ids": list(input_ids),
            "seq_len": len(input_ids),
            "prompt_hash": hashlib.sha256(context.encode("utf-8")).hexdigest(),
        })

    manifest = {
        "seed": seed,
        "math_count": len(indices["gsm8k_indices"]),
        "non_math_count": len(indices["hellaswag_indices"]),
        "total_probes": len(probes),
        "probes": probes,
    }

    if cache_path:
        p = Path(cache_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Saved probe inputs manifest to {p}.")

    return probes, manifest


def cache_base_model_activations(
    base_model: nn.Module,
    probes: List[Dict[str, Any]],
    target_layers: List[int] = [0, 13, 27],
    output_dir: Optional[Path | str] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Hooks down_proj in transformer layers 0, 13, 27 on the starting model.
    Caches input activation z at the last prompt token in FP32.

    Returns:
        activation_cache: Dict mapping f"{probe_id}_layer{layer_idx}" -> tensor of shape (d_in,) in FP32
    """
    device = device or next(base_model.parameters()).device
    base_model.eval()

    # Locate down_proj modules
    captured_inputs: Dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(layer_idx: int):
        def hook_fn(module, input_tuple, output):
            # input_tuple[0] is of shape (batch, seq_len, d_in)
            inp = input_tuple[0].detach()
            # Capture last non-padding token activation (seq_len - 1)
            captured_inputs[layer_idx] = inp[:, -1, :].to(torch.float32).cpu()
        return hook_fn

    for layer_idx in target_layers:
        down_proj = base_model.model.layers[layer_idx].mlp.down_proj
        h = down_proj.register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    activation_cache: Dict[str, torch.Tensor] = {}
    meta_records = []

    try:
        with torch.no_grad():
            for p in probes:
                probe_id = p["probe_id"]
                ids = torch.tensor([p["input_ids"]], dtype=torch.long, device=device)
                captured_inputs.clear()
                _ = base_model(input_ids=ids)

                for layer_idx in target_layers:
                    act_z = captured_inputs[layer_idx].squeeze(0)  # (d_in,) in FP32
                    key = f"{probe_id}_layer{layer_idx}"
                    activation_cache[key] = act_z
                    meta_records.append({
                        "key": key,
                        "probe_id": probe_id,
                        "group": p["group"],
                        "layer": layer_idx,
                        "z_norm": float(torch.norm(act_z, p=2).item()),
                        "shape": list(act_z.shape),
                    })
    finally:
        for h in hooks:
            h.remove()

    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        torch.save(activation_cache, str(out_p / "base_activations_z.pt"))
        with open(out_p / "activation_manifest.json", "w", encoding="utf-8") as f:
            json.dump({"total_activations": len(activation_cache), "records": meta_records}, f, indent=2)
        logger.info(f"Saved {len(activation_cache)} cached activation vectors to {out_p}.")

    return activation_cache

