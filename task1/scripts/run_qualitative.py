#!/usr/bin/env python3
"""
Qualitative evaluation runner:
- Loads the 24 prompts from first_plot_questions.yaml (8 questions x 3 formats: plain, json, template).
- Generates greedy responses (512-token cap, thinking disabled) for all 7 conditions (168 responses total).
- Generates the fixed 42-response annotation sheet and aligned comparison viewer.
- Enforces strict blocker reporting if first_plot_questions.yaml is absent.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
TASK1_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(TASK1_ROOT / "src"))

import pandas as pd
import torch
from task1.config import get_default_configs
from task1.model_loader import load_base_model, load_tokenizer
from task1.lora import apply_lora_to_model
from task1.oplora import apply_oplora_to_model
from task1.qualitative import build_42_annotation_sheet, generate_qualitative_responses, load_qualitative_prompts

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("run_qualitative")


def resolve_prompts_file(t1_cfg: Any, eval_cfg: dict) -> Path:
    """Locates first_plot_questions.yaml across designated locations."""
    candidates = [
        REPO_ROOT / "shared_assets" / "inputs" / "first_plot_questions.yaml",
        Path(t1_cfg.persistent_root) / "shared_assets" / "inputs" / "first_plot_questions.yaml",
        Path("/content/drive/MyDrive/ATML_PA3_Task1/shared_assets/inputs/first_plot_questions.yaml"),
        TASK1_ROOT / "configs" / "first_plot_questions.yaml",
        Path("first_plot_questions.yaml"),
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        f"Mandatory qualitative prompts file 'first_plot_questions.yaml' not found in designated locations: "
        f"{[str(c) for c in candidates]}. Per assignment rules, do not invent substitute questions."
    )


def load_model_for_condition(condition_id: str, t1_cfg: Any, methods_cfg: dict, device: torch.device):
    tokenizer = load_tokenizer(t1_cfg.model_name, revision=t1_cfg.model_revision, enable_thinking=False)
    if condition_id == "baseline":
        model = load_base_model(t1_cfg.model_name, revision=t1_cfg.model_revision, torch_dtype=torch.float16, device=device)
        return model, tokenizer

    step_map = {
        "full_ft_midpoint": ("full_ft", 625),
        "full_ft_final": ("full_ft", 1250),
        "l2sp_final": ("l2sp", 1250),
        "lora4_final": ("lora4", 1250),
        "lora16_final": ("lora16", 1250),
        "oplora16_final": ("oplora16", 1250),
    }
    method, step = step_map[condition_id]
    ckpt_path = t1_cfg.checkpoints_dir / method / f"checkpoint_step_{step:04d}" / "training_state.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint for {condition_id} missing at {ckpt_path}!")

    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model = load_base_model(t1_cfg.model_name, revision=t1_cfg.model_revision, torch_dtype=torch.float16, device=device)

    if method in ["lora4", "lora16"]:
        m_cfg = methods_cfg[method]
        model, _ = apply_lora_to_model(model, r=m_cfg["r"], alpha=m_cfg["alpha"], dropout=0.0, target_modules=m_cfg["target_modules"])
    elif method == "oplora16":
        m_cfg = methods_cfg[method]
        basis_cache_dir = t1_cfg.artifacts_dir / "oplora_bases"
        model, _, _ = apply_oplora_to_model(
            model, r=m_cfg["r"], alpha=m_cfg["alpha"], k=m_cfg["protection_rank_k"], dropout=0.0,
            target_modules=m_cfg["target_modules"], basis_cache_dir=basis_cache_dir
        )

    model.load_state_dict(state["model_state_dict"])
    model.to(device)

    # Merge adapters for generation
    if method in ["lora4", "lora16", "oplora16"]:
        for m in model.modules():
            if hasattr(m, "merge_weights"):
                m.merge_weights()

    return model, tokenizer


def generate_aligned_html_viewer(df_42: pd.DataFrame, output_html: Path) -> None:
    """Generates an aligned HTML comparison viewer for human inspection."""
    html_content = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>Qualitative 42-Response Aligned Comparison</title>",
        "<style>",
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 20px; line-height: 1.5; }",
        "h1 { color: #333; }",
        "table { border-collapse: collapse; width: 100%; margin-top: 15px; font-size: 13px; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }",
        "th { background-color: #f4f6f8; position: sticky; top: 0; }",
        "tr:nth-child(even) { background-color: #fbfbfb; }",
        ".badge-pass { background-color: #e6f4ea; color: #137333; padding: 2px 6px; border-radius: 4px; font-weight: 500; }",
        ".badge-warn { background-color: #fef7e0; color: #b06000; padding: 2px 6px; border-radius: 4px; font-weight: 500; }",
        ".badge-fail { background-color: #fce8e6; color: #c5221f; padding: 2px 6px; border-radius: 4px; font-weight: 500; }",
        ".pre-text { white-space: pre-wrap; word-break: break-word; font-family: monospace; font-size: 12px; max-height: 250px; overflow-y: auto; background: #fafafa; padding: 6px; border-radius: 4px; }",
        "</style></head><body>",
        "<h1>Qualitative Findings: Aligned 42-Response Subset</h1>",
        "<p>Aligned comparison across baseline, Full FT midpoint, and 5 final models for the first and last questions across plain, json, and template formats.</p>",
        "<table>",
        "<thead><tr><th>Condition</th><th>Question ID</th><th>Format</th><th>Objective Checks</th><th>Output Text</th></tr></thead>",
        "<tbody>",
    ]

    for _, row in df_42.iterrows():
        fmt_badge = "<span class='badge-pass'>Valid</span>" if row["format_compliant_auto"] else "<span class='badge-fail'>Invalid</span>"
        rep_badge = "<span class='badge-warn'>Repetition</span>" if row["repetition_detected_auto"] else "<span class='badge-pass'>Clean</span>"
        math_badge = "<span class='badge-warn'>Math Framing</span>" if row["math_framing_detected_auto"] else ""
        trunc_badge = "<span class='badge-warn'>Truncated</span>" if row["truncated_auto"] else ""

        html_content.append(f"<tr>")
        html_content.append(f"<td><b>{row['condition_id']}</b></td>")
        html_content.append(f"<td>{row['question_id']}</td>")
        html_content.append(f"<td>{row['format']}</td>")
        html_content.append(f"<td>{fmt_badge} {rep_badge} {math_badge} {trunc_badge}</td>")
        safe_text = str(row['output_text']).replace("<", "&lt;").replace(">", "&gt;")
        html_content.append(f"<td><div class='pre-text'>{safe_text}</div></td>")
        html_content.append(f"</tr>")

    html_content.extend(["</tbody></table></body></html>"])
    output_html.parent.mkdir(parents=True, exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html_content))


def main():
    parser = argparse.ArgumentParser(description="Generate qualitative outputs and 42-response annotation sheet")
    args = parser.parse_args()

    t1_cfg, methods_cfg, eval_cfg = get_default_configs(TASK1_ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        prompts_file = resolve_prompts_file(t1_cfg, eval_cfg)
        prompts = load_qualitative_prompts(prompts_file)
    except FileNotFoundError as e:
        logger.error(f"Qualitative evaluation blocked: {e}")
        # Record blocker in STATUS.md and manifest
        status_file = t1_cfg.results_dir / "STATUS.md"
        with open(status_file, "a", encoding="utf-8") as f:
            f.write(f"\n### Qualitative Evaluation Blocker\n- **Status**: Blocked\n- **Reason**: {e}\n")
        sys.exit(1)

    conditions = [
        "baseline",
        "full_ft_midpoint",
        "full_ft_final",
        "l2sp_final",
        "lora4_final",
        "lora16_final",
        "oplora16_final",
    ]

    all_responses = []

    for cond_id in conditions:
        logger.info(f"Generating qualitative responses for {cond_id}...")
        try:
            model, tokenizer = load_model_for_condition(cond_id, t1_cfg, methods_cfg, device)
            responses = generate_qualitative_responses(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                condition_id=cond_id,
                device=str(device),
                max_new_tokens=eval_cfg.get("qualitative", {}).get("max_new_tokens", 512),
            )
            all_responses.extend(responses)
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Could not generate qualitative responses for {cond_id}: {e}")

    qual_dir = t1_cfg.results_dir / "qualitative"
    qual_dir.mkdir(parents=True, exist_ok=True)

    if all_responses:
        # Save all responses (up to 168)
        resp_file = qual_dir / "all_168_responses.jsonl"
        with open(resp_file, "w", encoding="utf-8") as f:
            for r in all_responses:
                f.write(json.dumps(r) + "\n")
        logger.info(f"Saved {len(all_responses)} responses to {resp_file}.")

        # Build fixed 42-response annotation sheet
        annotation_csv = t1_cfg.results_dir / "tables" / "qualitative_42_annotations.csv"
        df_42 = build_42_annotation_sheet(all_responses, annotation_csv)

        # Generate HTML viewer
        viewer_html = qual_dir / "aligned_comparison.html"
        generate_aligned_html_viewer(df_42, viewer_html)
        logger.info(f"Generated HTML viewer at {viewer_html}.")


if __name__ == "__main__":
    main()
