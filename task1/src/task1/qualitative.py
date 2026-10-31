"""Qualitative prompt generation (168 total responses) and aligned 42-response annotation sheet."""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import torch
import yaml

from .formatting import format_chat_prompt

logger = logging.getLogger("task1.qualitative")


def load_qualitative_prompts(prompts_path: Path | str) -> List[Dict[str, Any]]:
    """
    Loads the 24 qualitative prompts from first_plot_questions.yaml.
    Supports both flat lists, nested questions lists, and questions dictionaries.
    If the file is absent, raises FileNotFoundError (hard blocker).
    """
    p = Path(prompts_path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Mandatory qualitative prompts file '{p}' not found! "
            f"Please supply 'first_plot_questions.yaml' at the designated path. "
            f"Per assignment instructions, do not invent substitute questions."
        )

    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    raw_items = []
    if isinstance(data, list):
        raw_items = data
    elif isinstance(data, dict):
        if "questions" in data:
            raw_items = data["questions"]
        elif "prompts" in data:
            raw_items = data["prompts"]
        else:
            raw_items = data

    prompts: List[Dict[str, Any]] = []

    # Case 1: Dict of questions keyed by ID
    if isinstance(raw_items, dict):
        for q_id, q_data in raw_items.items():
            if isinstance(q_data, dict):
                for fmt in ["plain", "json", "template"]:
                    if fmt in q_data:
                        prompts.append({
                            "question_id": str(q_id),
                            "format": fmt,
                            "prompt": q_data[fmt],
                        })
    # Case 2: List of questions
    elif isinstance(raw_items, list):
        for idx, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue
            # If already flattened (has both 'format' and ('prompt' or 'text'))
            if "format" in item and ("prompt" in item or "text" in item):
                prompts.append({
                    "question_id": str(item.get("question_id", item.get("id", f"q{idx}"))),
                    "format": item["format"],
                    "prompt": item.get("prompt", item.get("text", "")),
                })
            else:
                # It's an unflattened question with formats as keys
                q_id = str(item.get("question_id", item.get("id", f"q{idx+1}")))
                for fmt in ["plain", "json", "template"]:
                    if fmt in item:
                        prompts.append({
                            "question_id": q_id,
                            "format": fmt,
                            "prompt": item[fmt],
                        })

    logger.info(f"Loaded {len(prompts)} qualitative prompts from {p}.")
    return prompts


def evaluate_objective_checks(text: str, format_type: str, max_tokens: int = 512) -> Dict[str, Any]:
    """
    Performs deterministic checks on generated response:
    - JSON parseability
    - Template markers presence
    - Repetition check (repeated 4-grams)
    - Unnecessary mathematical framing (math symbols / equations in non-math answers)
    - Truncation flag
    """
    is_json_valid = False
    if format_type.lower() == "json":
        try:
            # Try parsing json from text or fenced block
            clean = text.strip()
            if "```json" in clean:
                clean = clean.split("```json")[1].split("```")[0].strip()
            elif "```" in clean:
                clean = clean.split("```")[1].split("```")[0].strip()
            json.loads(clean)
            is_json_valid = True
        except Exception:
            is_json_valid = False

    template_compliant = False
    if format_type.lower() == "template":
        # Look for typical template markers like [Title], [Summary], colons, etc.
        template_compliant = bool(re.search(r"(\[[A-Za-z\s]+\]|[A-Za-z\s]+:)", text))

    # Repetition: count unique 4-grams vs total 4-grams
    words = text.split()
    repetition_detected = False
    if len(words) >= 12:
        four_grams = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
        unique_four_grams = set(four_grams)
        rep_ratio = len(unique_four_grams) / len(four_grams)
        repetition_detected = rep_ratio < 0.65

    # Unnecessary math framing: excessive LaTeX, boxed formulas, or calculation steps
    math_patterns = [r"\\boxed", r"\\frac", r"\$+", r"\\sum", r"equation\b", r"calculate\b", r"let\s+[a-z]\s*="]
    math_framing = any(re.search(pat, text, re.IGNORECASE) for pat in math_patterns)

    # Truncation: ends without terminating punctuation
    truncated = False
    clean_stripped = text.strip()
    if clean_stripped and clean_stripped[-1] not in ".!?\"'”)}]>":
        truncated = True

    return {
        "format_compliant_auto": is_json_valid if format_type.lower() == "json" else (template_compliant if format_type.lower() == "template" else True),
        "repetition_detected_auto": repetition_detected,
        "math_framing_detected_auto": math_framing,
        "truncated_auto": truncated,
    }


def generate_qualitative_responses(
    model: Any,
    tokenizer: Any,
    prompts: List[Dict[str, Any]],
    condition_id: str,
    device: str = "cuda",
    max_new_tokens: int = 512,
) -> List[Dict[str, Any]]:
    """
    Generates 24 greedy responses for a model condition with thinking disabled.
    """
    model.eval()
    orig_use_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = True  # Enable KV cache during generation for speed

    responses: List[Dict[str, Any]] = []

    for item in prompts:
        q_id = str(item.get("question_id", item.get("id", "q")))
        fmt = item.get("format", "plain")
        prompt_text = item.get("prompt") or item.get(fmt) or item.get("text") or ""

        messages = format_chat_prompt(str(prompt_text), enable_thinking=False)
        try:
            prompt_encoded = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", enable_thinking=False
            )
        except Exception:
            prompt_encoded = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
            )

        if isinstance(prompt_encoded, dict) or hasattr(prompt_encoded, "input_ids"):
            input_ids = prompt_encoded["input_ids"].to(device)
            attention_mask = prompt_encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
        else:
            input_ids = prompt_encoded.to(device)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            attention_mask = torch.ones_like(input_ids, device=device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        }
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        with torch.no_grad():
            output_tokens = model.generate(input_ids=input_ids, **gen_kwargs)

        gen_tokens = output_tokens[0][input_ids.shape[-1] :]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        obj_checks = evaluate_objective_checks(gen_text, format_type=fmt, max_tokens=max_new_tokens)

        record = {
            "condition_id": condition_id,
            "question_id": q_id,
            "format": fmt,
            "prompt": prompt_text,
            "output_text": gen_text,
            "output_token_count": len(gen_tokens),
            "stop_reason": "length" if len(gen_tokens) >= max_new_tokens else "eos",
            **obj_checks,
        }
        responses.append(record)

    return responses


def build_42_annotation_sheet(
    all_responses: List[Dict[str, Any]],
    output_csv: Path | str,
) -> pd.DataFrame:
    """
    Filters and formats the fixed 42-response subset:
    - First and last underlying questions in plain-question order
    - All 3 formats (plain, json, template)
    - All 7 conditions
    - 2 questions x 3 formats x 7 conditions = 42 rows.
    """
    df = pd.DataFrame(all_responses)

    # Discover question order from plain format
    plain_df = df[df["format"] == "plain"]
    unique_questions = plain_df["question_id"].unique().tolist()
    if len(unique_questions) >= 2:
        first_q = unique_questions[0]
        last_q = unique_questions[-1]
        target_questions = [first_q, last_q]
    else:
        target_questions = unique_questions

    subset_df = df[df["question_id"].isin(target_questions)].copy()

    # Add required human-review columns clearly marked
    subset_df["format_compliance_human"] = "REQUIRES_HUMAN_REVIEW"
    subset_df["coherence_relevance_human"] = "REQUIRES_HUMAN_REVIEW"
    subset_df["inappropriate_content_human"] = "REQUIRES_HUMAN_REVIEW"
    subset_df["short_excerpt"] = subset_df["output_text"].apply(lambda t: t[:150].replace("\n", " ") + "...")

    p = Path(output_csv)
    p.parent.mkdir(parents=True, exist_ok=True)
    subset_df.to_csv(p, index=False)
    logger.info(f"Saved aligned 42-response annotation sheet to {p} (total rows={len(subset_df)}).")

    return subset_df

