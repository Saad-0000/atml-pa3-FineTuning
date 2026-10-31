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
    Supports all YAML layouts:
    - Format-first (keys: plain, json, template -> lists or dicts of questions)
    - Question-first (list or dict of questions, each with plain/json/template or formats: {...})
    - Flat list of 24 prompts
    If the file is absent, raises FileNotFoundError.
    """
    p = Path(prompts_path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Mandatory qualitative prompts file '{p}' not found! "
            f"Please supply 'first_plot_questions.yaml' at the designated path. "
            f"Per assignment instructions, do not invent substitute questions."
        )

    file_size = p.stat().st_size
    if file_size == 0:
        logger.error(
            f"Qualitative prompts file {p} is EMPTY (0 bytes)! "
            f"Please copy the authentic first_plot_questions.yaml from course materials into this location."
        )
        return []

    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        logger.error(
            f"Qualitative prompts file {p} parsed as None (empty or whitespace only). "
            f"Please check file contents."
        )
        return []

    logger.info(f"Inspecting {p} (size={file_size} bytes, type={type(data).__name__})")

    # Unwrap single top-level container if present (e.g., {"questions": ...} or {"first_plot_questions": ...})
    if isinstance(data, dict):
        for wrapper_key in ["questions", "prompts", "first_plot_questions", "items", "data", "benchmark"]:
            if wrapper_key in data and isinstance(data[wrapper_key], (dict, list)):
                data = data[wrapper_key]
                break

    prompts: List[Dict[str, Any]] = []

    # Format normalizer helper
    def norm_fmt(k: str) -> str:
        k_clean = str(k).strip().lower()
        if "json" in k_clean:
            return "json"
        elif "template" in k_clean:
            return "template"
        elif "plain" in k_clean or "raw" in k_clean or "text" in k_clean:
            return "plain"
        return k_clean

    # Layout 1: Format-first mapping (e.g. {plain: [...], json: [...], template: [...]})
    if isinstance(data, dict):
        keys_lower = {norm_fmt(k): k for k in data.keys()}
        if any(f in keys_lower for f in ["plain", "json", "template"]):
            for fmt in ["plain", "json", "template"]:
                if fmt in keys_lower:
                    orig_key = keys_lower[fmt]
                    fmt_items = data[orig_key]
                    if isinstance(fmt_items, list):
                        for idx, p_text in enumerate(fmt_items):
                            text = p_text if isinstance(p_text, str) else (p_text.get("prompt") or p_text.get("text", ""))
                            prompts.append({
                                "question_id": f"q{idx+1}",
                                "format": fmt,
                                "prompt": str(text),
                            })
                    elif isinstance(fmt_items, dict):
                        for q_id, p_text in fmt_items.items():
                            text = p_text if isinstance(p_text, str) else (p_text.get("prompt") or p_text.get("text", ""))
                            prompts.append({
                                "question_id": str(q_id),
                                "format": fmt,
                                "prompt": str(text),
                            })

    # Layout 2: Dict of questions keyed by ID (e.g. {q1: {plain: ..., json: ...}, ...})
    if not prompts and isinstance(data, dict):
        for q_id, q_data in data.items():
            if isinstance(q_data, dict):
                sub_fmts = q_data.get("formats", q_data)
                for k, v in sub_fmts.items():
                    fmt = norm_fmt(k)
                    if fmt in ["plain", "json", "template"]:
                        text = v if isinstance(v, str) else (v.get("prompt") or v.get("text", ""))
                        prompts.append({
                            "question_id": str(q_id),
                            "format": fmt,
                            "prompt": str(text),
                        })

    # Layout 3: List of questions or prompts
    if not prompts and isinstance(data, list):
        for idx, item in enumerate(data):
            if isinstance(item, str):
                prompts.append({
                    "question_id": f"q{idx+1}",
                    "format": "plain",
                    "prompt": item,
                })
            elif isinstance(item, dict):
                q_id = str(item.get("id", item.get("question_id", f"q{idx+1}")))

                # Sub-case 3a: Item contains 'paraphrases' list (standard benchmark layout)
                if "paraphrases" in item and isinstance(item["paraphrases"], list):
                    paras = item["paraphrases"]
                    assigned = [None] * len(paras)
                    used_fmts = set()

                    # Keyword-based format classification
                    for p_i, p_val in enumerate(paras):
                        p_str = str(p_val).lower()
                        if "json" in p_str and "json" not in used_fmts:
                            assigned[p_i] = "json"
                            used_fmts.add("json")
                        elif any(tok in p_str for tok in ["template", "format:", "<format>", "bracket"]) and "template" not in used_fmts:
                            assigned[p_i] = "template"
                            used_fmts.add("template")

                    fallback_fmts = ["plain", "json", "template"]
                    for p_i, p_val in enumerate(paras):
                        if assigned[p_i] is None:
                            for f_cand in fallback_fmts:
                                if f_cand not in used_fmts:
                                    assigned[p_i] = f_cand
                                    used_fmts.add(f_cand)
                                    break
                            if assigned[p_i] is None:
                                assigned[p_i] = fallback_fmts[min(p_i, len(fallback_fmts) - 1)]

                        prompts.append({
                            "question_id": q_id,
                            "format": assigned[p_i],
                            "prompt": str(p_val),
                        })

                # Sub-case 3b: Flat prompt item
                elif "format" in item and ("prompt" in item or "text" in item):
                    prompts.append({
                        "question_id": q_id,
                        "format": norm_fmt(item["format"]),
                        "prompt": str(item.get("prompt", item.get("text", ""))),
                    })
                else:
                    # Sub-case 3c: Item with format fields or 'formats' subdict
                    sub_fmts = item.get("formats", item)
                    for k, v in sub_fmts.items():
                        fmt = norm_fmt(k)
                        if fmt in ["plain", "json", "template"]:
                            text = v if isinstance(v, str) else (v.get("prompt") or v.get("text", ""))
                            prompts.append({
                                "question_id": q_id,
                                "format": fmt,
                                "prompt": str(text),
                            })

    if not prompts:
        logger.warning(
            f"Could not parse any qualitative prompts from {p}! "
            f"File size: {file_size} bytes. Raw type: {type(data).__name__}. "
            f"Raw content snippet: {str(data)[:200]}"
        )
    else:
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

