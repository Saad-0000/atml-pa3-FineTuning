"""Qualitative prompt generation (168 total responses) and aligned 42-response annotation sheet."""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import yaml

from .formatting import format_chat_prompt

logger = logging.getLogger("task1.qualitative")


def load_qualitative_prompts(prompts_path: Path | str) -> List[Dict[str, Any]]:
    """
    Loads the 24 qualitative prompts from first_plot_questions.yaml.
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

    # Normalize structure into list of 24 prompts
    # Expected: 8 underlying questions, each with plain, json, template formats
    prompts: List[Dict[str, Any]] = []

    if isinstance(data, list):
        prompts = data
    elif isinstance(data, dict):
        questions = data.get("questions", data.get("prompts", []))
        if isinstance(questions, list):
            prompts = questions
        elif isinstance(questions, dict):
            for q_id, q_data in questions.items():
                if isinstance(q_data, dict):
                    for fmt in ["plain", "json", "template"]:
                        if fmt in q_data:
                            prompts.append({
                                "question_id": q_id,
                                "format": fmt,
                                "prompt": q_data[fmt],
                            })

    if len(prompts) != 24:
        logger.warning(f"Loaded {len(prompts)} prompts from {p} (expected exactly 24: 8 questions x 3 formats).")

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
    responses: List[Dict[str, Any]] = []

    for item in prompts:
        q_id = item.get("question_id", "q")
        fmt = item.get("format", "plain")
        prompt_text = item.get("prompt", "")

        messages = format_chat_prompt(prompt_text, enable_thinking=False)
        try:
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", enable_thinking=False
            ).to(device)
        except TypeError:
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
            ).to(device)

        with torch.no_grad():
            output_tokens = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )

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
