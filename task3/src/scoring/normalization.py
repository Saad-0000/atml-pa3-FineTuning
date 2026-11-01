"""
Answer normalization, equivalence grouping, and correctness evaluation for MATH and MathVista.
"""

import re
from typing import Any, Dict, List, Optional
import sympy

try:
    import math_verify
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False

NO_ANSWER = "NO_ANSWER"


def clean_latex(text: Any) -> str:
    """Strips LaTeX formatting artifacts, dollar signs, and whitespace."""
    if text is None:
        return ""
    s = str(text).strip()
    s = s.replace("$", "").replace("\\%", "%")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\left", "", s)
    s = re.sub(r"\\right", "", s)
    s = s.strip()
    # Normalize common fraction \frac{a}{b} -> (a)/(b)
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
    return s.strip()


def normalize_math_answer(answer: Any) -> str:
    """Normalizes a MATH answer string for equivalence comparison."""
    if answer is None or not str(answer).strip():
        return NO_ANSWER

    raw = clean_latex(answer)
    if not raw or raw.lower() in ["none", "null", "no answer", "n/a"]:
        return NO_ANSWER

    # Try fast numeric conversion
    try:
        val = float(raw)
        if val.is_integer():
            return str(int(val))
        return f"{val:.4f}".rstrip("0").rstrip(".")
    except ValueError:
        pass

    # Simple alphanumeric / canonical cleanup
    s = raw.lower().replace(" ", "")
    s = s.rstrip(".,;")
    return s if s else NO_ANSWER


def normalize_mathvista_answer(answer: Any, meta: Optional[Dict[str, Any]] = None) -> str:
    """Normalizes a MathVista answer string according to question type (option letter, number, yes/no)."""
    if answer is None or not str(answer).strip():
        return NO_ANSWER

    raw = clean_latex(answer).strip()
    if not raw:
        return NO_ANSWER

    lower = raw.lower()
    q_type = (meta or {}).get("question_type", "")

    # Yes / No
    if lower in ["yes", "true", "(a) yes", "a"]:
        if q_type == "yes/no":
            return "yes"
    if lower in ["no", "false", "(b) no", "b"]:
        if q_type == "yes/no":
            return "no"

    # Single option letter e.g. "(B) 42 units" -> "B" or "Option C" -> "C"
    mc_match = re.search(r"^\(?([a-eA-E])\)?\.?($|\s)", raw) or re.search(r"\bOption\s+([a-eA-E])\b", raw, re.IGNORECASE)
    if mc_match:
        return mc_match.group(1).upper()

    # Number
    try:
        val = float(raw)
        if val.is_integer():
            return str(int(val))
        return f"{val:.4f}".rstrip("0").rstrip(".")
    except ValueError:
        pass

    return raw.lower()


def normalize_answer(raw_ans: Any, dataset_name: str, meta: Optional[Dict[str, Any]] = None) -> str:
    """Unified entry point for answer normalization."""
    if dataset_name.upper() == "MATH":
        return normalize_math_answer(raw_ans)
    else:
        return normalize_mathvista_answer(raw_ans, meta=meta)


def are_answers_equivalent(ans_a: Any, ans_b: Any, dataset: str = "math") -> bool:
    """Checks whether two generated answers are equivalent to form consensus answer groups.

    CRITICAL RULE: This function NEVER inspects gold answers!
    """
    if ans_a == NO_ANSWER or ans_b == NO_ANSWER:
        return ans_a == ans_b == NO_ANSWER

    str_a = str(ans_a).strip()
    str_b = str(ans_b).strip()
    if str_a == str_b:
        return True

    if dataset.lower() == "math":
        norm_a = normalize_math_answer(ans_a)
        norm_b = normalize_math_answer(ans_b)
        if norm_a == norm_b:
            return True
        # Fast numeric check before expensive sympy
        try:
            if abs(float(norm_a) - float(norm_b)) < 1e-6:
                return True
        except Exception:
            pass
        return norm_a == norm_b
    else:
        norm_a = normalize_mathvista_answer(ans_a)
        norm_b = normalize_mathvista_answer(ans_b)
        if norm_a == norm_b:
            return True
        try:
            val_a = float(norm_a)
            val_b = float(norm_b)
            if abs(val_a - val_b) < 1e-2 or abs(val_a - val_b) / max(abs(val_b), 1e-6) < 0.05:
                return True
        except Exception:
            pass
        return norm_a == norm_b


def grade_correctness(
    predicted_ans: Any,
    gold_ans: Any,
    dataset_name: str = "math",
    meta: Optional[Dict[str, Any]] = None,
    dataset: Optional[str] = None,
) -> bool:
    """Grades a predicted answer against the gold answer.

    Retrospective grading ONLY: never called inside candidate selection or voting.
    """
    ds = dataset if dataset is not None else dataset_name
    if isinstance(ds, dict):
        ds = "math"
    if not isinstance(ds, str):
        ds = str(ds)

    if predicted_ans is None or predicted_ans == NO_ANSWER:
        return False
    if gold_ans is None or gold_ans == NO_ANSWER:
        return False
    return are_answers_equivalent(predicted_ans, gold_ans, dataset=ds)


grade_answer = grade_correctness
