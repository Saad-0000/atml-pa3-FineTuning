"""
Balanced-brace extractor for mathematical answers inside \\boxed{...}.
"""

import re
from typing import Optional, Tuple


def extract_last_boxed_answer(text: str) -> Optional[str]:
    """Extracts the contents of the LAST \\boxed{...} expression using balanced brace counting.

    Supports arbitrarily nested braces, e.g. \\boxed{\\frac{1}{2}}, \\boxed{\\sqrt{x + 1}}.
    Returns None if no valid boxed expression is found.
    """
    if not text or "\\boxed" not in text:
        return None

    # Find all starting indices of \boxed
    boxed_starts = [m.start() for m in re.finditer(r"\\boxed\s*\{", text)]
    if not boxed_starts:
        return None

    # Check from the last occurrence backwards
    for start_idx in reversed(boxed_starts):
        # Find the opening brace
        brace_start = text.find("{", start_idx)
        if brace_start == -1:
            continue

        depth = 0
        end_idx = -1
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break

        if end_idx != -1:
            inside = text[brace_start + 1 : end_idx].strip()
            return inside

    return None
