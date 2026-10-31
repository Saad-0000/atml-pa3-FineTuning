"""Text and answer normalization routines for VQA and grounding evaluation."""

import re
import string
from typing import List


# Standard digit word to numeral mapping for VQA
DIGIT_MAP = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

CONTRACTIONS = {
    "aint": "aint",
    "ain't": "aint",
    "arent": "arent",
    "aren't": "arent",
    "cant": "cant",
    "can't": "cant",
    "couldve": "couldve",
    "could've": "couldve",
    "couldnt": "couldnt",
    "couldn't": "couldnt",
    "didnt": "didnt",
    "didn't": "didnt",
    "doesnt": "doesnt",
    "doesn't": "doesnt",
    "dont": "dont",
    "don't": "dont",
    "hadnt": "hadnt",
    "hadn't": "hadnt",
    "hasnt": "hasnt",
    "hasn't": "hasnt",
    "havent": "havent",
    "haven't": "havent",
    "isnt": "isnt",
    "isn't": "isnt",
    "it's": "its",
    "its": "its",
    "mightnt": "mightnt",
    "mightn't": "mightnt",
    "neednt": "neednt",
    "needn't": "neednt",
    "shant": "shant",
    "shan't": "shant",
    "shouldnt": "shouldnt",
    "shouldn't": "shouldnt",
    "wasnt": "wasnt",
    "wasn't": "wasnt",
    "werent": "werent",
    "weren't": "werent",
    "wont": "wont",
    "won't": "wont",
    "wouldnt": "wouldnt",
    "wouldn't": "wouldnt",
    "that's": "thats",
    "what's": "whats",
}

ARTICLES = {"a", "an", "the"}
PUNCTUATION = set(string.punctuation)


def normalize_vqa_answer(text: str) -> str:
    """
    Standard VQA evaluation answer normalization:
    1. Lowercase text.
    2. Convert contractions.
    3. Remove punctuation.
    4. Remove articles ('a', 'an', 'the').
    5. Convert number words ('one', 'two', etc.) to digits.
    6. Collapse whitespace.
    """
    if not text:
        return ""

    text = str(text).lower().strip()

    # Normalize contractions and apostrophes (e.g. don't -> dont, it's -> its)
    text = text.replace("'", "")
    words = text.split()
    words = [CONTRACTIONS.get(w, w) for w in words]
    text = " ".join(words)

    # Remove punctuation
    # Replace comma or period if between digits with dot or clean
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    cleaned_chars = [c if c not in PUNCTUATION else " " for c in text]
    text = "".join(cleaned_chars)

    # Process tokens
    tokens = text.split()
    processed_tokens: List[str] = []
    for tok in tokens:
        if tok in ARTICLES:
            continue
        tok = DIGIT_MAP.get(tok, tok)
        processed_tokens.append(tok)

    return " ".join(processed_tokens).strip()


def normalize_grounding_answer(text: str) -> str:
    """
    Normalizes presence ('yes'/'no') and counting ('1', '2', '3', '4') answers.
    """
    norm = normalize_vqa_answer(text)
    # Check for direct presence / counting canonical forms
    if norm in ["yes", "yeah", "yep", "true", "correct"]:
        return "yes"
    if norm in ["no", "nope", "false", "incorrect"]:
        return "no"
    if norm in DIGIT_MAP:
        return DIGIT_MAP[norm]
    # Extract first digit if string begins with or contains a number in 1-4
    m = re.search(r"\b([0-9])\b", norm)
    if m:
        return m.group(1)
    return norm


normalize_answer = normalize_vqa_answer
