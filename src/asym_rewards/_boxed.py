"""Dependency-free \\boxed{...} extraction.

Single source of truth for boxed-answer parsing, importable WITHOUT pulling in
math_verify/torch. Both reward.py (which re-exports it) and judge.py import from
here, so the parser is defined once (no duplicate to drift) while judge.py stays
importable on a host that lacks math_verify.
"""
from __future__ import annotations


def extract_boxed_answer(text: str) -> str:
    """Return the inner text of the LAST \\boxed{...} in `text`, or empty string.

    Brace-depth aware so \\boxed{\\frac{1}{2}} returns "\\frac{1}{2}".
    """
    if not text:
        return ""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""
    i = idx + len("\\boxed{")
    depth = 1
    start = i
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    if depth != 0:
        return ""
    return text[start:i].strip()
