#!/usr/bin/env python3
"""Subprocess worker for math_verify scoring."""

from __future__ import annotations

import json
import re
import sys


FINAL_MARKER_RE = re.compile(r"(?is)(?:final\s*answer|final|answer|\u53c2\u8003\u7b54\u6848)\s*[:\uff1a]\s*(.+)$")
ANSWER_TAG_RE = re.compile(r"(?is)<\s*(?:answer|\u7b54\u6848)\s*>(.*?)<\s*/\s*(?:answer|\u7b54\u6848)\s*>")


def _extract_boxed_content(text: str) -> str | None:
    marker_positions = [match.start() for match in re.finditer(r"\\(?:boxed|fbox)\s*\{", text)]
    for start in reversed(marker_positions):
        brace_start = text.find("{", start)
        depth = 0
        for idx in range(brace_start, len(text)):
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
                if depth == 0:
                    return text[brace_start + 1 : idx].strip()
    return None


def _extract_answer_tag_content(text: str) -> str | None:
    matches = list(ANSWER_TAG_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def _normalize_answer_text(value: str) -> str:
    tagged = _extract_answer_tag_content(value)
    if tagged:
        value = tagged
    boxed = _extract_boxed_content(value)
    if boxed:
        value = boxed
    value = re.sub(r"(?is)^(?:final\s*answer|final|answer|\u53c2\u8003\u7b54\u6848)\s*[:\uff1a]\s*", "", value.strip())
    replacements = {
        "\\left": "",
        "\\right": "",
        "\\,": "",
        "\\!": "",
        "$": "",
        ",": "",
        " ": "",
        "\n": "",
        "\t": "",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value.strip().rstrip(".").lower()


def _fallback_math_match(prediction: str, ground_truth: str) -> bool:
    pred_norm = _normalize_answer_text(prediction)
    gold_norm = _normalize_answer_text(ground_truth)
    if pred_norm and pred_norm == gold_norm:
        return True

    try:
        return abs(float(pred_norm) - float(gold_norm)) < 1e-9
    except Exception:
        pass

    try:
        import sympy as sp

        def parse_expr(value: str):
            value = value.replace("^", "**")
            value = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", value)
            value = value.replace("\\pi", "pi")
            return sp.sympify(value)

        return bool(sp.simplify(parse_expr(pred_norm) - parse_expr(gold_norm)) == 0)
    except Exception:
        return False


def _math_verify_model_output(solution_str: str, prediction: str) -> str:
    if "\\boxed" in solution_str or "\\fbox" in solution_str:
        return solution_str
    if prediction:
        return f"\\boxed{{{prediction}}}"
    return solution_str


def _run_math_verify(solution_str: str, ground_truth: str, prediction: str) -> float:
    from math_verify.grader import verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse

    ground_truth_boxed = f"\\boxed{{{ground_truth}}}"
    model_output = _math_verify_model_output(solution_str, prediction)
    extracted_gold = parse(ground_truth_boxed, (LatexExtractionConfig(),), parsing_timeout=None)
    extracted_pred = parse(
        model_output,
        (ExprExtractionConfig(), LatexExtractionConfig()),
        parsing_timeout=None,
    )
    if extracted_gold and extracted_pred and any(verify(gold, pred) for gold in extracted_gold for pred in extracted_pred):
        return 1.0
    return 1.0 if _fallback_math_match(prediction, ground_truth) else 0.0


def main() -> int:
    payload = json.loads(sys.stdin.read())
    score = _run_math_verify(
        solution_str=payload["solution_str"],
        ground_truth=payload["ground_truth"],
        prediction=payload["prediction"],
    )
    sys.stdout.write(json.dumps({"score": score}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
