"""Hybrid reward for Omni-Math-Diversity CISPO training."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

FINAL_MARKER_RE = re.compile(
    "(?is)(?:final\\s*answer|final|answer|\\u53c2\\u8003\\u7b54\\u6848)\\s*[:\\uff1a]\\s*(.+)$"
)
ANSWER_TAG_RE = re.compile(
    "(?is)<\\s*(?:answer|\\u7b54\\u6848)\\s*>(.*?)<\\s*/\\s*(?:answer|\\u7b54\\u6848)\\s*>"
)

GENRM_PROMPT_TEMPLATE = """
You are a strict but fair answer judge for math, science, humanities, and multilingual exam problems.

Problem:
{problem}

Reference answer:
{ground_truth}

Model final answer:
{prediction}

Decide whether the model final answer is semantically equivalent to the reference answer for the given problem.
Ignore harmless formatting differences, wording differences, and equivalent mathematical notation.
Do not give partial credit.

Return only one JSON object with exactly these keys:
{{"correct": true or false}}
""".strip()

REWARD_RESULT_DEFAULTS: dict[str, Any] = {
    "score": 0.0,
    "acc": False,
    "pred": "",
    "judge_type": "",
    "judge_error": "",
    "math_verify_score": -1.0,
    "genrm_response": "",
}


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


def extract_final_answer(solution_str: str) -> str:
    tagged = _extract_answer_tag_content(solution_str)
    if tagged:
        return tagged.strip("`$ ")

    boxed = _extract_boxed_content(solution_str)
    if boxed:
        return boxed

    marker_matches = list(FINAL_MARKER_RE.finditer(solution_str))
    if marker_matches:
        candidate = marker_matches[-1].group(1).strip()
        candidate = candidate.splitlines()[0].strip() if candidate else candidate
        return candidate.strip("`$ ")

    nonempty_lines = [line.strip() for line in solution_str.splitlines() if line.strip()]
    if nonempty_lines:
        return nonempty_lines[-1].strip("`$ ")
    return solution_str.strip("`$ ")


def _math_verify_model_output(solution_str: str, prediction: str) -> str:
    if "\\boxed" in solution_str or "\\fbox" in solution_str:
        return solution_str
    if prediction:
        return f"\\boxed{{{prediction}}}"
    return solution_str


def _normalize_answer_text(value: str) -> str:
    tagged = _extract_answer_tag_content(value)
    if tagged:
        value = tagged
    boxed = _extract_boxed_content(value)
    if boxed:
        value = boxed
    value = re.sub(
        "(?is)^(?:final\\s*answer|final|answer|\\u53c2\\u8003\\u7b54\\u6848)\\s*[:\\uff1a]\\s*",
        "",
        value.strip(),
    )
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


async def _score_math_verify(solution_str: str, ground_truth: str, timeout: float) -> dict[str, Any]:
    prediction = extract_final_answer(solution_str)

    try:
        worker_script = os.path.join(os.path.dirname(__file__), "math_verify_worker.py")
        payload = json.dumps(
            {
                "solution_str": solution_str,
                "ground_truth": ground_truth,
                "prediction": prediction,
            }
        ).encode("utf-8")
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            worker_script,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(payload), timeout=timeout)
        if proc.returncode != 0:
            err_text = stderr.decode("utf-8", errors="replace").strip() or "math_verify worker exited non-zero"
            raise RuntimeError(err_text)
        result = json.loads(stdout.decode("utf-8"))
        raw_score = float(result["score"])
        if raw_score not in (0.0, 1.0):
            raw_score = 1.0 if raw_score >= 0.5 else 0.0
    except asyncio.TimeoutError:
        raise
    except Exception as exc:
        logger.warning("math_verify failed: %s", exc)
        return {
            "score": 0.0,
            "acc": False,
            "judge_type": "math_verify",
            "judge_error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if "proc" in locals() and proc.returncode is None:
            proc.kill()
            await proc.wait()

    correct = raw_score >= 0.5
    return {
        "score": 1.0 if correct else 0.0,
        "acc": correct,
        "judge_type": "math_verify",
        "math_verify_score": raw_score,
    }


async def _post_chat_completion(
    router_address: str,
    payload: dict[str, Any],
    timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    url = f"http://{router_address}/v1/chat/completions"
    last_exception: Exception | None = None
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    return await response.json()
        except Exception as exc:
            last_exception = exc
            if attempt + 1 < max_retries:
                await asyncio.sleep(min(2**attempt, 8))
    assert last_exception is not None
    raise last_exception


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    decoder = json.JSONDecoder()
    for idx, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(stripped[idx:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise ValueError(f"GenRM did not return a JSON object: {text[:200]!r}")


def _parse_correct(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1", "correct"}
    return False


def _extract_correct_fallback(text: str) -> bool | None:
    match = re.search(r'"correct"\s*:\s*(true|false)', text, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower() == "true"
    stripped = text.strip().lower()
    if stripped in {"true", "false"}:
        return stripped == "true"
    return None


def _compact_text(value: Any, max_chars: int = 512) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _stable_reward_result(result: dict[str, Any]) -> dict[str, Any]:
    stable = dict(REWARD_RESULT_DEFAULTS)
    stable.update(result)
    stable["pred"] = _compact_text(stable["pred"])
    stable["judge_error"] = _compact_text(stable["judge_error"])
    stable["genrm_response"] = _compact_text(stable["genrm_response"])
    return stable


async def _score_semantic(
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any],
    reward_router_address: str | None,
    reward_model_tokenizer: Any,
    model_name: str | None,
    max_tokens: int,
    judge_temperature: float,
    request_timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    prediction = _compact_text(extract_final_answer(solution_str), max_chars=512)
    if not reward_router_address:
        return {
            "score": 0.0,
            "acc": False,
            "judge_type": "semantic_judge",
            "judge_error": "reward_router_address is not available",
        }

    problem = extra_info.get("problem") or extra_info.get("question") or ""
    prompt = GENRM_PROMPT_TEMPLATE.format(
        problem=problem,
        ground_truth=ground_truth,
        prediction=prediction,
    )
    resolved_model_name = (
        model_name
        or os.getenv("GENRM_MODEL_NAME")
        or getattr(reward_model_tokenizer, "name_or_path", None)
        or "reward-model"
    )
    payload = {
        "model": resolved_model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": judge_temperature,
        "top_p": 1.0,
        "max_tokens": max_tokens,
    }

    try:
        response = await _post_chat_completion(
            router_address=reward_router_address,
            payload=payload,
            timeout=request_timeout,
            max_retries=max_retries,
        )
        judge_response = response["choices"][0]["message"]["content"]
        try:
            judge_json = _extract_json_object(judge_response)
            correct = _parse_correct(judge_json.get("correct"))
        except Exception:
            fallback_correct = _extract_correct_fallback(judge_response)
            if fallback_correct is None:
                raise
            correct = fallback_correct
        return {
            "score": 1.0 if correct else 0.0,
            "acc": correct,
            "judge_type": "semantic_judge",
            "genrm_response": judge_response,
        }
    except Exception as exc:
        logger.warning("semantic judge failed: %s", exc)
        return {
            "score": 0.0,
            "acc": False,
            "judge_type": "semantic_judge",
            "judge_error": f"{type(exc).__name__}: {exc}",
        }


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    reward_router_address: str | None = None,
    reward_model_tokenizer: Any = None,
    model_name: str | None = None,
    max_tokens: int = 48,
    judge_temperature: float = 0.0,
    request_timeout: float = 120.0,
    max_retries: int = 3,
    math_verify_timeout: float = 30.0,
) -> dict[str, Any]:
    """Score one rollout response for Omni-Math-Diversity."""
    del data_source
    extra_info = extra_info or {}
    answer_kind = extra_info.get("answer_kind", "semantic_judge")

    if answer_kind == "math_verify":
        return _stable_reward_result(await _score_math_verify(solution_str, ground_truth, timeout=math_verify_timeout))

    return _stable_reward_result(await _score_semantic(
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        reward_router_address=reward_router_address,
        reward_model_tokenizer=reward_model_tokenizer,
        model_name=model_name,
        max_tokens=max_tokens,
        judge_temperature=judge_temperature,
        request_timeout=request_timeout,
        max_retries=max_retries,
    ))
