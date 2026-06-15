from __future__ import annotations

import math
import re
from typing import Any

import torch


NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    return str(value)


def _extract_answer(label: Any) -> str:
    text = _to_text(label).strip()
    if not text:
        return ""
    return text.splitlines()[0].strip()


def _numbers(text: str) -> set[str]:
    out = set()
    for match in NUMBER_RE.findall(text):
        try:
            value = float(match)
        except ValueError:
            out.add(match)
            continue
        if math.isfinite(value):
            out.add(f"{value:.8g}")
        out.add(match)
    return out


def score_response(response: str, prompt: str = "", label: Any = None) -> float:
    score = 0.0
    response = response or ""
    answer = _extract_answer(label)

    if len(response.strip()) >= 32:
        score += 0.10
    if any(marker in response for marker in ("解答", "推导", "步骤", "因此", "答案", "Answer")):
        score += 0.18
    if re.search(r"\$.*?\$|\\frac|\\sqrt|\\int|\\sum|=", response):
        score += 0.18
    if NUMBER_RE.search(response):
        score += 0.18
    if re.search(r"\b(m/s|m/s\^2|kg|N|J|Pa|K|eV|Hz|W|C|V|T|mol|rad|cm|mm|km|g|s)\b", response):
        score += 0.10

    if answer:
        if answer in response:
            score += 0.30
        elif _numbers(answer) and (_numbers(answer) & _numbers(response)):
            score += 0.24

    return max(0.0, min(1.0, score))


def reward_func(queries=None, prompts=None, labels=None, **kwargs):
    """OpenRLHF local reward function.

    The signature is intentionally permissive because OpenRLHF has changed
    argument names across releases. It returns a dict with a float tensor.
    """
    candidates = queries or kwargs.get("responses") or kwargs.get("texts") or []
    prompt_values = prompts or kwargs.get("prompt") or []
    label_values = labels or kwargs.get("label") or kwargs.get("answers") or []

    rewards = []
    for idx, item in enumerate(candidates):
        response = _to_text(item)
        prompt = _to_text(prompt_values[idx]) if idx < len(prompt_values) else ""
        label = label_values[idx] if idx < len(label_values) else None
        rewards.append(score_response(response, prompt, label))

    if len(rewards) == 1:
        value = float(rewards[0])
        return {"rewards": value, "scores": value}
    values = torch.tensor(rewards, dtype=torch.float32)
    return {"rewards": values, "scores": values}
