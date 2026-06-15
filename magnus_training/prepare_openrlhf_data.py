from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SYSTEM_PROMPT = "你是一位严谨的物理和数学解题助手。请逐步推理，并给出清晰答案。"


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".parquet":
        import pandas as pd

        rows = pd.read_parquet(path).to_dict(orient="records")
        return [row for row in rows if isinstance(row, dict)]

    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        return [row for row in data if isinstance(row, dict)]
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def normalize_messages(row: dict[str, Any]) -> list[dict[str, str]] | None:
    messages = row.get("messages")
    if isinstance(messages, list):
        cleaned = []
        for item in messages:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = str(item.get("content", "")).strip()
            if role in {"system", "user"} and content:
                cleaned.append({"role": role, "content": content})
        if cleaned:
            return cleaned

    prompt = row.get("prompt")
    if isinstance(prompt, list):
        cleaned = []
        for item in prompt:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = str(item.get("content", "")).strip()
            if role in {"system", "user"} and content:
                cleaned.append({"role": role, "content": content})
        if cleaned:
            return cleaned

    text = first_text(row, ("instruction", "question", "query", "prompt", "input", "problem"))
    if not text:
        return None
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def normalize_answer(row: dict[str, Any]) -> str:
    direct = first_text(row, ("answer", "final_answer", "label", "target"))
    solution = first_text(row, ("solution", "output", "response", "chosen"))
    if direct and solution and direct not in solution:
        return f"{direct}\n{solution}".strip()
    return direct or solution or ""


def convert(input_path: Path, output_path: Path, max_samples: int | None = None) -> int:
    rows = read_rows(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            messages = normalize_messages(row)
            if not messages:
                continue
            record = {
                "prompt": messages,
                "answer": normalize_answer(row),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if max_samples is not None and count >= max_samples:
                break
    if count == 0:
        raise ValueError(f"No usable prompt rows found in {input_path}")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert local RL data to OpenRLHF prompt JSONL")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    count = convert(Path(args.input), Path(args.output), args.max_samples)
    print(f"[prepare_openrlhf_data] wrote {count} rows -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
