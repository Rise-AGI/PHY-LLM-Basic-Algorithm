#!/usr/bin/env python3
"""
phy_lint.py — Phy-LLM 数据集 Schema 验证工具
评测组（丁辰一）从零开发需求；验证数据是否符合 Schema v10 要求。

用法：
    python phy_lint.py data.jsonl
    python phy_lint.py data.jsonl --strict          # warning 也算错误
    python phy_lint.py data.jsonl --json            # 输出机器可读 JSON
    python phy_lint.py data.jsonl --fix             # 自动修复可修复的问题

退出码：0=通过，1=失败
"""

import argparse
import hashlib
import json
import re
import sys
import uuid
from pathlib import Path


# ── Schema 枚举 ───────────────────────────────────────────────────────

VALID_SPLIT      = {"train", "val", "test"}
VALID_DIFFICULTY = {"high_school", "undergraduate", "graduate"}
VALID_CATEGORY   = {
    "mechanics", "em", "thermo", "quantum",
    "optics", "relativity", "nuclear", "statistical",
}
VALID_ROLE       = {"system", "user", "assistant"}
VALID_REJECTION  = {"skip_step", "wrong_answer", "format_error"}

VERSION_RE = re.compile(r"^v\d+\.\d+\.\d+$")
UUID_RE    = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)


# ── 单条记录验证 ──────────────────────────────────────────────────────

def validate_record(rec: dict, idx: int, strict: bool) -> list[dict]:
    issues = []

    def err(msg):
        issues.append({"idx": idx, "level": "error", "field": msg.split()[0], "msg": msg})

    def warn(msg):
        level = "error" if strict else "warning"
        issues.append({"idx": idx, "level": level, "field": msg.split()[0], "msg": msg})

    # ── P0：必填字段 ─────────────────────────────────────────────────

    # id
    id_val = rec.get("id")
    if not id_val:
        err("id: 缺少字段")
    elif not UUID_RE.match(str(id_val)):
        err(f"id: 不是合法 UUID: {id_val}")

    # version（Schema gap #5）
    ver = rec.get("version")
    if not ver:
        err("version: 缺少字段（Schema gap #5：数据集 version 字段缺失）")
    elif not VERSION_RE.match(str(ver)):
        err(f"version: 格式错误，应为 vX.Y.Z，实际: {ver}")

    # split
    split = rec.get("split")
    if not split:
        err("split: 缺少字段")
    elif split not in VALID_SPLIT:
        err(f"split: 非法值 '{split}'，合法: {sorted(VALID_SPLIT)}")

    # difficulty
    diff = rec.get("difficulty")
    if not diff:
        err("difficulty: 缺少字段")
    elif diff not in VALID_DIFFICULTY:
        err(f"difficulty: 非法值 '{diff}'，合法: {sorted(VALID_DIFFICULTY)}")

    # category
    cat = rec.get("category")
    if not cat:
        err("category: 缺少字段")
    elif cat not in VALID_CATEGORY:
        err(f"category: 非法值 '{cat}'，合法: {sorted(VALID_CATEGORY)}")

    # source
    if not rec.get("source"):
        err("source: 缺少字段或为空")

    # messages（Schema gap #1 🔴）
    messages = rec.get("messages")
    if messages is None:
        err("messages: 缺少字段（Schema gap #1 🔴 最高优先级缺口）")
    elif not isinstance(messages, list):
        err("messages: 必须是数组")
    elif len(messages) < 2:
        err(f"messages: 至少需要 2 条消息，实际 {len(messages)} 条")
    else:
        roles = [m.get("role") for m in messages]
        if "user" not in roles:
            err("messages: 缺少 role=user 的消息")
        if "assistant" not in roles:
            err("messages: 缺少 role=assistant 的消息")
        for i, msg in enumerate(messages):
            r = msg.get("role")
            if not r:
                err(f"messages[{i}]: 缺少 role 字段")
            elif r not in VALID_ROLE:
                err(f"messages[{i}]: role 非法值 '{r}'")
            c = msg.get("content")
            if c is None:
                err(f"messages[{i}]: 缺少 content 字段")
            elif not str(c).strip():
                err(f"messages[{i}]: content 为空")
            # 检查 assistant 是否有 <think> 标签（非强制，仅 warning）
            if r == "assistant" and "<think>" not in str(c):
                warn(f"messages[{i}]: assistant 回答缺少 <think> 推理链（推荐添加）")

    # ── P1：推荐字段（缺失为 warning）────────────────────────────────

    # content_hash（Schema gap #3）
    if "content_hash" not in rec:
        warn("content_hash: 缺少字段（Schema gap #3：内容哈希缺失）")

    # tool_calls（Schema gap #2 🔴）
    # 只在数据中有工具调用时检查
    if any("tool_call" in str(m.get("content", "")) for m in rec.get("messages", [])
           if isinstance(m, dict)):
        if "tool_calls" not in rec:
            err("tool_calls: 数据中含工具调用但缺少 tool_calls 字段（Schema gap #2 🔴）")

    # rejection_reason（Schema gap #4）
    rr = rec.get("rejection_reason")
    if rr is not None and rr not in VALID_REJECTION:
        err(f"rejection_reason: 非法值 '{rr}'，合法: {sorted(VALID_REJECTION)}")

    return issues


# ── 自动修复 ──────────────────────────────────────────────────────────

def auto_fix(rec: dict) -> dict:
    """修复可自动化处理的问题，返回修复后的记录。"""
    import time

    # 补 id
    if not rec.get("id"):
        rec["id"] = str(uuid.uuid4())

    # 补 version
    if not rec.get("version"):
        rec["version"] = "v1.0.0"

    # 补 content_hash
    if "content_hash" not in rec and "messages" in rec:
        content = json.dumps(rec["messages"], ensure_ascii=False, sort_keys=True)
        rec["content_hash"] = "sha256:" + hashlib.sha256(content.encode()).hexdigest()[:16]

    return rec


# ── 文件级 lint ───────────────────────────────────────────────────────

def lint_file(path: str, strict: bool = False, fix: bool = False) -> dict:
    p = Path(path)
    if not p.exists():
        return {"passed": False, "file": path, "fatal": "文件不存在"}

    try:
        raw = p.read_text(encoding="utf-8").strip()
        if path.endswith(".parquet"):
            import pandas as pd
            records = pd.read_parquet(path).to_dict(orient="records")
        elif raw.startswith("["):
            records = json.loads(raw)
        else:
            records = [json.loads(ln) for ln in raw.splitlines() if ln.strip()]
    except Exception as e:
        return {"passed": False, "file": path, "fatal": f"文件解析失败: {e}"}

    if not records:
        return {"passed": False, "file": path, "fatal": "数据集为空"}

    all_errors, all_warnings = [], []
    fixed_records = []

    for i, rec in enumerate(records):
        if fix:
            rec = auto_fix(rec)
        issues = validate_record(rec, i, strict)
        for issue in issues:
            if issue["level"] == "error":
                all_errors.append(issue)
            else:
                all_warnings.append(issue)
        fixed_records.append(rec)

    # 如果修复模式且无错误，写回文件
    if fix and not all_errors:
        out_path = path.replace(".jsonl", "_fixed.jsonl").replace(".json", "_fixed.json")
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in fixed_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fixed_to = out_path
    else:
        fixed_to = None

    passed = len(all_errors) == 0
    return {
        "passed":          passed,
        "file":            path,
        "total_records":   len(records),
        "error_count":     len(all_errors),
        "warning_count":   len(all_warnings),
        "error_details":   all_errors[:30],
        "warning_details": all_warnings[:10],
        "fixed_to":        fixed_to,
    }


# ── 主程序 ────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Phy-LLM Schema v10 数据验证工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python phy_lint.py sft_train.jsonl
  python phy_lint.py sft_train.jsonl --strict --json
  python phy_lint.py sft_train.jsonl --fix
        """,
    )
    p.add_argument("file",     help="数据文件路径（.jsonl / .json / .parquet）")
    p.add_argument("--strict", action="store_true", help="warning 升级为 error")
    p.add_argument("--json",   action="store_true", help="输出 JSON 格式报告")
    p.add_argument("--fix",    action="store_true", help="自动修复 id/version/content_hash")
    args = p.parse_args()

    result = lint_file(args.file, strict=args.strict, fix=args.fix)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        status = "✅ PASSED" if result["passed"] else "❌ FAILED"
        print(f"\nphy_lint.py — {status}")
        print(f"文件: {result['file']}")
        if "fatal" in result:
            print(f"致命错误: {result['fatal']}")
        else:
            print(f"记录数: {result['total_records']} 条 | "
                  f"错误: {result['error_count']} | "
                  f"警告: {result['warning_count']}")
            if result["error_details"]:
                print("\n── 错误详情 ──")
                for e in result["error_details"]:
                    print(f"  [ERROR] 第 {e['idx']+1} 条：{e['msg']}")
            if result["warning_details"]:
                print("\n── 警告详情 ──")
                for w in result["warning_details"]:
                    print(f"  [WARN]  第 {w['idx']+1} 条：{w['msg']}")
            if result.get("fixed_to"):
                print(f"\n✅ 已修复并写入：{result['fixed_to']}")
            elif not result["passed"]:
                print("\n提示：运行 --fix 可自动修复 id/version/content_hash 缺失问题")

    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
