"""
查询 Magnus 持久存储中已下载的模型，检查完整性及微调支持能力。

SFT  = 全参数微调 (Supervised Fine-Tuning)
LoRA = 低秩适配器微调
MoE  = Mixture of Experts

用法:
    python inspect_storage_model.py
    python inspect_storage_model.py --model-path /data/xxx/models/Qwen2.5-72B-Instruct
    python inspect_storage_model.py --address http://xxx:3011/ --token sk-xxx

结果写入 train/json/model_magnus.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import magnus
from config import notify_exe, SYSTEM_ENTRY_COMMAND, MAGNUS_ADDRESS, MAGNUS_TOKEN

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(HERE, "json", "model_magnus.json")

INSPECT_PY = r'''
import json, os, glob as _glob, hashlib, traceback

MODEL_DIRS = []   # 由 entry_command 注入
# 不限制，设为空列表则自动扫描 /data/*/models/

# ── 模型微调能力判定规则 ──

# MoE 架构特征键
MOE_CONFIG_KEYS = [
    "num_experts", "num_local_experts", "n_routed_experts",
    "moe_intermediate_size", "n_shared_experts",
]
MOE_ARCH_PATTERNS = ["Mixtral", "DeepSeek", "MoE", "Qwen2Moe", "dense_moe"]
MOE_MODEL_TYPE_PATTERNS = ["mixtral", "moe", "deepseek"]


def _is_moe(config: dict) -> tuple[bool, str]:
    """返回 (is_moe, reason)"""
    # 1) 直接在 config 顶层出现 num_experts 等
    for key in MOE_CONFIG_KEYS:
        if key in config:
            val = config[key]
            if isinstance(val, (int, float)) and val > 1:
                return True, f"config.{key}={val}"
            if isinstance(val, list) and len(val) > 1:
                return True, f"config.{key}=list(len={len(val)})"

    # 2) architectures 中包含 MoE 模式
    archs = config.get("architectures", [])
    for arch in archs:
        for pat in MOE_ARCH_PATTERNS:
            if pat.lower() in arch.lower():
                return True, f"architecture={arch}"

    # 3) model_type
    mt = config.get("model_type", "").lower()
    for pat in MOE_MODEL_TYPE_PATTERNS:
        if pat in mt:
            return True, f"model_type={config['model_type']}"

    return False, ""


def _read_config(model_dir: str) -> dict | None:
    """读取 config.json"""
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return {"_parse_error": str(e)}


def _is_safetensors_corrupt(path: str) -> str | None:
    """检查单个 safetensors 文件是否损坏。返回 None 表示完好，否则返回错误描述。"""
    try:
        import struct
        fsize = os.path.getsize(path)
        if fsize == 0:
            return "空文件 (size=0)"
        with open(path, "rb") as f:
            header_len_bytes = f.read(8)
            if len(header_len_bytes) < 8:
                return "文件头不足 8 字节"
            header_len = struct.unpack("<Q", header_len_bytes)[0]
            if header_len <= 0 or header_len > fsize - 8:
                return f"header_len={header_len} 异常 (fsize={fsize})"
            header_json = f.read(header_len)
            if len(header_json) != header_len:
                return f"header 读取不完整 ({len(header_json)}/{header_len})"
            try:
                json.loads(header_json.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                return f"header JSON 损坏: {e}"
    except Exception as e:
        return f"读取失败: {e}"
    return None


def _list_safetensors(model_dir: str) -> list[dict]:
    """列出所有 safetensors 文件及其状态。"""
    files = []
    pattern = os.path.join(model_dir, "*.safetensors")
    for fpath in sorted(_glob.glob(pattern)):
        fname = os.path.basename(fpath)
        fsize = os.path.getsize(fpath)
        corrupt_reason = _is_safetensors_corrupt(fpath)
        files.append({
            "name": fname,
            "size": fsize,
            "size_mb": round(fsize / 1024 / 1024, 2),
            "ok": corrupt_reason is None,
            "corrupt_reason": corrupt_reason,
        })
    return files


def _list_bins(model_dir: str) -> list[dict]:
    """列出 pytorch_model.bin 等文件。"""
    files = []
    for fname in sorted(os.listdir(model_dir)):
        if fname.endswith(".bin") and fname.startswith("pytorch_model"):
            fpath = os.path.join(model_dir, fname)
            fsize = os.path.getsize(fpath)
            files.append({
                "name": fname,
                "size": fsize,
                "size_mb": round(fsize / 1024 / 1024, 2),
                "ok": fsize > 0,
                "corrupt_reason": None if fsize > 0 else "空文件",
            })
    return files


def _list_weight_files(model_dir: str) -> list[str]:
    """列出所有权重文件扩展名。"""
    exts = set()
    try:
        names = os.listdir(model_dir)
    except PermissionError:
        return []
    except OSError:
        return []
    for fname in names:
        _, ext = os.path.splitext(fname)
        if ext in (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".h5"):
            exts.add(ext)
    return sorted(exts)


def _check_essential_files(model_dir: str) -> dict:
    """检查必需文件是否存在。"""
    has_config = os.path.isfile(os.path.join(model_dir, "config.json"))
    has_tokenizer = (
        os.path.isfile(os.path.join(model_dir, "tokenizer.json")) or
        os.path.isfile(os.path.join(model_dir, "tokenizer_config.json"))
    )
    has_generation = os.path.isfile(os.path.join(model_dir, "generation_config.json"))
    return {
        "config_json": has_config,
        "tokenizer": has_tokenizer,
        "generation_config": has_generation,
    }


def _estimate_params(config: dict) -> float | None:
    """从 config 估算参数量 (billion)。"""
    if "_parse_error" in config:
        return None
    # 直接声明
    for k in ("num_parameters", "num_params"):
        if k in config:
            v = config[k]
            if isinstance(v, (int, float)):
                return round(v / 1e9, 2)
    # 从 hidden_size, num_layers 估算 (粗略)
    hs = config.get("hidden_size")
    nl = config.get("num_hidden_layers")
    if hs and nl:
        # 粗略: params ≈ 12 * nl * hs^2  (GPT-like)
        rough = 12 * nl * (hs ** 2) / 1e9
        return round(rough, 2)
    return None


def _total_weight_size(st_files: list[dict], bin_files: list[dict]) -> int:
    """所有权重文件大小总和 (bytes)。"""
    return sum(f["size"] for f in st_files) + sum(f["size"] for f in bin_files)


def _any_corrupt(st_files: list[dict], bin_files: list[dict]) -> list[str]:
    """返回所有损坏的文件名。"""
    corrupt = []
    for f in st_files:
        if not f["ok"]:
            corrupt.append(f["name"])
    for f in bin_files:
        if not f["ok"]:
            corrupt.append(f["name"])
    return corrupt


def inspect_model(model_dir: str) -> dict:
    """检查单个模型目录，返回结构化结果。"""
    model_name = os.path.basename(model_dir.rstrip("/"))
    result = {
        "path": model_dir,
        "name": model_name,
        "exists": os.path.isdir(model_dir),
    }
    if not result["exists"]:
        result["error"] = "目录不存在"
        return result

    # 基础文件
    essential = _check_essential_files(model_dir)
    result["essential"] = essential

    # config
    config = _read_config(model_dir)
    if config is None:
        result["config_ok"] = False
        result["config_error"] = "config.json 不存在"
    elif "_parse_error" in config:
        result["config_ok"] = False
        result["config_error"] = config["_parse_error"]
    else:
        result["config_ok"] = True
        result["architecture"] = config.get("architectures", [])
        result["model_type"] = config.get("model_type", "unknown")
        result["hidden_size"] = config.get("hidden_size")
        result["num_hidden_layers"] = config.get("num_hidden_layers")

        # MoE 检测
        is_moe, moe_reason = _is_moe(config)
        result["is_moe"] = is_moe
        if is_moe:
            result["moe_reason"] = moe_reason

        # 参数量估算
        params_b = _estimate_params(config)
        if params_b is not None:
            result["estimated_params_b"] = params_b

    # 权重文件
    st_files = _list_safetensors(model_dir)
    bin_files = _list_bins(model_dir)
    weight_exts = _list_weight_files(model_dir)
    result["weight_format"] = weight_exts
    result["safetensors_files"] = st_files
    result["bin_files"] = bin_files
    result["weight_files_count"] = len(st_files) + len(bin_files)
    result["total_weight_size_gb"] = round(_total_weight_size(st_files, bin_files) / 1024**3, 3)

    # 损坏检测
    corrupt = _any_corrupt(st_files, bin_files)
    result["corrupt_files"] = corrupt
    result["all_files_ok"] = len(corrupt) == 0

    # ── 能力判定 ──
    has_weights = len(st_files) > 0 or len(bin_files) > 0
    result["capabilities"] = {
        "sft": essential["config_json"] and has_weights and len(corrupt) == 0,
        "lora": essential["config_json"] and has_weights and len(corrupt) == 0,
        "moe": result.get("is_moe", False),
    }

    return result


def scan_models() -> list[str]:
    """自动扫描 /data/*/models/ 下的所有模型目录。"""
    dirs = []
    for user_dir in _glob.glob("/data/*"):
        models_root = os.path.join(user_dir, "models")
        if os.path.isdir(models_root):
            try:
                names = sorted(os.listdir(models_root))
            except PermissionError:
                print(f"  [跳过] 无权限访问: {models_root}")
                continue
            except OSError as e:
                print(f"  [跳过] 读取失败 {models_root}: {e}")
                continue
            for name in names:
                full = os.path.join(models_root, name)
                if os.path.isdir(full):
                    dirs.append(full)
    return dirs


def main():
    if MODEL_DIRS:
        dirs = MODEL_DIRS
    else:
        dirs = scan_models()

    if not dirs:
        print("=== NO_MODEL_DIRS ===")
        return

    print(f"=== 扫描到 {len(dirs)} 个模型目录 ===")
    for d in dirs:
        print(f"  {d}")

    results = []
    for d in dirs:
        print()
        print(f"=== 检查: {d} ===")
        r = inspect_model(d)
        results.append(r)
        # 打印摘要
        if r.get("config_ok"):
            arch = r.get("architecture", [])
            mt = r.get("model_type", "?")
            params = r.get("estimated_params_b", "?")
            m = "MoE" if r.get("is_moe") else "Dense"
            caps = r["capabilities"]
            sft_ok = "✓" if caps["sft"] else "✗"
            lora_ok = "✓" if caps["lora"] else "✗"
            corrupt_n = len(r.get("corrupt_files", []))
            print(f"  type={mt}  arch={arch}  params≈{params}B  {m}")
            print(f"  weights: {r['weight_files_count']} files  {r['total_weight_size_gb']} GB  format={r['weight_format']}")
            print(f"  SFT={sft_ok}  LoRA={lora_ok}  corrupt={corrupt_n}")
        else:
            print(f"  config_ok=False  error={r.get('config_error', '?')}")

    # 输出 JSON 块供本地解析
    print()
    print("=== INSPECT_RESULT_JSON ===")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("=== END_INSPECT_RESULT_JSON ===")


if __name__ == "__main__":
    main()
'''


def _build_entry_command(model_dirs: list[str]) -> str:
    """注入模型路径列表到 INSPECT_PY。"""
    filled = INSPECT_PY.replace(
        "MODEL_DIRS = []   # 由 entry_command 注入",
        f"MODEL_DIRS = {json.dumps(model_dirs)}"
    )
    return (
        "set -e\n"
        "_log() { echo \"[$(date '+%Y-%m-%d %H:%M:%S')] $*\"; }\n\n"
        "_log \"开始检查模型...\"\n"
        "cat > /tmp/inspect_model.py << 'PYEOF'\n"
        + filled
        + "\nPYEOF\n"
        "python3 /tmp/inspect_model.py 2>&1\n"
        "echo '---DONE---'\n"
    )


def _poll_job(job_id: str, poll_interval: int = 30) -> dict:
    """轮询任务状态。"""
    last_status = None
    task_name = job_id[:8]

    while True:
        try:
            job = magnus.get_job(job_id)
        except Exception as exc:
            print(f"[{_ts()}] [{task_name}] 查询失败: {exc}")
            time.sleep(poll_interval)
            continue

        status = job.get("status", "Unknown")
        task_name = job.get("task_name", task_name)

        if status != last_status:
            if last_status is None:
                print(f"[{_ts()}] [{task_name}] [{status}]")
            else:
                print(f"[{_ts()}] [{task_name}] [{last_status}] -> [{status}]")
            last_status = status
        else:
            if status not in ("Success", "Failed", "Terminated"):
                print(f"[{_ts()}] [{task_name}] [{status}] ...")

        if status in ("Success", "Failed", "Terminated"):
            return job

        time.sleep(poll_interval)


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _parse_results(logs_text: str) -> list[dict] | None:
    """从日志中提取 INSPECT_RESULT_JSON 块。"""
    lines = logs_text.splitlines()
    capture = False
    json_lines = []
    for line in lines:
        if line.strip() == "=== INSPECT_RESULT_JSON ===":
            capture = True
            continue
        if line.strip() == "=== END_INSPECT_RESULT_JSON ===":
            break
        if capture:
            json_lines.append(line)
    if not json_lines:
        return None
    try:
        return json.loads("\n".join(json_lines))
    except json.JSONDecodeError:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="查询 Magnus 持久存储中的模型，检查微调能力"
    )
    parser.add_argument("--address", default=MAGNUS_ADDRESS)
    parser.add_argument("--token", default=MAGNUS_TOKEN)
    parser.add_argument("--model-path", action="append", default=None,
                        help="指定模型目录（可多次使用）。默认: 自动扫描 /data/*/models/")
    parser.add_argument("--poll-interval", type=int, default=30,
                        help="轮询间隔秒 (default: 30)")
    parser.add_argument("-o", "--output", default=OUTPUT_PATH,
                        help=f"输出 JSON 路径 (default: {OUTPUT_PATH})")
    args = parser.parse_args()

    model_dirs = args.model_path or []

    # ── 1. 配置 ──
    print(f"[1/3] 配置 Magnus 连接...")
    print(f"      地址: {args.address}")
    print(f"      Token: {args.token[:8]}...{args.token[-4:]}")
    magnus.configure(address=args.address, token=args.token)

    # ── 2. 提交作业 ──
    entry_command = _build_entry_command(model_dirs)
    scan_desc = "自动扫描" if not model_dirs else f"{len(model_dirs)} 个指定模型"
    print(f"[2/3] 提交模型检查作业 ({scan_desc})...")
    print()

    job_id = magnus.submit_job(
        task_name         = "Inspect-Model-Storage",
        description       = f"检查模型文件: {scan_desc}",
        entry_command     = entry_command,
        system_entry_command = SYSTEM_ENTRY_COMMAND,
        namespace         = "Rise-AGI",
        repo_name         = "OpenFundus",
        gpu_count         = 0,
        gpu_type          = "cpu",
        cpu_count         = 2,
        memory_demand     = "4G",
        ephemeral_storage = "10G",
        job_type          = "B2",
    )

    # ── 3. 轮询 ──
    print(f"[3/3] 提交成功，Job ID: {job_id}")
    notify_exe(job_id=job_id)
    print(f"      每 {args.poll_interval}s 轮询...")
    print()
    job = _poll_job(job_id, args.poll_interval)
    print()

    if job.get("status") != "Success":
        print(f"任务未成功 (status={job.get('status')})")
        try:
            log_page = magnus.get_job_logs(job_id, page=0)
            text = log_page.get("logs", "").strip()
            if text:
                print(f"最后日志:\n{text[:3000]}")
        except Exception:
            pass
        sys.exit(1)

    # ── 4. 获取日志并解析结果 ──
    all_logs = []
    page_num = 0
    while True:
        try:
            page = magnus.get_job_logs(job_id, page=page_num)
            text = page.get("logs", "").strip()
            if not text:
                break
            all_logs.append(text)
            page_num += 1
        except Exception:
            break

    full_logs = "\n".join(all_logs)
    if full_logs:
        print(full_logs)

    results = _parse_results(full_logs)
    if results is None:
        print("\n[WARN] 未从日志中解析到模型检查结果")
        sys.exit(1)

    # ── 5. 分组摘要 ──
    usable = []     # 权重完整 (SFT/LoRA 均可)
    sft_paths = []
    lora_paths = []
    moe_paths = []
    corrupt_paths = []
    unusable = []   # 缺 config 或权重

    for m in results:
        path = m.get("path", m.get("name", ""))
        caps = m.get("capabilities", {})
        if caps.get("sft"):
            usable.append(path)
            sft_paths.append(path)
            lora_paths.append(path)
        elif m.get("essential", {}).get("config_json") and len(m.get("corrupt_files", [])) > 0:
            corrupt_paths.append(path)
        else:
            unusable.append(path)
        if caps.get("moe"):
            moe_paths.append(path)

    summary = {
        "usable": usable,
        "sft": sft_paths,
        "lora": lora_paths,
        "moe": moe_paths,
        "corrupt": corrupt_paths,
        "unusable": unusable,
    }

    # ── 6. 写入输出 ──
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_data = {
        "inspected_at": datetime.now().isoformat(),
        "job_id": job_id,
        "model_count": len(results),
        "summary": summary,
        "models": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入: {args.output}  ({len(results)} 个模型)")

    # ── 7. 终端摘要 ──
    print("\n" + "=" * 60)
    print("  模型路径摘要")
    print("=" * 60)

    def _print_group(title, paths):
        if paths:
            print(f"\n-- {title} ({len(paths)}个)")
            for p in sorted(paths):
                print(f"    {p}")

    _print_group("可使用 (权重完整)", usable)
    _print_group("  SFT", sft_paths)
    _print_group("  LoRA", lora_paths)
    _print_group("  MoE", moe_paths)
    if corrupt_paths:
        _print_group("文件损坏", corrupt_paths)
    if unusable:
        _print_group("不可用 (缺配置/权重)", unusable)

    print("\n" + "-" * 60)
    for m in results:
        caps = m.get("capabilities", {})
        sft = "✓" if caps.get("sft") else "✗"
        lora = "✓" if caps.get("lora") else "✗"
        moe = "MoE" if caps.get("moe") else "Dense"
        corrupt_n = len(m.get("corrupt_files", []))
        corrupt_flag = f"  CORRUPT:{corrupt_n}" if corrupt_n else ""
        params = m.get("estimated_params_b", "?")
        print(f"  [{m['name']}]  params≈{params}B  {moe}  SFT={sft}  LoRA={lora}{corrupt_flag}")


if __name__ == "__main__":
    main()
