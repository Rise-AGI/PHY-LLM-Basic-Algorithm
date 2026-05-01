"""
在 Magnus 上通过蓝图提交 SFT 训练任务，并监控执行状态。

工作流: 读取 .magnus 蓝图文件 → 保存到 Magnus → 启动蓝图 → 监控任务

用法:
    python magnus_sft.py --model /data/magnus/models/Qwen2.5-1.5B
    python magnus_sft.py --model /data/magnus/models/internlm2-math-7b --epochs 5 --lr 1e-4
    python magnus_sft.py --model deepseek-ai/deepseek-math-7b --train-data /data/train.json
    python magnus_sft.py --blueprint OpenFundus_SFT.magnus --model /data/qwen --gpu-count 6
"""

import argparse
import os
import re
import sys
from datetime import datetime
from typing import Optional

import magnus

from monitor import Monitor, record_storage, check_model_version_exists, SFT_DATA_DIR, _ensure_record, auto_source, notify_exe


DEFAULT_ADDRESS = "http://162.105.151.134:3011/"
DEFAULT_TOKEN   = "sk-xxx"

HERE = os.path.dirname(os.path.abspath(__file__))


def _extract_model_short_name(model_path: str) -> str:
    """从模型路径或 Hub ID 提取简短模型名。"""
    name = model_path.rstrip("/").split("/")[-1]
    return name


def _auto_model_version(short_name: str) -> str:
    """自动生成下一个版本号 (model-v1, model-v2, ...)。"""
    record = _ensure_record()
    existing = [e for e in record.get("model-version", [])
                if e.get("model", "").startswith(short_name + "-v")]
    max_v = 0
    for e in existing:
        m = re.search(r'-v(\d+)$', e.get("model", ""))
        if m:
            v = int(m.group(1))
            if v > max_v:
                max_v = v
    return f"{short_name}-v{max_v + 1}"


def _download_report(model_version: str, secret_or_text: str) -> Optional[str]:
    """尝试将报告保存到 SFT_data/ 目录。返回本地路径或 None。"""
    os.makedirs(SFT_DATA_DIR, exist_ok=True)
    dest = os.path.join(SFT_DATA_DIR, model_version)

    # 如果结果是 custody secret，用 magnus download
    if secret_or_text.startswith("magnus-secret:"):
        try:
            magnus.download_file(secret_or_text, dest)
            return dest
        except Exception:
            pass

    # 否则当作文本保存
    try:
        with open(dest, "w", encoding="utf-8") as f:
            f.write(secret_or_text)
        return dest
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="通过蓝图提交 SFT 训练任务到 Magnus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── 连接 ──
    conn = parser.add_argument_group("连接参数")
    conn.add_argument("--address", default=DEFAULT_ADDRESS)
    conn.add_argument("--token", default=DEFAULT_TOKEN)

    # ── 蓝图 ──
    bp = parser.add_argument_group("蓝图参数")
    bp.add_argument("--blueprint", default=os.path.join(HERE, "OpenFundus_SFT_zyz.magnus"),
                    help="蓝图 .magnus 文件路径 (default: OpenFundus_SFT_zyz.magnus)")

    # ── SFT 参数 ──
    sft = parser.add_argument_group("SFT 训练参数")
    sft.add_argument("--model", required=True,
                     help="模型路径 (e.g. /data/magnus/models/Qwen2.5-1.5B) 或 Hub ID")
    sft.add_argument("--model-version", default=None,
                     help="模型版本名 (e.g. Qwen2.5-7B-v2，默认自动递增)")
    sft.add_argument("--train-data", default=None,
                     help="训练数据路径 (留空=假数据)")
    sft.add_argument("--test-data", default=None,
                     help="测试数据路径 (留空=假数据)")
    sft.add_argument("--output-dir", default="/data/magnus/models/general-sft-v1",
                     help="输出目录 (default: /data/magnus/models/general-sft-v1)")
    sft.add_argument("--epochs", type=int, default=3)
    sft.add_argument("--batch-size", type=int, default=2)
    sft.add_argument("--grad-accum", type=int, default=4)
    sft.add_argument("--lr", type=float, default=2e-5, help="学习率")
    sft.add_argument("--max-length", type=int, default=1024)
    sft.add_argument("--save-steps", type=int, default=200)

    # ── 硬件 ──
    hw = parser.add_argument_group("硬件与调度")
    hw.add_argument("--gpu-count", type=int, default=3)
    hw.add_argument("--gpu-type", default="a100", choices=["a100", "cpu"])
    hw.add_argument("--cpu-count", type=int, default=40)
    hw.add_argument("--memory", default="320G", help="内存需求 (e.g. 320G)")
    hw.add_argument("--storage", default="500G", help="临时存储 (e.g. 500G)")
    hw.add_argument("--priority", default="A2", choices=["A1", "A2", "B1", "B2"])

    # ── 其他 ──
    misc = parser.add_argument_group("其他")
    misc.add_argument("--container-image", default=None,
                      help="自定义容器镜像 URI")
    misc.add_argument("--resume-from", default=None,
                      help="从 checkpoint 恢复 (secret 或本地路径)")
    misc.add_argument("--poll-interval", type=int, default=60,
                      help="监控轮询间隔秒数 (default: 60)")

    args = parser.parse_args()

    # ── 0. model-version 去重 ────────────────────────────────
    short_name = _extract_model_short_name(args.model)
    model_version = args.model_version or _auto_model_version(short_name)
    print(f"[0/6] 模型版本: {model_version}")
    if check_model_version_exists(model_version):
        print(f"[FATAL] 模型版本 '{model_version}' 已存在于存储记录中，拒绝提交。")
        print(f"       如需重新训练，请使用 --model-version 指定新版本号")
        sys.exit(1)
    print(f"       版本检查通过")

    # ── 1. 配置 ──────────────────────────────────────────
    magnus.configure(address=args.address, token=args.token)
    print(f"[1/6] 配置 Magnus 连接")
    print(f"      地址 : {args.address}")
    print(f"      Token: {args.token[:8]}...{args.token[-4:]}")

    # ── 2. 读取并保存蓝图 ────────────────────────────────
    bp_path = args.blueprint
    if not os.path.exists(bp_path):
        print(f"[FATAL] 蓝图文件不存在: {bp_path}")
        sys.exit(1)

    with open(bp_path, "r", encoding="utf-8") as f:
        blueprint_code = f.read()

    bp_id = os.path.splitext(os.path.basename(bp_path))[0]
    bp_title = f"SFT-{bp_id}"

    print(f"[2/6] 保存蓝图到 Magnus")
    print(f"      ID: {bp_id}")
    print(f"      文件: {bp_path}")
    try:
        magnus.save_blueprint(
            blueprint_id=bp_id,
            title=bp_title,
            description=f"Auto-saved SFT blueprint: {bp_path}",
            code=blueprint_code,
        )
        print(f"      保存成功")
    except Exception as e:
        print(f"      保存失败: {e}")
        print(f"      继续尝试启动已有蓝图...")

    # ── 3. 准备参数 ──────────────────────────────────────
    bp_args = {
        "model_path":       args.model,
        "output_dir":       args.output_dir,
        "epochs":           args.epochs,
        "batch_size":       args.batch_size,
        "grad_accum":       args.grad_accum,
        "learning_rate":    args.lr,
        "max_length":       args.max_length,
        "save_steps":       args.save_steps,
        "gpu_count":        args.gpu_count,
        "gpu_type":         args.gpu_type,
        "cpu_count":        args.cpu_count,
        "memory_demand":    args.memory,
        "ephemeral_storage": args.storage,
        "priority":         args.priority,
    }
    if args.train_data:
        bp_args["train_data"] = args.train_data
    if args.test_data:
        bp_args["test_data"] = args.test_data
    if args.container_image:
        bp_args["container_image"] = args.container_image
    if args.resume_from:
        bp_args["resume_from"] = args.resume_from

    print(f"[3/6] 蓝图参数")
    for k, v in bp_args.items():
        print(f"      {k}: {v}")

    # ── 4. 提交 ──────────────────────────────────────────
    print(f"[4/6] 提交蓝图: {bp_id}")
    job_id = magnus.launch_blueprint(bp_id, args=bp_args)
    print(f"      Job ID: {job_id}")
    print()

    # ── 5. 监控 ──────────────────────────────────────────
    print(f"[5/6] 开始监控 (poll_interval={args.poll_interval}s)")
    print("-" * 60)
    notify_exe(job_id=job_id)
    Monitor(poll_interval=args.poll_interval, source=auto_source()).add(job_id).run()

    # ── 6. 后处理 ──────────────────────────────────────────
    print(f"[6/6] 后处理")
    job = magnus.get_job(job_id)
    if job.get("status") == "Success":
        result = None
        try:
            result = magnus.get_job_result(job_id)
        except Exception:
            pass

        if result:
            local_path = _download_report(model_version, result)
            if local_path:
                print(f"      报告已下载: {local_path}")
            else:
                print(f"      报告下载失败，原始结果: {result[:200]}")

        record_storage("model-version", {
            "time": datetime.now().isoformat(),
            "model": model_version,
            "local_path": args.output_dir,
            "base_model": args.model,
            "status": "success",
        })
        print(f"      已记录 model-version: {model_version}")
    else:
        print(f"      任务未成功，跳过记录")


if __name__ == "__main__":
    main()