"""将 OpenFundus_SFT_zyz.magnus 注册为 Blueprint 并提交 SFT 作业。

用法:
    SSL_CERT_FILE= python run_sft_blueprint.py [--model MODEL] [--output OUTPUT]
"""
import argparse
import magnus

DEFAULT_ADDRESS = "http://162.105.151.134:3011/"
DEFAULT_TOKEN   = "sk-xxx"


def main():
    parser = argparse.ArgumentParser(description="注册并运行 SFT Blueprint")
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--token",   default=DEFAULT_TOKEN)
    parser.add_argument("--model",   default="Qwen/Qwen2.5-7B-Instruct",
                        help="模型路径或 Hub ID")
    parser.add_argument("--output",  default="/data/magnus/models/qwen-sft-test-v1",
                        help="输出目录")
    parser.add_argument("--gpus",    type=int, default=3)
    parser.add_argument("--bp-id",   default="openfundus-sft-zyz",
                        help="Blueprint ID（在服务器上唯一）")
    args = parser.parse_args()

    magnus.configure(address=args.address, token=args.token)

    # 读取 .magnus 文件
    magnus_path = "OpenFundus_SFT_zyz.magnus"
    code = open(magnus_path, encoding="utf-8").read()

    # 注册 Blueprint（strip_imports 由 SDK 内部处理）
    # 如果已存在则先删除再重建
    print(f"[1/4] 注册 Blueprint: {args.bp_id} ...")
    try:
        magnus.delete_blueprint(args.bp_id)
    except Exception:
        pass  # 不存在则忽略
    bp = magnus.save_blueprint(
        blueprint_id=args.bp_id,
        title="通用大模型 SFT 训练 + 评估",
        description="自动下载模型 → SFT 训练 → 评估 → 上传日志到 File Custody",
        code=code,
    )
    print(f"      Title: {bp.get('title')}")
    print()

    # 运行 Blueprint
    model_name = args.model.split("/")[-1]
    params = {
        "model_path":       args.model,
        "train_data":       None,      # 使用假数据
        "output_dir":       args.output,
        "epochs":           1,
        "batch_size":       2,
        "grad_accum":       4,
        "gpu_count":        args.gpus,
        "gpu_type":         "a100",
        "cpu_count":        40,
        "memory_demand":    "256G",
        "ephemeral_storage": "500G",
        "priority":         "A2",
        "execute_action":   True,
    }

    print(f"[2/4] 提交 SFT 作业 ...")
    print(f"      模型: {args.model}")
    print(f"      输出: {args.output}")
    print(f"      GPU:  {args.gpus}x A100")
    print(f"      数据: 假数据 30 条（流程验证）")
    print()

    job_id = magnus.launch_blueprint(
        blueprint_id=args.bp_id,
        args=params,
        use_preference=False,
        save_preference=False,
    )
    print(f"      Job ID: {job_id}")
    print()
    print(f"[3/4] 监控作业（每 60s 轮询）...")

    from monitor import Monitor, auto_source, notify_exe
    notify_exe(job_id=job_id)
    Monitor(poll_interval=60, source=auto_source()).add(job_id).run()
    print()

    print(f"[4/4] 作业完成")
    result = magnus.get_job_result(job_id)
    if result:
        print(f"      结果: {result[:300]}")
    else:
        print(f"      无结果（可能失败）")


if __name__ == "__main__":
    main()
