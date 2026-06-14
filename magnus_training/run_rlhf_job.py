from __future__ import annotations

import argparse
from pathlib import Path

from common_runtime import (
    add_bool,
    add_flag,
    ensure_dependencies,
    launcher,
    log,
    make_fake_rlhf_data,
    receive_file_secret,
    receive_resume_checkpoint,
    resolve_model_path,
    run_training,
    runtime_dir,
    setup_environment,
    write_result,
)


SCRIPT_BY_ALGO = {
    "grpo": "grpo_train.py",
    "ppo": "ppo_train.py",
    "dpo": "dpo_train.py",
    "orpo": "orpo_train.py",
}


def parse_args():
    p = argparse.ArgumentParser(description="Magnus RLHF runtime wrapper")
    p.add_argument("--algorithm", choices=sorted(SCRIPT_BY_ALGO), required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--reward_model_path")
    p.add_argument("--train_data")
    p.add_argument("--train_data_secret")
    p.add_argument("--train_data_secret_name", default="uploaded_rlhf_train.json")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-7)
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--max_response_length", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--group_size", type=int, default=8)
    p.add_argument("--grpo_kl_coef", type=float, default=0.04)
    p.add_argument("--grpo_clip_range", type=float, default=0.2)
    p.add_argument("--ppo_kl_coef", type=float, default=0.1)
    p.add_argument("--ppo_clip_range", type=float, default=0.2)
    p.add_argument("--reward_ratio", type=float, default=0.3)
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--dpo_beta", type=float, default=0.1)
    p.add_argument("--dpo_loss_type", choices=["sigmoid", "hinge", "ipo"], default="sigmoid")
    p.add_argument("--orpo_lambda", type=float, default=0.1)
    p.add_argument("--reward_api_url")
    p.add_argument("--resume_from")
    p.add_argument("--cpu_offload", action="store_true")
    p.add_argument("--use_8bit_adam", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_environment()
    ensure_dependencies()
    actual_model = resolve_model_path(args.model_path)
    uploaded_train_data = receive_file_secret(
        args.train_data_secret,
        args.train_data_secret_name,
        "uploaded_rlhf_train.json",
    )
    train_data = uploaded_train_data or args.train_data or str(make_fake_rlhf_data(Path("/tmp/fake_rlhf_train.json"), args.algorithm))
    using_fake_data = uploaded_train_data is None and args.train_data is None
    resume_from = receive_resume_checkpoint(args.resume_from)
    reward_model = resolve_model_path(args.reward_model_path) if args.reward_model_path else None
    script = runtime_dir() / SCRIPT_BY_ALGO[args.algorithm]

    command = launcher(script)
    add_flag(command, "--model_path", actual_model)
    add_flag(command, "--train_data", train_data)
    add_flag(command, "--output_dir", args.output_dir)
    add_flag(command, "--epochs", args.epochs)
    add_flag(command, "--batch_size", args.batch_size)
    add_flag(command, "--gradient_accumulation_steps", args.grad_accum)
    add_flag(command, "--learning_rate", args.learning_rate)
    add_flag(command, "--max_prompt_length", args.max_prompt_length)
    add_flag(command, "--max_response_length", args.max_response_length)
    add_flag(command, "--logging_steps", 10)
    add_flag(command, "--num_workers", 0 if using_fake_data else args.num_workers)
    add_flag(command, "--resume_from_checkpoint", resume_from)
    add_bool(command, "--cpu_offload", args.cpu_offload)
    add_bool(command, "--use_8bit_adam", args.use_8bit_adam)

    if args.algorithm == "grpo":
        add_flag(command, "--group_size", args.group_size)
        add_flag(command, "--kl_coef", args.grpo_kl_coef)
        add_flag(command, "--clip_range", args.grpo_clip_range)
        add_flag(command, "--temperature", args.temperature)
        add_flag(command, "--top_p", args.top_p)
        add_flag(command, "--reward_api_url", args.reward_api_url)
    elif args.algorithm == "ppo":
        add_flag(command, "--reward_model_path", reward_model)
        add_flag(command, "--kl_coef", args.ppo_kl_coef)
        add_flag(command, "--clip_range", args.ppo_clip_range)
        add_flag(command, "--reward_ratio", args.reward_ratio)
        add_flag(command, "--value_coef", args.value_coef)
        add_flag(command, "--temperature", args.temperature)
        add_flag(command, "--top_p", args.top_p)
        add_flag(command, "--reward_api_url", args.reward_api_url)
    elif args.algorithm == "dpo":
        add_flag(command, "--dpo_beta", args.dpo_beta)
        add_flag(command, "--dpo_loss", args.dpo_loss_type)
    elif args.algorithm == "orpo":
        add_flag(command, "--orpo_lambda", args.orpo_lambda)

    try:
        run_training(command)
        write_result(args.output_dir)
        return 0
    except Exception as exc:
        log(f"Training failed: {exc}")
        write_result(args.output_dir, status="failed", error=str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
