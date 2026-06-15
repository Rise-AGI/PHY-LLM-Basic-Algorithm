from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from common_runtime import (
    log,
    receive_file_secret,
    receive_resume_checkpoint,
    resolve_model_path,
    runtime_dir,
    setup_environment,
    verified_gpu_count,
    write_result,
)
from prepare_openrlhf_data import convert as convert_prompt_data


DEFAULT_IMAGE_NOTE = "OpenRLHF/vLLM runtime expects docker tag sft-base:v5 or newer."
ONLINE_ALGORITHMS = {"ppo", "reinforce", "reinforce_baseline", "grpo", "dr_grpo", "rloo"}
ESTIMATOR_BY_ALGORITHM = {
    "ppo": "gae",
    "reinforce": "reinforce",
    "reinforce_baseline": "reinforce_baseline",
    "grpo": "group_norm",
    "dr_grpo": "dr_grpo",
    "rloo": "rloo",
}
VLLM_TEXT_ONLY_PATCH = r'''
"""Runtime patches for OpenRLHF + vLLM jobs on Magnus.

OpenRLHF creates vLLM AsyncEngineArgs internally and does not expose newer
vLLM multimodal/text-only switches.  vLLM 0.22 may classify Qwen3.5/Qwen3.6
text checkpoints through the Qwen3-VL renderer path and then fail when the
HF config is Qwen3_5TextConfig.  The PPO payload is text-only, so force vLLM
to skip multimodal processors for those rollout engines.

OpenRLHF also calls ray.init() internally without exposing Ray startup options.
On Apptainer/Magnus nodes Ray dashboard startup can fail and delay local node
registration until ray.init() times out.  Disable the dashboard for training
jobs; OpenRLHF does not need it.
"""

from __future__ import annotations

import os


def _enabled() -> bool:
    return os.environ.get("OPENRLHF_VLLM_TEXT_ONLY_PATCH", "1").lower() not in {
        "0",
        "false",
        "no",
    }


def _patch_ray_init() -> None:
    if os.environ.get("OPENRLHF_RAY_PATCH", "1").lower() in {"0", "false", "no"}:
        return
    try:
        import ray

        if getattr(ray.init, "_openrlhf_magnus_patched", False):
            return
        _original_ray_init = ray.init

        def _init_without_dashboard(*args, **kwargs):
            kwargs.setdefault("include_dashboard", False)
            kwargs.setdefault("dashboard_host", "127.0.0.1")
            kwargs.setdefault("ignore_reinit_error", True)
            temp_dir = os.environ.get("OPENRLHF_RAY_TEMP_DIR")
            if temp_dir:
                kwargs.setdefault("_temp_dir", temp_dir)
            return _original_ray_init(*args, **kwargs)

        _init_without_dashboard._openrlhf_magnus_patched = True
        _init_without_dashboard._openrlhf_original = _original_ray_init
        ray.init = _init_without_dashboard
        print("[openrlhf-ray-patch] ray.init dashboard disabled", flush=True)
    except Exception as exc:
        print(f"[openrlhf-ray-patch] failed to patch ray.init: {exc}", flush=True)


if _enabled():
    try:
        import vllm
        from vllm.engine import arg_utils as _arg_utils

        _OriginalAsyncEngineArgs = _arg_utils.AsyncEngineArgs

        class _TextOnlyAsyncEngineArgs(_OriginalAsyncEngineArgs):
            def __init__(self, *args, **kwargs):
                kwargs.setdefault("runner", "generate")
                kwargs["language_model_only"] = True
                kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
                kwargs["mm_processor_cache_gb"] = 0
                super().__init__(*args, **kwargs)

        _arg_utils.AsyncEngineArgs = _TextOnlyAsyncEngineArgs
        vllm.AsyncEngineArgs = _TextOnlyAsyncEngineArgs

        try:
            from vllm.multimodal.registry import MultiModalRegistry

            _original_supports_multimodal_inputs = (
                MultiModalRegistry.supports_multimodal_inputs
            )

            def _supports_multimodal_inputs_text_safe(self, model_config):
                mm_config = getattr(model_config, "multimodal_config", None)
                if getattr(mm_config, "language_model_only", False):
                    return False
                try:
                    return _original_supports_multimodal_inputs(self, model_config)
                except TypeError as exc:
                    if "Invalid type of HuggingFace config" in str(exc):
                        return False
                    raise

            MultiModalRegistry.supports_multimodal_inputs = (
                _supports_multimodal_inputs_text_safe
            )
        except Exception as exc:
            print(
                f"[openrlhf-vllm-patch] multimodal registry patch skipped: {exc}",
                flush=True,
            )

        print("[openrlhf-vllm-patch] text-only vLLM patch enabled", flush=True)
    except Exception as exc:
        print(f"[openrlhf-vllm-patch] failed to enable patch: {exc}", flush=True)

_patch_ray_init()
'''


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Magnus OpenRLHF/vLLM runtime wrapper")
    p.add_argument("--algorithm", choices=sorted(ONLINE_ALGORITHMS), required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--train_data")
    p.add_argument("--train_data_secret")
    p.add_argument("--train_data_secret_name", default="uploaded_rlhf_train.json")
    p.add_argument("--reward_model_path")
    p.add_argument("--reward_api_url")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--resume_from")

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--num_episodes", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--train_batch_size", type=int)
    p.add_argument("--rollout_batch_size", type=int)
    p.add_argument("--rollout_micro_batch_size", type=int)
    p.add_argument("--learning_rate", type=float, default=5e-7)
    p.add_argument("--critic_learning_rate", type=float, default=5e-6)
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--max_response_length", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_samples", type=int)

    p.add_argument("--group_size", type=int, default=8)
    p.add_argument("--kl_coef", type=float, default=0.02)
    p.add_argument("--kl_target", type=float)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--value_clip", type=float, default=0.5)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--normalize_reward", action="store_true")
    p.add_argument("--reward_clip_min", type=float, default=-5.0)
    p.add_argument("--reward_clip_max", type=float, default=5.0)

    p.add_argument("--zero_stage", type=int, default=3)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--flash_attn", action="store_true", default=True)
    p.add_argument("--packing_samples", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--adam_offload", action="store_true")
    p.add_argument("--ref_offload", action="store_true")
    p.add_argument("--disable_fast_tokenizer", action="store_true")
    p.add_argument("--save_steps", type=int, default=-1)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--train_max_tokens_per_gpu", type=int)
    p.add_argument("--rollout_max_tokens_per_gpu", type=int)

    p.add_argument("--vllm_num_engines", type=int, default=2)
    p.add_argument("--vllm_tensor_parallel_size", type=int, default=2)
    p.add_argument("--vllm_generate_batch_size", type=int)
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.45)
    p.add_argument("--vllm_sync_backend", default="nccl")
    p.add_argument("--vllm_enable_sleep", action="store_true", default=True)
    p.add_argument("--vllm_enforce_eager", action="store_true")
    p.add_argument("--vllm_text_only_patch", dest="vllm_text_only_patch", action="store_true", default=True)
    p.add_argument("--no_vllm_text_only_patch", dest="vllm_text_only_patch", action="store_false")
    p.add_argument("--ray_dashboard", dest="ray_dashboard", action="store_true")
    p.add_argument("--no_ray_dashboard", dest="ray_dashboard", action="store_false")
    p.set_defaults(ray_dashboard=False)

    p.add_argument("--colocate_all", action="store_true")
    p.add_argument("--ds_enable_sleep", action="store_true")
    p.add_argument("--openrlhf_cli_style", choices=["v5"], default="v5")
    p.add_argument("--ray_start", action="store_true")
    p.add_argument("--required_gpu_count", type=int, default=0)
    return p.parse_args()


def ensure_runtime_dependencies() -> None:
    missing = []
    for module in ("torch", "ray", "vllm", "deepspeed", "openrlhf"):
        try:
            importlib.import_module(module)
        except Exception as exc:
            missing.append(f"{module} ({exc})")
    if missing:
        raise RuntimeError(
            "Missing RLHF runtime dependencies: "
            + "; ".join(missing)
            + ". "
            + DEFAULT_IMAGE_NOTE
        )
    log("OpenRLHF/vLLM dependencies ready")


def add_flag(command: list[str], name: str, value) -> None:
    if value is None:
        return
    command.extend([name, str(value)])


def add_bool(command: list[str], name: str, enabled: bool) -> None:
    if enabled:
        command.append(name)


def prepare_train_data(args: argparse.Namespace) -> str:
    uploaded = receive_file_secret(
        args.train_data_secret,
        args.train_data_secret_name,
        "uploaded_rlhf_train.json",
    )
    source = uploaded or args.train_data
    if not source:
        raise ValueError("train_data or train_data_secret is required for OpenRLHF online RL")
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Training data does not exist: {source}")
    out = Path("/tmp/openrlhf_prompt_data.jsonl")
    count = convert_prompt_data(source_path, out, args.max_samples)
    log(f"Prepared OpenRLHF prompt data: {count} rows -> {out}")
    return str(out)


def install_vllm_text_only_patch(enabled: bool) -> None:
    if not enabled:
        os.environ["OPENRLHF_VLLM_TEXT_ONLY_PATCH"] = "0"
        log("vLLM text-only patch disabled")
        return

    patch_dir = Path("/tmp/openrlhf_vllm_text_only_patch")
    patch_dir.mkdir(parents=True, exist_ok=True)
    patch_file = patch_dir / "sitecustomize.py"
    patch_file.write_text(VLLM_TEXT_ONLY_PATCH, encoding="utf-8")
    current = os.environ.get("PYTHONPATH")
    patch_dir_str = str(patch_dir)
    parts = [patch_dir_str]
    if current:
        parts.append(current)
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)
    os.environ["OPENRLHF_VLLM_TEXT_ONLY_PATCH"] = "1"
    if patch_dir_str not in sys.path:
        sys.path.insert(0, patch_dir_str)
    spec = importlib.util.spec_from_file_location(
        "_openrlhf_vllm_text_only_sitecustomize",
        patch_file,
    )
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    log(f"Installed vLLM text-only patch via PYTHONPATH: {patch_dir}")


def configure_ray_environment(enable_dashboard: bool) -> None:
    job_id = os.environ.get("MAGNUS_JOB_ID") or "local"
    ray_tmp = Path("/tmp") / f"ray-openrlhf-{job_id}"
    ray_tmp.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    os.environ.setdefault("RAY_DEDUP_LOGS", "0")
    os.environ["OPENRLHF_RAY_PATCH"] = "0" if enable_dashboard else "1"
    os.environ["OPENRLHF_RAY_TEMP_DIR"] = str(ray_tmp)
    if enable_dashboard:
        log("Ray dashboard left enabled")
    else:
        log(f"Ray dashboard disabled; temp dir: {ray_tmp}")


def build_openrlhf_command(args: argparse.Namespace, model_path: str, train_data: str) -> list[str]:
    gpu_count = verified_gpu_count()
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    log(
        "CUDA visibility: "
        f"verified={gpu_count}, required={args.required_gpu_count or 'not set'}, "
        f"CUDA_VISIBLE_DEVICES={visible_devices}"
    )
    if gpu_count <= 0:
        raise RuntimeError("OpenRLHF/vLLM training requires CUDA GPUs")
    if args.required_gpu_count and gpu_count < args.required_gpu_count:
        raise RuntimeError(
            f"Magnus exposed only {gpu_count} usable CUDA GPU(s), but this PPO 27B preset "
            f"requires {args.required_gpu_count} GPU(s). Re-submit the blueprint with "
            f"GPU 数量={args.required_gpu_count} and GPU 类型=A100. Do not lower "
            "vLLM TP to 3 for Qwen 27B; tensor parallel size must match a valid model "
            "head split and the 27B PPO preset is configured for TP4."
        )
    vllm_gpu_slots = args.vllm_num_engines * args.vllm_tensor_parallel_size
    if vllm_gpu_slots > gpu_count:
        raise ValueError(
            "vllm_num_engines * vllm_tensor_parallel_size must be <= visible GPU count "
            f"({args.vllm_num_engines} * {args.vllm_tensor_parallel_size} > {gpu_count}). "
            "For the default Qwen 27B PPO preset, re-submit with GPU 数量=4 and "
            "vLLM TP size=4."
        )
    if args.colocate_all and vllm_gpu_slots != gpu_count:
        raise ValueError(
            "OpenRLHF v5 requires vllm_num_engines * vllm_tensor_parallel_size == gpu_count "
            f"when colocate_all is enabled ({vllm_gpu_slots} != {gpu_count})"
        )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ckpt_path = output / "ckpt"
    reward_url = args.reward_api_url or str(runtime_dir() / "reward_func.py")
    effective_train_batch = args.train_batch_size or max(1, args.batch_size * gpu_count * max(1, args.group_size))
    rollout_batch = args.rollout_batch_size or effective_train_batch
    rollout_micro_batch = args.rollout_micro_batch_size or max(1, args.batch_size * gpu_count)
    samples_per_prompt = 1 if args.algorithm == "ppo" else max(2, args.group_size)
    estimator = ESTIMATOR_BY_ALGORITHM[args.algorithm]
    max_total_len = int(args.max_prompt_length) + int(args.max_response_length)
    vllm_generate_batch = args.vllm_generate_batch_size or rollout_batch

    command = [sys.executable, "-m", "openrlhf.cli.train_ppo_ray"]
    add_flag(command, "--ref.num_nodes", 1)
    add_flag(command, "--ref.num_gpus_per_node", gpu_count)
    add_flag(command, "--actor.num_nodes", 1)
    add_flag(command, "--actor.num_gpus_per_node", gpu_count)
    add_flag(command, "--vllm.num_engines", args.vllm_num_engines)
    add_flag(command, "--vllm.tensor_parallel_size", args.vllm_tensor_parallel_size)
    add_flag(command, "--vllm.sync_backend", args.vllm_sync_backend)
    add_flag(command, "--vllm.gpu_memory_utilization", args.vllm_gpu_memory_utilization)
    add_flag(command, "--actor.model_name_or_path", model_path)
    add_flag(command, "--ckpt.output_dir", str(output))
    add_flag(command, "--ckpt.path", str(ckpt_path))
    add_flag(command, "--data.prompt_dataset", train_data)
    add_flag(command, "--data.prompt_split", "train")
    add_flag(command, "--data.input_key", "prompt")
    add_flag(command, "--data.label_key", "answer")
    add_flag(command, "--data.max_len", max_total_len)
    add_flag(command, "--rollout.max_new_tokens", args.max_response_length)
    add_flag(command, "--rollout.temperature", args.temperature)
    add_flag(command, "--rollout.top_p", args.top_p)
    add_flag(command, "--train.max_epochs", args.epochs)
    add_flag(command, "--train.num_episodes", args.num_episodes)
    add_flag(command, "--train.batch_size", effective_train_batch)
    add_flag(command, "--train.micro_batch_size", max(1, args.batch_size))
    add_flag(command, "--rollout.batch_size", rollout_batch)
    add_flag(command, "--rollout.micro_batch_size", rollout_micro_batch)
    add_flag(command, "--rollout.vllm_generate_batch_size", vllm_generate_batch)
    add_flag(command, "--rollout.n_samples_per_prompt", samples_per_prompt)
    add_flag(command, "--actor.adam.lr", args.learning_rate)
    add_flag(command, "--critic.adam.lr", args.critic_learning_rate)
    add_flag(command, "--algo.kl.init_coef", args.kl_coef)
    add_flag(command, "--algo.kl.target", args.kl_target)
    add_flag(command, "--actor.eps_clip", args.clip_range)
    add_flag(command, "--critic.value_clip", args.value_clip)
    add_flag(command, "--algo.advantage.lambd", args.gae_lambda)
    add_flag(command, "--algo.advantage.gamma", args.gamma)
    add_flag(command, "--reward.clip_range", args.reward_clip_min)
    command.append(str(args.reward_clip_max))
    add_flag(command, "--ds.zero_stage", args.zero_stage)
    add_flag(command, "--ds.param_dtype", "bf16" if args.bf16 else "fp16")
    add_flag(command, "--ckpt.save_steps", args.save_steps)
    add_flag(command, "--logger.logging_steps", args.logging_steps)
    add_flag(command, "--data.dataloader_num_workers", args.num_workers)
    add_flag(command, "--data.max_samples", args.max_samples)
    add_flag(command, "--reward.remote_url", reward_url)
    add_flag(command, "--train.max_tokens_per_gpu", args.train_max_tokens_per_gpu)
    add_flag(command, "--rollout.max_tokens_per_gpu", args.rollout_max_tokens_per_gpu)
    if estimator:
        add_flag(command, "--algo.advantage.estimator", estimator)
    if args.algorithm == "ppo":
        add_flag(command, "--critic.model_name_or_path", model_path)
        add_flag(command, "--critic.num_nodes", 1)
        add_flag(command, "--critic.num_gpus_per_node", gpu_count)
    if args.reward_model_path:
        reward_model = resolve_model_path(args.reward_model_path)
        add_flag(command, "--reward.model_name_or_path", reward_model)
        add_flag(command, "--reward.num_nodes", 1)
        add_flag(command, "--reward.num_gpus_per_node", gpu_count)
        try:
            idx = command.index("--reward.remote_url")
            del command[idx : idx + 2]
        except ValueError:
            pass

    add_bool(command, "--data.apply_chat_template", True)
    add_bool(command, "--reward.normalize_enable", args.normalize_reward)
    add_bool(command, "--ds.packing_samples", args.packing_samples)
    add_bool(command, "--actor.gradient_checkpointing_enable", args.gradient_checkpointing)
    add_bool(command, "--ds.adam_offload", args.adam_offload)
    add_bool(command, "--ref.offload", args.ref_offload)
    add_bool(command, "--ckpt.save_hf", True)
    add_bool(command, "--data.disable_fast_tokenizer", args.disable_fast_tokenizer)
    add_bool(command, "--train.colocate_actor_ref", True)
    add_bool(command, "--train.colocate_all", args.colocate_all)
    add_bool(command, "--ds.enable_sleep", args.ds_enable_sleep)
    add_bool(command, "--vllm.enable_sleep", args.vllm_enable_sleep)
    add_bool(command, "--vllm.enforce_eager", args.vllm_enforce_eager)
    if args.flash_attn:
        add_flag(command, "--ds.attn_implementation", "flash_attention_2")
    return command


def maybe_start_ray(gpu_count: int) -> None:
    log("Starting local Ray head")
    subprocess.run(["ray", "stop", "--force"], check=False)
    ray_tmp = os.environ.get("OPENRLHF_RAY_TEMP_DIR", "/tmp/ray-openrlhf-local")
    subprocess.run(
        [
            "ray",
            "start",
            "--head",
            "--node-ip-address",
            "127.0.0.1",
            "--include-dashboard=false",
            "--temp-dir",
            ray_tmp,
            "--num-gpus",
            str(gpu_count),
            "--disable-usage-stats",
        ],
        check=True,
    )


def main() -> int:
    args = parse_args()
    setup_environment()
    os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    configure_ray_environment(args.ray_dashboard)
    install_vllm_text_only_patch(args.vllm_text_only_patch)
    ensure_runtime_dependencies()

    model_path = resolve_model_path(args.model_path)
    train_data = prepare_train_data(args)
    resume_from = receive_resume_checkpoint(args.resume_from)
    if resume_from:
        log(f"Resume checkpoint received/resolved: {resume_from}")

    command = build_openrlhf_command(args, model_path, train_data)
    if args.ray_start:
        maybe_start_ray(verified_gpu_count())

    try:
        log("Running OpenRLHF command:")
        log(" ".join(command))
        subprocess.run(command, check=True)
        write_result(args.output_dir, status="success")
        return 0
    except Exception as exc:
        log(f"OpenRLHF training failed: {exc}")
        write_result(args.output_dir, status="failed", error=str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
