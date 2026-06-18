from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import re
import shlex
import socket
import subprocess
import sys
from collections import deque
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


DEFAULT_IMAGE_NOTE = "需要 sft-base:v5 或更新镜像。"
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
import sys


def _is_truthy(value: str | None) -> bool:
    return str(value or "").lower() in {"1", "true", "yes", "on"}


def _patch_log(message: str, *, worker: bool = False) -> None:
    verbose = _is_truthy(os.environ.get("OPENRLHF_PATCH_VERBOSE"))
    lower = message.lower()
    if not verbose and not any(key in lower for key in ("failed", "skipped", "失败", "跳过")):
        return
    print(message, flush=True)


def _patch_ray_node_start_timeout_source() -> None:
    timeout = os.environ.get("RAY_raylet_start_wait_time_s", "300")
    try:
        from pathlib import Path
        import importlib.util

        spec = importlib.util.find_spec("ray")
        if spec is None or not spec.submodule_search_locations:
            return
        node_path = Path(list(spec.submodule_search_locations)[0]) / "_private" / "node.py"
        source = node_path.read_text(encoding="utf-8")
        old = "raylet_start_wait_time_s = 30"
        new = (
            "raylet_start_wait_time_s = int(__import__(\"os\").environ.get("
            "\"RAY_raylet_start_wait_time_s\", \"300\"))"
        )
        if old in source:
            node_path.write_text(source.replace(old, new), encoding="utf-8")
            _patch_log(
                f"[补丁] Raylet 启动等待={timeout}s",
            )
    except Exception as exc:
        _patch_log(f"[补丁失败] Raylet 启动等待: {exc}")


def _enabled() -> bool:
    return os.environ.get("OPENRLHF_VLLM_TEXT_ONLY_PATCH", "1").lower() not in {
        "0",
        "false",
        "no",
    }


def _ray_control_process() -> bool:
    argv = " ".join(sys.argv)
    control_markers = (
        "/bin/ray",
        "\\bin\\ray",
        "ray/dashboard/",
        "ray\\dashboard\\",
        "ray/autoscaler/",
        "ray\\autoscaler\\",
        "ray/_private/log_monitor.py",
        "ray\\_private\\log_monitor.py",
        "ray/_private/runtime_env/agent/",
        "ray\\_private\\runtime_env\\agent\\",
    )
    return any(marker in argv for marker in control_markers)


def _ray_worker_process() -> bool:
    argv = " ".join(sys.argv)
    worker_markers = (
        "ray/_private/workers/setup_worker.py",
        "ray\\_private\\workers\\setup_worker.py",
        "ray/_private/workers/default_worker.py",
        "ray\\_private\\workers\\default_worker.py",
    )
    return any(marker in argv for marker in worker_markers)


def _patch_ray_init() -> None:
    if os.environ.get("OPENRLHF_RAY_PATCH", "1").lower() in {"0", "false", "no"}:
        return
    try:
        _patch_ray_node_start_timeout_source()
        import ray

        if getattr(ray.init, "_openrlhf_magnus_patched", False):
            return
        _original_ray_init = ray.init

        def _init_without_dashboard(*args, **kwargs):
            ray_address = os.environ.get("OPENRLHF_RAY_ADDRESS") or os.environ.get("RAY_ADDRESS")
            if ray_address and not args and "address" not in kwargs:
                kwargs["address"] = ray_address
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
        _patch_log("[补丁] Ray dashboard 已关闭", worker=_ray_worker_process())
    except Exception as exc:
        _patch_log(f"[补丁失败] Ray dashboard: {exc}")


def _patch_vllm_text_only() -> None:
    if getattr(_patch_vllm_text_only, "_done", False) or getattr(
        _patch_vllm_text_only, "_patching", False
    ):
        return
    _patch_vllm_text_only._patching = True
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
                disable_custom_ar = os.environ.get(
                    "OPENRLHF_VLLM_DISABLE_CUSTOM_ALL_REDUCE",
                    "1",
                ).lower() not in {"0", "false", "no"}
                if disable_custom_ar:
                    kwargs["disable_custom_all_reduce"] = True
                if kwargs.get("mamba_block_size") is None:
                    kwargs["mamba_block_size"] = int(kwargs.get("block_size") or 16)
                if kwargs.get("mamba_block_size") is not None:
                    kwargs["enable_prefix_caching"] = True
                if kwargs.get("mamba_cache_mode") in {None, "none"}:
                    kwargs["mamba_cache_mode"] = os.environ.get(
                        "OPENRLHF_VLLM_MAMBA_CACHE_MODE",
                        "align",
                    )
                super().__init__(*args, **kwargs)

        _arg_utils.AsyncEngineArgs = _TextOnlyAsyncEngineArgs
        vllm.AsyncEngineArgs = _TextOnlyAsyncEngineArgs

        def _patch_qwen35_text_weight_loader() -> None:
            try:
                from vllm.model_executor.models import qwen3_5

                model_cls = qwen3_5.Qwen3_5ForCausalLM
                if getattr(model_cls.load_weights, "_openrlhf_prefix_patched", False):
                    return

                original_load_weights = model_cls.load_weights

                def _normalize_text_checkpoint_names(weights):
                    for name, tensor in weights:
                        new_name = name
                        if name.startswith("language_model.lm_head."):
                            new_name = "lm_head." + name[len("language_model.lm_head.") :]
                        elif name.startswith("language_model.model."):
                            new_name = "model." + name[len("language_model.model.") :]
                        elif name.startswith("language_model."):
                            new_name = "model." + name[len("language_model.") :]
                        elif name.startswith("model.language_model.lm_head."):
                            new_name = "lm_head." + name[
                                len("model.language_model.lm_head.") :
                            ]
                        elif name.startswith("model.language_model.model."):
                            new_name = "model." + name[
                                len("model.language_model.model.") :
                            ]
                        elif name.startswith("model.language_model."):
                            new_name = "model." + name[len("model.language_model.") :]
                        yield new_name, tensor

                def _load_weights_with_prefix_fix(self, weights):
                    return original_load_weights(
                        self, _normalize_text_checkpoint_names(weights)
                    )

                _load_weights_with_prefix_fix._openrlhf_prefix_patched = True
                _load_weights_with_prefix_fix._openrlhf_original = original_load_weights
                model_cls.load_weights = _load_weights_with_prefix_fix
                _patch_log(
                    "[补丁] Qwen checkpoint 前缀修复已启用",
                    worker=_ray_worker_process(),
                )
            except Exception as exc:
                _patch_log(
                    f"[补丁跳过] Qwen checkpoint 前缀修复: {exc}",
                )

        try:
            from vllm.model_executor.models import ModelRegistry

            qwen35_text_models = {
                "Qwen3_5ForConditionalGeneration": (
                    "vllm.model_executor.models.qwen3_5:Qwen3_5ForCausalLM"
                ),
                "Qwen3_5MoeForConditionalGeneration": (
                    "vllm.model_executor.models.qwen3_5:Qwen3_5MoeForCausalLM"
                ),
            }
            for arch, model_cls in qwen35_text_models.items():
                ModelRegistry.register_model(arch, model_cls)
            _patch_qwen35_text_weight_loader()
            _patch_log(
                "[补丁] Qwen 文本模型注册已修复",
                worker=_ray_worker_process(),
            )
        except Exception as exc:
            _patch_log(
                f"[补丁跳过] Qwen 文本模型注册: {exc}",
            )

        try:
            import torch.distributed as dist
            from vllm.distributed.device_communicators.cuda_communicator import (
                CudaCommunicator,
            )

            if not getattr(CudaCommunicator.broadcast, "_openrlhf_fallback_patched", False):
                _original_cuda_broadcast = CudaCommunicator.broadcast

                def _broadcast_with_torch_fallback(self, tensor, src=0):
                    pynccl_comm = getattr(self, "pynccl_comm", None)
                    if pynccl_comm is not None and not getattr(pynccl_comm, "disabled", False):
                        return _original_cuda_broadcast(self, tensor, src)
                    dist.broadcast(tensor, self.ranks[src], self.device_group)
                    return tensor

                _broadcast_with_torch_fallback._openrlhf_fallback_patched = True
                CudaCommunicator.broadcast = _broadcast_with_torch_fallback
        except Exception as exc:
            _patch_log(
                f"[补丁跳过] CUDA communicator fallback: {exc}",
            )

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
            _patch_log(
                f"[补丁跳过] 多模态注册清理: {exc}",
            )

        try:
            from vllm.v1.core import kv_cache_utils as _kv_utils
            from dataclasses import replace as _dc_replace
            import math as _math

            if not getattr(
                _kv_utils.unify_kv_cache_spec_page_size, "_openrlhf_lcm_patched", False
            ):
                _original_unify = _kv_utils.unify_kv_cache_spec_page_size

                def _unify_with_lcm(kv_cache_spec):
                    page_sizes = {
                        layer.page_size_bytes for layer in kv_cache_spec.values()
                    }
                    if len(page_sizes) <= 1:
                        return kv_cache_spec
                    lcm_page = 0
                    for _ps in page_sizes:
                        lcm_page = _ps if lcm_page == 0 else (
                            lcm_page * _ps // _math.gcd(lcm_page, _ps)
                        )
                    max_page = max(page_sizes)
                    if lcm_page > max_page:
                        _patch_log(
                            "[补丁] 混合架构 KV page_size 无法用 max 整除，改用 LCM 统一: "
                            f"page_sizes={sorted(page_sizes)}, LCM={lcm_page}",
                        )
                    new_spec_map = {}
                    for _name, _spec in kv_cache_spec.items():
                        _lp = _spec.page_size_bytes
                        if _lp == lcm_page:
                            new_spec_map[_name] = _spec
                        else:
                            _ratio = lcm_page // _lp
                            new_spec_map[_name] = _dc_replace(
                                _spec, block_size=_spec.block_size * _ratio
                            )
                    return new_spec_map

                _unify_with_lcm._openrlhf_lcm_patched = True
                _unify_with_lcm._openrlhf_original = _original_unify
                _kv_utils.unify_kv_cache_spec_page_size = _unify_with_lcm
                _patch_log("[补丁] KV page_size LCM 统一策略已启用")
        except Exception as exc:
            _patch_log(f"[补丁跳过] KV page_size LCM 统一: {exc}")

        _patch_log(
            "[补丁] vLLM 文本模式已启用",
            worker=_ray_worker_process(),
        )
        _patch_vllm_text_only._done = True
    except Exception as exc:
        _patch_log(f"[补丁失败] vLLM 文本模式: {exc}")
    finally:
        _patch_vllm_text_only._patching = False


def _install_lazy_vllm_patch() -> None:
    if not _enabled():
        return
    if getattr(_install_lazy_vllm_patch, "_installed", False):
        return
    import builtins

    original_import = builtins.__import__

    def _import_with_vllm_patch(name, globals=None, locals=None, fromlist=(), level=0):
        module = original_import(name, globals, locals, fromlist, level)
        caller_name = ""
        if globals:
            caller_name = str(globals.get("__name__", ""))
        top_name = name.split(".", 1)[0]
        vllm_module = sys.modules.get("vllm")
        vllm_spec = getattr(vllm_module, "__spec__", None)
        vllm_initializing = bool(getattr(vllm_spec, "_initializing", False))
        if (
            top_name == "vllm"
            and not caller_name.startswith("vllm")
            and vllm_module is not None
            and not vllm_initializing
        ):
            _patch_vllm_text_only()
        return module

    _import_with_vllm_patch._openrlhf_original = original_import
    builtins.__import__ = _import_with_vllm_patch
    _install_lazy_vllm_patch._installed = True
    _patch_log(
        "[补丁] vLLM 文本模式延迟安装完成",
        worker=_ray_worker_process(),
    )


if not _ray_control_process():
    _install_lazy_vllm_patch()
    if not _ray_worker_process():
        _patch_ray_init()
'''


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Magnus RLHF 运行入口")
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
    p.add_argument("--raylet_start_wait_time_s", type=int, default=300)

    p.add_argument("--colocate_all", action="store_true")
    p.add_argument("--ds_enable_sleep", action="store_true")
    p.add_argument("--openrlhf_cli_style", choices=["v5"], default="v5")
    p.add_argument("--ray_start", action="store_true")
    p.add_argument("--no_ray_start", dest="ray_start", action="store_false")
    p.set_defaults(ray_start=False)
    p.add_argument("--ray_node_ip", default="auto")
    p.add_argument("--ray_head_port", type=int)
    p.add_argument("--ray_worker_port_count", type=int, default=1000)
    p.add_argument("--ray_object_store_memory_gb", type=int, default=16)
    p.add_argument("--ray_plasma_directory", default="/tmp")
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
            "缺少 RLHF 运行依赖: "
            + "; ".join(missing)
            + ". "
            + DEFAULT_IMAGE_NOTE
        )
    log("依赖检查通过")


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RAY_PID_PREFIX_RE = re.compile(
    r"^(?:\([\w.]+ pid=\d+\)\s*)+(?:ERROR|WARNING|INFO)\s+(?:\d{4}-)?\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+\[[\w./]+:\d+\]\s*"
)
RAY_ACTOR_PREFIX_RE = re.compile(r"^(?:\([\w.]+ pid=\d+\)\s*)+")
VLLM_LOG_PREFIX_RE = re.compile(r"^(?:ERROR|WARNING|INFO)\s+(?:\d{4}-)?\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+\[[\w./]+:\d+\]\s*")
PY_ERROR_ROOT_RE = re.compile(r"^([A-Za-z_]\w*(?:Error|Exception|Warning)):\s*(.+)")
PY_RAISE_RE = re.compile(r"^\s*raise\s+([A-Za-z_]\w*(?:Error|Exception))\(([^)]*)\)")
NOISY_RUNTIME_PATTERNS = (
    re.compile(r"Parameter .* not found in params_dict, skip loading"),
    re.compile(r"Loading safetensors checkpoint shards:"),
    re.compile(r"FutureWarning|SyntaxWarning|DeprecationWarning"),
    re.compile(r"torch\.cuda\.memory\._set_allocator_settings"),
    re.compile(r"WARNING: Overriding HOME environment variable"),
    re.compile(r"Model architecture Qwen3_5.*already registered"),
    re.compile(r"Unknown vLLM environment variable detected"),
    re.compile(r"ray\.init dashboard disabled"),
    re.compile(r"lazy text-only vLLM patch installed"),
    re.compile(r"Injected <class .*WorkerWrap"),
    re.compile(r"Missing `shared_worker_lock` argument"),
    re.compile(r"TORCH_NCCL_AVOID_RECORD_STREAMS"),
    re.compile(r"SymmMemCommunicator"),
    re.compile(r"Using \[\] all-reduce backends"),
    re.compile(r"rank \d+ in world size"),
    re.compile(r"Using address .*RAY_ADDRESS"),
    re.compile(r"Connecting to existing Ray cluster"),
    re.compile(r"Connected to Ray cluster"),
    re.compile(r"Using the existing placement group"),
    re.compile(r"Auto-prefetch is disabled"),
    re.compile(r"Exception in thread Thread-1 \(_report_usage_worker\)"),
    re.compile(r"vllm/usage/usage_lib\.py"),
    re.compile(r"cpuinfo\.py"),
    re.compile(r"json\.decoder\.JSONDecodeError"),
    re.compile(r"^\s*$"),
    re.compile(r"\[state-dump\]"),
    re.compile(r"Node with actor is already dead"),
    re.compile(r"KillActor RPC failed"),
    re.compile(r"Disconnecting worker, graceful=true"),
    re.compile(r"Disconnecting client due to connection error"),
    re.compile(r"Reporting worker exit"),
    re.compile(r"Worker exits because there was an exception"),
    re.compile(r"Formatted creation task exception"),
    re.compile(r"Exception raised in creation task"),
    re.compile(r"The actor died because of an error raised"),
    re.compile(r"CreationTaskError"),
    re.compile(r"Unintentional worker failures have been reported"),
    re.compile(r"gcs_actor_manager\.cc"),
    re.compile(r"node_manager\.cc:\d+: Disconnecting"),
    re.compile(r"store\.cc:\d+: Disconnecting client"),
    re.compile(r"raylet.*node_manager\.cc:\d+:.*has_creation_task_exception"),
    re.compile(r"^raise self\._exception"),
    re.compile(r"^\s*raise\s+\w*(?:Error|Exception)\($"),
    re.compile(r"^ray\.exceptions\.ActorDiedError"),
    re.compile(r"During handling of the above exception"),
    re.compile(r"Engine core initialization failed"),
    re.compile(r"Failed core proc\(s\)"),
    re.compile(r"^values, debugger_breakpoint = worker\.get_objects"),
    re.compile(r"Actor .* died because of an error raised"),
)
IMPORTANT_RUNTIME_PATTERNS = re.compile(
    r"("
    r"Traceback|RuntimeError|ValueError|AttributeError|NotImplementedError|"
    r"RayTaskError|OOM|out of memory|"
    r"reward|kl|loss|episode|global_step|saving|checkpoint|"
    r"Resolved architecture|Initializing a V1 LLM engine|"
    r"Starting to load model|FLASH_ATTN|FlashAttention|"
    r"Filesystem type for checkpoints|Loading weights took|Model loading took|"
    r"Dynamo bytecode transform time|Cache the graph|"
    r"No available shared memory broadcast block"
    r")",
    re.IGNORECASE,
)


def compact_runtime_line(line: str) -> str:
    line = ANSI_RE.sub("", line).strip()
    line = RAY_PID_PREFIX_RE.sub("", line)
    line = RAY_ACTOR_PREFIX_RE.sub("", line)
    line = VLLM_LOG_PREFIX_RE.sub("", line)
    line = line.strip()
    root_match = PY_ERROR_ROOT_RE.match(line)
    if root_match:
        return f"错误根因: {root_match.group(1)}: {root_match.group(2)}"
    raise_match = PY_RAISE_RE.match(line)
    if raise_match:
        return f"错误根因: {raise_match.group(1)}: {raise_match.group(2)}"
    if "Following weights were not initialized from checkpoint" in line:
        names = re.findall(r"'([^']+)'", line)
        if names:
            samples = ", ".join(names[:10])
            prefixes = sorted({".".join(name.split(".")[:2]) for name in names})
            return (
                f"ValueError: checkpoint 有 {len(names)} 个权重未初始化；"
                f"样例: {samples}; 前缀: {', '.join(prefixes[:8])}"
            )
    if "Loading custom `reward_func" in line:
        return "奖励函数: 内置规则奖励"
    if "Resolved architecture:" in line:
        match = re.search(r"Resolved architecture:\s*([^ ]+)", line)
        return f"vLLM模型架构: {match.group(1) if match else line.rsplit(':', 1)[-1].strip()}"
    if "Initializing a V1 LLM engine" in line:
        tp = re.search(r"tensor_parallel_size=(\d+)", line)
        max_len = re.search(r"max_seq_len=(\d+)", line)
        prefix = re.search(r"enable_prefix_caching=(True|False)", line)
        return (
            "vLLM引擎初始化: "
            f"TP={tp.group(1) if tp else '?'}, "
            f"max_len={max_len.group(1) if max_len else '?'}, "
            f"prefix_cache={prefix.group(1) if prefix else '?'}"
        )
    if "Starting to load model" in line:
        return "vLLM开始加载模型"
    if "Using FLASH_ATTN attention backend" in line or "Using FlashAttention version" in line:
        return "注意力后端: FlashAttention"
    if "Using Triton/FLA GDN prefill kernel" in line:
        return "GDN prefill: Triton/FLA"
    if "Filesystem type for checkpoints:" in line:
        fs = re.search(r"Filesystem type for checkpoints:\s*([^\.]+)", line)
        size = re.search(r"Checkpoint size:\s*([^\.]+\.[0-9]+ GiB|[^\.]+GiB)", line)
        ram = re.search(r"Available RAM:\s*([^\.]+\.[0-9]+ GiB|[^\.]+GiB)", line)
        parts = []
        if fs:
            parts.append(f"文件系统={fs.group(1).strip()}")
        if size:
            parts.append(f"checkpoint={size.group(1).strip()}")
        if ram:
            parts.append(f"可用RAM={ram.group(1).strip()}")
        return "checkpoint读取: " + ", ".join(parts)
    if "Loading weights took" in line:
        match = re.search(r"Loading weights took\s*([0-9.]+)\s*seconds", line)
        return f"权重加载完成: {match.group(1) if match else '?'}s"
    if "Model loading took" in line:
        match = re.search(r"Model loading took\s*([0-9.]+ GiB).*?([0-9.]+)\s*seconds", line)
        if match:
            return f"模型加载完成: 显存={match.group(1)}, 用时={match.group(2)}s"
        return "模型加载完成"
    if "Using cache directory:" in line and "torch.compile" in line:
        return "vLLM编译缓存已就绪"
    if "Dynamo bytecode transform time:" in line:
        match = re.search(r"Dynamo bytecode transform time:\s*([0-9.]+\s*s)", line)
        return f"torch.compile字节码转换: {match.group(1) if match else '?'}"
    if "Cache the graph of compile range" in line:
        match = re.search(r"compile range\s*\(([^)]+)\)", line)
        return f"torch.compile图缓存: {match.group(1) if match else '完成'}"
    if "No available shared memory broadcast block found in 60 seconds" in line:
        return "vLLM等待共享内存广播: 60s"
    if len(line) > 2200:
        return line[:2200] + " ... [截断]"
    return line


def useful_runtime_line(raw_line: str) -> str | None:
    raw = ANSI_RE.sub("", raw_line).strip()
    if not raw:
        return None
    if raw.startswith("namespace("):
        return None
    line = compact_runtime_line(raw)
    if any(pattern.search(line) for pattern in NOISY_RUNTIME_PATTERNS):
        return None
    if line != raw:
        return line
    if IMPORTANT_RUNTIME_PATTERNS.search(line):
        return line
    if line.startswith("[补丁"):
        return line
    return None


def compact_ray_start_output(output: str) -> str:
    useful: list[str] = []
    keep_markers = (
        "Local node IP:",
        "Ray runtime started",
        "object_store_memory",
        "Traceback",
        "ERROR",
        "Exception",
        "Failed",
    )
    for raw_line in output.splitlines():
        line = ANSI_RE.sub("", raw_line).strip()
        if not line:
            continue
        if "Local node IP:" in line:
            useful.append("Ray节点IP: " + line.rsplit(":", 1)[-1].strip())
        elif "Ray runtime started" in line:
            useful.append("Ray启动完成")
        elif "object_store_memory" in line:
            useful.append(line)
        elif any(marker in line for marker in keep_markers):
            useful.append(line)
    return "\n".join(useful[-20:])


def log_model_metadata(model_path: str) -> None:
    model_dir = Path(model_path)
    config_path = model_dir / "config.json"
    if not config_path.exists():
        log(f"模型配置不存在: {config_path}")
        return
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log(f"读取模型配置失败: {config_path}: {exc}")
        return

    text_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    source = text_config or config
    summary_keys = (
        "model_type",
        "architectures",
        "torch_dtype",
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "tie_word_embeddings",
    )
    summary = {key: source.get(key, config.get(key)) for key in summary_keys}
    log(
        "模型配置: "
        + ", ".join(f"{key}={value}" for key, value in summary.items() if value is not None)
    )

    index_candidates = (
        model_dir / "model.safetensors.index.json",
        model_dir / "pytorch_model.bin.index.json",
    )
    for index_path in index_candidates:
        if not index_path.exists():
            continue
        try:
            index_data = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index_data.get("weight_map", {})
            keys = sorted(weight_map)[:16]
            prefixes = sorted({".".join(key.split(".")[:2]) for key in keys})
            log(
                "checkpoint索引: "
                f"file={index_path.name}, weights={len(weight_map)}, "
                f"sample_prefixes={prefixes}, sample_keys={keys[:6]}"
            )
            return
        except Exception as exc:
            log(f"读取checkpoint索引失败: {index_path}: {exc}")
            return
    shards = sorted(path.name for path in model_dir.glob("*.safetensors"))[:6]
    if shards:
        log(f"checkpoint分片: {shards}")


def log_openrlhf_summary(args: argparse.Namespace, model_path: str, train_data: str, command: list[str]) -> None:
    gpu_count = verified_gpu_count()
    max_total_len = int(args.max_prompt_length) + int(args.max_response_length)
    effective_train_batch = args.train_batch_size or max(
        1, args.batch_size * gpu_count * max(1, args.group_size)
    )
    rollout_batch = args.rollout_batch_size or effective_train_batch
    rollout_micro_batch = args.rollout_micro_batch_size or max(1, args.batch_size * gpu_count)
    vllm_generate_batch = args.vllm_generate_batch_size or rollout_batch
    log(
        "训练启动: "
        f"算法={args.algorithm}, 优势估计={ESTIMATOR_BY_ALGORITHM[args.algorithm]}, "
        f"模型={model_path}, 数据={train_data}, 输出={args.output_dir}"
    )
    log(
        "批次与长度: "
        f"训练批={effective_train_batch}, 训练微批={args.batch_size}, "
        f"采样批={rollout_batch}, 采样微批={rollout_micro_batch}, "
        f"vLLM生成批={vllm_generate_batch}, 最大长度={max_total_len}"
    )
    log(
        "并行配置: "
        f"可见GPU={gpu_count}, vLLM引擎={args.vllm_num_engines}, "
        f"vLLM TP={args.vllm_tensor_parallel_size}, 同卡放置={args.colocate_all}, "
        f"prefix_cache=True, Ray预启动={args.ray_start}, "
        f"Ray worker端口={args.ray_worker_port_count}, vLLM eager={args.vllm_enforce_eager}"
    )
    if os.environ.get("OPENRLHF_VERBOSE_COMMAND", "0").lower() in {"1", "true", "yes"}:
        log("OpenRLHF命令: " + shlex.join(command))
    else:
        log("OpenRLHF命令已隐藏；设置 OPENRLHF_VERBOSE_COMMAND=1 可打印")


def run_openrlhf_command(command: list[str]) -> None:
    recent: deque[str] = deque(maxlen=120)
    repeat_counts: dict[str, int] = {}
    root_cause: str | None = None
    suppress_usage_trace = 0
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    if proc.stdout is not None:
        for raw in proc.stdout:
            for part in raw.replace("\r", "\n").splitlines():
                if "Exception in thread Thread-1 (_report_usage_worker)" in part:
                    suppress_usage_trace = 40
                    continue
                if suppress_usage_trace > 0:
                    suppress_usage_trace -= 1
                    continue
                useful = useful_runtime_line(part)
                if useful is None:
                    continue
                if useful.startswith("错误根因: "):
                    if root_cause is None:
                        root_cause = useful[len("错误根因: "):]
                        print(f"[训练] ★ {useful}", flush=True)
                        recent.append(f"★ {useful}")
                    continue
                count = repeat_counts.get(useful, 0) + 1
                repeat_counts[useful] = count
                if useful.startswith("vLLM等待共享内存广播"):
                    if count == 1 or count % 50 == 0:
                        useful = f"{useful}（已累计{count}次，约{count}分钟）" if count > 1 else useful
                        recent.append(useful)
                        print(f"[训练] {useful}", flush=True)
                    continue
                if count > 1 and count % 10 != 0:
                    continue
                if count >= 10:
                    useful = f"{useful}（重复{count}次）"
                recent.append(useful)
                print(f"[训练] {useful}", flush=True)
    return_code = proc.wait()
    if return_code != 0:
        if root_cause:
            log(f"★ 训练失败根因: {root_cause}")
        if recent:
            log("最近关键训练日志:\n" + "\n".join(list(recent)[-30:]))
        raise RuntimeError(f"OpenRLHF退出码={return_code}" + (f"，根因: {root_cause}" if root_cause else ""))


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
        raise ValueError("必须提供 train_data 或 train_data_secret")
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Training data does not exist: {source}")
    out = Path("/tmp/openrlhf_prompt_data.jsonl")
    count = convert_prompt_data(source_path, out, args.max_samples)
    log(f"训练数据: {count} 条 -> {out}")
    return str(out)


def install_vllm_text_only_patch(enabled: bool) -> None:
    if not enabled:
        os.environ["OPENRLHF_VLLM_TEXT_ONLY_PATCH"] = "0"
        log("vLLM文本修复: 关闭")
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
    log(f"vLLM文本修复: 已安装 {patch_dir}")


def configure_ray_environment(enable_dashboard: bool, raylet_start_wait_time_s: int) -> None:
    job_id = os.environ.get("MAGNUS_JOB_ID") or "local"
    ray_tmp = Path("/tmp") / f"ray-openrlhf-{job_id}"
    ray_tmp.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    os.environ.setdefault("RAY_DEDUP_LOGS", "1")
    os.environ["RAY_raylet_start_wait_time_s"] = str(max(30, raylet_start_wait_time_s))
    os.environ["OPENRLHF_RAY_PATCH"] = "0" if enable_dashboard else "1"
    os.environ["OPENRLHF_RAY_TEMP_DIR"] = str(ray_tmp)
    if enable_dashboard:
        log("Ray dashboard: 开启")
    else:
        log(f"Ray dashboard: 关闭，临时目录={ray_tmp}")


def build_openrlhf_command(args: argparse.Namespace, model_path: str, train_data: str) -> list[str]:
    gpu_count = verified_gpu_count()
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    log(
        "CUDA可见性: "
        f"实际={gpu_count}, 要求={args.required_gpu_count or '未设置'}, "
        f"CUDA_VISIBLE_DEVICES={visible_devices}"
    )
    if gpu_count <= 0:
        raise RuntimeError("OpenRLHF/vLLM 需要 CUDA GPU")
    if args.required_gpu_count and gpu_count < args.required_gpu_count:
        raise RuntimeError(
            f"当前可用 GPU={gpu_count}，PPO 27B 要求 {args.required_gpu_count} 张 A100。"
            "不要把 vLLM TP 改成 3。"
        )
    vllm_gpu_slots = args.vllm_num_engines * args.vllm_tensor_parallel_size
    if vllm_gpu_slots > gpu_count:
        raise ValueError(
            "vLLM engine*TP 不能超过可见 GPU 数: "
            f"{args.vllm_num_engines} * {args.vllm_tensor_parallel_size} > {gpu_count}。"
        )
    if args.colocate_all and vllm_gpu_slots != gpu_count:
        raise ValueError(
            "colocate_all=True 时，vLLM engine*TP 必须等于 GPU 数: "
            f"{vllm_gpu_slots} != {gpu_count}"
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
    add_bool(command, "--vllm.enable_prefix_caching", True)
    add_bool(command, "--vllm.enable_sleep", args.vllm_enable_sleep)
    add_bool(command, "--vllm.enforce_eager", args.vllm_enforce_eager)
    if args.flash_attn:
        add_flag(command, "--ds.attn_implementation", "flash_attention_2")
    return command


def dump_ray_logs(ray_tmp: str | None) -> None:
    if not ray_tmp:
        return
    logs_dir = Path(ray_tmp) / "session_latest" / "logs"
    if not logs_dir.exists():
        log(f"Ray日志不存在: {logs_dir}")
        return
    emitted = False
    for name in (
        "gcs_server.err",
        "gcs_server.out",
        "raylet.err",
        "raylet.out",
        "monitor.err",
        "monitor.out",
    ):
        path = logs_dir / name
        if not path.exists():
            continue
        try:
            raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            useful = []
            for raw_line in raw_lines[-1200:]:
                line = useful_runtime_line(raw_line)
                if line is not None:
                    useful.append(line)
            if useful:
                emitted = True
                log(f"Ray关键日志: {name}\n" + "\n".join(useful[-35:]))
        except Exception as exc:
            log(f"读取Ray日志失败: {path}: {exc}")
    if not emitted:
        log(f"Ray日志无关键错误: {logs_dir}")


def normalize_ray_node_ip(value: str | None) -> str | None:
    if value is None:
        return None
    node_ip = str(value).strip()
    if not node_ip or node_ip.lower() in {"auto", "default", "none", "null"}:
        return None
    return node_ip


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def extract_ray_addresses(output: str, fallback_port: int) -> list[str]:
    candidates: list[str] = []
    patterns = [
        r"--address[= ]'([^']+)'",
        r'--address[= ]"([^"]+)"',
        r"ray\.init\(address=['\"]([^'\"]+)['\"]",
        r"([0-9]{1,3}(?:\.[0-9]{1,3}){3}:\d+)",
    ]
    for pattern in patterns:
        candidates.extend(re.findall(pattern, output))

    for line in output.splitlines():
        if "Local node IP:" not in line:
            continue
        host = line.rsplit(":", 1)[-1].strip()
        if host:
            candidates.append(f"{host}:{fallback_port}")
    return unique_preserve_order(candidates)


def local_ray_address_candidates(base_port: int, explicit_node_ip: str | None) -> list[str]:
    candidates: list[str] = []
    if explicit_node_ip:
        candidates.append(f"{explicit_node_ip}:{base_port}")
    candidates.extend([f"127.0.0.1:{base_port}", f"localhost:{base_port}"])

    try:
        hostname = socket.gethostname()
        for host in socket.gethostbyname_ex(hostname)[2]:
            if host and not host.startswith("127.") and ":" not in host:
                candidates.append(f"{host}:{base_port}")
    except Exception:
        pass

    try:
        proc = subprocess.run(
            ["hostname", "-I"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0:
            for host in proc.stdout.split():
                if host and not host.startswith("127.") and ":" not in host:
                    candidates.append(f"{host}:{base_port}")
    except Exception:
        pass

    candidates.append("auto")
    return unique_preserve_order(candidates)


def wait_for_ray(address: str, timeout_s: int) -> tuple[bool, str]:
    env = os.environ.copy()
    env["RAY_ADDRESS"] = address
    probe = (
        "import ray; "
        f"ray.init(address={address!r}, ignore_reinit_error=True, include_dashboard=False); "
        "print(ray.cluster_resources()); "
        "ray.shutdown()"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            env=env,
            timeout=max(8, min(20, timeout_s)),
        )
    except subprocess.TimeoutExpired as exc:
        return False, f"probe timeout after {exc.timeout}s"
    if proc.returncode == 0:
        log(f"Ray已连接: {address}")
        return True, proc.stdout.strip()
    return False, (proc.stderr or proc.stdout or "").strip()


def wait_for_any_ray(candidates: list[str], timeout_s: int) -> str:
    import time

    deadline_s = max(30, timeout_s)
    started = time.monotonic()
    last_errors: dict[str, str] = {}
    candidates = unique_preserve_order(candidates)
    log("Ray候选地址: " + ", ".join(candidates))
    while time.monotonic() - started < deadline_s:
        for address in candidates:
            ok, detail = wait_for_ray(address, timeout_s=10)
            if ok:
                return address
            last_errors[address] = detail[-800:]
            if "No available ports" in detail:
                raise RuntimeError(
                    "Ray可连接，但 worker 端口不足。增加 --ray_worker_port_count。"
                )
        time.sleep(3)
    details = "\n".join(f"{addr}: {err}" for addr, err in last_errors.items())
    raise RuntimeError(f"等待Ray超时。候选地址失败信息:\n{details}")


def maybe_start_ray(args: argparse.Namespace, gpu_count: int) -> None:
    log("启动本地Ray")
    subprocess.run(["ray", "stop", "--force"], check=False)
    ray_tmp = os.environ.get("OPENRLHF_RAY_TEMP_DIR", "/tmp/ray-openrlhf-local")
    ray_tmp_path = Path(ray_tmp)
    ray_tmp_path.mkdir(parents=True, exist_ok=True)
    spill_dir = ray_tmp_path / "spill"
    spill_dir.mkdir(parents=True, exist_ok=True)
    base_port = args.ray_head_port or (20000 + (os.getpid() % 20000))
    worker_ports = max(32, args.ray_worker_port_count)
    node_ip = normalize_ray_node_ip(args.ray_node_ip)
    object_store_memory = max(1, args.ray_object_store_memory_gb) * 1024**3
    plasma_dir = args.ray_plasma_directory or "/tmp"
    command = [
        "ray",
        "start",
        "--head",
        "--port",
        str(base_port),
        "--min-worker-port",
        str(base_port + 100),
        "--max-worker-port",
        str(base_port + 100 + worker_ports - 1),
        "--include-dashboard=false",
        "--temp-dir",
        ray_tmp,
        "--plasma-directory",
        plasma_dir,
        "--object-spilling-directory",
        str(spill_dir),
        "--object-store-memory",
        str(object_store_memory),
        "--num-gpus",
        str(gpu_count),
        "--disable-usage-stats",
    ]
    if node_ip:
        command[3:3] = ["--node-ip-address", node_ip]
    log("Ray启动命令: " + " ".join(command))
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )
    start_output = "\n".join(part for part in (proc.stdout, proc.stderr) if part)
    compact_start_output = compact_ray_start_output(start_output)
    if compact_start_output:
        log("Ray启动输出:\n" + compact_start_output)
    if proc.returncode != 0:
        dump_ray_logs(ray_tmp)
        raise RuntimeError(f"ray start 退出码={proc.returncode}")
    candidates = (
        extract_ray_addresses(start_output, base_port)
        + local_ray_address_candidates(base_port, node_ip)
    )
    address = wait_for_any_ray(candidates, args.raylet_start_wait_time_s)
    os.environ["RAY_ADDRESS"] = address
    os.environ["OPENRLHF_RAY_ADDRESS"] = address
    log(f"OpenRLHF连接Ray: {address}")


def main() -> int:
    args = parse_args()
    setup_environment()
    env_keys = (
        "NCCL_IB_DISABLE",
        "NCCL_P2P_DISABLE",
        "NCCL_NET",
        "NCCL_SOCKET_IFNAME",
        "VLLM_DISABLE_PYNCCL",
        "VLLM_WORKER_MULTIPROC_METHOD",
        "OPENRLHF_VLLM_DISABLE_CUSTOM_ALL_REDUCE",
    )
    log("分布式环境: " + ", ".join(f"{key}={os.environ.get(key)}" for key in env_keys))
    os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    configure_ray_environment(args.ray_dashboard, args.raylet_start_wait_time_s)
    install_vllm_text_only_patch(args.vllm_text_only_patch)
    ensure_runtime_dependencies()

    model_path = resolve_model_path(args.model_path)
    log_model_metadata(model_path)
    train_data = prepare_train_data(args)
    resume_from = receive_resume_checkpoint(args.resume_from)
    if resume_from:
        log(f"恢复checkpoint: {resume_from}")

    command = build_openrlhf_command(args, model_path, train_data)
    try:
        if args.ray_start:
            maybe_start_ray(args, verified_gpu_count())
        else:
            os.environ.pop("OPENRLHF_RAY_ADDRESS", None)
        log_openrlhf_summary(args, model_path, train_data, command)
        run_openrlhf_command(command)
        write_result(args.output_dir, status="success")
        return 0
    except Exception as exc:
        log(f"训练失败: {type(exc).__name__}: {exc}")
        dump_ray_logs(os.environ.get("OPENRLHF_RAY_TEMP_DIR"))
        write_result(args.output_dir, status="failed", error=str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
