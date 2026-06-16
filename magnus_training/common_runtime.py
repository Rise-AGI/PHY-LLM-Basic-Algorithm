from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def runtime_dir() -> Path:
    return Path(__file__).resolve().parent


def setup_environment() -> None:
    defaults = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPYCACHEPREFIX": "/tmp/.pycache",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "600",
        "NCCL_DEBUG": "WARN",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_P2P_DISABLE": "1",
        "NCCL_IB_DISABLE": "1",
        "NCCL_NET": "Socket",
        "NCCL_NET_GDR_LEVEL": "0",
        "NCCL_SOCKET_IFNAME": "^docker,lo,virbr",
        "NCCL_ALGO": "Ring",
        "NCCL_PROTO": "Simple",
        "NCCL_MIN_NCHANNELS": "2",
        "NCCL_SOCKET_NTHREADS": "4",
        "NCCL_NTHREADS": "512",
        "NCCL_BUFFSIZE": "4194304",
        "NCCL_NCHANNELS_PER_PEER": "8",
        "VLLM_DISABLE_PYNCCL": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "OPENRLHF_VLLM_DISABLE_CUSTOM_ALL_REDUCE": "1",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    os.environ.setdefault("MAGNUS_METRICS_DIR", "/magnus/workspace/metrics")
    Path(os.environ["MAGNUS_METRICS_DIR"]).mkdir(parents=True, exist_ok=True)


def ensure_dependencies() -> None:
    missing = []
    for pkg in ("torch", "transformers", "datasets", "accelerate"):
        try:
            __import__(pkg)
        except Exception:
            missing.append(pkg)
    if not missing:
        log("Dependencies ready")
        return
    log("Installing missing dependencies: " + " ".join(missing))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-i",
            "https://pypi.tuna.tsinghua.edu.cn/simple",
            *missing,
        ],
        check=True,
    )


def resolve_model_path(model_path: str) -> str:
    if (Path(model_path) / "config.json").exists():
        log(f"Using local model path: {model_path}")
        return model_path
    if model_path.startswith("/"):
        raise FileNotFoundError(f"Local model path does not exist: {model_path}")

    log(f"Local model not found, trying ModelScope: {model_path}")
    try:
        from modelscope import snapshot_download
    except Exception:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "-i",
                "https://pypi.tuna.tsinghua.edu.cn/simple",
                "modelscope",
            ],
            check=True,
        )
        from modelscope import snapshot_download

    actual = snapshot_download(model_path, cache_dir="/tmp/models")
    log(f"ModelScope download complete: {actual}")
    return actual


def receive_resume_checkpoint(resume_from: str | None) -> str | None:
    if not resume_from:
        return None
    token = resume_from.strip()
    if not token:
        return None
    if token.startswith("magnus-secret:"):
        out = "/tmp/resume_ckpt"
        log("Receiving checkpoint from Magnus secret")
        subprocess.run(["magnus", "receive", token, "-o", out], check=True)
        return out
    return token


def receive_file_secret(secret: str | None, output_name: str | None, default_name: str) -> str | None:
    if not secret:
        return None
    token = secret.strip()
    if not token:
        return None
    name = (output_name or default_name).strip().replace("\\", "/").split("/")[-1]
    if not name or name in {".", ".."}:
        name = default_name
    out = Path("/tmp/magnus_uploads") / name
    out.parent.mkdir(parents=True, exist_ok=True)
    log(f"Receiving uploaded file secret to {out}")
    subprocess.run(["magnus", "receive", token, "-o", str(out)], check=True)
    return str(out)


def make_fake_sft_data(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fake = [
        {
            "instruction": f"题目{i}: 请计算 {i}+{i} 的结果。",
            "output": f"答案: {i+i}\n\n解答: {i}+{i}={i+i}。",
        }
        for i in range(30)
    ]
    path.write_text(json.dumps(fake, ensure_ascii=False), encoding="utf-8")
    return path


def make_fake_rlhf_data(path: Path, algorithm: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if algorithm in {"dpo", "orpo"}:
        fake = [
            {
                "instruction": f"题目{i}: 计算 {i}+{i}",
                "chosen": f"答案: {i+i}\n\n解答: {i}+{i}={i+i}",
                "rejected": f"错误答案: {i*i}",
            }
            for i in range(30)
        ]
    else:
        fake = [
            {"instruction": f"数学题{i}: 请计算 {i}+{i} 的结果并展示推导过程。"}
            for i in range(30)
        ]
    path.write_text(json.dumps(fake, ensure_ascii=False), encoding="utf-8")
    return path


def verified_gpu_count() -> int:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        total = torch.cuda.device_count()
        ok = 0
        for i in range(total):
            try:
                with torch.cuda.device(i):
                    t = torch.zeros(1, device="cuda")
                    del t
                torch.cuda.synchronize(i)
                ok += 1
            except Exception as exc:
                log(f"GPU {i} CUDA context failed: {exc}")
        torch.cuda.empty_cache()
        return ok
    except Exception as exc:
        log(f"CUDA verification failed: {exc}")
        return 0


def launcher(script_path: Path) -> list[str]:
    gpu_count = verified_gpu_count()
    log(f"Verified GPU count: {gpu_count}")
    if gpu_count <= 0:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        return [sys.executable, str(script_path)]
    if gpu_count == 1:
        return [sys.executable, str(script_path)]
    master_port = str(30000 + (os.getpid() % 20000))
    os.environ.setdefault("MASTER_PORT", master_port)
    return [
        "torchrun",
        f"--nproc_per_node={gpu_count}",
        f"--rdzv_endpoint=localhost:{master_port}",
        str(script_path),
    ]


def run_training(command: list[str]) -> None:
    log("Running: " + " ".join(command))
    subprocess.run(command, check=True)


def write_result(output_dir: str, *, status: str = "success", error: str | None = None) -> None:
    result_path = os.environ.get("MAGNUS_RESULT")
    if not result_path:
        return
    summary = {"status": status, "output_dir": output_dir}
    log_path = Path(output_dir) / "training_log.json"
    if log_path.exists():
        try:
            data = json.loads(log_path.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                summary["last_log"] = data[-1]
        except Exception as exc:
            summary["log_read_error"] = str(exc)
    if error:
        summary["error"] = error
    Path(result_path).write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")


def add_flag(command: list[str], name: str, value) -> None:
    if value is None:
        return
    command.extend([name, str(value)])


def add_bool(command: list[str], name: str, enabled: bool) -> None:
    if enabled:
        command.append(name)
