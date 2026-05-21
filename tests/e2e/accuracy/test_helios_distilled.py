from __future__ import annotations

import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from tests.e2e.accuracy.helpers import reset_artifact_dir
from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_NAME = "BestWishYsh/Helios-Distilled"
PROMPT = "A cat wearing sunglasses dances on a beach at sunset, cinematic lighting."
HEIGHT = 384
WIDTH = 640
NUM_FRAMES = 99
FPS = 16
SEED = 42
GUIDANCE_SCALE = 1.0
PYRAMID_NUM_INFERENCE_STEPS = (1, 1, 1)
MIN_PSNR_DB = 35.0
MAX_MAE = 3.0
MAX_TIME_RATIO = 1.03
WARMUP_REQUESTS = 1
WARMUP_NUM_INFERENCE_STEPS = 1


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _open_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _wait_for_service(base_url: str, process: subprocess.Popen, *, timeout_s: float = 1200.0) -> None:
    import requests

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited with code {process.returncode} before becoming ready.")
        try:
            response = requests.get(f"{base_url}/health", timeout=1)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"Server at {base_url} did not become ready within {timeout_s:.0f}s.")


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except Exception:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def _extract_frames(video_path: Path, output_dir: Path) -> list[Image.Image]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_pattern = output_dir / "frame_%05d.png"
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vsync",
            "0",
            str(frame_pattern),
        ],
        cwd=output_dir,
    )
    frame_paths = sorted(output_dir.glob("frame_*.png"))
    if not frame_paths:
        raise RuntimeError(f"No frames extracted from {video_path}")
    return [Image.open(path).convert("RGB").copy() for path in frame_paths]


def _image_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    diff = reference.astype(np.float64) - candidate.astype(np.float64)
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    psnr_db = float("inf") if mse == 0.0 else 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)
    ref_flat = reference.astype(np.float64).reshape(-1)
    cand_flat = candidate.astype(np.float64).reshape(-1)
    ref_norm = float(np.linalg.norm(ref_flat))
    cand_norm = float(np.linalg.norm(cand_flat))
    if ref_norm == 0.0 and cand_norm == 0.0:
        cosine = 1.0
    else:
        cosine = float(np.dot(ref_flat, cand_flat) / (ref_norm * cand_norm))
    return {
        "mae": mae,
        "mse": mse,
        "max_abs": max_abs,
        "psnr_db": psnr_db,
        "cosine_similarity": cosine,
    }


def _video_metrics(reference_frames: list[Image.Image], candidate_frames: list[Image.Image]) -> dict[str, Any]:
    if len(reference_frames) != len(candidate_frames):
        raise ValueError(f"Frame count mismatch: reference={len(reference_frames)} candidate={len(candidate_frames)}")

    ref_stack = np.stack([np.asarray(frame, dtype=np.uint8) for frame in reference_frames], axis=0)
    cand_stack = np.stack([np.asarray(frame, dtype=np.uint8) for frame in candidate_frames], axis=0)
    mid = len(reference_frames) // 2
    return {
        "num_frames": len(reference_frames),
        "frame0_metrics": _image_metrics(ref_stack[0], cand_stack[0]),
        "mid_frame_index": mid,
        "mid_frame_metrics": _image_metrics(ref_stack[mid], cand_stack[mid]),
        "all_frames_metrics": _image_metrics(ref_stack, cand_stack),
    }


def _reference_repo(output_root: Path) -> Path:
    configured = os.environ.get("VLLM_HELIOS_REFERENCE_REPO")
    if configured:
        reference = Path(configured).expanduser().resolve()
        if not reference.exists():
            raise FileNotFoundError(f"VLLM_HELIOS_REFERENCE_REPO does not exist: {reference}")
        return reference

    reference = output_root / "origin_main"
    if reference.exists():
        subprocess.run(["git", "worktree", "remove", "--force", str(reference)], cwd=str(REPO_ROOT), check=False)
        if reference.exists():
            shutil.rmtree(reference)
    _run(["git", "fetch", "origin", "main"], cwd=REPO_ROOT)
    _run(["git", "worktree", "add", "--detach", str(reference), "origin/main"], cwd=REPO_ROOT)
    return reference


def _server_env(repo: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    return env


def _benchmark_config() -> str:
    return json.dumps(
        [
            {
                "width": WIDTH,
                "height": HEIGHT,
                "num_inference_steps": 50,
                "num_frames": NUM_FRAMES,
                "fps": FPS,
                "guidance_scale": GUIDANCE_SCALE,
                "is_enable_stage2": True,
                "pyramid_num_stages": 3,
                "pyramid_num_inference_steps_list": list(PYRAMID_NUM_INFERENCE_STEPS),
                "is_amplify_first_chunk": False,
                "weight": 1,
            }
        ]
    )


def _start_server(*, repo: Path, variant_dir: Path) -> tuple[subprocess.Popen, str]:
    port = _open_port()
    base_url = f"http://127.0.0.1:{port}"
    log_file = (variant_dir / "server.log").open("w", encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "vllm_omni.entrypoints.cli.main",
        "serve",
        MODEL_NAME,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--omni",
    ]
    print("+ " + " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=str(repo),
        env=_server_env(repo),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        text=True,
    )
    try:
        _wait_for_service(base_url, process)
    except Exception:
        log_file.close()
        _stop_process(process)
        raise
    log_file.close()
    return process, base_url


def _run_serving_benchmark(*, repo: Path, variant_dir: Path, base_url: str) -> dict[str, Any]:
    metrics_path = variant_dir / "metrics.json"
    response_dir = variant_dir / "responses"
    command = [
        sys.executable,
        str(REPO_ROOT / "benchmarks/diffusion/diffusion_benchmark_serving.py"),
        "--base-url",
        base_url,
        "--endpoint",
        "/v1/videos",
        "--model",
        MODEL_NAME,
        "--dataset",
        "random",
        "--task",
        "t2v",
        "--num-prompts",
        "1",
        "--max-concurrency",
        "1",
        "--warmup-requests",
        str(WARMUP_REQUESTS),
        "--warmup-num-inference-steps",
        str(WARMUP_NUM_INFERENCE_STEPS),
        "--warmup-concurrency",
        "1",
        "--random-prompt",
        PROMPT,
        "--seed",
        str(SEED),
        "--disable-tqdm",
        "--random-request-config",
        _benchmark_config(),
        "--output-file",
        str(metrics_path),
        "--save-response-dir",
        str(response_dir),
    ]
    completed = _run(command, cwd=repo, env=_server_env(repo))
    (variant_dir / "benchmark.log").write_text(completed.stdout, encoding="utf-8")
    metrics = _read_json(metrics_path)
    saved_responses = metrics.get("saved_responses") or []
    if len(saved_responses) != 1:
        raise RuntimeError(f"Expected one saved video response, got {saved_responses}")
    return {
        "repo": str(repo),
        "metrics": metrics,
        "video": saved_responses[0],
        "log": str(variant_dir / "benchmark.log"),
        "server_log": str(variant_dir / "server.log"),
    }


def _run_variant(*, label: str, repo: Path, output_root: Path) -> dict[str, Any]:
    variant_dir = output_root / label
    variant_dir.mkdir(parents=True, exist_ok=True)
    process, base_url = _start_server(repo=repo, variant_dir=variant_dir)
    try:
        return _run_serving_benchmark(repo=repo, variant_dir=variant_dir, base_url=base_url)
    finally:
        _stop_process(process)


@pytest.mark.benchmark
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_helios_distilled_pr_matches_main_acc_perf(accuracy_artifact_root: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("Helios-Distilled accuracy/perf comparison requires CUDA.")

    output_root = reset_artifact_dir(accuracy_artifact_root / "helios_distilled_acc_perf")
    reference = _reference_repo(output_root)
    output_json = output_root / "result.json"

    reference_result = _run_variant(label="reference", repo=reference, output_root=output_root)
    candidate_result = _run_variant(label="candidate", repo=REPO_ROOT, output_root=output_root)

    with TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        reference_frames = _extract_frames(Path(reference_result["video"]), tmp_dir / "reference_frames")
        candidate_frames = _extract_frames(Path(candidate_result["video"]), tmp_dir / "candidate_frames")
        output_metrics = _video_metrics(reference_frames, candidate_frames)

    reference_time = float(reference_result["metrics"]["latency_mean"])
    candidate_time = float(candidate_result["metrics"]["latency_mean"])
    time_ratio = candidate_time / reference_time
    result = {
        "reference": reference_result,
        "candidate": candidate_result,
        "sampling": {
            "model": MODEL_NAME,
            "prompt": PROMPT,
            "height": HEIGHT,
            "width": WIDTH,
            "num_frames": NUM_FRAMES,
            "fps": FPS,
            "seed": SEED,
            "guidance_scale": GUIDANCE_SCALE,
            "pyramid_num_inference_steps_list": list(PYRAMID_NUM_INFERENCE_STEPS),
            "is_amplify_first_chunk": False,
        },
        "performance": {
            "reference_measured_time_s": reference_time,
            "candidate_measured_time_s": candidate_time,
            "candidate_over_reference_time_ratio": time_ratio,
        },
        "output_metrics": output_metrics,
        "thresholds": {
            "min_psnr_db": MIN_PSNR_DB,
            "max_mae": MAX_MAE,
            "max_time_ratio": MAX_TIME_RATIO,
        },
    }
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8")
    all_metrics = output_metrics["all_frames_metrics"]
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "candidate_over_reference_time_ratio": time_ratio,
                "all_frames_psnr_db": all_metrics["psnr_db"],
                "all_frames_mae": all_metrics["mae"],
                "all_frames_cosine_similarity": all_metrics["cosine_similarity"],
            },
            indent=2,
            allow_nan=True,
        )
    )

    assert all_metrics["psnr_db"] >= MIN_PSNR_DB
    assert all_metrics["mae"] <= MAX_MAE
    assert time_ratio <= MAX_TIME_RATIO
    assert output_json.exists(), f"Expected Helios comparison result at {output_json}"
