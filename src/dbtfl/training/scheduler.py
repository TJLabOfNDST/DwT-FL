'One-process-per-GPU scheduling for concurrent local client training.'

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ClientTrainingJob:
    'A client command executed once in one bounded GPU resource slot.'

    client_id: str
    command: tuple[str, ...]
    log_path: Path

    def __post_init__(self) -> None:
        'Reject empty job contracts before subprocesses are created.'
        if not self.client_id or not self.command:
            raise ValueError("client_id and command must not be empty / client_id ")


@dataclass(frozen=True, slots=True)
class ClientTrainingResult:
    'Observable outcome for one locally scheduled client process.'

    client_id: str
    gpu_id: int
    slot_index: int
    gpu_memory_fraction: float
    mps_partitioning_enabled: bool
    mps_active_thread_percentage: int | None
    timed_out: bool
    cancelled: bool
    return_code: int
    elapsed_seconds: float
    log_path: Path


def run_client_training_jobs(
    jobs: Sequence[ClientTrainingJob],
    gpu_ids: Sequence[int],
    *,
    clients_per_gpu: int = 1,
    gpu_memory_fraction: float | None = None,
    require_mps_partitioning: bool = False,
    timeout_seconds: float | None = None,
    on_job_start: Callable[[ClientTrainingJob], None] | None = None,
    cancel_requested: Callable[[ClientTrainingJob], bool] | None = None,
) -> list[ClientTrainingResult]:
    'Run a bounded, evenly shared number of client processes per GPU.\n    Each subprocess sees its assigned physical GPU as ``cuda:0`` through\n    ``CUDA_VISIBLE_DEVICES``.  By default it receives ``1 / clients_per_gpu``\n    as its PyTorch memory budget; an explicit ``gpu_memory_fraction`` keeps the\n    per-client allocation fixed across scale cases.  For example, a 20-percent\n    share with five slots per GPU makes 4-, 8-, and 10-client cases comparable\n    without queuing. ``on_job_start`` runs immediately before a child process\n    is launched, allowing an evaluator to inject a genuine training-stage\n    fault rather than mislabel the preceding protocol as a training failure.\n    The callback must return promptly. ``cancel_requested`` is polled while a\n    child runs and terminates only deliberately faulted jobs; it is not used\n    for ordinary scheduler errors.\n    ``gpu_memory_fraction``\n    20%'
    normalized_jobs = tuple(jobs)
    normalized_gpu_ids = tuple(gpu_ids)
    if not normalized_jobs:
        return []
    if not normalized_gpu_ids or any(
        isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or gpu_id < 0
        for gpu_id in normalized_gpu_ids
    ):
        raise ValueError("gpu_ids must be non-empty non-negative integers / gpu_ids ")
    if len(set(normalized_gpu_ids)) != len(normalized_gpu_ids):
        raise ValueError("gpu_ids must not repeat / gpu_ids ")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive when set /  timeout_seconds ")
    if isinstance(clients_per_gpu, bool) or not isinstance(clients_per_gpu, int):
        raise TypeError("clients_per_gpu must be an integer /  GPU ")
    if clients_per_gpu < 1:
        raise ValueError("clients_per_gpu must be positive /  GPU ")
    if gpu_memory_fraction is None:
        effective_memory_fraction = 1.0 / clients_per_gpu
    else:
        if isinstance(gpu_memory_fraction, bool) or not isinstance(
            gpu_memory_fraction, int | float
        ):
            raise TypeError(
                "gpu_memory_fraction must be numeric / GPU "
            )
        effective_memory_fraction = float(gpu_memory_fraction)
        if not 0.0 < effective_memory_fraction <= 1.0:
            raise ValueError(
                "gpu_memory_fraction must be in (0, 1] / GPU  (0, 1]"
            )
        if clients_per_gpu * effective_memory_fraction > 1.0 + 1e-9:
            raise ValueError(
                "concurrent GPU shares exceed one device /  GPU "
            )
    mps_status = mps_partitioning_status()
    if require_mps_partitioning and not mps_status["available"]:
        raise RuntimeError(
            "CUDA MPS is required to enforce the fixed compute share: "
            f"{mps_status['reason']} /  CUDA MPS{mps_status['reason']}"
        )

    pending_by_gpu: dict[int, queue.Queue[ClientTrainingJob]] = {
        gpu_id: queue.Queue() for gpu_id in normalized_gpu_ids
    }
    for position, job in enumerate(normalized_jobs):
        pending_by_gpu[normalized_gpu_ids[position % len(normalized_gpu_ids)]].put(job)
    results: list[ClientTrainingResult] = []
    result_lock = threading.Lock()

    def worker(gpu_id: int, slot_index: int) -> None:
        'Drain queued jobs serially for one bounded GPU share.'
        pending = pending_by_gpu[gpu_id]
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            try:
                if on_job_start is not None:
                    on_job_start(job)
                result = _run_one_job(
                    job,
                    gpu_id,
                    slot_index,
                    effective_memory_fraction,
                    bool(mps_status["available"]),
                    timeout_seconds,
                    cancel_requested,
                )
                with result_lock:
                    results.append(result)
            finally:
                pending.task_done()

    workers = [
        threading.Thread(
            target=worker,
            args=(gpu_id, slot_index),
            name=f"dbtfl-gpu-{gpu_id}-slot-{slot_index}",
        )
        for gpu_id in normalized_gpu_ids
        for slot_index in range(clients_per_gpu)
    ]
    for worker_thread in workers:
        worker_thread.start()
    for worker_thread in workers:
        worker_thread.join()
    return sorted(results, key=lambda result: result.client_id)


def _run_one_job(
    job: ClientTrainingJob,
    gpu_id: int,
    slot_index: int,
    gpu_memory_fraction: float,
    mps_partitioning_enabled: bool,
    timeout_seconds: float | None,
    cancel_requested: Callable[[ClientTrainingJob], bool] | None,
) -> ClientTrainingResult:
    'Run one client process with an isolated bounded-device environment.'
    log_path = Path(job.log_path).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    environment["DBTFL_GPU_MEMORY_FRACTION"] = str(gpu_memory_fraction)
    active_thread_percentage = int(round(gpu_memory_fraction * 100))
    if mps_partitioning_enabled:
        # NVIDIA MPS reads this before the CUDA context is created and limits
        # this client process to the requested portion of GPU threads. NVIDIA
        # MPS
        
        environment["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(active_thread_percentage)
    environment["DBTFL_GPU_SLOT_INDEX"] = str(slot_index)
    environment.setdefault("PYTHONUNBUFFERED", "1")
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8", newline="") as log_stream:
        try:
            process = subprocess.Popen(
                list(job.command),
                stdin=subprocess.DEVNULL,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                env=environment,
            )
            deadline = None if timeout_seconds is None else started + timeout_seconds
            cancelled = False
            timed_out = False
            while process.poll() is None:
                if cancel_requested is not None and cancel_requested(job):
                    process.terminate()
                    cancelled = True
                    break
                if deadline is not None and time.perf_counter() >= deadline:
                    process.terminate()
                    timed_out = True
                    break
                time.sleep(0.02)
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            return_code = process.returncode
            # Windows and POSIX report different exit values after terminate.
            # Preserve one portable scheduler contract for a true timeout;
            # deliberate fault cancellation remains separately observable via
            # ``cancelled``. Windows
            
            
            if timed_out:
                return_code = 124
                log_stream.write(
                    "training subprocess timed out and was terminated / "
                    "\n"
                )
                log_stream.flush()
            if cancelled:
                log_stream.write(
                    "training subprocess was deliberately cancelled after client disconnect / "
                    "\n"
                )
                log_stream.flush()
        except subprocess.TimeoutExpired:
            # subprocess.run terminates the child before raising on Windows,
            # WSL, and Ubuntu.  Preserve an explicit observable outcome rather
            # than letting a worker thread disappear and stall the evaluator.
            # subprocess.run
            
            log_stream.write(
                "training subprocess timed out and was terminated / "
                "\n"
            )
            log_stream.flush()
            return_code = 124
            timed_out = True
            cancelled = False
    return ClientTrainingResult(
        client_id=job.client_id,
        gpu_id=gpu_id,
        slot_index=slot_index,
        gpu_memory_fraction=gpu_memory_fraction,
        mps_partitioning_enabled=mps_partitioning_enabled,
        mps_active_thread_percentage=(
            active_thread_percentage if mps_partitioning_enabled else None
        ),
        timed_out=timed_out,
        cancelled=cancelled,
        return_code=return_code,
        elapsed_seconds=time.perf_counter() - started,
        log_path=log_path,
    )


def mps_partitioning_status() -> dict[str, object]:
    'Probe whether a local CUDA MPS control daemon accepts client commands.\n    A memory cap alone does not partition GPU compute.  The evaluator therefore\n    requires this probe for real fixed-share runs, while protocol-only and\n    portable scheduler tests can keep it disabled.'
    if sys.platform == "win32":
        return {
            "available": False,
            "reason": "CUDA MPS requires a Linux CUDA environment",
        }
    executable = shutil.which("nvidia-cuda-mps-control")
    if executable is None:
        return {"available": False, "reason": "nvidia-cuda-mps-control was not found"}
    try:
        completed = subprocess.run(
            [executable],
            input="get_server_list\n",
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except OSError as error:
        return {"available": False, "reason": type(error).__name__}
    detail = (completed.stderr + "\n" + completed.stdout).strip()
    missing_daemon_markers = (
        "cannot find mps control daemon",
        "failed to connect",
        "connection refused",
        "not running",
    )
    if completed.returncode != 0 or any(marker in detail.lower() for marker in missing_daemon_markers):
        detail = detail or "MPS control rejected probe"
        return {"available": False, "reason": detail}
    return {"available": True, "reason": "MPS control daemon accepted probe"}
