"""Dedicated, paired dynamic-client-join evaluation for DwT-FL.

DwT-FL 动态客户端加入的专用配对评估。

This benchmark intentionally does not reuse the legacy ``dynamic_join`` suite.
It first establishes a real 10-client FL round outside the measured interval,
then measures only the incremental join interval shared with the NDSS'25
baseline: the first additional client actually starts its join protocol until
every additional client has reached a real local-training subprocess launch.
该基准刻意不复用旧的 ``dynamic_join`` 套件。它先在计时区间外建立真实的
10 客户端联邦学习轮次，随后只测量与 NDSS'25 基线共享的增量加入区间：从第一名
额外客户端实际启动加入协议，到所有额外客户端均到达真实本地训练子进程启动点。
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

from dbtfl.communication import TrafficRecorder
from dbtfl.entities import ClientConfig, ClientEntity, KeyServerConfig, KeyServerEntity
from dbtfl.evaluation.runner import (
    EvaluationPlan,
    EvaluationRunner,
    _Case,
    _LocalAsProcess,
    _current_training_participants,
    _fetch_ks_metrics,
    _parallel_phase,
)
from dbtfl.evaluation.oprf_precompute import allocate_joining_records


_RESULT_SCHEMA = "1.0"
_SCHEDULE_SCHEMA = "1.0"


@dataclass(frozen=True, slots=True)
class DynamicJoinEvaluationPlan:
    """Configuration for one fair DwT-FL dynamic-join benchmark.

    一个公平 DwT-FL 动态加入基准的配置。
    """

    output_directory: Path
    prepared_data_path: Path
    arrival_schedule_path: Path
    joining_client_counts: tuple[int, ...] = (1, 3, 5, 7)
    repetitions: int = 4
    seed: int = 17
    base_clients: int = 10
    records_per_client: int = 1024
    duplicate_ratio: float = 0.30
    as_backend_workers: int = 4
    arrival_min_delay_seconds: float = 1.0
    arrival_max_delay_seconds: float = 5.0
    rpc_timeout_seconds: float = 300.0
    heartbeat_rpc_timeout_seconds: float = 2.0
    heartbeat_interval_seconds: float = 5.0
    heartbeat_timeout_seconds: float = 300.0
    oprf_batch_size: int = 1024
    model_chunk_bytes: int = 2 * 1024 * 1024
    # The paired NDSS'25 runner starts every joining model read concurrently.
    # Keep the same cap above the largest 1/3/5/7 case rather than serializing
    # DwT-FL downloads with an implementation-only two-reader limit. 配对的
    # NDSS'25 评估器会并发启动所有加入者模型读取；因此将上限保持在最大 1/3/5/7
    # 用例之上，避免使用实现特有的双读取器限制串行化 DwT-FL 下载。
    global_model_download_workers: int = 10
    gpu_ids: tuple[int, ...] = (0, 1)
    clients_per_gpu: int = 5
    gpu_memory_fraction_per_client: float = 0.20
    require_mps_partitioning: bool = False
    gpt_batch_size: int = 8
    gpt_gradient_accumulation: int = 4
    gpt_local_epochs: int = 1
    gpt_max_length: int = 128
    gpt_precision: str = "bf16"
    gpt_checkpoint_precision: str = "fp16"
    training_job_timeout_seconds: float = 1800.0

    def __post_init__(self) -> None:
        """Reject invalid experiments before creating live protocol services.

        在创建实时协议服务前拒绝无效实验。
        """
        if not self.joining_client_counts or any(value < 1 for value in self.joining_client_counts):
            raise ValueError("joining client counts must be positive / 加入客户端数必须为正数")
        if self.repetitions < 1 or self.base_clients < 1 or self.records_per_client < 1:
            raise ValueError("count values must be positive / 计数参数必须为正数")
        if self.base_clients != 10 or self.records_per_client != 1024:
            raise ValueError(
                "the paired dynamic-join benchmark is fixed to 10 base clients and 1024 records each / "
                "配对动态加入基准固定为 10 个基础客户端、每客户端 1024 条记录"
            )
        if not 0.0 <= self.duplicate_ratio <= 1.0:
            raise ValueError("duplicate ratio must be in [0, 1] / 重复比例必须位于 [0, 1]")
        if self.arrival_min_delay_seconds <= 0.0:
            raise ValueError("joining arrival delays must be positive / 加入上线延迟必须为正秒数")
        if self.arrival_max_delay_seconds < self.arrival_min_delay_seconds:
            raise ValueError("maximum arrival delay precedes minimum / 最大上线延迟小于最小值")
        if self.as_backend_workers < 1 or self.global_model_download_workers < 1:
            raise ValueError("worker counts must be positive / 工作线程数必须为正数")
        if not self.gpu_ids or any(value < 0 for value in self.gpu_ids):
            raise ValueError("GPU IDs must be non-negative / GPU 标识必须为非负数")
        if self.clients_per_gpu * self.gpu_memory_fraction_per_client > 1.0 + 1e-9:
            raise ValueError("GPU memory shares exceed one / GPU 显存份额超过一")


class DynamicJoinEvaluationRunner:
    """Run the live DwT-FL dynamic join protocol and retain compact evidence.

    运行实时 DwT-FL 动态加入协议并保留紧凑证据。
    """

    def __init__(self, plan: DynamicJoinEvaluationPlan, *, progress: Any = print) -> None:
        """Store a validated plan and an optional progress callback.

        保存已验证的计划和可选进度回调。
        """
        self.plan = plan
        self.progress = progress
        self.root = plan.output_directory.resolve()
        self._arrival_schedule = _load_or_create_arrival_schedule(plan)

    def run(self) -> Path:
        """Execute every join-count case, persisting each completed case at once.

        执行每个加入数量用例，并在每个用例完成后立即持久化。
        """
        self.root.mkdir(parents=True, exist_ok=True)
        runner = EvaluationRunner(self._runner_plan())
        runner._ensure_prepared_data()
        runner._verify_configured_cuda_devices(self.root)
        # A report may be interrupted after JSON/CSV persistence but before a
        # secondary Markdown write. Keep only the completed contiguous prefix
        # and continue at the next join count, never repeating finished CUDA
        # workloads. 报告可能在 JSON/CSV 已落盘、次要 Markdown 写入前中断；仅保留
        # 已完成的连续前缀并从下一个加入数量继续，绝不重跑已完成的 CUDA 工作负载。
        results = self._load_completed_prefix()
        total = len(self.plan.joining_client_counts)
        if results:
            self.progress(
                f"[resume] retained {len(results)} completed cases; continuing at "
                f"{len(results) + 1}/{total} / [续跑] 已保留 {len(results)} 个完成用例；"
                f"将从 {len(results) + 1}/{total} 继续"
            )
        for ordinal, joining_count in enumerate(self.plan.joining_client_counts, start=1):
            if ordinal <= len(results):
                continue
            self.progress(f"[{ordinal}/{total}] dynamic_join: joining_clients={joining_count}")
            raw_results: list[dict[str, Any]] = []
            failures: list[dict[str, Any]] = []
            for repetition in range(self.plan.repetitions):
                self.progress(f"  [repeat {repetition + 1}/{self.plan.repetitions}]")
                try:
                    raw = self._run_case(runner, joining_count, repetition)
                    raw_results.append(raw)
                    self._append_raw_repeat({
                        "status": "completed",
                        "joining_clients": joining_count,
                        "repetition": repetition + 1,
                        "result": raw,
                    })
                except Exception as error:  # Keep independent cases observable.
                    failure = {
                        "repetition": repetition + 1,
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                        "diagnostic_paths": list(
                            getattr(error, "dbtfl_diagnostic_paths", ())
                        ),
                    }
                    failures.append(failure)
                    self._append_raw_repeat({
                        "status": "failed",
                        "joining_clients": joining_count,
                        "repetition": repetition + 1,
                        "failure": failure,
                    })
                    self.progress(
                        f"[repeat failed] dynamic_join: joining_clients={joining_count}; "
                        f"repeat={repetition + 1}; {type(error).__name__}: {error} / "
                        f"[重复失败] dynamic_join：joining_clients={joining_count}；"
                        f"第 {repetition + 1} 次；{type(error).__name__}: {error}"
                    )
            if failures:
                result: dict[str, Any] = {
                    "schema_version": _RESULT_SCHEMA,
                    # Preserve all successful repetitions in the case report.
                    # They are intentionally not trimmed or averaged when one
                    # repetition fails, but remain available for diagnosis and
                    # a targeted rerun. 某次重复失败时，保留该用例已成功的全部
                    # 重复；它们不会被截尾或平均，却可用于诊断及定向重跑。
                    "status": "partial" if raw_results else "failed",
                    "suite": "dynamic_join",
                    "variable": "joining_clients",
                    "value": joining_count,
                    "configuration": self._configuration(joining_count),
                    "failure": {
                        "completed_repetitions": len(raw_results),
                        "failed_repetitions": failures,
                    },
                    "successful_repetitions": raw_results,
                    "raw_repetition_log": "raw_dynamic_join_repetitions.jsonl",
                }
            else:
                result = _trimmed_result(raw_results)
            results.append(result)
            self._write_reports(results)
        return self.root

    def _load_completed_prefix(self) -> list[dict[str, Any]]:
        """Load only a matching completed prefix from a previous interrupted run.

        仅加载此前中断运行中与当前计划匹配的完成前缀。
        """
        path = self.root / "results.json"
        if not path.is_file():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            prior = payload.get("results", [])
            if not isinstance(prior, list):
                return []
        except (OSError, ValueError, TypeError):
            return []
        retained: list[dict[str, Any]] = []
        for joining_count, result in zip(self.plan.joining_client_counts, prior):
            if (
                not isinstance(result, dict)
                or result.get("status") != "completed"
                or result.get("suite") != "dynamic_join"
                or result.get("value") != joining_count
            ):
                break
            retained.append(result)
        return retained

    def _runner_plan(self) -> EvaluationPlan:
        """Create the existing real-GPT runner configuration used by this benchmark.

        创建本基准使用的既有真实 GPT 评估器配置。
        """
        return EvaluationPlan(
            output_directory=self.root,
            repetitions=1,
            seed=self.plan.seed,
            base_clients=self.plan.base_clients,
            base_duplicate_ratio=self.plan.duplicate_ratio,
            base_backend_workers=self.plan.as_backend_workers,
            base_records_per_client=self.plan.records_per_client,
            client_counts=(self.plan.base_clients,),
            duplicate_ratios=(self.plan.duplicate_ratio,),
            backend_worker_counts=(self.plan.as_backend_workers,),
            records_per_client_values=(self.plan.records_per_client,),
            join_client_counts=self.plan.joining_client_counts,
            join_duplicate_ratio=self.plan.duplicate_ratio,
            client_arrival_min_delay_seconds=0.0,
            client_arrival_max_delay_seconds=0.0,
            client_arrival_anchor_seconds=0.0,
            heartbeat_interval_seconds=self.plan.heartbeat_interval_seconds,
            heartbeat_timeout_seconds=self.plan.heartbeat_timeout_seconds,
            training_mode="gpt",
            prepared_data_path=self.plan.prepared_data_path,
            training_job_timeout_seconds=self.plan.training_job_timeout_seconds,
            rpc_timeout_seconds=self.plan.rpc_timeout_seconds,
            heartbeat_rpc_timeout_seconds=self.plan.heartbeat_rpc_timeout_seconds,
            oprf_batch_size=self.plan.oprf_batch_size,
            model_chunk_bytes=self.plan.model_chunk_bytes,
            global_model_download_workers=self.plan.global_model_download_workers,
            gpu_ids=self.plan.gpu_ids,
            clients_per_gpu=self.plan.clients_per_gpu,
            gpu_memory_fraction_per_client=self.plan.gpu_memory_fraction_per_client,
            require_mps_partitioning=self.plan.require_mps_partitioning,
            gpt_batch_size=self.plan.gpt_batch_size,
            gpt_gradient_accumulation=self.plan.gpt_gradient_accumulation,
            gpt_local_epochs=self.plan.gpt_local_epochs,
            gpt_max_length=self.plan.gpt_max_length,
            gpt_precision=self.plan.gpt_precision,
            gpt_checkpoint_precision=self.plan.gpt_checkpoint_precision,
            require_cuda=True,
            federated_rounds=1,
            include_ablations=False,
            service_mode="isolated",
            require_live_oprf=True,
        )

    def _run_case(
        self,
        runner: EvaluationRunner,
        joining_count: int,
        repetition: int,
    ) -> dict[str, Any]:
        """Establish a base real FL round, then measure one live incremental join.

        建立一轮基础真实 FL，然后测量一次实时增量加入。
        """
        max_joiners = max(self.plan.joining_client_counts)
        case = _Case(
            suite="dynamic_join",
            variable="joining_clients",
            value=joining_count,
            clients=self.plan.base_clients,
            request_workers=self.plan.base_clients,
            duplicate_ratio=self.plan.duplicate_ratio,
            backend_workers=self.plan.as_backend_workers,
            records_per_client=self.plan.records_per_client,
            joining_clients=max_joiners,
        )
        with TemporaryDirectory(prefix="dbtfl-dynamic-join-") as temporary_name:
            runtime_root = Path(temporary_name)
            as_process = _LocalAsProcess(
                runtime_root, case, runner.plan, self.plan.heartbeat_timeout_seconds
            )
            ks = KeyServerEntity(KeyServerConfig(
                key_path=runtime_root / "ks-ristretto255-key.json", host="127.0.0.1", port=0
            ))
            clients: list[ClientEntity] = []
            try:
                ks.start()
                base_clients = runner._create_clients(runtime_root, as_process.base_url, ks.base_url, case)
                # All allocations are created for seven potential joiners first.
                # Thus the 10 base splits and the first 1/3/5/7 joiner splits are
                # identical across join-count cases and repetitions. 先为七个潜在
                # 加入者生成全部划分，确保 10 个基础划分及前 1/3/5/7 个加入者划分在
                # 不同加入数量用例和重复之间完全一致。
                base_allocation_case = _Case(
                    suite="dynamic_join_base_allocation",
                    variable="base_clients",
                    value=self.plan.base_clients,
                    clients=self.plan.base_clients,
                    request_workers=self.plan.base_clients,
                    duplicate_ratio=self.plan.duplicate_ratio,
                    backend_workers=self.plan.as_backend_workers,
                    records_per_client=self.plan.records_per_client,
                    joining_clients=0,
                )
                base_records = runner._records_for_case(base_allocation_case, repetition=0)
                # Allocate joiners from the already established base corpus.
                # This preserves all ten base splits while requiring only
                # 8,705 base records plus 7 x 717 fresh joining records, which
                # fits the 14,414-record prepared pool. 从已经建立的基础语料中
                # 分配加入者，既保持十个基础划分不变，又只需要 8,705 条基础记录和
                # 7 x 717 条新加入记录，适配 14,414 条训练池。
                joining_records = allocate_joining_records(
                    runner._prepared_records,
                    base_records,
                    joining_clients=max_joiners,
                    duplicate_ratio=self.plan.duplicate_ratio,
                    seed=self.plan.seed,
                )
                records = {**base_records, **joining_records}
                base_claims, base_queues = self._complete_base_protocol(base_clients, records)
                clients.extend(base_clients)
                base_participants = _current_training_participants(
                    base_clients, base_claims, base_queues
                )
                if len(base_participants) != self.plan.base_clients:
                    raise RuntimeError("base round has a non-training client / 基础轮存在未训练客户端")
                base_training = runner._run_training(
                    runtime_root, base_participants, base_queues, repetition
                )
                base_sids = tuple(
                    client.as_session.sid for client in base_participants if client.as_session is not None
                )
                if len(base_sids) != len(base_participants):
                    raise RuntimeError("base participant lost its AS session / 基础参与者丢失 AS 会话")
                base_participants[0].configure_federated_round_at_as(1, base_sids)
                base_submissions = runner._submit_updates(
                    runtime_root, base_participants, base_claims, base_queues,
                    base_training,
                    repetition,
                    round_id=1,
                    # This base round is established before the measured
                    # dynamic-join clock. Keep actual uploads concurrent, but
                    # cap physical streams at the configured AS capacity so a
                    # 10-client checkpoint burst cannot starve the control
                    # plane. 该基础轮在动态加入计时前建立；实际上传仍并发执行，
                    # 但物理流数被限制为配置的 AS 容量，避免 10 客户端检查点
                    # 突发挤占控制平面。
                    max_submit_workers=self.plan.as_backend_workers,
                )
                if set(base_submissions["submitted_sids"]) != set(base_sids):
                    raise RuntimeError("base FedAvg roster is incomplete / 基础 FedAvg 名册不完整")
                base_participants[0].aggregate_federated_round_at_as(1, base_sids)

                joiners = self._create_joiners(
                    runtime_root, as_process.base_url, ks.base_url, joining_count
                )
                clients.extend(joiners)
                before_join_ks = _fetch_ks_metrics(ks.base_url)
                origin = time.perf_counter()
                join_claims, join_queues, arrivals = self._run_join_protocols(
                    joiners, records, origin
                )
                first_started_at = min(item["join_started_at"] for item in arrivals.values())
                last_registered_at = max(item["registered_at"] for item in arrivals.values())
                dedup_finished_at = max(item["protocol_completed_at"] for item in arrivals.values())

                destinations = {
                    client.config.client_id: runtime_root / "joining-global-models" /
                    f"{client.config.client_id}-global-round-1.safetensors"
                    for client in joiners
                }
                downloads = _parallel_phase(
                    joiners,
                    min(self.plan.global_model_download_workers, len(joiners)),
                    lambda client: client.download_global_model_from_as(
                        1, destinations[client.config.client_id]
                    ),
                )
                training_started_at: dict[str, float] = {}
                training_start_lock = threading.Lock()

                def on_training_start(job: Any) -> None:
                    """Record the real subprocess-launch point once per joiner.

                    仅为每名加入者记录一次真实子进程启动点。
                    """
                    with training_start_lock:
                        training_started_at.setdefault(job.client_id, time.perf_counter())

                join_participants = _current_training_participants(joiners, join_claims, join_queues)
                if {client.config.client_id for client in join_participants} != {
                    client.config.client_id for client in joiners
                }:
                    raise RuntimeError(
                        "every joining client must retain training data at r=0.3 / "
                        "在 r=0.3 下每名加入客户端必须保留训练数据"
                    )
                join_training = runner._run_training(
                    runtime_root,
                    join_participants,
                    join_queues,
                    repetition,
                    initial_checkpoints=destinations,
                    on_job_start=on_training_start,
                )
                expected_join_ids = {client.config.client_id for client in joiners}
                if set(training_started_at) != expected_join_ids:
                    raise RuntimeError(
                        "not every joining client reached a real training start / "
                        "并非每个加入客户端都到达真实训练启动点"
                    )
                last_training_started_at = max(training_started_at.values())
                ks_after = _fetch_ks_metrics(ks.base_url)
                return {
                    "schema_version": _RESULT_SCHEMA,
                    "status": "completed",
                    "suite": "dynamic_join",
                    "variable": "joining_clients",
                    "value": joining_count,
                    "configuration": self._configuration(joining_count),
                    "repetition": repetition + 1,
                    "oprf_suite": "ristretto255-sha512-libsodium-v1",
                    "live_oprf_required": True,
                    "base_client_count": self.plan.base_clients,
                    "joining_client_count": joining_count,
                    "input_record_count": joining_count * self.plan.records_per_client,
                    "arrival_schedule": _selected_schedule(self._arrival_schedule, joining_count),
                    "client_arrivals": _serializable_arrivals(arrivals),
                    "first_join_started_after_base_seconds": first_started_at - origin,
                    "last_join_registered_after_base_seconds": last_registered_at - origin,
                    "join_arrival_span_seconds": last_registered_at - first_started_at,
                    "join_end_to_end_to_training_start_seconds": (
                        last_training_started_at - first_started_at
                    ),
                    "join_dedup_after_first_join_seconds": dedup_finished_at - first_started_at,
                    "join_dedup_after_all_joined_seconds": dedup_finished_at - last_registered_at,
                    "join_global_model_download_wall_seconds": downloads["wall_seconds"],
                    "join_training_launch_span_seconds": (
                        last_training_started_at - min(training_started_at.values())
                    ),
                    "join_training_completion_wall_seconds": join_training["wall_seconds"],
                    "join_training_completion_accumulated_seconds": join_training["accumulated_seconds"],
                    "ks_oprf_delta": _ks_delta(before_join_ks, ks_after),
                    "joining_training": join_training["client_metrics"],
                }
            except Exception as error:
                # Preserve the isolated AS log before TemporaryDirectory removes
                # it. A failed raw repeat must remain diagnosable without
                # retaining checkpoints or model-update payloads. 在临时目录清理
                # 前保留隔离 AS 日志；失败的原始重复必须可诊断，但不保留检查点或模型
                # 更新载荷。
                diagnostic = (
                    self.root / "failed-case-diagnostics" /
                    f"dynamic-join-{joining_count}-repeat-{repetition + 1}-as.log"
                )
                try:
                    as_process.copy_log_to(diagnostic)
                    setattr(error, "dbtfl_diagnostic_paths", (str(diagnostic),))
                except OSError:
                    pass
                raise
            finally:
                for client in clients:
                    try:
                        client.close()
                    except Exception:
                        pass
                try:
                    ks.close()
                finally:
                    as_process.close()

    def _complete_base_protocol(
        self,
        clients: Sequence[ClientEntity],
        records: Mapping[str, Sequence[str]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run the unmeasured base protocol concurrently without arrival sleeps.

        并发运行不计入动态加入时钟的基础协议，且不加入上线等待。
        """
        def workflow(client: ClientEntity) -> tuple[list[Any], Any]:
            """Connect, evaluate OPRF, register labels, and claim immediately.

            连接、执行 OPRF、登记标签并立即抢占。
            """
            client.connect_to_as()
            labels = client.register_records_with_as(
                records[client.config.client_id], created_round=1, refresh_lease=False
            )
            decisions = client.claim_registered_labels_at_as(labels, refresh_lease=False)
            return decisions, client.route_claim_decisions(decisions)

        phase = _parallel_phase(list(clients), len(clients), workflow)
        claims = {
            "values": {
                client.config.client_id: (phase["values"][client.config.client_id][0][0], 0.0)
                for client in clients
            }
        }
        queues = {
            client.config.client_id: phase["values"][client.config.client_id][0][1]
            for client in clients
        }
        return claims, queues

    def _create_joiners(
        self, root: Path, as_url: str, ks_url: str, joining_count: int
    ) -> list[ClientEntity]:
        """Create only the selected prefix of the fixed potential joiner set.

        仅创建固定潜在加入者集合中被选定的前缀。
        """
        return [
            ClientEntity(ClientConfig(
                client_id=f"client-{self.plan.base_clients + offset}",
                ks_base_url=ks_url,
                as_base_url=as_url,
                label_store_path=root / f"client-{self.plan.base_clients + offset}-labels.json",
                timeout_seconds=self.plan.rpc_timeout_seconds,
                heartbeat_rpc_timeout_seconds=self.plan.heartbeat_rpc_timeout_seconds,
                oprf_batch_size=self.plan.oprf_batch_size,
                model_chunk_bytes=self.plan.model_chunk_bytes,
                traffic_recorder=TrafficRecorder(),
            ))
            for offset in range(joining_count)
        ]

    def _run_join_protocols(
        self,
        joiners: Sequence[ClientEntity],
        records: Mapping[str, Sequence[str]],
        origin: float,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, float]]]:
        """Run each selected joiner's live protocol at its own persisted arrival.

        在每名加入者各自持久化的上线时刻运行实时协议。
        """
        delays = self._arrival_schedule["scheduled_delay_seconds"]

        def workflow(client: ClientEntity) -> tuple[list[Any], Any, dict[str, float]]:
            """Perform one independent OPRF, AS update, and CAS workflow.

            执行一条独立的 OPRF、AS 更新与 CAS 工作流。
            """
            offset = int(client.config.client_id.rsplit("-", 1)[1]) - self.plan.base_clients
            schedule_id = f"joining-client-{offset}"
            delay = float(delays[schedule_id])
            time.sleep(delay)
            started_at = time.perf_counter()
            client.connect_to_as()
            registered_at = time.perf_counter()
            oprf_started_at = time.perf_counter()
            labels = client.generate_protected_labels(records[client.config.client_id])
            oprf_finished_at = time.perf_counter()
            registrations = client.register_protected_labels_with_as(
                tuple(dict.fromkeys(labels)), created_round=2
            )
            registrations_by_label = {
                registration.protected_label: registration for registration in registrations
            }
            decisions = client.claim_registered_labels_at_as(
                [registrations_by_label[label] for label in labels], refresh_lease=False
            )
            completed_at = time.perf_counter()
            return decisions, client.route_claim_decisions(decisions), {
                "scheduled_delay_seconds": delay,
                "join_started_at": started_at,
                "registered_at": registered_at,
                "oprf_started_at": oprf_started_at,
                "oprf_finished_at": oprf_finished_at,
                "protocol_completed_at": completed_at,
                "join_started_after_base_seconds": started_at - origin,
                "registered_after_base_seconds": registered_at - origin,
                "registration_seconds": registered_at - started_at,
                "oprf_seconds": oprf_finished_at - oprf_started_at,
                "protocol_seconds": completed_at - started_at,
            }

        phase = _parallel_phase(list(joiners), len(joiners), workflow)
        claims = {
            "values": {
                client.config.client_id: (phase["values"][client.config.client_id][0][0], 0.0)
                for client in joiners
            }
        }
        queues = {
            client.config.client_id: phase["values"][client.config.client_id][0][1]
            for client in joiners
        }
        arrivals = {
            client.config.client_id: phase["values"][client.config.client_id][0][2]
            for client in joiners
        }
        return claims, queues, arrivals

    def _configuration(self, joining_count: int) -> dict[str, object]:
        """Return the paper-facing fixed configuration for one result row.

        返回单条结果行的论文配置。
        """
        return {
            "base_clients": self.plan.base_clients,
            "joining_clients": joining_count,
            "records_per_client": self.plan.records_per_client,
            "duplicate_ratio": self.plan.duplicate_ratio,
            "as_backend_workers": self.plan.as_backend_workers,
            "arrival_delay_range_seconds": [
                self.plan.arrival_min_delay_seconds,
                self.plan.arrival_max_delay_seconds,
            ],
            "repetitions": self.plan.repetitions,
            "trim_policy": "discard_min_and_max_then_mean",
            "live_oprf": True,
        }

    def _write_reports(self, results: Sequence[Mapping[str, Any]]) -> None:
        """Persist compact JSON, CSV, and bilingual Markdown reports.

        持久化紧凑的 JSON、CSV 与双语 Markdown 报告。
        """
        payload = {"plan": _jsonable(asdict(self.plan)), "results": list(results)}
        _write_json(self.root / "results.json", payload)
        _write_csv(self.root, results)
        _write_markdown(self.root, results)

    def _append_raw_repeat(self, payload: Mapping[str, Any]) -> None:
        """Durably append one repetition before the next GPU workload starts.

        The JSONL file is an append-only evidence log: successful raw metrics
        and failure diagnostics survive a later failed repeat or process
        interruption. 每次 GPU 工作负载开始前，向 JSONL 追加并强制落盘一条
        重复证据：后续重复失败或进程中断不会抹去已成功指标和失败诊断。
        """
        path = self.root / "raw_dynamic_join_repetitions.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def _load_or_create_arrival_schedule(plan: DynamicJoinEvaluationPlan) -> dict[str, Any]:
    """Reuse the NDSS-compatible schedule or create it exactly once.

    复用与 NDSS 兼容的上线计划，或仅在首次运行时创建一次。
    """
    path = Path(plan.arrival_schedule_path).resolve()
    identifiers = tuple(f"joining-client-{index}" for index in range(max(plan.joining_client_counts)))
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != _SCHEDULE_SCHEMA
            or tuple(payload.get("joining_client_ids", ())) != identifiers
            or float(payload.get("minimum_delay_seconds", -1.0)) != plan.arrival_min_delay_seconds
            or float(payload.get("maximum_delay_seconds", -1.0)) != plan.arrival_max_delay_seconds
            or set(payload.get("scheduled_delay_seconds", {})) != set(identifiers)
        ):
            raise ValueError(
                "persisted arrival schedule is incompatible; choose a new path rather than changing it / "
                "持久化上线计划不兼容；请使用新路径而非修改已有计划"
            )
        return payload
    digest = hashlib.sha256(
        f"ndss25-dynamic-join-arrival|{plan.seed}|{identifiers}|"
        f"{plan.arrival_min_delay_seconds}|{plan.arrival_max_delay_seconds}".encode("utf-8")
    ).digest()
    generator = random.Random(int.from_bytes(digest[:8], "big"))
    payload = {
        "schema_version": _SCHEDULE_SCHEMA,
        "seed": plan.seed,
        "joining_client_ids": list(identifiers),
        "minimum_delay_seconds": plan.arrival_min_delay_seconds,
        "maximum_delay_seconds": plan.arrival_max_delay_seconds,
        "scheduled_delay_seconds": {
            identifier: generator.uniform(
                plan.arrival_min_delay_seconds, plan.arrival_max_delay_seconds
            ) for identifier in identifiers
        },
    }
    _write_json(path, payload)
    return payload


def _selected_schedule(schedule: Mapping[str, Any], joining_count: int) -> dict[str, object]:
    """Expose exactly the selected persisted delay prefix for auditability.

    为审计公开精确选取的持久化延迟前缀。
    """
    delays = dict(schedule["scheduled_delay_seconds"])
    identifiers = tuple(f"joining-client-{index}" for index in range(joining_count))
    return {
        "schedule_path_policy": "created_once_then_reused",
        "scheduled_delay_seconds": {identifier: float(delays[identifier]) for identifier in identifiers},
    }


def _serializable_arrivals(
    arrivals: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    """Strip monotonic timestamps while retaining durations and schedule evidence.

    删除不可跨运行比较的单调时间戳，仅保留时长与调度证据。
    """
    return {
        client_id: {
            key: float(value) for key, value in metrics.items()
            if not key.endswith("_at")
        }
        for client_id, metrics in arrivals.items()
    }


def _ks_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, float | int]:
    """Return the join-only KS OPRF computation delta.

    返回仅属于加入阶段的 KS OPRF 计算增量。
    """
    return {
        "oprf_evaluated_element_count": int(after["oprf_evaluated_element_count"])
        - int(before["oprf_evaluated_element_count"]),
        "oprf_evaluation_compute_seconds": float(after["oprf_evaluation_compute_seconds"])
        - float(before["oprf_evaluation_compute_seconds"]),
    }


def _trimmed_result(raw_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Discard one high and one low endpoint sample, then average remaining runs.

    剔除端到端指标的一个最高值和一个最低值，再平均剩余运行。
    """
    if not raw_results:
        raise ValueError("raw results must not be empty / 原始结果不得为空")
    ordered = sorted(raw_results, key=lambda item: float(item["join_end_to_end_to_training_start_seconds"]))
    retained = ordered[1:-1] if len(ordered) >= 3 else ordered
    first = retained[0]
    numeric_fields = (
        "first_join_started_after_base_seconds",
        "last_join_registered_after_base_seconds",
        "join_arrival_span_seconds",
        "join_end_to_end_to_training_start_seconds",
        "join_dedup_after_first_join_seconds",
        "join_dedup_after_all_joined_seconds",
        "join_global_model_download_wall_seconds",
        "join_training_launch_span_seconds",
        "join_training_completion_wall_seconds",
        "join_training_completion_accumulated_seconds",
    )
    result = {
        "schema_version": _RESULT_SCHEMA,
        "status": "completed",
        "suite": "dynamic_join",
        "variable": "joining_clients",
        "value": first["value"],
        "configuration": first["configuration"],
        "oprf_suite": first["oprf_suite"],
        "live_oprf_required": True,
        "base_client_count": first["base_client_count"],
        "joining_client_count": first["joining_client_count"],
        "input_record_count": first["input_record_count"],
        "arrival_schedule": first["arrival_schedule"],
        "trimmed_repetitions": len(retained),
        "discarded_repetitions": len(raw_results) - len(retained),
    }
    for field in numeric_fields:
        result[field] = sum(float(item[field]) for item in retained) / len(retained)
    result["ks_oprf_delta"] = {
        key: sum(float(item["ks_oprf_delta"][key]) for item in retained) / len(retained)
        for key in ("oprf_evaluated_element_count", "oprf_evaluation_compute_seconds")
    }
    return result


def _write_csv(root: Path, results: Sequence[Mapping[str, Any]]) -> None:
    """Write one compact table suitable for direct paper-table transcription.

    写入可直接转录到论文表格的紧凑表。
    """
    columns = (
        "status", "joining_clients", "input_records", "arrival_span_s",
        "end_to_end_to_training_start_s", "dedup_after_first_join_s",
        "dedup_after_all_joined_s", "global_model_download_s", "training_launch_span_s",
        "training_completion_wall_s", "ks_oprf_elements", "ks_oprf_compute_s",
    )
    with (root / "dynamic_join_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for result in results:
            if result.get("status") != "completed":
                writer.writerow({
                    "status": result.get("status", "failed"),
                    "joining_clients": result.get("value"),
                })
                continue
            ks = result["ks_oprf_delta"]
            writer.writerow({
                "status": "completed",
                "joining_clients": result["joining_client_count"],
                "input_records": result["input_record_count"],
                "arrival_span_s": result["join_arrival_span_seconds"],
                "end_to_end_to_training_start_s": result["join_end_to_end_to_training_start_seconds"],
                "dedup_after_first_join_s": result["join_dedup_after_first_join_seconds"],
                "dedup_after_all_joined_s": result["join_dedup_after_all_joined_seconds"],
                "global_model_download_s": result["join_global_model_download_wall_seconds"],
                "training_launch_span_s": result["join_training_launch_span_seconds"],
                "training_completion_wall_s": result["join_training_completion_wall_seconds"],
                "ks_oprf_elements": ks["oprf_evaluated_element_count"],
                "ks_oprf_compute_s": ks["oprf_evaluation_compute_seconds"],
            })


def _write_markdown(root: Path, results: Sequence[Mapping[str, Any]]) -> None:
    """Write a concise bilingual metric-boundary description.

    写入简洁的双语指标边界说明。
    """
    lines = [
        "# DwT-FL dynamic-client-join evaluation / DwT-FL 动态客户端加入评估",
        "",
        "The measured endpoint is from the first additional client's actual join-protocol start "
        "to the last additional client's actual local-training subprocess launch. "
        "Base-round establishment is completed before this clock. / 计时边界为首个额外客户端实际启动加入协议，"
        "至最后一个额外客户端实际启动本地训练子进程；基础轮次在该时钟前完成。",
        "",
        "| Joining clients / 加入客户端数 | Status / 状态 | End-to-end to training start (s) / 至训练启动端到端时间（秒） |",
        "|---:|---|---:|",
    ]
    for result in results:
        value = result.get("value", "-")
        if result.get("status") == "completed":
            metric = float(result["join_end_to_end_to_training_start_seconds"])
            lines.append(f"| {value} | completed / 完成 | {metric:.6f} |")
        else:
            status = str(result.get("status", "failed"))
            lines.append(f"| {value} | {status} / {status} | - |")
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a compact JSON report. / 原子写入紧凑 JSON 报告。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _jsonable(value: Any) -> Any:
    """Convert paths and nested immutable objects into JSON-safe values.

    将路径和嵌套不可变对象转换为 JSON 安全值。
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    return value
