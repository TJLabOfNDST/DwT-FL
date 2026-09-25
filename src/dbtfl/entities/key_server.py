'Deployable Key Server entity for the DwT-FL OPRF service.\nDwT-FL OPRF'

from __future__ import annotations

import ssl
from dataclasses import dataclass
from pathlib import Path

from dbtfl.communication import RequestRejected, ThreadedJsonServer, WireMessage
from dbtfl.communication.endpoints import KeyServerPath
from dbtfl.oprf import KeyServerOprfService, OprfKeyStore


KS_METRICS_REQUEST = "ks.evaluation.metrics.request"
KS_METRICS_RESPONSE = "ks.evaluation.metrics.response"


@dataclass(frozen=True, slots=True)
class KeyServerConfig:
    'Network and local-key configuration owned only by the KS process.'

    key_path: Path
    host: str = "0.0.0.0"
    port: int = 18081
    advertised_host: str | None = None
    ssl_context: ssl.SSLContext | None = None

    def __post_init__(self) -> None:
        'Reject invalid deployment binding parameters early.'
        if not self.host:
            raise ValueError("KS host must not be empty / KS ")
        if not 0 <= self.port <= 65535:
            raise ValueError("KS port must be in 0..65535 / KS  0..65535")


class KeyServerEntity:
    'Own a KS private scalar and expose only blind OPRF evaluation.'

    def __init__(self, config: KeyServerConfig) -> None:
        'Load the host-local secret and bind, but do not start, the listener.'
        self.config = config
        material = OprfKeyStore(config.key_path).load_or_create()
        self._oprf_service = KeyServerOprfService(material)
        self._oprf_service.router.add(
            "POST",
            KeyServerPath.METRICS.value,
            self.evaluation_metrics,
        )
        self._server = ThreadedJsonServer(
            config.host,
            config.port,
            self._oprf_service.router,
            ssl_context=config.ssl_context,
            advertised_host=config.advertised_host,
        )

    @property
    def base_url(self) -> str:
        'Return the active HTTP(S) endpoint without exposing KS key material.'
        return self._server.base_url

    @property
    def port(self) -> int:
        'Return the actual bound port, including an OS-selected port zero.'
        return self._server.port

    def evaluation_metrics(self, message: WireMessage) -> WireMessage:
        'Return the private-key storage size without exposing key material.'
        if message.message_type != KS_METRICS_REQUEST or dict(message.payload):
            raise RequestRejected(
                400,
                "invalid_ks_metrics_request",
                "KS metrics request must have an empty payload / KS ",
            )
        key_path = Path(self.config.key_path)
        metrics = self._oprf_service.metrics_snapshot()
        return WireMessage.create(
            KS_METRICS_RESPONSE,
            {
                "private_key_bytes": key_path.stat().st_size if key_path.is_file() else 0,
                "oprf_evaluated_element_count": metrics["evaluated_element_count"],
                "oprf_evaluation_compute_seconds": metrics["evaluation_compute_seconds"],
            },
            request_id=message.request_id,
        )

    def start(self) -> None:
        'Start the KS listener on its managed daemon thread.'
        self._server.start()

    def close(self) -> None:
        'Stop the KS listener and release its network resources.'
        self._server.close()

    def __enter__(self) -> "KeyServerEntity":
        'Start the entity when entering a managed lifetime.'
        self.start()
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        'Close the listener when leaving a managed lifetime.'
        self.close()
