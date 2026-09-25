'Persistent Ristretto255 secret handling for the KS OPRF service.\nKS OPRF'

from __future__ import annotations

import base64
import binascii
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .group14 import OprfValidationError
from .ristretto255 import OPRF_SUITE_IDENTIFIER, random_scalar, validate_scalar


KEY_FILE_VERSION: Final[int] = 2
"""Version for the native Ristretto255 KS key file. /  Ristretto255 KS """

GROUP_IDENTIFIER: Final[str] = OPRF_SUITE_IDENTIFIER
"""Explicit cryptographic-suite binding for KS secrets. / KS """


class OprfKeyStoreError(RuntimeError):
    'Raised when the KS secret file does not meet the selected suite contract.'


@dataclass(frozen=True, slots=True)
class OprfKeyMaterial:
    'Validated 32-byte Ristretto255 scalar held only by the KS process.'

    scalar: bytes

    def __post_init__(self) -> None:
        'Reject malformed scalar material at the persistence trust boundary.'
        validate_scalar(self.scalar)


class OprfKeyStore:
    'Load one Ristretto255 secret or atomically create a new local secret.'

    def __init__(self, path: str | Path) -> None:
        'Bind the store to one concrete local filesystem path.'
        self.path = Path(path)

    def load_or_create(self) -> OprfKeyMaterial:
        'Load a valid key and never replace malformed or legacy material silently.'
        if self.path.exists():
            return self._load_existing()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        material = OprfKeyMaterial(random_scalar())
        serialized = self._serialize(material)
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return self._load_existing()
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                output.write(serialized)
                output.flush()
                os.fsync(output.fileno())
        except Exception:
            try:
                self.path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        return material

    def _load_existing(self) -> OprfKeyMaterial:
        'Load and validate an existing Ristretto255 key without changing it.'
        if self.path.is_symlink():
            raise OprfKeyStoreError("KS key path must not be a symlink / KS ")
        try:
            parsed = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise OprfKeyStoreError("cannot read KS key file /  KS ") from error
        if not isinstance(parsed, dict):
            raise OprfKeyStoreError("unsupported KS key-file schema /  KS ")
        if parsed.get("version") == 1:
            raise OprfKeyStoreError(
                "legacy MODP KS key cannot be used with Ristretto255; configure a new key path / "
                " MODP KS  Ristretto255"
            )
        if parsed.get("version") != KEY_FILE_VERSION or parsed.get("suite") != GROUP_IDENTIFIER:
            raise OprfKeyStoreError("KS key is bound to another OPRF suite / KS  OPRF ")
        serialized_scalar = parsed.get("k_b64")
        if not isinstance(serialized_scalar, str):
            raise OprfKeyStoreError("KS private scalar is invalid / KS ")
        try:
            scalar = base64.b64decode(serialized_scalar.encode("ascii"), validate=True)
            return OprfKeyMaterial(scalar)
        except (UnicodeEncodeError, ValueError, binascii.Error, OprfValidationError) as error:
            raise OprfKeyStoreError("KS private scalar is invalid / KS ") from error

    @staticmethod
    def _serialize(material: OprfKeyMaterial) -> str:
        'Serialize a suite-bound key without exposing derived public material.'
        return json.dumps(
            {
                "version": KEY_FILE_VERSION,
                "suite": GROUP_IDENTIFIER,
                "k_b64": base64.b64encode(material.scalar).decode("ascii"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n"
