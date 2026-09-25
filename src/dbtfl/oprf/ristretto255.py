"Constant-time Ristretto255 helpers for the DwT-FL blind OPRF.\nThe module deliberately accepts only the ``oblivious.ristretto.sodium`` path\nwhich delegates group operations to libsodium.  It never falls back to the\npackage's pure-Python arithmetic: a silent fallback would make a deployment's\nthroughput hardware-dependent and would reintroduce the original bottleneck.\nlibsodium"

from __future__ import annotations

import hashlib
from typing import Any, Final

from .group14 import MAX_INPUT_BYTES, OprfValidationError, normalize_record


OPRF_SUITE_IDENTIFIER: Final[str] = "ristretto255-sha512-libsodium-v1"
"""Suite identifier bound to keys and client-local label stores. / """

LEGACY_OPRF_SUITE_IDENTIFIER: Final[str] = "rfc3526-modp-group14-legacy-v1"
"""Read-only identifier for pre-migration MODP label stores. /  MODP """

POINT_BYTES: Final[int] = 32
SCALAR_BYTES: Final[int] = 32
_HASH_DOMAIN: Final[bytes] = b"DwT-FL/OPRF/ristretto255/SHA-512/H1/v1\x00"
_OUTPUT_DOMAIN: Final[bytes] = b"DwT-FL/OPRF/ristretto255/SHAKE-256/output/v1\x00"


class RistrettoBackendUnavailable(RuntimeError):
    'Raised when the required native libsodium Ristretto backend is absent.'


def native_backend_available() -> bool:
    'Return whether a libsodium-backed Ristretto implementation is loaded.'
    try:
        from oblivious.ristretto import sodium
    except ImportError:
        return False
    return sodium is not None


def _backend() -> Any:
    'Return the native backend or fail before handling any OPRF secret.'
    try:
        from oblivious.ristretto import sodium
    except ImportError as error:
        raise RistrettoBackendUnavailable(
            "the fast OPRF requires oblivious with a libsodium backend; install oblivious[rbcl] and libsodium / "
            " OPRF  libsodium  oblivious oblivious[rbcl]  libsodium"
        ) from error
    if sodium is None:
        raise RistrettoBackendUnavailable(
            "oblivious loaded without libsodium; refusing the slow pure-Python fallback / "
            "oblivious  libsodium Python "
        )
    return sodium


def random_scalar() -> bytes:
    'Generate one non-zero native Ristretto scalar.'
    value = bytes(_backend().rnd())
    return validate_scalar(value)


def validate_scalar(value: bytes) -> bytes:
    'Validate canonical non-zero scalar bytes.'
    if not isinstance(value, bytes) or len(value) != SCALAR_BYTES:
        raise OprfValidationError("Ristretto scalar must contain 32 bytes / Ristretto  32 ")
    try:
        scalar = _backend().scl(value)
    except (TypeError, ValueError) as error:
        raise OprfValidationError("Ristretto scalar encoding is invalid / Ristretto ") from error
    if scalar is None or bytes(scalar) != value or value == b"\x00" * SCALAR_BYTES:
        raise OprfValidationError("Ristretto scalar is not canonical and non-zero / Ristretto ")
    return value


def hash_to_group(record: str | bytes) -> bytes:
    'Hash one framed record to a prime-order Ristretto element.'
    encoded = normalize_record(record)
    # ``sodium.pnt`` maps a 64-byte uniform string to Ristretto. Framing keeps
    # records unambiguous before SHA-512. ``sodium.pnt``
    
    framed = _HASH_DOMAIN + len(encoded).to_bytes(4, "big") + encoded
    point = _backend().pnt(hashlib.sha512(framed).digest())
    if point is None:
        raise OprfValidationError("native Ristretto hash-to-group failed /  Ristretto ")
    return validate_point(bytes(point))


def validate_point(value: bytes) -> bytes:
    'Validate one canonical non-identity Ristretto point encoding.'
    if not isinstance(value, bytes) or len(value) != POINT_BYTES:
        raise OprfValidationError("Ristretto point must contain 32 bytes / Ristretto  32 ")
    try:
        point = _backend().point.from_bytes(value)
    except (TypeError, ValueError) as error:
        raise OprfValidationError("Ristretto point encoding is invalid / Ristretto ") from error
    if point is None or bytes(point) != value or value == b"\x00" * POINT_BYTES:
        raise OprfValidationError("Ristretto point is not canonical and non-identity / Ristretto ")
    return value


def scalar_multiply(scalar_bytes: bytes, point_bytes: bytes) -> bytes:
    'Compute a constant-time native scalar multiplication.'
    scalar = validate_scalar(scalar_bytes)
    point = validate_point(point_bytes)
    try:
        result = _backend().mul(scalar, point)
    except (TypeError, ValueError) as error:
        raise OprfValidationError("native Ristretto scalar multiplication failed /  Ristretto ") from error
    return validate_point(bytes(result))


def scalar_inverse(scalar_bytes: bytes) -> bytes:
    'Return the multiplicative inverse of one non-zero scalar.'
    scalar = validate_scalar(scalar_bytes)
    try:
        inverse = _backend().inv(scalar)
    except (TypeError, ValueError) as error:
        raise OprfValidationError("native Ristretto scalar inversion failed /  Ristretto ") from error
    return validate_scalar(bytes(inverse))


def encode_wire_point(value: bytes) -> str:
    'Encode one point compactly for the JSON compatibility envelope.'
    import base64
    return base64.b64encode(validate_point(value)).decode("ascii")


def decode_wire_point(value: str) -> bytes:
    'Decode and validate one point from the JSON compatibility envelope.'
    import base64
    import binascii
    if not isinstance(value, str):
        raise OprfValidationError("Ristretto point wire value must be text / Ristretto ")
    try:
        return validate_point(base64.b64decode(value.encode("ascii"), validate=True))
    except (UnicodeEncodeError, ValueError, binascii.Error) as error:
        raise OprfValidationError("Ristretto point wire encoding is invalid / Ristretto ") from error


def protected_label(point_bytes: bytes) -> str:
    "Expand the 32-byte OPRF output to the native index's 512-hex contract.\n    The expansion is domain-separated and preserves equality exactly; it is not\n    a second OPRF evaluation.\n    OPRF"
    return hashlib.shake_256(_OUTPUT_DOMAIN + validate_point(point_bytes)).hexdigest(256)


def validate_protected_label(label: str) -> str:
    'Validate the native-index-compatible expanded output label.'
    if not isinstance(label, str) or len(label) != 512 or any(character not in "0123456789abcdef" for character in label):
        raise OprfValidationError("protected label must contain 512 lowercase hexadecimal characters /  512 ")
    return label
