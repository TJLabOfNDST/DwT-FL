'Paper-compatible OPRF arithmetic in the RFC 3526 group 14 subgroup.'

from __future__ import annotations

import hashlib
import secrets
from typing import Final


# RFC 3526, section 3: 2048-bit MODP group (group 14).  Squaring a non-zero
# field element maps it into the prime-order quadratic-residue subgroup.
# RFC 3526

MODP_GROUP14_PRIME: Final[int] = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E08"
    "8A67CC74020BBEA63B139B22514A08798E3404DD"
    "EF9519B3CD3A431B302B0A6DF25F14374FE1356D"
    "6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
    "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8"
    "A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961"
    "C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA"
    "18217C32905E462E36CE3BE39E772C180E86039B2783A2EC07A28FB5"
    "C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898"
    "FA051015728E5A8AACAA68FFFFFFFFFFFFFFFF",
    16,
)
"""The RFC 3526 group 14 safe prime. / RFC 3526  14 """

SUBGROUP_ORDER: Final[int] = (MODP_GROUP14_PRIME - 1) // 2
"""Prime order q of the quadratic-residue subgroup. /  q"""

ELEMENT_HEX_LENGTH: Final[int] = 512
"""Fixed lower-hex element encoding length. / """

MAX_INPUT_BYTES: Final[int] = 65_535
"""Bound for one application record before hashing. / """

_HASH_DOMAIN: Final[bytes] = b"DwT-FL/OPRF/RFC3526-group14/H1/v1\x00"


class OprfValidationError(ValueError):
    'Raised for non-canonical OPRF inputs or subgroup elements.'


def normalize_record(record: str | bytes) -> bytes:
    'Encode one client-only record deterministically and enforce its bound.'
    if isinstance(record, str):
        encoded = record.encode("utf-8")
    elif isinstance(record, bytes):
        encoded = record
    else:
        raise TypeError("record must be str or bytes /  str  bytes")
    if not encoded:
        raise OprfValidationError("record must not be empty / ")
    if len(encoded) > MAX_INPUT_BYTES:
        raise OprfValidationError(
            f"record exceeds {MAX_INPUT_BYTES} bytes /  {MAX_INPUT_BYTES} "
        )
    return encoded


def random_scalar() -> int:
    'Return a cryptographically random non-zero scalar in Z_q.'
    return secrets.randbelow(SUBGROUP_ORDER - 1) + 1


def validate_scalar(value: int) -> int:
    'Validate a private or blinding scalar without normalizing it.'
    if not isinstance(value, int) or not 1 <= value < SUBGROUP_ORDER:
        raise OprfValidationError("scalar is outside Z_q* /  Z_q* ")
    return value


def hash_to_subgroup(record: str | bytes) -> int:
    "Compute the paper's H(m) as a domain-separated subgroup element.\n    SHAKE-256 rejection sampling avoids modulo bias when mapping to the field.\n    SHAKE-256"
    encoded = normalize_record(record)
    framed_record = _HASH_DOMAIN + len(encoded).to_bytes(4, "big") + encoded
    counter = 0
    while True:
        digest = hashlib.shake_256(
            framed_record + counter.to_bytes(4, "big")
        ).digest(ELEMENT_HEX_LENGTH // 2)
        candidate = int.from_bytes(digest, "big")
        if 1 < candidate < MODP_GROUP14_PRIME - 1:
            element = pow(candidate, 2, MODP_GROUP14_PRIME)
            if element != 1:
                return element
        counter += 1
        if counter >= 2**32:
            raise RuntimeError(
                "hash-to-subgroup counter exhausted / "
            )


def is_subgroup_element(element: int) -> bool:
    'Return whether an integer is a non-identity member of the q subgroup.'
    return (
        isinstance(element, int)
        and 1 < element < MODP_GROUP14_PRIME - 1
        and pow(element, SUBGROUP_ORDER, MODP_GROUP14_PRIME) == 1
    )


def encode_element(element: int) -> str:
    'Return one canonical lower-hex subgroup encoding for the native index.'
    if not is_subgroup_element(element):
        raise OprfValidationError(
            "value is not a valid subgroup element / "
        )
    return f"{element:0{ELEMENT_HEX_LENGTH}x}"


def decode_element(encoded_element: str) -> int:
    'Decode and validate one canonical lower-hex subgroup encoding.'
    is_canonical = (
        isinstance(encoded_element, str)
        and len(encoded_element) == ELEMENT_HEX_LENGTH
    )
    if not is_canonical or any(
        character not in "0123456789abcdef" for character in encoded_element
    ):
        raise OprfValidationError(
            "element must contain exactly 512 lowercase hexadecimal characters / "
            " 512 "
        )
    element = int(encoded_element, 16)
    if not is_subgroup_element(element):
        raise OprfValidationError("element is outside the q subgroup /  q ")
    return element
