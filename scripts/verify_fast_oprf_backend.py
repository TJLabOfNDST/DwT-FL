'Verify that DwT-FL can use the required native Ristretto255 OPRF backend.'

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dbtfl.oprf.ristretto255 import (
    RistrettoBackendUnavailable,
    hash_to_group,
    native_backend_available,
    random_scalar,
    scalar_inverse,
    scalar_multiply,
)


def main() -> int:
    'Check backend identity and one blind/unblind algebra round.'
    if not native_backend_available():
        print(
            "FAILED: native libsodium Ristretto255 backend is unavailable; pure-Python fallback is refused / "
            " libsodium Ristretto255  Python ",
            file=sys.stderr,
        )
        return 2
    try:
        point = hash_to_group(b"DwT-FL native backend preflight")
        blind = random_scalar()
        restored = scalar_multiply(scalar_inverse(blind), scalar_multiply(blind, point))
    except RistrettoBackendUnavailable as error:
        print(f"FAILED: {error}", file=sys.stderr)
        return 2
    if restored != point:
        print("FAILED: Ristretto blind/unblind self-check failed / Ristretto /", file=sys.stderr)
        return 3
    print("OK: native libsodium Ristretto255 backend is active; pure-Python fallback is disabled /  libsodium Ristretto255  Python ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
