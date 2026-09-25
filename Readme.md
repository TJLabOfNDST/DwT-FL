# DwT-FL

This project runs clients through blind OPRF labeling, protected-label registration, CAS task claiming, model-update upload, FedAvg, and global-model download.

## Requirements

- Python 3.10 or later.
- The Ristretto255 OPRF backend supplied by `oblivious[rbcl]`.
- PyTorch and Safetensors for the demo's tiny model updates.
- A native DwT-FL index library. This package includes
  `src/dbtfl/native/atomic_word.dll` for Windows. Linux users must provide a
  compatible `atomic_word.so` through `DBTFL_NATIVE_LIBRARY` or the demo's
  `--native-library` option.

## Installation

Create an isolated environment and install the demo dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-training.txt
```

For library-only use without model-update aggregation, install the smaller
dependency set instead:

```powershell
python -m pip install -r requirements.txt
```

## Run the local protocol demo

Run one complete local protocol round:

```powershell
python scripts/run_protocol_demo.py
```

The script starts an in-process Key Server and Aggregation Server on local
ephemeral ports, creates two clients, and verifies the downloaded FedAvg model.
Each run writes retained artifacts and `summary.json` below
`results/protocol-demo/`.

When the native library is stored elsewhere, pass its explicit path:

```powershell
python scripts/run_protocol_demo.py --native-library C:\path\to\atomic_word.dll
```

## Troubleshooting

- If the OPRF backend is unavailable, reinstall the dependency with
  `python -m pip install "oblivious[rbcl]>=7.0"` and make sure its libsodium
  backend is available on the host.
- If the native index cannot be found, use the bundled Windows DLL or provide
  a matching shared library with `--native-library`.
- The demo requires the `requirements-training.txt` dependencies because it
  performs real Safetensors FedAvg. It intentionally does not download a GPT
  model or run paper experiments.
