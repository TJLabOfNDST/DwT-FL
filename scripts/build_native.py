'Build the DwT-FL native index shared library.'

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def default_compiler() -> str:
    'Choose a compiler available on Windows, WSL, or Ubuntu.'
    configured = os.environ.get("CXX")
    if configured:
        return configured
    return "g++" if shutil.which("g++") is not None else "cl"


def default_library_path() -> Path:
    'Return the packaged library location for the active platform.'
    suffix = ".dll" if platform.system() == "Windows" else ".so"
    return PROJECT_ROOT / "src" / "dbtfl" / "native" / f"atomic_word{suffix}"


def parse_args() -> argparse.Namespace:
    'Parse deterministic native-build options.'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "native" / "atomic_word.cpp",
        help="C++ source file / C++ ",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_library_path(),
        help="Shared-library output path / ",
    )
    parser.add_argument(
        "--compiler",
        default=default_compiler(),
        help="C++ compiler command / C++ ",
    )
    return parser.parse_args()


def build(source: Path, output: Path, compiler: str) -> None:
    'Compile one source file with reproducible release flags.'
    source = source.resolve()
    output = output.resolve()
    compiler_path = shutil.which(compiler)
    if compiler_path is None:
        raise RuntimeError(f"C++ compiler was not found: {compiler}")
    if not source.is_file():
        raise FileNotFoundError(f"Native source was not found: {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    is_msvc = Path(compiler_path).name.lower() in {"cl", "cl.exe"}
    if is_msvc:
        command = [
            compiler_path,
            "/std:c++20",
            "/O2",
            "/LD",
            str(source),
            f"/Fe:{output}",
        ]
    else:
        command = [compiler_path, "-std=c++20", "-O2", "-shared", "-pthread"]
        if platform.system() == "Windows":
            # Ship MinGW C++ runtime support inside the DLL. This avoids a
            # hidden PATH dependency when the benchmark is invoked by a
            # different Python distribution.
            
            command.extend(["-static-libstdc++", "-static-libgcc"])
        else:
            command.append("-fPIC")
        command.extend([str(source), "-o", str(output)])
    subprocess.run(command, check=True)


def main() -> int:
    'Build the requested artifact and print its absolute path.'
    args = parse_args()
    build(args.source, args.output, args.compiler)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
