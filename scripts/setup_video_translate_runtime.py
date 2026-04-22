#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Установщик зависимостей video-translate в локальный runtime Python.

Примеры:
  python scripts/setup_video_translate_runtime.py --runtime-python e:/Neyro/ShortsCreatorStudio/runtime/python.exe --profile cpu
  python scripts/setup_video_translate_runtime.py --runtime-python e:/Neyro/ShortsCreatorStudio/runtime/python.exe --profile gpu-cu124
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


CORE_PACKAGES = [
    "coqui-tts==0.27.5",
    "faster-whisper==1.2.1",
    "pyannote.audio==4.0.4",
    "transformers==4.57.1",
    "tokenizers==0.22.1",
]


def run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_with_env(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("[RUN]", " ".join(cmd))
    merged = os.environ.copy()
    if env:
        merged.update(env)
    subprocess.run(cmd, check=True, env=merged)


def install_torch(runtime_python: str, profile: str) -> None:
    if profile == "cpu":
        run([runtime_python, "-m", "pip", "install", "--upgrade", "torch", "torchaudio"])
        return

    if profile == "gpu-cu124":
        run(
            [
                runtime_python,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "torch",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cu124",
            ]
        )
        return

    raise ValueError(f"Unknown profile: {profile}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-python", required=True, help="Путь к runtime/python.exe")
    parser.add_argument(
        "--profile",
        default="cpu",
        choices=["cpu", "gpu-cu124"],
        help="Профиль установки PyTorch",
    )
    parser.add_argument("--upgrade-pip", action="store_true", help="Обновить pip перед установкой")
    parser.add_argument(
        "--skip-xtts-preload",
        action="store_true",
        help="Не выполнять предзагрузку XTTS v2 (по умолчанию предзагрузка включена)",
    )
    args = parser.parse_args()

    runtime_python = str(Path(args.runtime_python))
    if not Path(runtime_python).exists():
        print(f"[ERROR] runtime python not found: {runtime_python}")
        return 2

    run(
        [
            runtime_python,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "setuptools<82",
            "wheel",
        ]
        if args.upgrade_pip
        else [runtime_python, "-m", "pip", "--version"]
    )

    install_torch(runtime_python, args.profile)
    run([runtime_python, "-m", "pip", "install", "--upgrade", *CORE_PACKAGES])

    if not args.skip_xtts_preload:
        # Неинтерактивное принятие CPML/TOS для XTTS при автоподготовке.
        # Это убирает блокирующий input() в Coqui TTS manager.
        run_with_env(
            [runtime_python, "scripts\\download_xtts_model.py"],
            env={"COQUI_TOS_AGREED": "1"},
        )

    # Быстрая проверка импортов
    run(
        [
            runtime_python,
            "-c",
            "import torch,transformers,tokenizers,faster_whisper;"
            "from TTS.api import TTS;"
            "import pyannote.audio;"
            "print('OK', torch.__version__, 'cuda=', torch.cuda.is_available())",
        ]
    )

    print("\n[OK] Runtime готов для модуля video translate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
