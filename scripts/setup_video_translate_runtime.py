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
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


CORE_PACKAGES = [
    "coqui-tts==0.27.5",
    "faster-whisper==1.2.1",
    "pyannote.audio==4.0.4",
    "demucs==4.0.1",
    "transformers==4.57.1",
    "tokenizers==0.22.1",
]

ONNXRUNTIME_CANDIDATES = [
    # На разных зеркалах/архитектурах конкретный pin может отсутствовать.
    "onnxruntime==1.22.1",
    "onnxruntime==1.21.1",
    "onnxruntime",
]

RVC_NATIVE_PACKAGES = [
    # Нативный RVC-конвертер (без MMVC-сервера)
    "rvc-python",
]

STUDIO_ADDON_PACKAGES = [
    # Улучшение профиля «студийного» дубляжа и QA-аудио
    "pyloudnorm==0.1.1",
    "noisereduce==3.0.3",
    "pedalboard==0.9.19",
    "audio-separator==0.30.1",
]

STUDIO_ADDON_MODULES = [
    "pyloudnorm",
    "noisereduce",
    "pedalboard",
    "audio_separator",
]

# Эти пакеты часто конфликтуют/требуют сборку на Windows (Python headers/VS Build Tools).
# Ставим их отдельно, опционально и в best-effort режиме.
EXPERIMENTAL_ADDON_PACKAGES = [
    "whisperx==3.3.1",
    "resemblyzer==0.1.4",
    "webrtcvad-wheels==2.0.14",
]

EXPERIMENTAL_ADDON_MODULES = [
    "whisperx",
    "resemblyzer",
    "webrtcvad",
]


def run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def try_run(cmd: list[str]) -> bool:
    print("[TRY]", " ".join(cmd))
    p = subprocess.run(cmd)
    return p.returncode == 0


def install_packages_best_effort(
    runtime_python: str,
    packages: list[str],
    group_name: str,
    env: dict[str, str] | None = None,
) -> tuple[list[str], list[str]]:
    ok: list[str] = []
    failed: list[str] = []
    for pkg in packages:
        cmd = [runtime_python, "-m", "pip", "install", "--upgrade", pkg]
        print("[TRY]", " ".join(cmd))
        merged = os.environ.copy()
        if env:
            merged.update(env)
        p = subprocess.run(cmd, env=merged)
        if p.returncode == 0:
            ok.append(pkg)
        else:
            failed.append(pkg)
            print(f"[WARN] {group_name}: failed to install {pkg}")
    return ok, failed


def install_onnxruntime_with_fallback(runtime_python: str) -> str:
    last_err = ""
    for pkg in ONNXRUNTIME_CANDIDATES:
        cmd = [runtime_python, "-m", "pip", "install", "--upgrade", pkg]
        if try_run(cmd):
            print(f"[OK] onnxruntime selected: {pkg}")
            return pkg
        last_err = pkg
    raise RuntimeError(f"Failed to install onnxruntime (last attempt: {last_err})")


def run_with_env(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("[RUN]", " ".join(cmd))
    merged = os.environ.copy()
    if env:
        merged.update(env)
    subprocess.run(cmd, check=True, env=merged)


def maybe_download_uvr_models(runtime_python: str) -> None:
    downloader = SCRIPT_DIR / "download_uvr_models.py"
    if not downloader.exists():
        print(f"[WARN] UVR downloader script not found: {downloader}")
        return
    run([runtime_python, str(downloader)])


def install_torch(runtime_python: str, profile: str) -> None:
    if profile == "none":
        print("[INFO] Torch install skipped (profile=none)")
        return

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

    if profile == "gpu-cu128":
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
                "https://download.pytorch.org/whl/cu128",
            ]
        )
        return

    if profile == "gpu-auto":
        # Порядок попыток: cu128 -> cu124 -> cpu
        attempts = [
            (
                "gpu-cu128",
                [
                    runtime_python,
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "torch",
                    "torchaudio",
                    "--index-url",
                    "https://download.pytorch.org/whl/cu128",
                ],
            ),
            (
                "gpu-cu124",
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
                ],
            ),
            (
                "cpu",
                [runtime_python, "-m", "pip", "install", "--upgrade", "torch", "torchaudio"],
            ),
        ]
        for label, cmd in attempts:
            if try_run(cmd):
                print(f"[OK] Torch profile selected: {label}")
                return
        raise RuntimeError("Failed to install torch for all auto profiles (cu128/cu124/cpu)")

    raise ValueError(f"Unknown profile: {profile}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-python", required=True, help="Путь к runtime/python.exe")
    parser.add_argument(
        "--profile",
        default="cpu",
        choices=["cpu", "gpu-cu124", "gpu-cu128", "gpu-auto", "none"],
        help="Профиль установки PyTorch",
    )
    parser.add_argument("--upgrade-pip", action="store_true", help="Обновить pip перед установкой")
    parser.add_argument(
        "--skip-xtts-preload",
        action="store_true",
        help="Не выполнять предзагрузку XTTS v2 (по умолчанию предзагрузка включена)",
    )
    parser.add_argument(
        "--skip-core-packages",
        action="store_true",
        help="Не ставить CORE_PACKAGES (TTS/faster-whisper/pyannote/demucs/transformers)",
    )
    parser.add_argument(
        "--with-studio-addons",
        action="store_true",
        help="Установить безопасные studio-дополнения (loudness/denoise/mix fx)",
    )
    parser.add_argument(
        "--with-experimental-addons",
        action="store_true",
        help="Установить экспериментальные пакеты (whisperx/resemblyzer/webrtcvad) в best-effort режиме",
    )
    parser.add_argument(
        "--restore-core-after-experimental",
        action="store_true",
        help="После experimental-установки вернуть production-версии core-пакетов",
    )
    parser.add_argument(
        "--with-uvr-models",
        action="store_true",
        help="Скачать UVR/MDX модели (Inst HQ3 + Kim Vocal 2) в AppData/models/uvr",
    )
    parser.add_argument(
        "--with-rvc-native",
        action="store_true",
        help="Установить зависимости для native RVC inference (без MMVC)",
    )
    args = parser.parse_args()

    runtime_python = str(Path(args.runtime_python))
    if not Path(runtime_python).exists():
        print(f"[ERROR] runtime python not found: {runtime_python}")
        return 2

    if args.upgrade_pip:
        # Для rvc-python нужен fairseq -> omegaconf==2.0.6.
        # У pip>=24.1 строгая валидация метаданных, из-за чего эта версия
        # omegaconf отбрасывается (ResolutionImpossible). Поэтому при включенном
        # native RVC удерживаем pip на ветке <24.1.
        pip_spec = "pip<24.1" if args.with_rvc_native else "pip"
        pip_upgrade_cmd = [
            runtime_python,
            "-m",
            "pip",
            "install",
            "--upgrade",
            pip_spec,
            "setuptools<82",
            "wheel",
        ]
        # Сетевые/зеркальные проблемы не должны ломать весь installer.
        if not try_run(pip_upgrade_cmd):
            print("[WARN] pip/setuptools/wheel upgrade failed, continue with current versions")
            run([runtime_python, "-m", "pip", "--version"])
    else:
        run([runtime_python, "-m", "pip", "--version"])

    install_torch(runtime_python, args.profile)
    if not args.skip_core_packages:
        run([runtime_python, "-m", "pip", "install", "--upgrade", *CORE_PACKAGES])
        install_onnxruntime_with_fallback(runtime_python)
    else:
        print("[INFO] CORE packages install skipped")

    failed_groups: dict[str, list[str]] = {}

    if args.with_studio_addons:
        _, failed = install_packages_best_effort(
            runtime_python,
            STUDIO_ADDON_PACKAGES,
            group_name="studio-addons",
        )
        if failed:
            failed_groups["studio-addons"] = failed

    if args.with_experimental_addons:
        _, failed = install_packages_best_effort(
            runtime_python,
            EXPERIMENTAL_ADDON_PACKAGES,
            group_name="experimental-addons",
        )
        if failed:
            failed_groups["experimental-addons"] = failed

    if args.with_experimental_addons and args.restore_core_after_experimental:
        print("[INFO] Restoring production core package pins after experimental install...")
        run([runtime_python, "-m", "pip", "install", "--upgrade", *CORE_PACKAGES])

    if not args.skip_xtts_preload:
        # Неинтерактивное принятие CPML/TOS для XTTS при автоподготовке.
        # Это убирает блокирующий input() в Coqui TTS manager.
        run_with_env(
            [runtime_python, str(SCRIPT_DIR / "download_xtts_model.py")],
            env={"COQUI_TOS_AGREED": "1"},
        )

    if args.with_uvr_models:
        maybe_download_uvr_models(runtime_python)

    rvc_modules_for_smoke: list[str] = []
    if args.with_rvc_native:
        if sys.version_info >= (3, 11):
            print(
                "[WARN] Python >= 3.11: rvc-python/fairseq в текущем виде нестабильны "
                "(ломают импорт и тянут конфликтные версии). Пропускаем rvc-native в этом runtime."
            )
            failed_groups["rvc-native"] = ["unsupported-python>=3.11"]
        else:
            rvc_modules_for_smoke = ["rvc_python", "rvc"]
            _, failed = install_packages_best_effort(
                runtime_python,
                RVC_NATIVE_PACKAGES,
                group_name="rvc-native",
                # fairseq из rvc-python на Windows часто пытается собирать C/C++
                # расширения (libbleu) и падает на portable runtime без Python.h.
                # READTHEDOCS=1 отключает сборку extensions в setup.py fairseq.
                env={"READTHEDOCS": "1"},
            )
            if failed:
                failed_groups["rvc-native"] = failed

            # Важно: rvc-python может понижать numpy/scipy/omegaconf и ломать
            # production-стек XTTS/pyannote/faster-whisper в общем runtime.
            # Возвращаем рабочие пины сразу после попытки RVC-установки.
            print("[INFO] Restoring production core package pins after rvc-native step...")
            run([runtime_python, "-m", "pip", "install", "--upgrade", *CORE_PACKAGES])
            install_onnxruntime_with_fallback(runtime_python)

    # Быстрая проверка импортов
    addon_modules: list[str] = []
    if args.with_studio_addons:
        addon_modules += STUDIO_ADDON_MODULES
    if args.with_experimental_addons and not args.restore_core_after_experimental:
        addon_modules += EXPERIMENTAL_ADDON_MODULES

    addon_modules_expr = repr(addon_modules + rvc_modules_for_smoke)
    smoke_code = (
        "import torch,importlib\n"
        "mods=[]\n"
        "try:\n"
        "  [importlib.import_module(m) for m in ['transformers','tokenizers','faster_whisper','demucs','TTS.api','pyannote.audio']]\n"
        "  mods.append('core=ok')\n"
        "except Exception as e:\n"
        "  mods.append('core=skip:'+str(e)[:120])\n"
        f"addon_mods={addon_modules_expr}\n"
        "if addon_mods:\n"
        "  try:\n"
        "    [importlib.import_module(m) for m in addon_mods]\n"
        "    mods.append('studio=ok')\n"
        "  except Exception as e:\n"
        "    mods.append('studio=skip:'+str(e)[:120])\n"
        "print('OK', torch.__version__, 'cuda=', torch.cuda.is_available(), ' '.join(mods))\n"
    )
    run([runtime_python, "-c", smoke_code])

    if failed_groups:
        print("\n[WARN] Некоторые addon-пакеты не установлены:")
        for g, items in failed_groups.items():
            print(f"  - {g}: {', '.join(items)}")

    print("\n[OK] Runtime готов для модуля video translate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
