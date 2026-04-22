#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Проверка готовности окружения для модуля перевода видео."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _safe_print(text: str):
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        data = str(text).encode(enc, errors="backslashreplace")
        sys.stdout.buffer.write(data + b"\n")


def _run_python(runtime_python: str, code: str) -> tuple[int, str]:
    p = subprocess.run(
        [runtime_python, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode, out.strip()


def main() -> int:
    try:
        # Чтобы рус/кит текст не ломал вывод в cp1251-консоли Windows
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-python", required=True, help="Путь к runtime/python.exe")
    parser.add_argument("--model", default="tiny", help="Whisper модель для preflight")
    args = parser.parse_args()

    runtime_python = str(Path(args.runtime_python))
    if not Path(runtime_python).exists():
        _safe_print(f"[ERROR] runtime python not found: {runtime_python}")
        return 2

    project_root = Path(__file__).resolve().parents[1]
    code = (
        "import json,sys; "
        f"sys.path.insert(0, r'{project_root.as_posix()}'); "
        "from app.core.video_translate.bootstrap import VideoTranslateBootstrap; "
        f"r=VideoTranslateBootstrap().readiness_report(model_name={args.model!r}, auto_download=False); "
        "print(json.dumps(r, ensure_ascii=False))"
    )

    rc, out = _run_python(runtime_python, code)
    if rc != 0:
        _safe_print("[ERROR] Не удалось выполнить preflight.")
        _safe_print(out)
        return 3

    # Берём последнюю JSON-строку (до неё могут идти предупреждения библиотек)
    report_line = ""
    for line in out.splitlines()[::-1]:
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            report_line = line
            break

    if not report_line:
        _safe_print("[ERROR] Не удалось распарсить preflight JSON")
        _safe_print(out)
        return 4

    report = json.loads(report_line)
    _safe_print("=== Video Translate Environment Report ===")
    _safe_print(f"ok: {report.get('ok')}")
    _safe_print(f"asr_runtime: {report.get('asr_runtime')}")
    _safe_print(f"clone_quality: {report.get('clone_quality')}")
    _safe_print(f"diarization_quality: {report.get('diarization_quality')}")
    _safe_print(f"torch_version: {report.get('torch_version')}")
    _safe_print(f"torch_cuda: {report.get('torch_cuda')}")

    checks = report.get("checks", [])
    failed = [c for c in checks if not c.get("ok")]
    if failed:
        _safe_print("\nПроблемы:")
        for c in failed:
            _safe_print(f"- {c.get('name')}: {c.get('hint')}")

    _safe_print("\nРекомендации:")
    if report.get("asr_runtime") == "missing":
        _safe_print(
            "- Установите зависимости: "
            f"{runtime_python} scripts\\setup_video_translate_runtime.py --runtime-python {runtime_python} --profile cpu"
        )
    if not report.get("torch_cuda"):
        _safe_print(
            "- Для GPU установите CUDA-профиль: "
            f"{runtime_python} scripts\\setup_video_translate_runtime.py --runtime-python {runtime_python} --profile gpu-cu124"
        )
    if report.get("clone_quality") != "high":
        _safe_print("- Для точного клонирования тембра проверьте Coqui TTS и модель XTTS v2")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
