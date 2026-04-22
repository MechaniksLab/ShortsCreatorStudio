#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Скачивание UVR/MDX моделей для разделения речи и фона.

Сохраняет модели в: AppData/models/uvr
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path


DEFAULT_MODELS = {
    "UVR-MDX-NET-Inst_HQ_3.onnx": [
        "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_3.onnx",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/UVR-MDX-NET-Inst_HQ_3.onnx",
    ],
    "Kim_Vocal_2.onnx": [
        "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Kim_Vocal_2.onnx",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/Kim_Vocal_2.onnx",
    ],
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_output_dir() -> Path:
    return _project_root() / "AppData" / "models" / "uvr"


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _download_one(url: str, dst: Path) -> bool:
    tmp = Path(tempfile.gettempdir()) / f"uvr_dl_{dst.name}"
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass
    try:
        print(f"[RUN] Download: {url}")
        with urllib.request.urlopen(url, timeout=240) as r, tmp.open("wb") as f:
            shutil.copyfileobj(r, f)
        if not tmp.exists() or tmp.stat().st_size < 1_000_000:
            print(f"[WARN] Too small file from {url}")
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp), str(dst))
        print(f"[OK] {dst.name}: {dst.stat().st_size} bytes, sha1={_sha1(dst)[:12]}")
        return True
    except Exception as e:
        print(f"[WARN] Download failed from {url}: {e}")
        return False
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(_default_output_dir()))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for model_name, urls in DEFAULT_MODELS.items():
        dst = out_dir / model_name
        if dst.exists() and dst.stat().st_size > 1_000_000:
            print(f"[SKIP] {model_name} already exists ({dst.stat().st_size} bytes)")
            continue

        ok = False
        for u in urls:
            if _download_one(u, dst):
                ok = True
                break
        if not ok:
            missing.append(model_name)

    if missing:
        print("[ERROR] Не удалось скачать модели:", ", ".join(missing))
        return 2

    print(f"[OK] UVR-модели готовы: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
