#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Локальный XTTS синтез в отдельном runtime-процессе.

Используется как обходной путь, когда импорт XTTS из frozen EXE
падает по DLL, но в runtime/python.exe работает.
"""

from __future__ import annotations

import argparse
import datetime
import os
import traceback
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--speaker-wav", default="")
    parser.add_argument("--log-file", default="")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    log_file = Path(args.log_file) if str(args.log_file).strip() else None

    # Новый запуск — очищаем прошлый debug, чтобы видеть актуальную причину.
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text("", encoding="utf-8")

    def _log(msg: str):
        print(msg)
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with log_file.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")

    _log("=== XTTS synth start ===")
    _log(f"time={datetime.datetime.now().isoformat()}")
    _log(f"out={out}")
    _log(f"lang={args.lang}")

    # torchaudio>=2.8 по умолчанию может идти через torchcodec.
    # В нашем runtime это ломается из-за отсутствующих torchcodec DLL,
    # поэтому принудительно маршрутизируем загрузку через soundfile backend.
    os.environ.setdefault("TORCHAUDIO_USE_BACKEND_DISPATCHER", "0")

    import soundfile as sf  # type: ignore
    import torch  # type: ignore
    import torchaudio  # type: ignore

    _log(f"torch={getattr(torch, '__version__', 'unknown')}")
    _log(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            _log(f"cuda_device={torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    try:
        if hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend("soundfile")
    except Exception as e:
        _log(f"set_audio_backend_warning={e}")

    try:
        _orig_torchaudio_load = torchaudio.load

        def _safe_torchaudio_load(*a, **kw):
            kw.setdefault("backend", "soundfile")
            return _orig_torchaudio_load(*a, **kw)

        torchaudio.load = _safe_torchaudio_load
        _log("torchaudio.load patched -> backend=soundfile")
    except Exception as e:
        _log(f"torchaudio_patch_warning={e}")

    # Жёсткий monkey-patch конкретно для XTTS: обходим torchaudio/torchcodec полностью.
    def _sf_load_for_xtts(audiopath, *_, **__):
        wav, sr = sf.read(str(audiopath), dtype="float32", always_2d=False)
        if isinstance(wav, np.ndarray):
            if wav.ndim == 1:
                wav = wav[np.newaxis, :]
            elif wav.ndim == 2:
                wav = wav.T
        wav_t = torch.from_numpy(wav) if isinstance(wav, np.ndarray) else wav
        return wav_t, int(sr)

    from TTS.api import TTS  # type: ignore
    import TTS.tts.models.xtts as xtts_mod  # type: ignore

    xtts_mod.torchaudio.load = _sf_load_for_xtts
    _log("xtts.torchaudio.load patched -> soundfile (no torchcodec)")

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    if torch.cuda.is_available():
        try:
            tts.to("cuda")
            _log("xtts_device=cuda")
        except Exception as e:
            _log(f"xtts_cuda_move_warning={e}")
    else:
        _log("xtts_device=cpu")
    speaker_wav = Path(args.speaker_wav) if str(args.speaker_wav).strip() else None
    if speaker_wav:
        _log(f"speaker_wav={speaker_wav}, exists={speaker_wav.exists()}")
        if speaker_wav.exists():
            _log(f"speaker_wav_size={speaker_wav.stat().st_size}")
    try:
        if speaker_wav and speaker_wav.exists():
            tts.tts_to_file(
                text=args.text,
                speaker_wav=str(speaker_wav),
                language=str(args.lang or "en"),
                file_path=str(out),
            )
        else:
            # Без референса будет обычный TTS тембр, но не клонирование.
            tts.tts_to_file(
                text=args.text,
                language=str(args.lang or "en"),
                file_path=str(out),
            )
    except Exception as e:
        _log(f"xtts_exception={e}")
        _log(traceback.format_exc())
        raise

    if not out.exists() or out.stat().st_size <= 0:
        raise RuntimeError("XTTS did not produce output wav")
    _log("=== XTTS synth success ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
