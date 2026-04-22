#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import soundfile as sf  # type: ignore
import torch  # type: ignore
import torchaudio  # type: ignore
from TTS.api import TTS  # type: ignore
import TTS.tts.models.xtts as xtts_mod  # type: ignore


def _patch_audio_loading():
    os.environ.setdefault("TORCHAUDIO_USE_BACKEND_DISPATCHER", "0")
    try:
        if hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend("soundfile")
    except Exception:
        pass

    def _sf_load_for_xtts(audiopath, *_, **__):
        wav, sr = sf.read(str(audiopath), dtype="float32", always_2d=False)
        if isinstance(wav, np.ndarray):
            if wav.ndim == 1:
                wav = wav[np.newaxis, :]
            elif wav.ndim == 2:
                wav = wav.T
        wav_t = torch.from_numpy(wav) if isinstance(wav, np.ndarray) else wav
        return wav_t, int(sr)

    xtts_mod.torchaudio.load = _sf_load_for_xtts


def _read_json(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    raw = handler.rfile.read(length) if length > 0 else b"{}"
    try:
        return json.loads(raw.decode("utf-8", errors="ignore"))
    except Exception:
        return {}


class _State:
    tts: TTS | None = None
    device: str = "cpu"
    requested_device: str = "auto"
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"


class Handler(BaseHTTPRequestHandler):
    server_version = "XTTSRuntime/1.0"

    def _send_json(self, obj: dict, code: int = 200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(
                {
                    "ok": True,
                    "device": _State.device,
                    "requested_device": _State.requested_device,
                    "cuda_available": bool(torch.cuda.is_available()),
                    "model": _State.model_name,
                    "ready": _State.tts is not None,
                }
            )
            return
        self._send_json({"ok": False, "error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/tts":
            self._send_json({"ok": False, "error": "not found"}, 404)
            return
        data = _read_json(self)
        text = str(data.get("text") or "").strip()
        language = str(data.get("language") or "en").strip() or "en"
        speaker_wav = str(data.get("speaker_wav") or "").strip()
        if not text:
            self._send_json({"ok": False, "error": "text required"}, 400)
            return

        try:
            if _State.tts is None:
                raise RuntimeError("XTTS not initialized")

            tmp_out = Path(os.getenv("TEMP", ".")) / f"xtts_rt_{os.getpid()}.wav"
            if speaker_wav and Path(speaker_wav).exists():
                _State.tts.tts_to_file(
                    text=text,
                    speaker_wav=speaker_wav,
                    language=language,
                    file_path=str(tmp_out),
                )
            else:
                _State.tts.tts_to_file(
                    text=text,
                    language=language,
                    file_path=str(tmp_out),
                )

            audio = tmp_out.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(audio)))
            self.end_headers()
            self.wfile.write(audio)
        except Exception as e:
            self._send_json({"ok": False, "error": str(e), "trace": traceback.format_exc()[-4000:]}, 500)

    def log_message(self, fmt, *args):
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8021)
    parser.add_argument("--model", default="tts_models/multilingual/multi-dataset/xtts_v2")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    _State.model_name = str(args.model)
    _State.requested_device = str(args.device or "auto")
    _patch_audio_loading()
    _State.tts = TTS(_State.model_name)
    if _State.requested_device == "cpu":
        _State.device = "cpu"
    elif _State.requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for XTTS runtime server, but torch.cuda.is_available() is False")
        _State.tts.to("cuda")
        _State.device = "cuda"
    else:
        if torch.cuda.is_available():
            try:
                _State.tts.to("cuda")
                _State.device = "cuda"
            except Exception:
                _State.device = "cpu"

    httpd = ThreadingHTTPServer((args.host, int(args.port)), Handler)
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
