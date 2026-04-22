#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import inspect
from pathlib import Path


def _pick_param(sig: inspect.Signature, names: list[str]) -> str | None:
    for n in names:
        if n in sig.parameters:
            return n
    return None


def _call_with_supported_kwargs(fn, kwargs: dict):
    sig = inspect.signature(fn)
    call_kwargs = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            call_kwargs[k] = v
    return fn(**call_kwargs)


def run_native_rvc(
    *,
    input_wav: Path,
    output_wav: Path,
    model_path: Path,
    index_path: Path | None,
    f0_up_key: int,
    index_rate: float,
    protect: float,
    filter_radius: int,
    device: str,
):
    rvc_mod = None
    import_errors = []
    for mod_name in ("rvc_python", "rvc"):
        try:
            rvc_mod = __import__(mod_name)
            break
        except Exception as e:
            import_errors.append(f"{mod_name}: {e}")

    if rvc_mod is None:
        raise RuntimeError(
            "RVC модуль не установлен в runtime (пробовали rvc_python/rvc). "
            "Запустите install_video_translate_*.cmd с поддержкой RVC. "
            f"Ошибки: {' | '.join(import_errors)[:300]}"
        )

    cls = None
    for name in ("RVCInference", "RVC", "VoiceConverter"):
        cls = getattr(rvc_mod, name, None)
        if cls is not None:
            break
    if cls is None:
        raise RuntimeError("RVC модуль найден, но не содержит совместимого класса inference")

    init_sig = inspect.signature(cls)
    init_kwargs = {}
    if _pick_param(init_sig, ["model_path", "model"]):
        init_kwargs[_pick_param(init_sig, ["model_path", "model"])] = str(model_path)
    if index_path and _pick_param(init_sig, ["index_path", "index_file", "index"]):
        init_kwargs[_pick_param(init_sig, ["index_path", "index_file", "index"])] = str(index_path)
    if _pick_param(init_sig, ["device"]):
        init_kwargs[_pick_param(init_sig, ["device"])] = str(device)

    engine = cls(**init_kwargs)

    methods = [
        getattr(engine, "infer_file", None),
        getattr(engine, "convert_file", None),
        getattr(engine, "vc_single", None),
        getattr(engine, "infer", None),
        getattr(engine, "convert", None),
    ]
    methods = [m for m in methods if callable(m)]
    if not methods:
        raise RuntimeError("Не найден метод infer/convert в rvc_python")

    last_error = None
    for m in methods:
        try:
            kwargs = {
                "input_path": str(input_wav),
                "audio_path": str(input_wav),
                "wav_path": str(input_wav),
                "input_wav": str(input_wav),
                "output_path": str(output_wav),
                "output_wav": str(output_wav),
                "f0_up_key": int(f0_up_key),
                "pitch": int(f0_up_key),
                "index_rate": float(index_rate),
                "protect": float(protect),
                "filter_radius": int(filter_radius),
            }
            _call_with_supported_kwargs(m, kwargs)
            if output_wav.exists() and output_wav.stat().st_size > 0:
                return
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"rvc_python inference failed: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", default="")
    parser.add_argument("--f0-up-key", type=int, default=0)
    parser.add_argument("--index-rate", type=float, default=0.0)
    parser.add_argument("--protect", type=float, default=0.5)
    parser.add_argument("--filter-radius", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_native_rvc(
        input_wav=Path(args.input),
        output_wav=Path(args.output),
        model_path=Path(args.model),
        index_path=Path(args.index) if str(args.index).strip() else None,
        f0_up_key=int(args.f0_up_key),
        index_rate=float(args.index_rate),
        protect=float(args.protect),
        filter_radius=int(args.filter_radius),
        device=str(args.device),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
