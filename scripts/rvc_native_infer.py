#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path


def _prepare_torch_compat() -> None:
    """Совместимость с torch>=2.6 (weights_only) для старых RVC/fairseq чекпоинтов."""
    try:
        import torch
    except Exception:
        return

    # Разрешаем нужные классы для безопасной десериализации (best effort)
    try:
        from fairseq.data.dictionary import Dictionary  # type: ignore
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([Dictionary])
    except Exception:
        pass

    # Для старых чекпоинтов RVC принудительно отключаем weights_only по умолчанию.
    try:
        _orig_torch_load = torch.load

        def _torch_load_compat(*args, **kwargs):
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return _orig_torch_load(*args, **kwargs)

        torch.load = _torch_load_compat  # type: ignore
    except Exception:
        pass


def _force_utf8_stdio() -> None:
    """Принудительно выставляет UTF-8 для stdout/stderr, чтобы traceback не превращался в кракозябры."""
    try:
        os.environ["PYTHONIOENCODING"] = "utf-8"
    except Exception:
        pass
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


_force_utf8_stdio()
_prepare_torch_compat()


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
    for mod_name in ("rvc_python.infer", "rvc_python", "rvc"):
        try:
            rvc_mod = __import__(mod_name)
            # Для вложенного импорта __import__('a.b') вернёт пакет "a",
            # поэтому отдельно подтягиваем реальный модуль по пути.
            if mod_name == "rvc_python.infer":
                import importlib
                rvc_mod = importlib.import_module("rvc_python.infer")
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
    # 1) Прямой поиск в импортированном модуле
    for name in ("RVCInference", "RVC", "VoiceConverter"):
        cls = getattr(rvc_mod, name, None)
        if cls is not None:
            break
    # 2) Частый кейс для rvc_python: класс лежит в rvc_python.infer.RVCInference
    if cls is None:
        try:
            import importlib
            infer_mod = importlib.import_module("rvc_python.infer")
            for name in ("RVCInference", "RVC", "VoiceConverter"):
                cls = getattr(infer_mod, name, None)
                if cls is not None:
                    break
        except Exception:
            cls = None
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

    # Спец-ветка для rvc_python==0.1.x: infer_file иногда падает из-за tuple на записи.
    # Делаем прямой вызов vc.vc_single и сохраняем wav сами.
    try:
        vc_obj = getattr(engine, "vc", None)
        vc_single = getattr(vc_obj, "vc_single", None) if vc_obj is not None else None
        if callable(vc_single):
            result = vc_single(
                sid=0,
                input_audio_path=str(input_wav),
                f0_up_key=int(f0_up_key),
                f0_method=getattr(engine, "f0method", "harvest"),
                file_index=str(index_path) if index_path else "",
                index_rate=float(index_rate),
                filter_radius=int(filter_radius),
                resample_sr=0,
                rms_mix_rate=1,
                protect=float(protect),
                f0_file="",
                file_index2="",
            )

            sr = int(getattr(vc_obj, "tgt_sr", 0) or 0)
            audio = None

            # Обычный кейс: numpy-массив
            if hasattr(result, "dtype"):
                audio = result
            # Некоторые версии возвращают tuple
            elif isinstance(result, tuple):
                # (sr, audio)
                if len(result) >= 2 and isinstance(result[0], (int, float)) and hasattr(result[1], "dtype"):
                    sr = int(result[0])
                    audio = result[1]
                # ("info", (sr, audio))
                elif len(result) >= 2 and isinstance(result[1], (tuple, list)) and len(result[1]) >= 2:
                    sub = result[1]
                    if isinstance(sub[0], (int, float)) and hasattr(sub[1], "dtype"):
                        sr = int(sub[0])
                        audio = sub[1]

            if audio is not None and sr > 0:
                try:
                    import soundfile as sf
                    sf.write(str(output_wav), audio, sr)
                except Exception:
                    from scipy.io import wavfile
                    wavfile.write(str(output_wav), sr, audio)

                if output_wav.exists() and output_wav.stat().st_size > 0:
                    return
    except Exception:
        # Если спец-ветка не сработала, ниже остаётся общий fallback
        pass

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
            result = _call_with_supported_kwargs(m, kwargs)

            # Если вызвали vc_single и получили сырое аудио, сохраним вручную.
            if not (output_wav.exists() and output_wav.stat().st_size > 0) and callable(m) and getattr(m, "__name__", "") == "vc_single":
                try:
                    sr = int(getattr(getattr(engine, "vc", None), "tgt_sr", 0) or 0)
                    audio = None
                    if hasattr(result, "dtype"):
                        audio = result
                    elif isinstance(result, tuple):
                        if len(result) >= 2 and isinstance(result[0], (int, float)) and hasattr(result[1], "dtype"):
                            sr = int(result[0])
                            audio = result[1]
                        elif len(result) >= 2 and isinstance(result[1], (tuple, list)) and len(result[1]) >= 2:
                            sub = result[1]
                            if isinstance(sub[0], (int, float)) and hasattr(sub[1], "dtype"):
                                sr = int(sub[0])
                                audio = sub[1]
                    if audio is not None and sr > 0:
                        try:
                            import soundfile as sf
                            sf.write(str(output_wav), audio, sr)
                        except Exception:
                            from scipy.io import wavfile
                            wavfile.write(str(output_wav), sr, audio)
                except Exception:
                    pass

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
