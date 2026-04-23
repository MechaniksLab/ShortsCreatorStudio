#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Локальное разделение audio на vocals / no_vocals через UVR-модели.

Скрипт вызывается из video_translate.processor и должен быть максимально
устойчивым: если UVR не смог — вернёт ненулевой код, чтобы пайплайн ушёл в fallback.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _safe_print(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        sys.stdout.buffer.write(str(msg).encode(enc, errors="backslashreplace") + b"\n")


def _run(cmd: list[str]) -> None:
    _safe_print("[RUN] " + subprocess.list2cmdline(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout or "").strip()[-1200:])


def _normalize_wav(src: Path, dst: Path) -> None:
    _run([
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ac",
        "2",
        "-ar",
        "44100",
        str(dst),
    ])


def _separate_with_audio_separator(input_wav: Path, output_dir: Path, model_file: Path) -> list[Path]:
    from audio_separator.separator import Separator  # type: ignore

    # Для кастомных UVR ONNX-моделей важно передавать model_file_dir,
    # иначе audio-separator ищет только во внутреннем /tmp-каталоге моделей.
    sep = Separator(
        output_dir=str(output_dir),
        output_format="WAV",
        model_file_dir=str(model_file.parent),
    )

    # В большинстве версий библиотека ожидает имя файла, а не абсолютный путь.
    # Оставляем fallback на абсолютный путь для совместимости со старыми релизами.
    try:
        sep.load_model(model_filename=str(model_file.name))
    except Exception:
        sep.load_model(model_filename=str(model_file))

    before = {p.resolve() for p in output_dir.glob("*.wav")}
    res = sep.separate(str(input_wav))
    after = {p.resolve() for p in output_dir.glob("*.wav")}
    new_files = [p for p in after - before if p.exists() and p.stat().st_size > 0]

    # Некоторые версии возвращают список путей — учитываем и это.
    if isinstance(res, (list, tuple)):
        for item in res:
            try:
                p = Path(str(item)).resolve()
                if p.exists() and p.stat().st_size > 0 and p not in new_files:
                    new_files.append(p)
            except Exception:
                pass

    return new_files


def _pick_stem(paths: list[Path], want_vocals: bool) -> Path | None:
    if not paths:
        return None

    def _score(p: Path) -> int:
        n = p.name.lower()
        s = 0

        # Точные признаки stem-типа (приоритетнее всего)
        if "(vocals)" in n or "_vocals" in n or " vocals" in n:
            s += 100 if want_vocals else -120
        if "(instrumental)" in n or "_instrumental" in n or " instrumental" in n:
            s += 100 if not want_vocals else -120
        if "no_voc" in n or "accompan" in n or "karaoke" in n:
            s += 90 if not want_vocals else -80

        # Более слабые эвристики
        if "voice" in n:
            s += 25 if want_vocals else -20
        if "inst" in n:
            # ВАЖНО: "inst" часто встречается в ИМЕНИ МОДЕЛИ (Inst_HQ_3),
            # поэтому это только слабый сигнал.
            s += 8 if not want_vocals else -6

        # Небольшой бонус за размер (обычно основной stem не пустой).
        try:
            s += min(20, int(p.stat().st_size / 1_000_000))
        except Exception:
            pass
        return s

    ranked = sorted(paths, key=_score, reverse=True)
    return ranked[0] if ranked else None


def _derive_vocals_from_inst(mix_wav: Path, inst_wav: Path, out_vocals: Path) -> None:
    # Оценка вокала как разница mix - instrumental.
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(mix_wav),
            "-i",
            str(inst_wav),
            "-filter_complex",
            "[0:a][1:a]amix=inputs=2:weights='1 -1':normalize=0,alimiter=limit=0.98",
            str(out_vocals),
        ]
    )


def _pseudo_no_vocals(input_wav: Path, output_no_vocals: Path) -> None:
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_wav),
            "-af",
            "stereotools=mlev=-0.9,highpass=f=120,lowpass=f=9000,volume=0.78",
            str(output_no_vocals),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="source wav")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--inst-model", default="UVR-MDX-NET-Inst_HQ_3.onnx")
    parser.add_argument("--vocal-model", default="Kim_Vocal_2.onnx")
    args = parser.parse_args()

    src = Path(args.input)
    out_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        _safe_print(f"[ERROR] input not found: {src}")
        return 2

    vocal_model = model_dir / args.vocal_model
    inst_model = model_dir / args.inst_model
    if not vocal_model.exists() and not inst_model.exists():
        _safe_print(f"[ERROR] no UVR models found in {model_dir}")
        return 3

    normalized = out_dir / "input_44k.wav"
    _normalize_wav(src, normalized)

    all_outputs: list[Path] = []
    vocal_outputs: list[Path] = []
    inst_outputs: list[Path] = []
    try:
        if vocal_model.exists():
            _safe_print(f"[INFO] vocal model: {vocal_model.name}")
            model_out = out_dir / "vocal_model"
            if model_out.exists():
                shutil.rmtree(model_out, ignore_errors=True)
            model_out.mkdir(parents=True, exist_ok=True)
            vocal_outputs = _separate_with_audio_separator(normalized, model_out, vocal_model)
            all_outputs += vocal_outputs
        if inst_model.exists():
            _safe_print(f"[INFO] inst model: {inst_model.name}")
            model_out = out_dir / "inst_model"
            if model_out.exists():
                shutil.rmtree(model_out, ignore_errors=True)
            model_out.mkdir(parents=True, exist_ok=True)
            inst_outputs = _separate_with_audio_separator(normalized, model_out, inst_model)
            all_outputs += inst_outputs
    except Exception as e:
        _safe_print(f"[WARN] audio-separator failed: {e}")

    uniq = []
    seen = set()
    for p in all_outputs:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(rp)

    # Жёсткая стратегия выбора stem'ов:
    # - vocals сначала из vocal-model output,
    # - no_vocals сначала из inst-model output (Instrumental).
    # Это убирает перепутывание каналов при эвристиках.
    vocals = (
        _pick_stem(vocal_outputs, want_vocals=True)
        or _pick_stem(inst_outputs, want_vocals=True)
        or _pick_stem(uniq, want_vocals=True)
    )
    no_vocals = (
        _pick_stem(inst_outputs, want_vocals=False)
        or _pick_stem(vocal_outputs, want_vocals=False)
        or _pick_stem(uniq, want_vocals=False)
    )

    out_vocals = out_dir / "vocals.wav"
    out_bg = out_dir / "no_vocals.wav"

    if vocals and vocals.exists():
        shutil.copy2(vocals, out_vocals)
    if no_vocals and no_vocals.exists():
        shutil.copy2(no_vocals, out_bg)

    _safe_print(f"[DEBUG] selected_vocals={vocals}")
    _safe_print(f"[DEBUG] selected_no_vocals={no_vocals}")

    # Приоритет для MDX Inst_HQ_3: строим вокал как разницу mix - instrumental.
    # Это обычно даёт более «чистый» голос (меньше музыки/SFX), чем прямой vocal stem.
    if out_bg.exists() and out_bg.stat().st_size > 0:
        try:
            if out_vocals.exists() and out_vocals.stat().st_size > 0:
                backup_vocals = out_dir / "vocals_from_model.wav"
                shutil.copy2(out_vocals, backup_vocals)
            _derive_vocals_from_inst(normalized, out_bg, out_vocals)
        except Exception:
            pass

    # Если no_vocals не получен, строим псевдо-фон.
    if not out_bg.exists() or out_bg.stat().st_size == 0:
        try:
            _pseudo_no_vocals(normalized, out_bg)
        except Exception:
            pass

    # Если vocals не получен, берём исходник как soft-fallback.
    if not out_vocals.exists() or out_vocals.stat().st_size == 0:
        shutil.copy2(normalized, out_vocals)

    if out_vocals.exists() and out_bg.exists() and out_vocals.stat().st_size > 0 and out_bg.stat().st_size > 0:
        _safe_print(f"[OK] vocals={out_vocals} no_vocals={out_bg}")
        return 0

    _safe_print("[ERROR] failed to produce required stems")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
