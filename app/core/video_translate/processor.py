import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import requests
from openai import OpenAI

from app.config import PROJECT_ROOT
from app.core.bk_asr.asr_data import ASRData
from app.core.bk_asr.asr_data import ASRDataSeg
from app.core.bk_asr.transcribe import transcribe
from app.core.entities import TranslatorServiceEnum, VideoTranslateTask
from app.core.subtitle_processor.translate import TranslatorFactory, TranslatorType
from app.core.utils import json_repair
from app.core.utils.logger import setup_logger
from app.core.utils.video_utils import video2audio
from app.core.video_translate.bootstrap import VideoTranslateBootstrap
from app.core.video_translate.diarization import (
    DiarizerFactory,
    HeuristicPauseDiarizer,
    SpeakerTurn,
)
from app.core.video_translate.service_manager import VideoTranslateServiceManager
from app.core.video_translate.rvc_model_registry import default_rvc_model_root, scan_rvc_models

logger = setup_logger("video_translate_processor")

_SPEAKER_ANALYSIS_CACHE = PROJECT_ROOT / "AppData" / "cache" / "video_translate_last_speakers.json"
_LAST_TRANSLATION_CACHE = PROJECT_ROOT / "AppData" / "cache" / "video_translate_last_translation.json"


def _safe_run(cmd: List[str]):
    logger.info("CMD: %s", subprocess.list2cmdline(cmd))
    p = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=(
            subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        ),
    )
    if p.returncode != 0:
        err = (p.stderr or p.stdout or "").strip()
        tail = err[-1800:] if err else "no stderr"
        raise RuntimeError(
            f"Command failed (rc={p.returncode}): {subprocess.list2cmdline(cmd)}\n{tail}"
        )


def _probe_audio_duration_ms(path: Path) -> int:
    """Длительность аудио через ffprobe (мс)."""
    try:
        r = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=(
                subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            ),
        )
        dur = float((r.stdout or "0").strip() or 0)
        if dur > 0:
            return int(dur * 1000)
    except Exception:
        pass
    return 0


def _probe_mean_db_with_filter(path: Path, afilter: str) -> Optional[float]:
    """Оценка средней громкости (dB) после фильтра ffmpeg volumedetect."""
    try:
        p = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-i",
                str(path),
                "-af",
                f"{afilter},volumedetect",
                "-f",
                "null",
                "NUL",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=(
                subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            ),
        )
        text = (p.stdout or "") + "\n" + (p.stderr or "")
        m_mean = re.search(r"mean_volume:\s*(-?[\d\.]+)\s*dB", text)
        if m_mean:
            return float(m_mean.group(1))
    except Exception:
        return None
    return None


def _estimate_vocal_leak_score(path: Path) -> float:
    """Грубая оценка остаточного голоса в фоне: чем меньше, тем лучше."""
    # Диапазон, где обычно много речевой информации.
    mid = _probe_mean_db_with_filter(path, "highpass=f=150,lowpass=f=3800")
    full = _probe_mean_db_with_filter(path, "anull")
    if mid is None and full is None:
        return 1e9
    if mid is None:
        mid = -120.0
    if full is None:
        full = -120.0
    # Если середина относительно общего сигнала громкая -> вероятна утечка вокала.
    relative_mid = mid - full
    # Наказываем слишком тихие треки, чтобы не выбрать «почти тишину» как лучший фон.
    quiet_penalty = 8.0 if full < -48.0 else 0.0
    return float(mid + 0.65 * relative_mid + quiet_penalty)


def _pick_best_background_candidate(candidates: List[Path], source_audio: Path) -> Path:
    valid = [c for c in candidates if c and c.exists() and c.stat().st_size > 0]
    if not valid:
        return source_audio
    scored: List[tuple[float, Path]] = []
    for c in valid:
        score = _estimate_vocal_leak_score(c)
        scored.append((score, c))
        logger.info("BG candidate score: %s -> %.3f", c, score)
    scored.sort(key=lambda x: x[0])
    return scored[0][1]


def _build_atempo_filter(speed: float) -> str:
    """Собирает безопасную цепочку atempo для произвольного коэффициента скорости."""
    # ffmpeg atempo поддерживает 0.5..2.0 на один фильтр.
    speed = max(0.35, min(3.2, float(speed)))
    parts: List[str] = []
    cur = speed
    while cur > 2.0:
        parts.append("atempo=2.0")
        cur /= 2.0
    while cur < 0.5:
        parts.append("atempo=0.5")
        cur /= 0.5
    parts.append(f"atempo={cur:.5f}")
    return ",".join(parts)


def _runtime_python() -> Optional[Path]:
    p = PROJECT_ROOT / "runtime" / "python.exe"
    return p if p.exists() else None


def _runtime_python_for_rvc(cfg) -> Optional[Path]:
    raw = str(getattr(cfg, "rvc_runtime_python", "") or "").strip()
    if raw:
        p = Path(raw)
        if p.exists() and p.is_file():
            return p
    candidates = [
        PROJECT_ROOT / "runtime_rvc" / "Scripts" / "python.exe",
        PROJECT_ROOT / "runtime" / "python.exe",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def _find_demucs_stem(demucs_out: Path, model_name: str, track_name: str, stem_name: str) -> Optional[Path]:
    p = demucs_out / model_name / track_name / stem_name
    if p.exists() and p.stat().st_size > 0:
        return p
    return None


def _try_demucs_split(source_audio: Path, work_dir: Path) -> tuple[Optional[Path], Optional[Path]]:
    """Пытается получить stems (vocals, no_vocals/accompaniment) через demucs (локально)."""
    py = _runtime_python()
    if py is None:
        return None, None

    # Проверяем наличие demucs в runtime.
    try:
        chk = subprocess.run(
            [str(py), "-c", "import demucs,separate; print('ok')"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=25,
            creationflags=(
                subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            ),
        )
        if chk.returncode != 0:
            # Иногда пакет доступен как entrypoint demucs, а импорт в frozen окружениях ломается.
            # Продолжаем и пробуем CLI ниже.
            pass
    except Exception:
        pass

    demucs_out = work_dir / "demucs"
    demucs_out.mkdir(parents=True, exist_ok=True)
    track_name = source_audio.stem
    model_name = "htdemucs"

    demucs_cli = shutil.which("demucs")
    if demucs_cli:
        cmd = [
            demucs_cli,
            "--two-stems",
            "vocals",
            "-n",
            model_name,
            "-o",
            str(demucs_out),
            str(source_audio),
        ]
    else:
        cmd = [
            str(py),
            "-m",
            "demucs.separate",
            "--two-stems",
            "vocals",
            "-n",
            model_name,
            "-o",
            str(demucs_out),
            str(source_audio),
        ]

    try:
        _safe_run(cmd)
    except Exception as e:
        logger.warning("Demucs separation failed: %s", e)
        return None, None

    vocals = _find_demucs_stem(demucs_out, model_name, track_name, "vocals.wav")
    no_vocals = _find_demucs_stem(demucs_out, model_name, track_name, "no_vocals.wav")
    if no_vocals is None:
        # Некоторые сборки именуют стем как accompaniment.wav
        no_vocals = _find_demucs_stem(demucs_out, model_name, track_name, "accompaniment.wav")

    if vocals is not None:
        logger.info("Demucs vocals stem: %s", vocals)
    if no_vocals is not None:
        logger.info("Demucs no_vocals/accompaniment stem: %s", no_vocals)
    return vocals, no_vocals


def _default_uvr_model_dir() -> Path:
    return PROJECT_ROOT / "AppData" / "models" / "uvr"


def _resolve_uvr_model_dir(cfg) -> Path:
    raw = str(getattr(cfg, "uvr_model_dir", "") or "").strip()
    return Path(raw) if raw else _default_uvr_model_dir()


def _try_uvr_mdx_kim_split(source_audio: Path, work_dir: Path, cfg) -> tuple[Optional[Path], Optional[Path]]:
    """Пытается разделить дорожку через локальный UVR/MDX каскад (Inst HQ3 + Kim Vocal 2)."""
    py = _runtime_python()
    if py is None:
        return None, None

    # Жёсткая проверка зависимости: без audio_separator UVR-скрипт не сможет реально разделить стемы.
    try:
        chk = subprocess.run(
            [str(py), "-c", "import audio_separator; print('ok')"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            creationflags=(
                subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            ),
        )
        if chk.returncode != 0:
            raise RuntimeError(
                "В runtime не установлен пакет 'audio_separator' для UVR/MDX. "
                "Запустите scripts\\install_video_translate_experimental_addons.cmd "
                "или установите зависимости UVR вручную."
            )
    except Exception as e:
        logger.warning("UVR runtime dependency check failed: %s", e)
        return None, None

    uvr_script = PROJECT_ROOT / "scripts" / "uvr_separate.py"
    if not uvr_script.exists():
        return None, None

    model_dir = _resolve_uvr_model_dir(cfg)
    inst_model = str(
        getattr(cfg, "uvr_inst_hq3_model_name", "UVR-MDX-NET-Inst_HQ_3.onnx")
        or "UVR-MDX-NET-Inst_HQ_3.onnx"
    )
    vocal_model = str(
        getattr(cfg, "uvr_kim_vocal_model_name", "Kim_Vocal_2.onnx")
        or "Kim_Vocal_2.onnx"
    )

    uvr_out = work_dir / "uvr"
    if uvr_out.exists():
        try:
            shutil.rmtree(uvr_out, ignore_errors=True)
        except Exception:
            pass
    uvr_out.mkdir(parents=True, exist_ok=True)
    vocals = uvr_out / "vocals.wav"
    vocals_from_model = uvr_out / "vocals_from_model.wav"
    no_vocals = uvr_out / "no_vocals.wav"

    cmd = [
        str(py),
        str(uvr_script),
        "--input",
        str(source_audio),
        "--output-dir",
        str(uvr_out),
        "--model-dir",
        str(model_dir),
        "--inst-model",
        inst_model,
        "--vocal-model",
        vocal_model,
    ]
    try:
        _safe_run(cmd)
    except Exception as e:
        logger.warning("UVR/MDX split failed: %s", e)
        return None, None

    # Для ASR/референса предпочтительнее «чистый» vocal stem напрямую из модели,
    # если он сохранён скриптом (vocals_from_model.wav).
    if vocals_from_model.exists() and vocals_from_model.stat().st_size > 0:
        v = vocals_from_model
    else:
        v = vocals if vocals.exists() and vocals.stat().st_size > 0 else None
    nv = no_vocals if no_vocals.exists() and no_vocals.stat().st_size > 0 else None
    if v is not None:
        logger.info("UVR vocals stem: %s", v)
    if nv is not None:
        logger.info("UVR no_vocals stem: %s", nv)
    return v, nv


def _finalize_selected_stems(work_dir: Path, speech_audio: Path, background_audio: Path) -> tuple[Path, Path]:
    """Нормализует пути итоговых stem-файлов в рабочей папке."""
    out_speech = work_dir / "speech_audio.wav"
    out_bg = work_dir / "background_audio.wav"
    try:
        if speech_audio.exists() and speech_audio != out_speech:
            shutil.copy2(speech_audio, out_speech)
        elif speech_audio.exists():
            out_speech = speech_audio
    except Exception:
        out_speech = speech_audio

    try:
        if background_audio.exists() and background_audio != out_bg:
            shutil.copy2(background_audio, out_bg)
        elif background_audio.exists():
            out_bg = background_audio
    except Exception:
        out_bg = background_audio

    return out_speech, out_bg


def _split_model_tokens(raw: str) -> List[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    return [p.strip() for p in re.split(r"[;,\n\r]+", text) if p.strip()]


def _guess_gender_from_name(name: str) -> str:
    n = str(name or "").lower()
    female_hints = [
        "female", "girl", "woman", "дев", "жен", "оля", "olesya", "yumilia", "мейбл", "sandy",
    ]
    male_hints = [
        "male", "boy", "man", "муж", "диппер", "putin", "стэтхэм", "kuplinov", "брежнев", "levitan", "neo", "loki", "dominic", "shrek", "spongebob",
    ]
    if any(h in n for h in female_hints):
        return "female"
    if any(h in n for h in male_hints):
        return "male"
    return "unknown"


def _resolve_mmvc_root(cfg) -> Optional[Path]:
    candidates: List[Path] = []
    raw_server = str(getattr(cfg, "rvc_server_exe", "") or "").strip()
    if raw_server:
        p = Path(raw_server)
        if p.exists() and p.is_file():
            return p.parent

    # Приоритет: встроенный tools-каталог проекта
    candidates.append(PROJECT_ROOT / "AppData" / "tools" / "MMVCServerSIO")
    # Путь пользователя из задачи
    candidates.append(Path(r"E:\Downloads\voice-changer-windows-amd64-cuda.zip.001\MMVCServerSIO"))

    for c in candidates:
        exe = c / "MMVCServerSIO.exe"
        if exe.exists():
            return c
    return None


def _load_msgpack_from_mmvc(mmvc_root: Path):
    internal = mmvc_root / "_internal"
    if internal.exists():
        path_s = str(internal)
        if path_s not in sys.path:
            sys.path.insert(0, path_s)
    import msgpack  # type: ignore

    return msgpack


@dataclass
class VoiceSegment:
    idx: int
    start_ms: int
    end_ms: int
    text: str
    speaker_id: str = "spk_0"


class BaseVoiceCloneProvider:
    def synthesize(
        self,
        segments: List[VoiceSegment],
        work_dir: Path,
        task: VideoTranslateTask,
        progress_cb: Optional[Callable[[int, str], None]] = None,
    ) -> Path:
        raise NotImplementedError


class LocalXTTSProvider(BaseVoiceCloneProvider):
    """Локальный бесплатный провайдер через HTTP endpoint (XTTS/OpenVoice/FishSpeech server)."""

    def _synthesize_one(
        self,
        endpoint: str,
        text: str,
        speaker_wav: Path,
        lang: str,
        device_preference: str,
        out_path: Path,
        require_xtts: bool,
        prefer_xtts: bool,
        allow_non_strict_fallback: bool,
        max_retries: int,
        req_timeout: int,
    ):
        endpoint = endpoint.rstrip("/")

        # Прямой JSON /tts (clone_tts endpoint в локальном сервере сейчас не multipart-aware)
        payload = {
            "text": text,
            "language": lang,
            "speaker_wav": str(speaker_wav),
            "device_preference": str(device_preference or "auto"),
            "require_xtts": bool(require_xtts),
            "prefer_xtts": bool(prefer_xtts),
        }
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                r = requests.post(f"{endpoint}/tts", json=payload, timeout=req_timeout)
                if r.status_code < 300 and r.content:
                    out_path.write_bytes(r.content)
                    return
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
            except Exception as e:
                last_err = str(e)
            time.sleep(0.45 * attempt)

        # Для мягкого режима клонирования можно откатиться на обычный локальный TTS.
        if require_xtts and allow_non_strict_fallback:
            logger.warning(
                "Strict XTTS failed for segment, fallback to non-strict local tts. error=%s",
                last_err,
            )
            payload["require_xtts"] = False
            for attempt in range(1, 3):
                try:
                    r = requests.post(f"{endpoint}/tts", json=payload, timeout=90)
                    if r.status_code < 300 and r.content:
                        out_path.write_bytes(r.content)
                        return
                    last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                except Exception as e:
                    last_err = str(e)
                time.sleep(0.4 * attempt)

        raise RuntimeError(f"Local TTS endpoint error: {last_err}")

    @staticmethod
    def _probe_volumedetect(path: Path) -> Dict[str, float]:
        """Снимает базовые аудио-метрики через ffmpeg volumedetect."""
        result: Dict[str, float] = {}
        try:
            p = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-i",
                    str(path),
                    "-af",
                    "volumedetect",
                    "-f",
                    "null",
                    "NUL",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=(
                    subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
                ),
            )
            text = (p.stdout or "") + "\n" + (p.stderr or "")
            m_mean = re.search(r"mean_volume:\s*(-?[\d\.]+)\s*dB", text)
            m_max = re.search(r"max_volume:\s*(-?[\d\.]+)\s*dB", text)
            if m_mean:
                result["mean_db"] = float(m_mean.group(1))
            if m_max:
                result["max_db"] = float(m_max.group(1))
        except Exception:
            pass
        return result

    @classmethod
    def _segment_qa_check(
        cls,
        wav_path: Path,
        target_ms: int,
        cfg,
        quality: str,
    ) -> tuple[bool, str]:
        if not wav_path.exists():
            return False, "missing-file"

        min_size = int(getattr(cfg, "segment_min_size_bytes", 2500) or 2500)
        if wav_path.stat().st_size < min_size:
            return False, f"too-small:{wav_path.stat().st_size}"

        dur_ms = _probe_audio_duration_ms(wav_path)
        min_dur = int(getattr(cfg, "segment_min_duration_ms", 180) or 180)
        if dur_ms <= min_dur:
            return False, f"too-short:{dur_ms}"

        # Слишком короткая реплика относительно окна — индикатор «схлопнувшегося» синтеза.
        if target_ms > 0:
            ratio = dur_ms / float(target_ms)
            min_ratio = 0.42 if quality in {"studio", "high"} else 0.30
            if ratio < min_ratio:
                return False, f"bad-ratio:{ratio:.2f}"

        stats = cls._probe_volumedetect(wav_path)
        mean_db = stats.get("mean_db")
        max_db = stats.get("max_db")

        min_mean_db = float(getattr(cfg, "segment_min_mean_db", -43.0) or -43.0)
        max_peak_db = float(getattr(cfg, "segment_max_peak_db", -0.2) or -0.2)

        if mean_db is not None and mean_db < min_mean_db:
            return False, f"too-quiet:{mean_db:.2f}dB"
        if max_db is not None and max_db > max_peak_db:
            return False, f"clipping-risk:{max_db:.2f}dB"

        return True, "ok"

    @staticmethod
    def _fit_segment_to_target_window(src_wav: Path, dst_wav: Path, target_ms: int) -> Path:
        """Подгоняет длительность сегмента под окно тайминга: time-stretch + мягкий tail-safe.

        Важно: не режем фразу "в ноль" по границе сегмента, оставляем небольшой хвост,
        чтобы не съедать последние согласные/слоги.
        """
        try:
            target_ms = max(120, int(target_ms))
            tail_guard_ms = 35
            soft_target_ms = target_ms + tail_guard_ms
            cur_ms = max(1, _probe_audio_duration_ms(src_wav))
            ratio = cur_ms / float(soft_target_ms)

            # Если уже близко к окну — лишний ffmpeg не гоняем.
            if 0.93 <= ratio <= 1.04:
                return src_wav

            if ratio > 1.0:
                # Слегка сжимаем реплики, но НЕ обрезаем конец atrim'ом,
                # чтобы не терять последние слоги/согласные.
                max_speedup = 1.22
                speed = min(ratio, max_speedup)
                atempo = _build_atempo_filter(speed)
                af = atempo
            else:
                # Важно: не дополняем короткие фразы тишиной,
                # иначе появляются искусственные паузы между словами/репликами.
                return src_wav

            _safe_run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(src_wav),
                    "-af",
                    af,
                    "-ac",
                    "1",
                    "-ar",
                    "24000",
                    str(dst_wav),
                ]
            )
            if dst_wav.exists() and dst_wav.stat().st_size > 0:
                return dst_wav
        except Exception:
            pass
        return src_wav

    @classmethod
    def _enhance_reference_clip(cls, src_wav: Path, dst_wav: Path) -> Path:
        """Мягкая очистка/нормализация эталона голоса для XTTS."""
        try:
            _safe_run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(src_wav),
                    "-af",
                    "highpass=f=70,lowpass=f=7600,afftdn=nf=-24,acompressor=threshold=-18dB:ratio=2.2:attack=8:release=140,loudnorm=I=-20:TP=-2:LRA=10",
                    "-ac",
                    "1",
                    "-ar",
                    "24000",
                    str(dst_wav),
                ]
            )
            if dst_wav.exists() and dst_wav.stat().st_size > 0:
                return dst_wav
        except Exception:
            pass
        return src_wav

    @staticmethod
    def _run_native_rvc_convert(
        *,
        runtime_python: Optional[Path],
        input_wav: Path,
        output_wav: Path,
        model_path: Path,
        index_path: Optional[Path],
        f0_up_key: int,
        index_rate: float,
        protect: float,
        filter_radius: int,
        device: str,
    ) -> None:
        runtime_py = runtime_python
        if runtime_py is None:
            raise RuntimeError("Не найден Python для native RVC (ожидался config RVCRuntimePython или runtime_rvc)")
        script = PROJECT_ROOT / "scripts" / "rvc_native_infer.py"
        if not script.exists():
            raise RuntimeError(f"Не найден скрипт native RVC: {script}")

        cmd = [
            str(runtime_py),
            str(script),
            "--input",
            str(input_wav),
            "--output",
            str(output_wav),
            "--model",
            str(model_path),
            "--f0-up-key",
            str(int(f0_up_key)),
            "--index-rate",
            f"{float(index_rate):.4f}",
            "--protect",
            f"{float(protect):.4f}",
            "--filter-radius",
            str(int(filter_radius)),
            "--device",
            str(device or "cuda"),
        ]
        if index_path and index_path.exists():
            cmd += ["--index", str(index_path)]

        _safe_run(cmd)
        if not output_wav.exists() or output_wav.stat().st_size <= 0:
            raise RuntimeError("native RVC не создал выходной wav")

    @staticmethod
    def _merge_segments_for_rvc_passthrough(segments: List[VoiceSegment]) -> List[VoiceSegment]:
        """
        Склеивает слишком короткие соседние сегменты в фразы для RVC-only режима,
        чтобы не было эффекта «по слову» и обрывов на каждой границе.
        """
        if not segments:
            return segments
        ordered = sorted(segments, key=lambda s: (s.start_ms, s.end_ms))
        merged: List[VoiceSegment] = []

        max_chunk_ms = 7200
        max_gap_ms = 320
        cur = VoiceSegment(
            idx=int(ordered[0].idx),
            start_ms=int(ordered[0].start_ms),
            end_ms=int(ordered[0].end_ms),
            text=str(ordered[0].text or "").strip(),
            speaker_id=str(ordered[0].speaker_id or "spk_0"),
        )

        for seg in ordered[1:]:
            same_spk = str(seg.speaker_id) == str(cur.speaker_id)
            gap = int(seg.start_ms) - int(cur.end_ms)
            next_total = int(seg.end_ms) - int(cur.start_ms)
            can_merge = same_spk and gap <= max_gap_ms and next_total <= max_chunk_ms
            if can_merge:
                add_text = str(seg.text or "").strip()
                if add_text:
                    cur.text = (f"{cur.text} {add_text}").strip()
                cur.end_ms = max(int(cur.end_ms), int(seg.end_ms))
            else:
                merged.append(cur)
                cur = VoiceSegment(
                    idx=int(seg.idx),
                    start_ms=int(seg.start_ms),
                    end_ms=int(seg.end_ms),
                    text=str(seg.text or "").strip(),
                    speaker_id=str(seg.speaker_id or "spk_0"),
                )
        merged.append(cur)
        return merged

    @staticmethod
    def _merge_segments_for_speaker_chunks(segments: List[VoiceSegment]) -> List[VoiceSegment]:
        """Мягко объединяет соседние реплики одного спикера в более цельные фразы.

        Нужен для режима перевода (TTS->RVC/XTTS), чтобы речь звучала менее «рвано».
        """
        if not segments:
            return segments
        ordered = sorted(segments, key=lambda s: (s.start_ms, s.end_ms))
        merged: List[VoiceSegment] = []

        max_gap_ms = 220
        max_chunk_ms = 12000
        cur = VoiceSegment(
            idx=int(ordered[0].idx),
            start_ms=int(ordered[0].start_ms),
            end_ms=int(ordered[0].end_ms),
            text=str(ordered[0].text or "").strip(),
            speaker_id=str(ordered[0].speaker_id or "spk_0"),
        )

        for seg in ordered[1:]:
            same_spk = str(seg.speaker_id) == str(cur.speaker_id)
            gap = int(seg.start_ms) - int(cur.end_ms)
            next_total = int(seg.end_ms) - int(cur.start_ms)
            can_merge = same_spk and gap <= max_gap_ms and next_total <= max_chunk_ms
            if can_merge:
                add_text = str(seg.text or "").strip()
                if add_text:
                    cur.text = (f"{cur.text} {add_text}").strip()
                cur.end_ms = max(int(cur.end_ms), int(seg.end_ms))
            else:
                merged.append(cur)
                cur = VoiceSegment(
                    idx=int(seg.idx),
                    start_ms=int(seg.start_ms),
                    end_ms=int(seg.end_ms),
                    text=str(seg.text or "").strip(),
                    speaker_id=str(seg.speaker_id or "spk_0"),
                )
        merged.append(cur)
        return merged

    def synthesize(
        self,
        segments: List[VoiceSegment],
        work_dir: Path,
        task: VideoTranslateTask,
        progress_cb: Optional[Callable[[int, str], None]] = None,
    ) -> Path:
        cfg = task.video_translate_config
        endpoint = (cfg.local_tts_endpoint or "http://127.0.0.1:8020").strip()
        if not endpoint:
            raise RuntimeError("Не задан local TTS endpoint")

        provider_mode = str(getattr(cfg, "voice_clone_provider", "auto") or "auto").strip().lower()
        xtts_clone_mode = provider_mode == "xtts"
        rvc_pipeline_mode = not xtts_clone_mode

        quality = "rvc_pipeline"
        tts_device_pref = str(getattr(task.transcribe_config, "faster_whisper_device", "auto") or "auto").strip().lower()
        gpu_pref = tts_device_pref == "cuda"
        allow_overlap_cfg = False
        qa_enabled = False
        qa_retry_count = 0
        strict_xtts = bool(xtts_clone_mode)
        prefer_xtts = bool(xtts_clone_mode)
        allow_non_strict_fallback = False
        max_retries = 2
        req_timeout = 90

        translation_enabled = bool(getattr(cfg, "translation_enabled", True))
        # Для RVC-пайплайна:
        # - без перевода: сразу RVC по оригинальному speech stem
        # - с переводом: быстрый local TTS -> RVC
        rvc_passthrough_mode = bool(rvc_pipeline_mode and (not translation_enabled))

        # В RVC-пайплайне с переводом intentionally НЕ используем XTTS-клонирование:
        # сначала быстрый обычный local TTS, затем тембр даёт RVC.
        if rvc_pipeline_mode and translation_enabled:
            strict_xtts = False
            prefer_xtts = False

        diarization_enabled = bool(getattr(cfg, "enable_diarization", False))
        if rvc_passthrough_mode:
            # В RVC-only режиме overlap даёт эффект «двойных слов» на стыках сегментов.
            allow_overlap_cfg = False
            qa_enabled = False
            if not diarization_enabled:
                # Без diarization — один цельный кусок голосовой дорожки одной RVC моделью.
                _speech = work_dir / "speech_audio.wav"
                _source = work_dir / "source_audio.wav"
                _full_src = _speech if _speech.exists() else _source
                full_ms = max(1, _probe_audio_duration_ms(_full_src))
                segments = [
                    VoiceSegment(
                        idx=1,
                        start_ms=0,
                        end_ms=full_ms,
                        text="_",
                        speaker_id="spk_0",
                    )
                ]
            else:
                segments = self._merge_segments_for_rvc_passthrough(segments)
        else:
            speaker_chunk_mode = bool(getattr(cfg, "speaker_chunk_synthesis_enabled", True))
            if speaker_chunk_mode:
                # Для режима с переводом объединяем соседние куски одного спикера,
                # чтобы убрать «рубленую» подачу по одной короткой фразе.
                segments = self._merge_segments_for_speaker_chunks(segments)
        manual_map_raw = str(getattr(cfg, "manual_voice_map_json", "") or "").strip()
        speaker_slot_map: Dict[str, str] = {}
        if manual_map_raw:
            try:
                parsed = json.loads(manual_map_raw)
                if isinstance(parsed, dict):
                    speaker_slot_map = {str(k): str(v) for k, v in parsed.items() if v is not None}
            except Exception:
                speaker_slot_map = {}
        single_voice_mode = str(speaker_slot_map.get("__single_voice__", "")).strip().lower() in {"1", "true", "yes", "on"}

        rvc_models_by_slot = {}
        model_root_raw = str(getattr(cfg, "rvc_model_dir", "") or "").strip()
        model_root = Path(model_root_raw) if model_root_raw else default_rvc_model_root()
        if rvc_pipeline_mode:
            for m in scan_rvc_models(model_root):
                rvc_models_by_slot[str(m.slot)] = m

            if not speaker_slot_map and rvc_models_by_slot:
                default_raw = str(getattr(cfg, "rvc_default_model", "") or "").strip()
                default_slot = ""
                if default_raw:
                    if default_raw in rvc_models_by_slot:
                        default_slot = default_raw
                    else:
                        for _slot, _m in rvc_models_by_slot.items():
                            if str(getattr(_m, "name", "")).strip().lower() == default_raw.lower():
                                default_slot = _slot
                                break
                if not default_slot:
                    default_slot = sorted(rvc_models_by_slot.keys())[0]
                speaker_slot_map["default"] = default_slot

        if single_voice_mode and str(speaker_slot_map.get("default", "")).strip():
            # Принудительно одним голосом для всех спикеров.
            default_slot = str(speaker_slot_map.get("default", "")).strip()
            for seg in segments:
                speaker_slot_map[str(seg.speaker_id)] = default_slot

        if rvc_pipeline_mode and progress_cb:
            progress_cb(
                1,
                f"RVC registry: {len(rvc_models_by_slot)} моделей ({model_root})",
            )

        if rvc_pipeline_mode and (not rvc_models_by_slot):
            raise RuntimeError(
                "Для режима RVC требуется хотя бы одна RVC-модель. Откройте менеджер RVC и добавьте модели/назначения спикеров."
            )

        if progress_cb:
            if xtts_clone_mode:
                progress_cb(0, "XTTS mode: отдельный режим клонирования голоса")
            elif rvc_passthrough_mode:
                progress_cb(0, f"RVC-only mode: переозвучка без XTTS ({quality})")
            else:
                progress_cb(
                    0,
                    f"TTS -> RVC mode: {quality}",
                )

        rvc_runtime_py = _runtime_python_for_rvc(cfg)
        if progress_cb:
            progress_cb(2, "Подготовка эталонов спикеров")

        # Для стабильного voice cloning XTTS нужен более длинный референс (обычно 3-6с).
        ref_duration_sec = 6.0 if quality == "studio" else (4.0 if quality == "high" else 3.0)

        # Эталон голоса: при наличии используем speech stem после separation.
        source_audio = work_dir / "source_audio.wav"
        if not source_audio.exists():
            raise RuntimeError("Не найден source_audio.wav для клонирования голоса")
        speech_audio = work_dir / "speech_audio.wav"
        reference_audio = speech_audio if speech_audio.exists() else source_audio

        # Speaker-aware reference wav: собираем лучший эталон из нескольких кусков спикера.
        speaker_refs: Dict[str, Path] = {}
        by_speaker: Dict[str, List[VoiceSegment]] = {}
        for seg in segments:
            by_speaker.setdefault(seg.speaker_id, []).append(seg)

        target_ref_total = float(
            getattr(
                cfg,
                "reference_target_total_sec",
                18.0 if quality == "studio" else (10.0 if quality == "high" else 5.0),
            )
            or 18.0
        )
        reference_enhance = bool(getattr(cfg, "reference_enhancement_enabled", quality in {"high", "studio"}))
        # Быстрый режим подготовки эталонов для GPU-пайплайна: меньше CPU-нагрузки и быстрее старт TTS.
        fast_ref_mode = bool(gpu_pref and quality in {"high", "studio"})
        if fast_ref_mode:
            target_ref_total = min(target_ref_total, 3.2)
            reference_enhance = False
        # В режиме без перевода сам тембр вносится RVC-конвертером,
        # поэтому не тратим лишний CPU на слишком длинные эталоны XTTS.
        if rvc_passthrough_mode:
            target_ref_total = min(target_ref_total, 6.0)
            reference_enhance = False
        reference_min_mean_db = float(getattr(cfg, "reference_min_mean_db", -48.0) or -48.0)
        total_speakers = max(1, len(by_speaker))
        for spk_idx, (spk, spk_segs) in enumerate(by_speaker.items(), start=1):
            if progress_cb:
                progress_cb(2 + int(8 * spk_idx / total_speakers), f"Эталоны спикеров {spk_idx}/{total_speakers}")
            ref_wav = work_dir / f"ref_{spk}.wav"
            # Берём самые длинные реплики как более стабильный voiceprint.
            ordered = sorted(spk_segs, key=lambda s: max(0, s.end_ms - s.start_ms), reverse=True)
            parts: List[Path] = []
            acc = 0.0
            max_ref_parts = 1 if fast_ref_mode else 5
            for n, seg in enumerate(ordered, start=1):
                if acc >= target_ref_total:
                    break
                if len(parts) >= max_ref_parts:
                    break
                clip = work_dir / f"ref_{spk}_{n:02d}.wav"
                start_sec = max(0.0, (seg.start_ms / 1000.0) - 0.30)
                dur_sec = max(1.2, min(2.8 if fast_ref_mode else 6.5, (seg.end_ms - seg.start_ms) / 1000.0 + 0.6))
                try:
                    _safe_run(
                        [
                            "ffmpeg",
                            "-y",
                            "-ss",
                            f"{start_sec:.3f}",
                            "-t",
                            f"{dur_sec:.3f}",
                            "-i",
                            str(reference_audio),
                            "-ac",
                            "1",
                            "-ar",
                            "24000",
                            str(clip),
                        ]
                    )
                    if clip.exists() and clip.stat().st_size > 0:
                        if not fast_ref_mode:
                            stats = self._probe_volumedetect(clip)
                            mean_db = stats.get("mean_db", -120.0)
                            if mean_db < reference_min_mean_db:
                                continue
                        if reference_enhance:
                            clip = self._enhance_reference_clip(
                                clip,
                                work_dir / f"ref_{spk}_{n:02d}_enh.wav",
                            )
                        parts.append(clip)
                        acc += dur_sec
                except Exception:
                    continue

            if not parts:
                continue
            if len(parts) == 1:
                try:
                    ref_wav.write_bytes(parts[0].read_bytes())
                except Exception:
                    pass
            else:
                concat_list = work_dir / f"ref_{spk}_concat.txt"
                concat_list.write_text(
                    "\n".join(f"file '{p.as_posix()}'" for p in parts),
                    encoding="utf-8",
                )
                try:
                    _safe_run(
                        [
                            "ffmpeg",
                            "-y",
                            "-f",
                            "concat",
                            "-safe",
                            "0",
                            "-i",
                            str(concat_list),
                            "-ac",
                            "1",
                            "-ar",
                            "24000",
                            str(ref_wav),
                        ]
                    )
                except Exception:
                    pass

            if ref_wav.exists() and ref_wav.stat().st_size > 0:
                logger.info(
                    "Speaker ref prepared: %s (%s bytes, parts=%s, target=%.1fs)",
                    ref_wav,
                    ref_wav.stat().st_size,
                    len(parts),
                    target_ref_total,
                )
                speaker_refs[spk] = ref_wav

        tts_dir = work_dir / "tts_segments"
        tts_dir.mkdir(parents=True, exist_ok=True)
        wav_segments: List[Path] = []

        total = max(1, len(segments))
        lang = str(cfg.target_language or "英语")
        configured_workers = int(getattr(cfg, "tts_parallel_workers", 3) or 3)
        configured_workers = max(1, min(8, configured_workers))
        # Уважаем настройку UI "Параллельных TTS задач".
        max_workers = configured_workers
        if gpu_pref:
            qa_enabled = False

        def _job(i_seg):
            i, seg = i_seg
            wav_path = tts_dir / f"seg_{i:04d}.wav"
            fixed_wav = tts_dir / f"seg_{i:04d}_fixed.wav"
            target_ms = max(220, int(seg.end_ms - seg.start_ms))
            attempts = max(1, qa_retry_count + 1)
            last_reason = ""

            # Если назначен RVC-slot для текущего спикера — берём его params.json.
            slot = str(
                speaker_slot_map.get(seg.speaker_id)
                or speaker_slot_map.get("default")
                or ""
            ).strip()
            model_meta = rvc_models_by_slot.get(slot) if slot else None
            model_tune = int(getattr(model_meta, "default_tune", 0) or 0)
            # Безопасный clamp, чтобы не «ломать» голос.
            model_tune = max(-6, min(6, model_tune))

            for attempt in range(1, attempts + 1):
                # На повторных попытках делаем синтез мягче/стабильнее.
                soft_pass = attempt > 1

                if rvc_passthrough_mode:
                    # Без перевода: не делаем Local TTS, берём оригинальный speech-сегмент и прогоняем через RVC.
                    # Даём заметный запас, чтобы не «съедать» окончания слов на границах сегмента.
                    pre_pad_sec = 0.00
                    post_pad_sec = 0.16
                    clip_start_sec = max(0.0, (seg.start_ms / 1000.0) - pre_pad_sec)
                    clip_dur_sec = max(0.20, (seg.end_ms - seg.start_ms) / 1000.0 + pre_pad_sec + post_pad_sec)
                    _safe_run(
                        [
                            "ffmpeg",
                            "-y",
                            "-ss",
                            f"{clip_start_sec:.3f}",
                            "-t",
                            f"{clip_dur_sec:.3f}",
                            "-i",
                            str(reference_audio),
                            "-ac",
                            "1",
                            "-ar",
                            "24000",
                            str(wav_path),
                        ]
                    )
                    fixed_wav = wav_path
                else:
                    tts_text = re.sub(r"\s+", " ", str(seg.text or "")).strip()
                    # Для очень коротких реплик уменьшаем пунктуационные паузы TTS.
                    if len(tts_text.split()) <= 3:
                        tts_text = re.sub(r"[,:;.!?…]+", "", tts_text)
                        tts_text = re.sub(r"\s+", " ", tts_text).strip()
                    if not tts_text:
                        tts_text = str(seg.text or "").strip() or "_"
                    self._synthesize_one(
                        endpoint=endpoint,
                        text=tts_text,
                        speaker_wav=speaker_refs.get(seg.speaker_id, reference_audio),
                        lang=lang,
                        device_preference=tts_device_pref,
                        out_path=wav_path,
                        require_xtts=(strict_xtts if quality == "studio" else (strict_xtts and not soft_pass)),
                        prefer_xtts=(prefer_xtts or quality in {"high", "studio"}),
                        allow_non_strict_fallback=allow_non_strict_fallback,
                        max_retries=max_retries,
                        req_timeout=req_timeout + (12 if soft_pass else 0),
                    )

                    fixed_wav = wav_path

                # Native RVC post-convert по params.json и назначению спикера.
                if rvc_pipeline_mode and (model_meta is not None):
                    model_pth = model_meta.slot_dir / str(model_meta.model_file)
                    index_file_name = str(model_meta.index_file or "").strip()
                    model_index = model_meta.slot_dir / index_file_name if index_file_name else None
                    rvc_wav = tts_dir / f"seg_{i:04d}_rvc.wav"
                    if gpu_pref:
                        # Жёстко держим RVC на GPU: сначала с index, затем (если нужно) без index,
                        # но не откатываемся на CPU.
                        try:
                            self._run_native_rvc_convert(
                                runtime_python=rvc_runtime_py,
                                input_wav=fixed_wav,
                                output_wav=rvc_wav,
                                model_path=model_pth,
                                index_path=model_index,
                                f0_up_key=model_tune,
                                index_rate=float(getattr(model_meta, "default_index_ratio", 0.0) or 0.0),
                                protect=float(getattr(model_meta, "default_protect", 0.5) or 0.5),
                                filter_radius=int(getattr(cfg, "rvc_filter_radius", 3) or 3),
                                device="cuda",
                            )
                            fixed_wav = rvc_wav
                        except Exception as e:
                            try:
                                self._run_native_rvc_convert(
                                    runtime_python=rvc_runtime_py,
                                    input_wav=fixed_wav,
                                    output_wav=rvc_wav,
                                    model_path=model_pth,
                                    index_path=None,
                                    f0_up_key=model_tune,
                                    index_rate=0.0,
                                    protect=float(getattr(model_meta, "default_protect", 0.5) or 0.5),
                                    filter_radius=int(getattr(cfg, "rvc_filter_radius", 3) or 3),
                                    device="cuda",
                                )
                                fixed_wav = rvc_wav
                            except Exception as e2:
                                raise RuntimeError(
                                    f"RVC GPU convert failed for slot={slot}, seg={i}. "
                                    f"Проверьте CUDA runtime RVC / модель / index. ({e} | {e2})"
                                )
                    else:
                        self._run_native_rvc_convert(
                            runtime_python=rvc_runtime_py,
                            input_wav=fixed_wav,
                            output_wav=rvc_wav,
                            model_path=model_pth,
                            index_path=None,
                            f0_up_key=model_tune,
                            index_rate=0.0,
                            protect=float(getattr(model_meta, "default_protect", 0.5) or 0.5),
                            filter_radius=int(getattr(cfg, "rvc_filter_radius", 3) or 3),
                            device="cpu",
                        )
                        fixed_wav = rvc_wav

                if not qa_enabled or rvc_passthrough_mode:
                    break

                ok, reason = self._segment_qa_check(
                    fixed_wav,
                    target_ms=target_ms,
                    cfg=cfg,
                    quality=quality,
                )
                if ok:
                    break
                last_reason = reason
                logger.warning(
                    "Segment QA failed (seg=%s, spk=%s, attempt=%s/%s, reason=%s)",
                    i,
                    seg.speaker_id,
                    attempt,
                    attempts,
                    reason,
                )

            # В режиме перевода (TTS->RVC) фиксируем длительность в окно сегмента,
            # чтобы речь попадала в исходные тайминги, а не «ехала» по порядку.
            if (not rvc_passthrough_mode) and target_ms > 0:
                fitted_wav = tts_dir / f"seg_{i:04d}_fit.wav"
                fixed_wav = self._fit_segment_to_target_window(fixed_wav, fitted_wav, target_ms)

            if qa_enabled and last_reason:
                logger.warning("Segment %s accepted with last QA warning: %s", i, last_reason)
            return i, fixed_wav

        if max_workers > 1:
            produced = {}
            done_cnt = 0
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(_job, (i, seg)) for i, seg in enumerate(segments, start=1)]
                for fut in as_completed(futures):
                    i, fixed_wav = fut.result()
                    produced[i] = fixed_wav
                    done_cnt += 1
                    if progress_cb:
                        mode_name = "RVC" if rvc_passthrough_mode else "Local TTS"
                        progress_cb(int(done_cnt * 100 / total), f"{mode_name} {done_cnt}/{total} ({quality}, x{max_workers})")
            wav_segments = [produced[i] for i in sorted(produced.keys())]
        else:
            for i, seg in enumerate(segments, start=1):
                _, fixed_wav = _job((i, seg))
                wav_segments.append(fixed_wav)
                if progress_cb:
                    mode_name = "RVC" if rvc_passthrough_mode else "Local TTS"
                    progress_cb(int(i * 100 / total), f"{mode_name} {i}/{total} ({quality})")

        wav_durations_ms: List[int] = [max(1, _probe_audio_duration_ms(w)) for w in wav_segments]

        # Перестановка стартов под реальную длительность синтезированных кусков,
        # чтобы не было наложений/обрубания окончаний при миксе.
        adjusted_starts: List[int] = []
        if not rvc_passthrough_mode:
            # Гибкий режим: стараемся держать исходные таймкоды,
            # но слегка сдвигаем только если предыдущий сегмент реально наползает.
            min_gap_ms = 6
            max_shift_ms = 520
            overlap_tolerance_ms = 60
            prev_end = 0
            for seg, dur_ms in zip(segments, wav_durations_ms):
                nominal_start = max(0, int(seg.start_ms))
                if prev_end <= nominal_start + overlap_tolerance_ms:
                    start_ms = nominal_start
                else:
                    spill = prev_end - nominal_start
                    start_ms = nominal_start + min(max_shift_ms, max(0, spill))
                adjusted_starts.append(start_ms)
                target_ms = max(1, int(seg.end_ms - seg.start_ms))
                effective_sched_dur = min(int(dur_ms), target_ms + 420)
                prev_end = start_ms + effective_sched_dur + min_gap_ms
        elif allow_overlap_cfg:
            adjusted_starts = [max(0, int(seg.start_ms)) for seg in segments]
        else:
            min_gap_ms = 55
            prev_end = 0
            for seg, dur_ms in zip(segments, wav_durations_ms):
                start_ms = max(int(seg.start_ms), prev_end)
                adjusted_starts.append(start_ms)
                prev_end = start_ms + int(dur_ms) + min_gap_ms

        total_duration_sec = max(
            [
                (st + dur) for st, dur in zip(adjusted_starts, wav_durations_ms)
            ] + [max((s.end_ms for s in segments), default=1000)]
        ) / 1000.0
        silence = work_dir / "silence.wav"
        _safe_run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=24000:cl=mono",
                "-t",
                f"{total_duration_sec:.3f}",
                str(silence),
            ]
        )

        mixed = work_dir / "dubbed_voice.wav"
        cmd = ["ffmpeg", "-y", "-i", str(silence)]
        for p in wav_segments:
            cmd += ["-i", str(p)]

        filters = []
        for i, start_ms in enumerate(adjusted_starts, start=1):
            filters.append(f"[{i}:a]adelay={start_ms}|{start_ms}[a{i}]")

        overlap_aware = False
        speaker_map: Dict[str, List[int]] = {}
        if overlap_aware:
            for idx, seg in enumerate(segments, start=1):
                speaker_map.setdefault(seg.speaker_id, []).append(idx)

        if overlap_aware and len(speaker_map) > 1:
            bus_names: List[str] = []
            for bus_idx, (speaker_id, inds) in enumerate(speaker_map.items(), start=1):
                bus = f"spk{bus_idx}"
                bus_names.append(f"[{bus}]")
                if len(inds) == 1:
                    filters.append(f"[a{inds[0]}]anull[{bus}]")
                else:
                    ins = "".join(f"[a{n}]" for n in inds)
                    filters.append(
                        f"{ins}amix=inputs={len(inds)}:normalize=0,alimiter=limit=0.96[{bus}]"
                    )

                # Мягкая стабилизация отдельных speaker-bus.
                filters.append(f"[{bus}]dynaudnorm=f=75:g=7[{bus}n]")
                bus_names[-1] = f"[{bus}n]"
                logger.debug("Speaker bus created: %s -> %s segments", speaker_id, len(inds))

            inputs = "[0:a]" + "".join(bus_names)
            filters.append(
                f"{inputs}amix=inputs={1 + len(bus_names)}:normalize=0,alimiter=limit=0.98[out]"
            )
        else:
            inputs = "[0:a]" + "".join(f"[a{i}]" for i in range(1, len(segments) + 1))
            filters.append(f"{inputs}amix=inputs={len(segments)+1}:normalize=0[out]")

        cmd += ["-filter_complex", ";".join(filters), "-map", "[out]", str(mixed)]
        _safe_run(cmd)
        return mixed


class ElevenLabsProvider(BaseVoiceCloneProvider):
    def synthesize(
        self,
        segments: List[VoiceSegment],
        work_dir: Path,
        task: VideoTranslateTask,
        progress_cb: Optional[Callable[[int, str], None]] = None,
    ) -> Path:
        cfg = task.video_translate_config
        api_key = (cfg.elevenlabs_api_key or "").strip()
        if not api_key:
            raise RuntimeError("Не указан ElevenLabs API key")

        mapping_raw = (cfg.manual_voice_map_json or "").strip()
        if mapping_raw:
            try:
                mapping = json.loads(mapping_raw)
            except Exception as e:
                raise RuntimeError(f"Некорректный JSON Voice Map: {e}")
        else:
            mapping = {}

        default_voice_id = str(mapping.get("default") or "").strip()
        if not default_voice_id:
            raise RuntimeError(
                "Для ElevenLabs укажите ManualVoiceMapJson, например: {\"default\":\"VOICE_ID\"}"
            )

        tts_dir = work_dir / "tts_segments"
        tts_dir.mkdir(parents=True, exist_ok=True)
        wav_segments: List[Path] = []

        total = max(1, len(segments))
        for i, seg in enumerate(segments, start=1):
            voice_id = str(mapping.get(seg.speaker_id) or default_voice_id).strip()
            mp3_path = tts_dir / f"seg_{i:04d}.mp3"
            wav_path = tts_dir / f"seg_{i:04d}.wav"

            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            payload = {
                "text": seg.text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {"stability": 0.45, "similarity_boost": 0.85},
            }
            headers = {
                "xi-api-key": api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code >= 300:
                raise RuntimeError(
                    f"ElevenLabs TTS ошибка ({resp.status_code}): {resp.text[:300]}"
                )
            mp3_path.write_bytes(resp.content)

            _safe_run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(mp3_path),
                    "-ac",
                    "1",
                    "-ar",
                    "24000",
                    str(wav_path),
                ]
            )
            wav_segments.append(wav_path)

            if progress_cb:
                progress_cb(int(i * 100 / total), f"TTS {i}/{total}")

        total_duration_sec = max((s.end_ms for s in segments), default=1000) / 1000.0
        silence = work_dir / "silence.wav"
        _safe_run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=24000:cl=mono",
                "-t",
                f"{total_duration_sec:.3f}",
                str(silence),
            ]
        )

        mixed = work_dir / "dubbed_voice.wav"
        cmd = ["ffmpeg", "-y", "-i", str(silence)]
        for p in wav_segments:
            cmd += ["-i", str(p)]

        filters = []
        for i, seg in enumerate(segments, start=1):
            filters.append(f"[{i}:a]adelay={seg.start_ms}|{seg.start_ms}[a{i}]")

        inputs = "[0:a]" + "".join(f"[a{i}]" for i in range(1, len(segments) + 1))
        filters.append(f"{inputs}amix=inputs={len(segments)+1}:normalize=0[out]")

        cmd += [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            str(mixed),
        ]
        _safe_run(cmd)
        return mixed


class VideoTranslationProcessor:
    def __init__(self):
        pass

    @staticmethod
    def _asr_data_from_manual_json(json_path: str) -> Optional[ASRData]:
        p = Path(str(json_path or "").strip())
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore") or "[]")
            if not isinstance(data, list):
                return None
            segs: List[ASRDataSeg] = []
            for row in data:
                if not isinstance(row, dict):
                    continue
                start_ms = int(row.get("start_ms", 0) or 0)
                end_ms = int(row.get("end_ms", start_ms + 1) or (start_ms + 1))
                txt = str(row.get("text", "") or "").strip()
                if not txt:
                    continue
                segs.append(
                    ASRDataSeg(
                        text=txt,
                        start_time=max(0, start_ms),
                        end_time=max(start_ms + 1, end_ms),
                        word_timestamps=[],
                    )
                )
            if not segs:
                return None
            return ASRData(segs)
        except Exception:
            return None

    @staticmethod
    def _runtime_cuda_available() -> bool:
        """Проверка CUDA через runtime/python.exe (важно для frozen exe)."""
        try:
            runtime_py = PROJECT_ROOT / "runtime" / "python.exe"
            if not runtime_py.exists():
                return False
            p = subprocess.run(
                [
                    str(runtime_py),
                    "-c",
                    "import torch; print('1' if torch.cuda.is_available() else '0')",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
                ),
            )
            out = (p.stdout or "").strip()
            return p.returncode == 0 and out.endswith("1")
        except Exception:
            return False

    @staticmethod
    def _detect_torch_device(prefer: str = "cuda") -> str:
        prefer = (prefer or "cuda").strip().lower()
        if prefer not in {"cuda", "cpu"}:
            prefer = "cuda"
        if prefer == "cpu":
            return "cpu"
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            # В frozen-процессе импорт torch может падать, но в runtime всё ок.
            return "cuda" if VideoTranslationProcessor._runtime_cuda_available() else "cpu"

    def _python_faster_whisper_fallback(
        self,
        audio_path: str,
        task: VideoTranslateTask,
        progress_cb: Optional[Callable[[int, str], None]] = None,
    ) -> ASRData:
        """Fallback ASR через python-пакет faster_whisper, если CLI не найден."""
        from faster_whisper import WhisperModel

        cfg = task.transcribe_config
        model_name = str(getattr(cfg, "faster_whisper_model", "tiny") or "tiny")
        model_dir = str(getattr(cfg, "faster_whisper_model_dir", "") or "").strip() or None
        language = str(getattr(cfg, "transcribe_language", "en") or "en")
        need_word_ts = bool(getattr(cfg, "need_word_time_stamp", False))
        device = self._detect_torch_device(getattr(cfg, "faster_whisper_device", "cuda"))
        compute_type = "float16" if device == "cuda" else "int8"

        self._emit(progress_cb, 15, f"ASR fallback (python faster-whisper, device={device})")
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            download_root=model_dir,
        )

        segments_iter, _ = model.transcribe(
            audio_path,
            language=language,
            vad_filter=bool(getattr(cfg, "faster_whisper_vad_filter", True)),
            word_timestamps=need_word_ts,
            initial_prompt=str(getattr(cfg, "faster_whisper_prompt", "") or "") or None,
        )

        segments: List[ASRDataSeg] = []
        for s in segments_iter:
            words = []
            for w in (s.words or []):
                if w.start is None or w.end is None:
                    continue
                words.append(
                    {
                        "word": str(w.word or ""),
                        "start": int(float(w.start) * 1000),
                        "end": int(float(w.end) * 1000),
                    }
                )

            segments.append(
                ASRDataSeg(
                    text=(s.text or "").strip(),
                    start_time=int(float(s.start) * 1000),
                    end_time=int(float(s.end) * 1000),
                    word_timestamps=words,
                )
            )

        asr_data = ASRData(segments)
        if not need_word_ts:
            asr_data.optimize_timing()
        return asr_data

    def _emit(
        self,
        cb: Optional[Callable[[int, str], None]],
        p: int,
        msg: str,
    ):
        if cb:
            cb(max(0, min(100, int(p))), msg)

    @staticmethod
    def _translator_type(service: TranslatorServiceEnum) -> TranslatorType:
        mapping = {
            TranslatorServiceEnum.OPENAI: TranslatorType.OPENAI,
            TranslatorServiceEnum.DEEPLX: TranslatorType.DEEPLX,
            TranslatorServiceEnum.BING: TranslatorType.BING,
            TranslatorServiceEnum.GOOGLE: TranslatorType.GOOGLE,
        }
        return mapping[service]

    @staticmethod
    def _contains_cyrillic(text: str) -> bool:
        return bool(re.search(r"[А-Яа-яЁё]", str(text or "")))

    @staticmethod
    def _target_prefers_non_cyrillic(target_language: str) -> bool:
        t = str(target_language or "").strip().lower()
        # Для русской/кириллической цели не делаем жёсткую проверку.
        if any(x in t for x in ["рус", "russian", "укра", "ukrain", "белар", "serbian", "bulgarian"]):
            return False
        return True

    @staticmethod
    def _strip_reasoning_tags(text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.DOTALL).strip()

    @staticmethod
    def _build_translation_cache_key(asr_data: ASRData, task: VideoTranslateTask) -> str:
        cfg = task.video_translate_config
        payload = {
            "target_language": str(getattr(cfg, "target_language", "") or ""),
            "translator_service": str(getattr(cfg, "translator_service", "") or ""),
            "llm_model": str(getattr(task.subtitle_config, "llm_model", "") or ""),
            "rows": [
                {
                    "start_ms": int(seg.start_time),
                    "end_ms": int(seg.end_time),
                    "text": str(seg.text or ""),
                }
                for seg in (asr_data.segments or [])
            ],
        }
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()

    @staticmethod
    def _try_apply_translation_cache(asr_data: ASRData, task: VideoTranslateTask, cache_key: str) -> bool:
        try:
            if not _LAST_TRANSLATION_CACHE.exists():
                return False
            raw = json.loads(_LAST_TRANSLATION_CACHE.read_text(encoding="utf-8", errors="ignore") or "{}")
            if not isinstance(raw, dict):
                return False
            if str(raw.get("cache_key", "")) != str(cache_key):
                return False
            rows = raw.get("rows", [])
            if not isinstance(rows, list) or len(rows) != len(asr_data.segments):
                return False
            for i, seg in enumerate(asr_data.segments):
                tr = str((rows[i] or {}).get("translated", "") or "").strip()
                seg.translated_text = tr if tr else str(seg.text or "")
            return True
        except Exception:
            return False

    def _translate_asr_openai_holistic(self, asr_data: ASRData, task: VideoTranslateTask) -> Optional[ASRData]:
        """Переводит весь ASR единым JSON-запросом (id + тайминги), без дробления на чанки."""
        sub_cfg = task.subtitle_config
        base_url = str(getattr(sub_cfg, "base_url", "") or "").strip()
        api_key = str(getattr(sub_cfg, "api_key", "") or "").strip()
        model = str(getattr(sub_cfg, "llm_model", "") or "gpt-4o-mini").strip()
        if not (base_url and api_key):
            return None

        target_language = str(getattr(sub_cfg, "target_language", "") or "English")
        items = []
        for i, seg in enumerate(asr_data.segments, start=1):
            items.append(
                {
                    "id": i,
                    "start_ms": int(seg.start_time),
                    "end_ms": int(seg.end_time),
                    "text": str(seg.text or ""),
                }
            )
        if not items:
            return asr_data

        strict_en = ""
        if str(target_language).strip().lower() in {"english", "en", "英语"}:
            strict_en = " For English target use only English words/alphabet; never output Cyrillic."

        system_prompt = (
            "You are a subtitle translator. "
            "Translate each item text to target language while preserving meaning and subtitle style. "
            "Return ONLY valid JSON object in this exact schema: "
            "{\"items\":[{\"id\":1,\"translated_text\":\"...\"}]}. "
            "Do not add/remove ids, do not merge/split items, no explanations, no markdown. "
            "You MUST return all ids from input exactly once. Never omit any id."
            + strict_en
        )
        user_payload = {
            "target_language": target_language,
            "items": items,
        }

        # На длинных роликах часть id может пропадать из-за обрезки ответа по токенам.
        # Даём достаточный бюджет completion, сохраняя один строгий запрос без fallback.
        est_chars = sum(max(8, len(str(x.get("text", "")))) for x in items)
        est_tokens = int(est_chars * 0.9) + len(items) * 16 + 800
        max_tokens = max(2500, min(16000, est_tokens))

        client = OpenAI(base_url=base_url, api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
            timeout=120,
        )
        content = self._strip_reasoning_tags(response.choices[0].message.content)
        parsed = json_repair.loads(content)
        rows = parsed.get("items", []) if isinstance(parsed, dict) else []
        if not isinstance(rows, list):
            raise RuntimeError("Holistic translate: invalid JSON format")

        translated_by_id: Dict[int, str] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                idx = int(row.get("id"))
            except Exception:
                continue
            tr = str(row.get("translated_text") or "").strip()
            if idx > 0 and tr:
                translated_by_id[idx] = tr

        if len(translated_by_id) != len(items):
            expected_ids = {int(x["id"]) for x in items}
            got_ids = set(translated_by_id.keys())
            missing = sorted(expected_ids - got_ids)
            extra = sorted(got_ids - expected_ids)
            missing_preview = ",".join(str(x) for x in missing[:40])
            extra_preview = ",".join(str(x) for x in extra[:40])
            raise RuntimeError(
                "ПРЕДУПРЕЖДЕНИЕ: модель вернула неполный перевод (id mismatch). "
                f"Получено {len(translated_by_id)} из {len(items)} сегментов. "
                f"missing_ids=[{missing_preview}] extra_ids=[{extra_preview}]. "
                "Fallback отключён: обработка остановлена, проверьте модель/промпт/API."
            )

        non_cyr_target = self._target_prefers_non_cyrillic(target_language)
        for i, seg in enumerate(asr_data.segments, start=1):
            tr = str(translated_by_id.get(i) or "").strip()
            src = str(seg.text or "").strip()
            if non_cyr_target and self._contains_cyrillic(src) and self._contains_cyrillic(tr):
                raise RuntimeError(
                    f"ПРЕДУПРЕЖДЕНИЕ: сегмент {i} не переведён (обнаружена кириллица в результате для non-cyr target). "
                    "Fallback отключён: обработка остановлена."
                )
            seg.translated_text = tr if tr else src
        return asr_data

    def _translate_asr(self, asr_data: ASRData, task: VideoTranslateTask) -> ASRData:
        sub_cfg = task.subtitle_config
        translator_type = self._translator_type(sub_cfg.translator_service)

        # Строгий режим: для OpenAI используем ТОЛЬКО holistic JSON-перевод
        # без fallback-веток. При любой проблеме сразу понятная ошибка.
        if translator_type == TranslatorType.OPENAI:
            try:
                holistic = self._translate_asr_openai_holistic(asr_data, task)
                if holistic is not None:
                    logger.info("Holistic OpenAI translation used for video revoice")
                    return holistic
                raise RuntimeError(
                    "Не настроен OpenAI-перевод: проверьте Base URL/API Key в настройках."
                )
            except Exception as e:
                raise RuntimeError(
                    "Ошибка перевода (строгий режим, без fallback): "
                    f"{e}. Проверьте модель/ключ/API и формат ответа JSON."
                )

        # Для video translate с LLM-переводом уменьшаем количество запросов:
        # один поток + максимально крупный batch.
        # Это убирает «залп» параллельных запросов к LLM для коротких роликов.
        thread_num = sub_cfg.thread_num
        batch_num = sub_cfg.batch_size
        if translator_type == TranslatorType.OPENAI:
            thread_num = 1
            batch_num = 9999

        translator = TranslatorFactory.create_translator(
            translator_type=translator_type,
            thread_num=thread_num,
            batch_num=batch_num,
            target_language=sub_cfg.target_language,
            model=sub_cfg.llm_model,
            custom_prompt=sub_cfg.custom_prompt_text or "",
            is_reflect=sub_cfg.need_reflect,
            use_cache=sub_cfg.use_cache,
            openai_base_url=sub_cfg.base_url,
            openai_api_key=sub_cfg.api_key,
            deeplx_endpoint=sub_cfg.deeplx_endpoint,
        )
        return translator.translate_subtitle(asr_data)

    def _retry_untranslated_segments(self, asr_data: ASRData, task: VideoTranslateTask) -> ASRData:
        """Повторно переводит проблемные сегменты (пусто/идентично/явно кириллица для non-cyr target)."""
        target = str(getattr(task.subtitle_config, "target_language", "") or "")
        non_cyr_target = self._target_prefers_non_cyrillic(target)

        bad_idx: List[int] = []
        for i, seg in enumerate(asr_data.segments):
            src = str(seg.text or "").strip()
            tr = str(seg.translated_text or "").strip()
            if not src:
                continue
            same = (not tr) or (tr == src)
            cyr_bad = non_cyr_target and self._contains_cyrillic(src) and self._contains_cyrillic(tr)
            if same or cyr_bad:
                bad_idx.append(i)

        if not bad_idx:
            return asr_data

        sub_cfg = task.subtitle_config
        translator_type = self._translator_type(sub_cfg.translator_service)
        translator = TranslatorFactory.create_translator(
            translator_type=translator_type,
            thread_num=1,
            batch_num=max(1, min(24, len(bad_idx))),
            target_language=sub_cfg.target_language,
            model=sub_cfg.llm_model,
            custom_prompt=sub_cfg.custom_prompt_text or "",
            is_reflect=True,
            use_cache=False,
            openai_base_url=sub_cfg.base_url,
            openai_api_key=sub_cfg.api_key,
            deeplx_endpoint=sub_cfg.deeplx_endpoint,
        )

        subset = ASRData([
            ASRDataSeg(
                text=str(asr_data.segments[i].text or ""),
                start_time=int(asr_data.segments[i].start_time),
                end_time=int(asr_data.segments[i].end_time),
                word_timestamps=list(getattr(asr_data.segments[i], "word_timestamps", []) or []),
            )
            for i in bad_idx
        ])
        translated_subset = translator.translate_subtitle(subset)

        replaced = 0
        for off, idx in enumerate(bad_idx):
            new_tr = str(translated_subset.segments[off].translated_text or translated_subset.segments[off].text or "").strip()
            if new_tr:
                asr_data.segments[idx].translated_text = new_tr
                replaced += 1
        logger.info("Retry translated segments: %s/%s", replaced, len(bad_idx))
        return asr_data

    @staticmethod
    def _apply_manual_translation_json(asr_data: ASRData, json_path: str) -> ASRData:
        p = Path(str(json_path or "").strip())
        if not p.exists():
            return asr_data
        try:
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore") or "[]")
        except Exception:
            return asr_data
        rows = data if isinstance(data, list) else []
        for i, row in enumerate(rows):
            if i >= len(asr_data.segments):
                break
            seg = asr_data.segments[i]
            tr = str((row or {}).get("translated") or "").strip()
            if tr:
                seg.translated_text = tr
        return asr_data

    @staticmethod
    def _apply_manual_transcription_json(asr_data: ASRData, json_path: str) -> ASRData:
        p = Path(str(json_path or "").strip())
        if not p.exists():
            return asr_data
        try:
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore") or "[]")
        except Exception:
            return asr_data
        rows = data if isinstance(data, list) else []
        for i, row in enumerate(rows):
            if i >= len(asr_data.segments):
                break
            seg = asr_data.segments[i]
            txt = str((row or {}).get("text") or "").strip()
            if txt:
                seg.text = txt
        return asr_data

    def _build_segments(self, asr_data: ASRData) -> List[VoiceSegment]:
        result: List[VoiceSegment] = []
        for i, seg in enumerate(asr_data.segments, start=1):
            text = (seg.translated_text or seg.text or "").strip()
            if not text:
                continue
            # Если diarization выключен, все сегменты относятся к одному спикеру.
            speaker_id = "spk_0"
            result.append(
                VoiceSegment(
                    idx=i,
                    start_ms=int(seg.start_time),
                    end_ms=int(seg.end_time),
                    text=text,
                    speaker_id=speaker_id,
                )
            )
        return result

    def _build_segments_from_turns(self, asr_data: ASRData, turns: List[SpeakerTurn]) -> List[VoiceSegment]:
        """Строит voice-сегменты по diarization turn (целые куски спикеров, без ограничения в 2 спикера)."""
        if not turns:
            return []
        result: List[VoiceSegment] = []
        idx = 1
        ordered_turns = sorted(turns, key=lambda t: (int(t.start_ms), int(t.end_ms)))
        for t in ordered_turns:
            t_start = int(t.start_ms)
            t_end = int(t.end_ms)
            if t_end <= t_start:
                continue
            pieces: List[str] = []
            for seg in asr_data.segments:
                s = int(seg.start_time)
                e = int(seg.end_time)
                overlap = min(t_end, e) - max(t_start, s)
                if overlap <= 0:
                    continue
                txt = str(seg.translated_text or seg.text or "").strip()
                if txt:
                    pieces.append(txt)
            text = " ".join(pieces).strip()
            if not text:
                continue
            result.append(
                VoiceSegment(
                    idx=idx,
                    start_ms=t_start,
                    end_ms=t_end,
                    text=text,
                    speaker_id=str(t.speaker_id or "spk_0"),
                )
            )
            idx += 1
        return result

    @staticmethod
    def _enforce_non_overlap(segments: List[VoiceSegment], min_gap_ms: int = 60) -> List[VoiceSegment]:
        if not segments:
            return segments
        ordered = sorted(segments, key=lambda s: (s.start_ms, s.end_ms))
        prev_end = -1
        for seg in ordered:
            if seg.start_ms < prev_end + min_gap_ms:
                shift = (prev_end + min_gap_ms) - seg.start_ms
                seg.start_ms += shift
                seg.end_ms += shift
            prev_end = max(prev_end, seg.end_ms)
        return ordered

    def _apply_speaker_turns(
        self,
        segments: List[VoiceSegment],
        turns: List[SpeakerTurn],
    ) -> List[VoiceSegment]:
        if not segments or not turns:
            return segments

        # Присваиваем speaker по максимальному overlap с diarization turn
        for seg in segments:
            best_spk = seg.speaker_id
            best_overlap = -1
            for t in turns:
                overlap = min(seg.end_ms, t.end_ms) - max(seg.start_ms, t.start_ms)
                if overlap > best_overlap and overlap > 0:
                    best_overlap = overlap
                    best_spk = t.speaker_id
            seg.speaker_id = best_spk
        return sorted(segments, key=lambda s: (s.start_ms, s.end_ms))

    def _select_provider(self, task: VideoTranslateTask) -> BaseVoiceCloneProvider:
        cfg = task.video_translate_config
        endpoint = (cfg.local_tts_endpoint or "").strip()
        provider_mode = str(getattr(cfg, "voice_clone_provider", "auto") or "auto").strip().lower()
        translation_enabled = bool(getattr(cfg, "translation_enabled", True))

        # Нужен local TTS endpoint:
        # - XTTS mode (клонирование)
        # - RVC mode при переводе (быстрый TTS -> RVC)
        need_local_tts = (provider_mode == "xtts") or translation_enabled
        if need_local_tts:
            ok = VideoTranslateServiceManager.instance().ensure_local_tts(endpoint)
            if not ok:
                if provider_mode == "xtts":
                    raise RuntimeError(
                        f"Не удалось поднять локальный XTTS сервис: {endpoint or 'http://127.0.0.1:8020'}"
                    )
                raise RuntimeError(
                    f"Не удалось поднять локальный TTS сервис для режима TTS->RVC: {endpoint or 'http://127.0.0.1:8020'}"
                )
        return LocalXTTSProvider()

    def _prepare_audio_stems(
        self,
        task: VideoTranslateTask,
        source_audio: Path,
        work_dir: Path,
    ) -> tuple[Path, Path]:
        """Готовит speech/bg дорожки:
        - speech_audio: очищенная дорожка для ASR/diarization/voice-reference
        - background_audio: фон без оригинального голоса для финального микса
        """
        cfg = task.video_translate_config
        uvr_vocals, uvr_bg = _try_uvr_mdx_kim_split(source_audio, work_dir, cfg)
        if not (uvr_vocals and uvr_vocals.exists() and uvr_vocals.stat().st_size > 0):
            raise RuntimeError(
                "MDX separation не смог получить voice stem. "
                "Проверьте UVR-зависимости и модели Kim_Vocal_2.onnx + UVR-MDX-NET-Inst_HQ_3.onnx."
            )
        if not (uvr_bg and uvr_bg.exists() and uvr_bg.stat().st_size > 0):
            raise RuntimeError("MDX separation не смог получить background stem (no_vocals.wav).")
        return _finalize_selected_stems(work_dir, uvr_vocals, uvr_bg)

    def _mux_video(self, task: VideoTranslateTask, background_audio: Path, dubbed_audio: Path, output_path: str):
        # Финальная сборка видео: добавлены настраиваемые decode/encode backend'ы.
        cfg = task.video_translate_config
        decode_backend = str(getattr(cfg, "video_decode_backend", "auto") or "auto").strip().lower()
        encode_backend = str(getattr(cfg, "video_encode_backend", "copy") or "copy").strip().lower()

        base_cmd = ["ffmpeg", "-y"]
        if decode_backend == "cuda":
            base_cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]

        cmd = base_cmd + [
            "-i",
            task.video_path,
            "-i",
            str(background_audio),
            "-i",
            str(dubbed_audio),
            "-filter_complex",
            "[1:a][2:a]amix=inputs=2:normalize=0[mix]",
            "-map",
            "0:v:0",
            "-map",
            "[mix]",
        ]

        if encode_backend == "nvenc":
            # GPU encode (NVIDIA): сохраняем визуально качественный профиль.
            cmd += ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "19", "-b:v", "0"]
        elif encode_backend == "cpu":
            cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]
        else:
            # copy = без перекодирования видео (быстро/без потери от re-encode).
            cmd += ["-c:v", "copy"]

        cmd += ["-c:a", "aac", "-shortest", output_path]

        try:
            _safe_run(cmd)
        except Exception as e:
            # Безопасный fallback: если CUDA/NVENC недоступны, не роняем пайплайн.
            if decode_backend == "cuda" or encode_backend == "nvenc":
                logger.warning(
                    "GPU mux failed (decode=%s, encode=%s), fallback to CPU/copy. reason=%s",
                    decode_backend,
                    encode_backend,
                    e,
                )
                fallback_cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    task.video_path,
                    "-i",
                    str(background_audio),
                    "-i",
                    str(dubbed_audio),
                    "-filter_complex",
                    "[1:a][2:a]amix=inputs=2:normalize=0[mix]",
                    "-map",
                    "0:v:0",
                    "-map",
                    "[mix]",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-shortest",
                    output_path,
                ]
                _safe_run(fallback_cmd)
            else:
                raise

    @staticmethod
    def _save_speaker_analysis_cache(segments: List[VoiceSegment], turns: Optional[List[SpeakerTurn]] = None):
        try:
            _SPEAKER_ANALYSIS_CACHE.parent.mkdir(parents=True, exist_ok=True)
            unique_ids = sorted({str(s.speaker_id) for s in segments if str(s.speaker_id).strip()})
            payload = {
                "speaker_ids": unique_ids,
                "segments_count": len(segments),
                "turns_count": len(turns or []),
            }
            _SPEAKER_ANALYSIS_CACHE.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("Failed to save speaker analysis cache: %s", e)

    @staticmethod
    def _save_last_translation_cache(asr_data: ASRData, task: VideoTranslateTask, cache_key: str = ""):
        try:
            _LAST_TRANSLATION_CACHE.parent.mkdir(parents=True, exist_ok=True)
            rows = []
            for seg in asr_data.segments:
                rows.append(
                    {
                        "start_ms": int(seg.start_time),
                        "end_ms": int(seg.end_time),
                        "original": str(seg.text or ""),
                        "translated": str(seg.translated_text or seg.text or ""),
                    }
                )
            payload = {
                "video_path": str(task.video_path or ""),
                "cache_key": str(cache_key or ""),
                "rows": rows,
            }
            _LAST_TRANSLATION_CACHE.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("Failed to save translation cache: %s", e)

    def process(
        self,
        task: VideoTranslateTask,
        progress_cb: Optional[Callable[[int, str], None]] = None,
    ) -> VideoTranslateTask:
        started = datetime.datetime.now()
        task.started_at = started
        work_dir = Path(tempfile.gettempdir()) / "shorts_video_translate"
        work_dir.mkdir(parents=True, exist_ok=True)

        self._emit(progress_cb, 2, "Подготовка")

        if getattr(task.video_translate_config, "autonomous_mode", False):
            self._emit(progress_cb, 4, "Проверка локальных моделей")
            model_name = (
                getattr(task.transcribe_config, "faster_whisper_model", "tiny")
                or "tiny"
            )
            model_name = str(model_name)
            auto_download = bool(getattr(task.video_translate_config, "auto_download_models", True))
            readiness = VideoTranslateBootstrap().readiness_report(
                model_name=model_name,
                auto_download=auto_download,
            )
            failed = readiness.get("critical_failed", [])
            if failed:
                details = "; ".join(f"{c.get('name')}: {c.get('hint')}" for c in failed)
                raise RuntimeError(f"Preflight не пройден: {details}")

            if readiness.get("clone_quality") != "high":
                self._emit(
                    progress_cb,
                    6,
                    "XTTS не найден: будет fallback-озвучка (без точного тембра).",
                )
            if readiness.get("diarization_quality") != "high":
                self._emit(
                    progress_cb,
                    7,
                    "Pyannote не найден: diarization в heuristic режиме.",
                )

        audio_path = work_dir / "source_audio.wav"
        if not video2audio(task.video_path, str(audio_path)):
            raise RuntimeError("Не удалось извлечь аудио из видео")

        # Подготовка дорожек: speech для ASR/clone + фон без речи для финального микса.
        speech_audio, bg_audio = self._prepare_audio_stems(task, audio_path, work_dir)

        # Автовыбор ASR device: не даём падать, если CUDA недоступна
        current_device = str(getattr(task.transcribe_config, "faster_whisper_device", "cuda") or "cuda").strip().lower()
        if current_device == "cuda" and not self._runtime_cuda_available():
            raise RuntimeError(
                "Выбран режим GPU (CUDA), но CUDA недоступна в runtime. "
                "Установите/исправьте GPU-окружение, либо вручную выберите CPU в интерфейсе."
            )
        task.transcribe_config.faster_whisper_device = current_device

        manual_tx_json = str(getattr(task.video_translate_config, "manual_transcription_json", "") or "").strip()
        asr_data: Optional[ASRData] = None
        if manual_tx_json and Path(manual_tx_json).exists():
            asr_data = self._asr_data_from_manual_json(manual_tx_json)
            if asr_data is not None:
                self._emit(progress_cb, 18, "ASR пропущен: использовано готовое распознавание (этап 2)")

        if asr_data is None:
            self._emit(progress_cb, 12, "Распознавание речи")
            try:
                asr_data = transcribe(str(speech_audio), task.transcribe_config)
            except Exception as e:
                msg = str(e)
                fw_cli_missing = (
                    "faster-whisper" in msg.lower()
                    and ("未找到" in msg or "not found" in msg.lower())
                )
                fw_cli_compute_mismatch = (
                    "float16" in msg.lower() and "cpu" in msg.lower()
                )
                if fw_cli_missing or fw_cli_compute_mismatch:
                    reason = "CLI не найден" if fw_cli_missing else "CLI compute_type не подходит для CPU"
                    self._emit(progress_cb, 14, f"Faster-Whisper {reason}, fallback на python-пакет")
                    asr_data = self._python_faster_whisper_fallback(
                        str(speech_audio),
                        task,
                        progress_cb=progress_cb,
                    )
                else:
                    raise

        translation_enabled = bool(getattr(task.video_translate_config, "translation_enabled", True))
        translation_cache_key = ""
        self._emit(progress_cb, 35, "Перевод текста" if translation_enabled else "Режим без перевода")
        if translation_enabled:
            manual_tr_json = str(getattr(task.video_translate_config, "manual_translation_json", "") or "").strip()
            if manual_tr_json and Path(manual_tr_json).exists():
                asr_data = self._apply_manual_translation_json(asr_data, manual_tr_json)
                self._emit(progress_cb, 38, "Применён подтверждённый перевод из предпросмотра")
            else:
                translation_cache_key = self._build_translation_cache_key(asr_data, task)
                use_cache = bool(getattr(task.video_translate_config, "use_translation_cache", True))
                if use_cache and self._try_apply_translation_cache(asr_data, task, translation_cache_key):
                    self._emit(progress_cb, 38, "Перевод взят из кэша")
                else:
                    asr_data = self._translate_asr(asr_data, task)
        else:
            for seg in asr_data.segments:
                seg.translated_text = str(seg.text or "")
            self._emit(progress_cb, 38, "Озвучка будет выполнена по оригинальному тексту")

        if translation_enabled and not translation_cache_key:
            translation_cache_key = self._build_translation_cache_key(asr_data, task)
        self._save_last_translation_cache(asr_data, task, translation_cache_key)

        if task.output_subtitle_path:
            asr_data.to_srt(task.output_subtitle_path)

        self._emit(progress_cb, 55, "Клонирование голоса / TTS")
        segments = self._build_segments(asr_data)
        if not segments:
            raise RuntimeError("После перевода не найдено сегментов для озвучки")

        # Если выбран режим «один голос для всех», diarization не нужен:
        # он только замедляет пайплайн, а итоговое назначение голоса всё равно единое.
        manual_map_raw = str(getattr(task.video_translate_config, "manual_voice_map_json", "") or "").strip()
        single_voice_mode = False
        if manual_map_raw:
            try:
                manual_map = json.loads(manual_map_raw)
                if isinstance(manual_map, dict):
                    single_voice_mode = str(manual_map.get("__single_voice__", "")).strip().lower() in {"1", "true", "yes", "on"}
            except Exception:
                single_voice_mode = False

        # Speaker-aware assign
        diarization_enabled = bool(getattr(task.video_translate_config, "enable_diarization", False))
        if diarization_enabled and not single_voice_mode:
            try:
                self._emit(progress_cb, 52, "Diarization спикеров")
                diar_started = time.time()
                diarizer = DiarizerFactory.create_preferred()
                turns = diarizer.diarize(
                    str(speech_audio),
                    asr_data,
                    expected_speaker_count=int(
                        getattr(task.video_translate_config, "expected_speaker_count", 0) or 0
                    ),
                )
                logger.info(
                    "Diarization completed: turns=%s, seconds=%.2f",
                    len(turns),
                    max(0.0, time.time() - diar_started),
                )
                turn_segments = self._build_segments_from_turns(asr_data, turns)
                if turn_segments:
                    segments = turn_segments
                else:
                    segments = self._apply_speaker_turns(segments, turns)
                self._save_speaker_analysis_cache(segments, turns)
            except Exception as e:
                logger.warning("Diarization failed, fallback to heuristic: %s", e)
                try:
                    turns = HeuristicPauseDiarizer().diarize(str(speech_audio), asr_data)
                    turn_segments = self._build_segments_from_turns(asr_data, turns)
                    if turn_segments:
                        segments = turn_segments
                    else:
                        segments = self._apply_speaker_turns(segments, turns)
                    self._save_speaker_analysis_cache(segments, turns)
                except Exception:
                    pass
        else:
            if diarization_enabled and single_voice_mode:
                self._emit(progress_cb, 52, "Diarization пропущен: режим одного голоса")
            self._save_speaker_analysis_cache(segments, None)

        # Управление overlap-сценариями: для studio/high оставляем оригинальные пересечения,
        # для fast/balanced можно включать строгую раздвижку сегментов.
        quality = str(
            getattr(task.video_translate_config, "voice_clone_quality", "high") or "high"
        ).strip().lower()
        allow_overlap = bool(
            getattr(task.video_translate_config, "allow_speaker_overlap", quality in {"high", "studio"})
        )
        if not allow_overlap:
            segments = self._enforce_non_overlap(segments)
        else:
            segments = sorted(segments, key=lambda s: (s.start_ms, s.end_ms))

        provider = self._select_provider(task)
        dubbed_audio = provider.synthesize(
            segments,
            work_dir=work_dir,
            task=task,
            progress_cb=lambda p, m: self._emit(progress_cb, 55 + int(p * 0.25), m),
        )

        self._emit(progress_cb, 85, "Сборка итогового видео")
        self._mux_video(task, bg_audio, dubbed_audio, task.output_path)

        task.completed_at = datetime.datetime.now()
        self._emit(progress_cb, 100, "Перевод видео завершён")
        return task
