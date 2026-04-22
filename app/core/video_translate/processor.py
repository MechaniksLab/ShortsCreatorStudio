import datetime
import json
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import requests

from app.config import PROJECT_ROOT
from app.core.bk_asr.asr_data import ASRData
from app.core.bk_asr.asr_data import ASRDataSeg
from app.core.bk_asr.transcribe import transcribe
from app.core.entities import TranslatorServiceEnum, VideoTranslateTask
from app.core.subtitle_processor.translate import TranslatorFactory, TranslatorType
from app.core.utils.logger import setup_logger
from app.core.utils.video_utils import video2audio
from app.core.video_translate.bootstrap import VideoTranslateBootstrap
from app.core.video_translate.diarization import (
    DiarizerFactory,
    HeuristicPauseDiarizer,
    SpeakerTurn,
)
from app.core.video_translate.service_manager import VideoTranslateServiceManager

logger = setup_logger("video_translate_processor")


def _safe_run(cmd: List[str]):
    logger.info("CMD: %s", subprocess.list2cmdline(cmd))
    subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=(
            subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        ),
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

        quality = str(getattr(cfg, "voice_clone_quality", "balanced") or "balanced").strip().lower()
        if quality == "fast":
            strict_xtts = False
            prefer_xtts = False
            allow_non_strict_fallback = True
            max_retries = 1
            req_timeout = 45
        elif quality == "high":
            strict_xtts = False
            prefer_xtts = True
            allow_non_strict_fallback = True
            max_retries = 2
            req_timeout = 90
        elif quality == "studio":
            # Для реального продакшна важнее стабильный качественный вывод,
            # чем fail-fast в strict режиме.
            strict_xtts = False
            prefer_xtts = True
            allow_non_strict_fallback = True
            max_retries = 2
            req_timeout = 90
        else:  # balanced
            strict_xtts = False
            prefer_xtts = False
            allow_non_strict_fallback = True
            max_retries = 2
            req_timeout = 70

        if progress_cb:
            progress_cb(
                0,
                f"Local TTS mode: {quality}, clone={'strict' if strict_xtts else ('soft' if prefer_xtts else 'off')}",
            )

        # Для стабильного voice cloning XTTS нужен более длинный референс (обычно 3-6с).
        ref_duration_sec = 6.0 if quality == "studio" else (4.0 if quality == "high" else 3.0)

        # Эталон голоса: берём исходную дорожку (доступно локально и бесплатно)
        source_audio = work_dir / "source_audio.wav"
        if not source_audio.exists():
            raise RuntimeError("Не найден source_audio.wav для клонирования голоса")

        # Speaker-aware reference wav: собираем лучший эталон из нескольких кусков спикера.
        speaker_refs: Dict[str, Path] = {}
        by_speaker: Dict[str, List[VoiceSegment]] = {}
        for seg in segments:
            by_speaker.setdefault(seg.speaker_id, []).append(seg)

        target_ref_total = 12.0 if quality == "studio" else (8.0 if quality == "high" else 5.0)
        for spk, spk_segs in by_speaker.items():
            ref_wav = work_dir / f"ref_{spk}.wav"
            # Берём самые длинные реплики как более стабильный voiceprint.
            ordered = sorted(spk_segs, key=lambda s: max(0, s.end_ms - s.start_ms), reverse=True)
            parts: List[Path] = []
            acc = 0.0
            for n, seg in enumerate(ordered, start=1):
                if acc >= target_ref_total:
                    break
                clip = work_dir / f"ref_{spk}_{n:02d}.wav"
                start_sec = max(0.0, (seg.start_ms / 1000.0) - 0.30)
                dur_sec = max(1.6, min(6.5, (seg.end_ms - seg.start_ms) / 1000.0 + 0.9))
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
                            str(source_audio),
                            "-ac",
                            "1",
                            "-ar",
                            "24000",
                            str(clip),
                        ]
                    )
                    if clip.exists() and clip.stat().st_size > 0:
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
        max_workers = configured_workers if quality == "studio" else 1

        def _job(i_seg):
            i, seg = i_seg
            wav_path = tts_dir / f"seg_{i:04d}.wav"
            self._synthesize_one(
                endpoint=endpoint,
                text=seg.text,
                speaker_wav=speaker_refs.get(seg.speaker_id, source_audio),
                lang=lang,
                out_path=wav_path,
                require_xtts=strict_xtts,
                prefer_xtts=prefer_xtts,
                allow_non_strict_fallback=allow_non_strict_fallback,
                max_retries=max_retries,
                req_timeout=req_timeout,
            )
            if quality == "studio":
                # XTTS обычно уже даёт mono/24k. Пропускаем лишний ffmpeg per-segment,
                # чтобы не тратить CPU и ускорить pipeline.
                fixed_wav = wav_path
            else:
                fixed_wav = tts_dir / f"seg_{i:04d}_fixed.wav"
                _safe_run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(wav_path),
                        "-ac",
                        "1",
                        "-ar",
                        "24000",
                        str(fixed_wav),
                    ]
                )
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
                        progress_cb(int(done_cnt * 100 / total), f"Local TTS {done_cnt}/{total} ({quality}, x{max_workers})")
            wav_segments = [produced[i] for i in sorted(produced.keys())]
        else:
            for i, seg in enumerate(segments, start=1):
                _, fixed_wav = _job((i, seg))
                wav_segments.append(fixed_wav)
                if progress_cb:
                    progress_cb(int(i * 100 / total), f"Local TTS {i}/{total} ({quality})")

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

        # Жёстко устраняем наложения между сегментами по фактической длительности синтеза.
        adjusted_starts: List[int] = []
        min_gap_ms = 120
        prev_end = 0
        for i, seg in enumerate(segments, start=1):
            wav_dur = _probe_audio_duration_ms(wav_segments[i - 1])
            if wav_dur <= 0:
                wav_dur = max(220, seg.end_ms - seg.start_ms)
            start_ms = max(int(seg.start_ms), prev_end + min_gap_ms)
            adjusted_starts.append(start_ms)
            prev_end = start_ms + wav_dur

        filters = []
        for i, start_ms in enumerate(adjusted_starts, start=1):
            filters.append(f"[{i}:a]adelay={start_ms}|{start_ms}[a{i}]")

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

    def _translate_asr(self, asr_data: ASRData, task: VideoTranslateTask) -> ASRData:
        sub_cfg = task.subtitle_config
        translator_type = self._translator_type(sub_cfg.translator_service)

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

    def _build_segments(self, asr_data: ASRData) -> List[VoiceSegment]:
        result: List[VoiceSegment] = []
        for i, seg in enumerate(asr_data.segments, start=1):
            text = (seg.translated_text or seg.text or "").strip()
            if not text:
                continue
            speaker_id = f"spk_{(i - 1) % 2}" if i > 1 else "spk_0"
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
        return self._enforce_non_overlap(segments)

    def _select_provider(self, task: VideoTranslateTask) -> BaseVoiceCloneProvider:
        provider = (task.video_translate_config.voice_clone_provider or "auto").strip().lower()
        if provider in {"auto", "xtts", "openvoice", "fish_speech"}:
            endpoint = (task.video_translate_config.local_tts_endpoint or "").strip()
            ok = VideoTranslateServiceManager.instance().ensure_local_tts(endpoint)
            if not ok:
                raise RuntimeError(
                    f"Не удалось поднять локальный TTS сервис: {endpoint or 'http://127.0.0.1:8020'}"
                )
            return LocalXTTSProvider()
        if provider in {"elevenlabs"}:
            return ElevenLabsProvider()
        raise RuntimeError(
            "Выбранный voice provider пока не реализован в текущей сборке. Используйте xtts/auto (локально) или elevenlabs."
        )

    def _mux_video(self, task: VideoTranslateTask, source_audio: Path, dubbed_audio: Path, output_path: str):
        keep_bg = bool(getattr(task.video_translate_config, "keep_background_music", True))
        suppress_vocals = bool(getattr(task.video_translate_config, "enable_source_separation", True))
        if keep_bg:
            if suppress_vocals:
                # Псевдо-vocal removal без внешних моделей: ослабляем mid-channel (центр),
                # где обычно находится основной голос. Музыка/стерео остаются заметнее.
                _safe_run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        task.video_path,
                        "-i",
                        str(source_audio),
                        "-i",
                        str(dubbed_audio),
                        "-filter_complex",
                        "[1:a]pan=stereo|c0=0.35*c0-0.65*c1|c1=0.35*c1-0.65*c0,highpass=f=140,lowpass=f=9000,volume=0.55[bg];"
                        "[2:a]loudnorm=I=-16:TP=-1.5:LRA=11[dub];"
                        "[bg][dub]amix=inputs=2:weights='0.35 1.65':normalize=0[mix]",
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
                )
                return

            _safe_run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    task.video_path,
                    "-i",
                    str(source_audio),
                    "-i",
                    str(dubbed_audio),
                    "-filter_complex",
                    "[1:a]volume=0.22[bg];[2:a]loudnorm=I=-16:TP=-1.5:LRA=11[dub];"
                    "[bg][dub]amix=inputs=2:weights='0.7 1.3':normalize=0[mix]",
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
            )
            return

        _safe_run(
            [
                "ffmpeg",
                "-y",
                "-i",
                task.video_path,
                "-i",
                str(dubbed_audio),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-af",
                "loudnorm=I=-16:TP=-1.5:LRA=11",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                output_path,
            ]
        )

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

        # Автовыбор ASR device: не даём падать, если CUDA недоступна
        current_device = getattr(task.transcribe_config, "faster_whisper_device", "cuda")
        resolved_device = self._detect_torch_device(current_device)
        task.transcribe_config.faster_whisper_device = resolved_device
        if resolved_device != str(current_device):
            self._emit(progress_cb, 10, f"ASR device: {current_device} -> {resolved_device}")

        self._emit(progress_cb, 12, "Распознавание речи")
        try:
            asr_data = transcribe(str(audio_path), task.transcribe_config)
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
                    str(audio_path),
                    task,
                    progress_cb=progress_cb,
                )
            else:
                raise

        self._emit(progress_cb, 35, "Перевод текста")
        asr_data = self._translate_asr(asr_data, task)

        if task.output_subtitle_path:
            asr_data.to_srt(task.output_subtitle_path)

        self._emit(progress_cb, 55, "Клонирование голоса / TTS")
        segments = self._build_segments(asr_data)
        if not segments:
            raise RuntimeError("После перевода не найдено сегментов для озвучки")
        segments = self._enforce_non_overlap(segments)

        # Speaker-aware assign
        if getattr(task.video_translate_config, "enable_diarization", False):
            try:
                self._emit(progress_cb, 52, "Diarization спикеров")
                diar_started = time.time()
                diarizer = DiarizerFactory.create_preferred()
                turns = diarizer.diarize(
                    str(audio_path),
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
                segments = self._apply_speaker_turns(segments, turns)
            except Exception as e:
                logger.warning("Diarization failed, fallback to heuristic: %s", e)
                try:
                    turns = HeuristicPauseDiarizer().diarize(str(audio_path), asr_data)
                    segments = self._apply_speaker_turns(segments, turns)
                except Exception:
                    pass

        provider = self._select_provider(task)
        dubbed_audio = provider.synthesize(
            segments,
            work_dir=work_dir,
            task=task,
            progress_cb=lambda p, m: self._emit(progress_cb, 55 + int(p * 0.25), m),
        )

        self._emit(progress_cb, 85, "Сборка итогового видео")
        self._mux_video(task, audio_path, dubbed_audio, task.output_path)

        task.completed_at = datetime.datetime.now()
        self._emit(progress_cb, 100, "Перевод видео завершён")
        return task
