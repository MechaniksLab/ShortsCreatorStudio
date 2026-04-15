import os
import shutil
import subprocess
import tempfile
import hashlib
import json
from pathlib import Path
from typing import Dict, List

from PyQt5.QtCore import QThread, pyqtSignal

from app.common.config import cfg
from app.config import CACHE_PATH, LOG_PATH
from app.core.bk_asr.asr_data import ASRData
from app.core.bk_asr.transcribe import transcribe
from app.core.entities import LLMServiceEnum, TranscribeModelEnum
from app.core.shorts import ShortCandidate, ShortsProcessor, render_shorts
from app.core.task_factory import TaskFactory
from app.core.utils.logger import setup_logger
from app.core.utils.video_utils import video2audio

logger = setup_logger("auto_shorts_thread")
RENDER_DEBUG_LOG = LOG_PATH / "auto_shorts_render.log"


class AutoShortsAnalyzeThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(
        self,
        video_path: str,
        min_duration_s: int,
        max_duration_s: int,
        range_enabled: bool = False,
        range_start_s: int = 0,
        range_end_s: int = 0,
    ):
        super().__init__()
        self.video_path = video_path
        self.min_duration_s = min_duration_s
        self.max_duration_s = max_duration_s
        self.range_enabled = bool(range_enabled)
        self.range_start_s = max(0, int(range_start_s or 0))
        self.range_end_s = max(0, int(range_end_s or 0))

    def run(self):
        temp_wav = None
        temp_clip = None
        try:
            if not Path(self.video_path).exists():
                raise FileNotFoundError("Видео файл не найден")

            source_video_for_asr = self.video_path
            source_offset_ms = 0
            if self.range_enabled and self.range_end_s > self.range_start_s:
                self.progress.emit(3, "Подготовка тестового фрагмента...")
                fd_clip, temp_clip = tempfile.mkstemp(suffix=".mp4")
                os.close(fd_clip)
                self._cut_video_segment(
                    self.video_path,
                    temp_clip,
                    self.range_start_s,
                    self.range_end_s,
                )
                source_video_for_asr = temp_clip
                source_offset_ms = int(self.range_start_s * 1000)

            self.progress.emit(5, "Подготовка аудио...")
            fd, temp_wav = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            if not video2audio(source_video_for_asr, output=temp_wav):
                raise RuntimeError("Не удалось извлечь аудио из видео")

            self.progress.emit(15, "Распознавание речи...")
            transcribe_task = TaskFactory.create_transcribe_task(self.video_path, need_next_task=False)
            # Для плотного монтажа по речи нужны тайминги слов
            transcribe_task.transcribe_config.need_word_time_stamp = True
            cache_key = self._build_asr_cache_key(
                video_path=self.video_path,
                range_enabled=self.range_enabled,
                range_start_s=self.range_start_s,
                range_end_s=self.range_end_s,
                transcribe_model=str(transcribe_task.transcribe_config.transcribe_model),
                llm_service=str(cfg.llm_service.value),
            )
            asr_data = self._load_asr_cache(cache_key)
            if asr_data is None:
                asr_data = self._transcribe_with_fast_profile(temp_wav, transcribe_task.transcribe_config)
                self._save_asr_cache(cache_key, asr_data)
            else:
                self.progress.emit(45, "ASR кеш: найдено, повторный Whisper пропущен")

            llm_cfg = self._resolve_llm_config()
            processor = ShortsProcessor(
                min_duration_s=self.min_duration_s,
                max_duration_s=self.max_duration_s,
                llm_base_url=llm_cfg["base_url"],
                llm_api_key=llm_cfg["api_key"],
                llm_model=llm_cfg["model"],
            )

            candidates = processor.find_candidates(
                asr_data,
                progress_cb=lambda p, m: self.progress.emit(min(95, 70 + int(p * 0.25)), m),
            )
            if source_offset_ms > 0:
                for c in candidates:
                    c.start_ms += source_offset_ms
                    c.end_ms += source_offset_ms
                    if c.speech_ranges:
                        c.speech_ranges = [
                            (int(a) + source_offset_ms, int(b) + source_offset_ms)
                            for a, b in c.speech_ranges
                        ]

            self.progress.emit(100, f"Найдено {len(candidates)} кандидатов")
            self.finished.emit([c.to_dict() for c in candidates])
        except Exception as e:
            logger.exception("Auto shorts analyze failed: %s", e)
            self.error.emit(str(e))
        finally:
            if temp_wav and Path(temp_wav).exists():
                try:
                    Path(temp_wav).unlink()
                except Exception:
                    pass
            if temp_clip and Path(temp_clip).exists():
                try:
                    Path(temp_clip).unlink()
                except Exception:
                    pass

    def _transcribe_with_fast_profile(self, wav_path: str, config):
        """Пробуем быстрый профиль (FasterWhisper), при проблемах откатываемся к исходной модели."""

        def _cb(value, message):
            self.progress.emit(min(70, 15 + int(value * 0.55)), message)

        orig_model = config.transcribe_model
        fw_program = str(getattr(config, "faster_whisper_program", "") or "").strip()
        fw_available = bool((fw_program and Path(fw_program).exists()) or shutil.which(fw_program))

        if fw_available and orig_model != TranscribeModelEnum.FASTER_WHISPER:
            try:
                self.progress.emit(18, "Auto Shorts: быстрый ASR-профиль (FasterWhisper)...")
                config.transcribe_model = TranscribeModelEnum.FASTER_WHISPER
                config.faster_whisper_one_word = True
                config.faster_whisper_vad_filter = True
                config.faster_whisper_ff_mdx_kim2 = False
                return transcribe(wav_path, config, callback=_cb)
            except Exception as e:
                logger.warning("Fast ASR profile failed, fallback to original model: %s", e)
                config.transcribe_model = orig_model

        return transcribe(wav_path, config, callback=_cb)

    @staticmethod
    def _asr_cache_dir() -> Path:
        p = CACHE_PATH / "auto_shorts_asr"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def _build_asr_cache_key(
        video_path: str,
        range_enabled: bool,
        range_start_s: int,
        range_end_s: int,
        transcribe_model: str,
        llm_service: str,
    ) -> str:
        vp = Path(video_path)
        try:
            st = vp.stat()
            file_sig = f"{vp.resolve()}|{int(st.st_size)}|{int(st.st_mtime)}"
        except Exception:
            file_sig = str(vp)

        raw = "|".join(
            [
                file_sig,
                f"range={int(bool(range_enabled))}:{int(range_start_s)}-{int(range_end_s)}",
                f"model={transcribe_model}",
                f"llm={llm_service}",
                "word_ts=1",
            ]
        )
        return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()

    def _load_asr_cache(self, cache_key: str):
        path = self._asr_cache_dir() / f"{cache_key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            asr_json = data.get("asr") if isinstance(data, dict) else None
            if not asr_json:
                return None
            return ASRData.from_json(asr_json)
        except Exception as e:
            logger.warning("AutoShorts ASR cache read failed: %s", e)
            return None

    def _save_asr_cache(self, cache_key: str, asr_data):
        path = self._asr_cache_dir() / f"{cache_key}.json"
        try:
            payload = {
                "version": 1,
                "asr": asr_data.to_json(),
            }
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.warning("AutoShorts ASR cache write failed: %s", e)

    @staticmethod
    def _cut_video_segment(input_video: str, output_video: str, start_s: int, end_s: int):
        # 1) Быстрый путь: remux без перекодирования (самый быстрый, почти без нагрузки)
        cmd_copy = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(start_s),
            "-to",
            str(end_s),
            "-i",
            input_video,
            "-map",
            "0",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-avoid_negative_ts",
            "1",
            "-y",
            output_video,
        ]
        proc = subprocess.run(
            cmd_copy,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=(
                subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            ),
        )
        if proc.returncode == 0 and Path(output_video).exists() and Path(output_video).stat().st_size > 0:
            return

        # 2) Fallback: быстрый NVENC (RTX), если copy не сработал
        cmd_nvenc = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(start_s),
            "-to",
            str(end_s),
            "-i",
            input_video,
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p5",
            "-cq",
            "23",
            "-b:v",
            "0",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-y",
            output_video,
        ]
        proc = subprocess.run(
            cmd_nvenc,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=(
                subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            ),
        )
        if proc.returncode == 0 and Path(output_video).exists() and Path(output_video).stat().st_size > 0:
            return

        # 3) Последний fallback: CPU x264
        cmd_cpu = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(start_s),
            "-to",
            str(end_s),
            "-i",
            input_video,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-y",
            output_video,
        ]
        proc = subprocess.run(
            cmd_cpu,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=(
                subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            ),
        )
        if proc.returncode != 0 or not Path(output_video).exists() or Path(output_video).stat().st_size <= 0:
            raise RuntimeError(f"Не удалось вырезать тестовый диапазон: {proc.stderr[-500:]}")

    @staticmethod
    def _resolve_llm_config() -> Dict[str, str]:
        service = cfg.llm_service.value
        if service == LLMServiceEnum.OPENAI:
            return {
                "base_url": cfg.openai_api_base.value,
                "api_key": cfg.openai_api_key.value,
                "model": cfg.openai_model.value,
            }
        if service == LLMServiceEnum.SILICON_CLOUD:
            return {
                "base_url": cfg.silicon_cloud_api_base.value,
                "api_key": cfg.silicon_cloud_api_key.value,
                "model": cfg.silicon_cloud_model.value,
            }
        if service == LLMServiceEnum.DEEPSEEK:
            return {
                "base_url": cfg.deepseek_api_base.value,
                "api_key": cfg.deepseek_api_key.value,
                "model": cfg.deepseek_model.value,
            }
        if service == LLMServiceEnum.OLLAMA:
            return {
                "base_url": cfg.ollama_api_base.value,
                "api_key": cfg.ollama_api_key.value,
                "model": cfg.ollama_model.value,
            }
        if service == LLMServiceEnum.LM_STUDIO:
            return {
                "base_url": cfg.lm_studio_api_base.value,
                "api_key": cfg.lm_studio_api_key.value,
                "model": cfg.lm_studio_model.value,
            }
        if service == LLMServiceEnum.GEMINI:
            return {
                "base_url": cfg.gemini_api_base.value,
                "api_key": cfg.gemini_api_key.value,
                "model": cfg.gemini_model.value,
            }
        if service == LLMServiceEnum.CHATGLM:
            return {
                "base_url": cfg.chatglm_api_base.value,
                "api_key": cfg.chatglm_api_key.value,
                "model": cfg.chatglm_model.value,
            }
        return {
            "base_url": cfg.public_api_base.value,
            "api_key": cfg.public_api_key.value,
            "model": cfg.public_model.value,
        }


class AutoShortsRenderThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(
        self,
        video_path: str,
        selected_candidates: List[Dict],
        output_dir: str,
        layout_template: Dict = None,
        render_backend: str = "auto",
    ):
        super().__init__()
        self.video_path = video_path
        self.selected_candidates = selected_candidates
        self.output_dir = output_dir
        self.layout_template = layout_template or {}
        self.render_backend = (render_backend or "auto").strip().lower()
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    def run(self):
        try:
            candidates = [
                ShortCandidate(
                    start_ms=int(c["start_ms"]),
                    end_ms=int(c["end_ms"]),
                    score=float(c["score"]),
                    title=str(c.get("title", "")),
                    reason=str(c.get("reason", "")),
                    excerpt=str(c.get("excerpt", "")),
                    speech_ranges=c.get("speech_ranges") or [],
                )
                for c in self.selected_candidates
            ]
            result = render_shorts(
                input_video=self.video_path,
                candidates=candidates,
                output_dir=self.output_dir,
                progress_cb=lambda p, m: self.progress.emit(p, m),
                layout_template=self.layout_template,
                render_backend=self.render_backend,
                cancel_cb=lambda: self._cancel_requested,
            )
            if self._cancel_requested:
                self.progress.emit(0, "Рендер остановлен пользователем")
                self.finished.emit(result)
                return
            if not result:
                raise RuntimeError(
                    "Рендер не создал ни одного шортса. Проверьте FFmpeg, параметры шаблона и путь вывода. "
                    f"Детальный лог: {RENDER_DEBUG_LOG}"
                )
            self.finished.emit(result)
        except Exception as e:
            logger.exception("Auto shorts render failed: %s", e)
            self.error.emit(str(e))
