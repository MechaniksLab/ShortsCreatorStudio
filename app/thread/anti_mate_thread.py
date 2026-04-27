import os
import tempfile
from pathlib import Path
from typing import Dict, List

from PyQt5.QtCore import QThread, pyqtSignal

from app.core.anti_mate import CensorRegion, detect_profanity_regions, render_censored_video
from app.core.anti_mate import render_beep_preview
from app.core.bk_asr.transcribe import transcribe
from app.core.task_factory import TaskFactory
from app.core.utils.logger import setup_logger
from app.core.utils.video_utils import video2audio

logger = setup_logger("anti_mate_thread")


class AntiMateAnalyzeThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    notice = pyqtSignal(str)

    def __init__(
        self,
        video_path: str,
        *,
        forced_forbidden_words: List[str] | None = None,
        forced_allowed_words: List[str] | None = None,
        use_llm: bool = True,
        llm_config: Dict[str, str] | None = None,
        use_asr_cache: bool = True,
        asr_device: str = "cuda",
        asr_cache_tag: str = "anti_mate",
        default_mode: str = "beep",
        pad_before_ms: int = 120,
        pad_after_ms: int = 180,
        merge_gap_ms: int = 220,
        min_region_ms: int = 120,
        max_region_ms: int = 2200,
    ):
        super().__init__()
        self.video_path = str(video_path or "")
        self.forced_forbidden_words = list(forced_forbidden_words or [])
        self.forced_allowed_words = list(forced_allowed_words or [])
        self.use_llm = bool(use_llm)
        self.llm_config = dict(llm_config or {})
        self.use_asr_cache = bool(use_asr_cache)
        self.asr_device = str(asr_device or "cuda").strip().lower()
        self.asr_cache_tag = str(asr_cache_tag or "anti_mate")
        self.default_mode = str(default_mode or "beep")
        self.pad_before_ms = int(pad_before_ms or 120)
        self.pad_after_ms = int(pad_after_ms or 180)
        self.merge_gap_ms = int(merge_gap_ms or 220)
        self.min_region_ms = int(min_region_ms or 120)
        self.max_region_ms = int(max_region_ms or 2200)

    def run(self):
        temp_wav = None
        try:
            if not self.video_path or not Path(self.video_path).exists():
                raise FileNotFoundError("Видео/аудио файл не найден")

            self.progress.emit(5, "Подготовка задачи ASR")
            transcribe_task = TaskFactory.create_transcribe_task(self.video_path, need_next_task=False)
            transcribe_task.transcribe_config.use_asr_cache = bool(self.use_asr_cache)
            transcribe_task.transcribe_config.asr_cache_tag = str(self.asr_cache_tag)
            transcribe_task.transcribe_config.need_word_time_stamp = True
            transcribe_task.transcribe_config.faster_whisper_device = (
                "cpu" if self.asr_device == "cpu" else "cuda"
            )

            self.progress.emit(10, "Извлечение аудио")
            fd, temp_wav = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            if not video2audio(self.video_path, output=temp_wav):
                raise RuntimeError("Не удалось извлечь аудио")

            self.progress.emit(
                15,
                f"Whisper: распознавание речи ({'CPU' if self.asr_device == 'cpu' else 'GPU/CUDA'}, "
                f"кэш {'вкл' if self.use_asr_cache else 'выкл'})",
            )

            def _asr_cb(value, message):
                p = min(65, 15 + int(float(value or 0) * 0.5))
                self.progress.emit(p, str(message or "ASR"))

            asr_data = transcribe(temp_wav, transcribe_task.transcribe_config, callback=_asr_cb)
            asr_json = asr_data.to_json()

            self.progress.emit(68, "Анализ текста: поиск нецензурной брани")

            def _det_cb(value, message):
                p = min(100, 68 + int(float(value or 0) * 0.32))
                self.progress.emit(p, str(message or "Анализ"))

            regions = detect_profanity_regions(
                asr_json,
                forced_forbidden_words=self.forced_forbidden_words,
                forced_allowed_words=self.forced_allowed_words,
                use_llm=self.use_llm,
                llm_config=self.llm_config,
                default_mode=self.default_mode,
                pad_before_ms=self.pad_before_ms,
                pad_after_ms=self.pad_after_ms,
                merge_gap_ms=self.merge_gap_ms,
                min_region_ms=self.min_region_ms,
                max_region_ms=self.max_region_ms,
                progress_cb=_det_cb,
                notify_cb=lambda txt: self.notice.emit(str(txt)),
            )

            self.progress.emit(100, f"Готово: найдено областей {len(regions)}")
            self.finished.emit(
                {
                    "asr_json": asr_json,
                    "regions": [r.to_dict() for r in regions],
                }
            )
        except Exception as e:
            logger.exception("AntiMate analyze failed: %s", e)
            self.error.emit(str(e))
        finally:
            if temp_wav and Path(temp_wav).exists():
                try:
                    Path(temp_wav).unlink()
                except Exception:
                    pass


class AntiMateRenderThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, video_path: str, output_path: str, regions: List[Dict], render_device: str = "cpu"):
        super().__init__()
        self.video_path = str(video_path or "")
        self.output_path = str(output_path or "")
        self.regions = list(regions or [])
        self.render_device = str(render_device or "cpu").strip().lower()
        self.beep_profile: str = "classic"
        self.beep_frequency: int = 1000
        self.beep_volume: float = 0.9
        self.beep_duck_level: float = 0.08

    def run(self):
        try:
            if not self.video_path or not Path(self.video_path).exists():
                raise FileNotFoundError("Исходное видео не найдено")
            if not self.output_path:
                raise RuntimeError("Не задан путь выходного файла")

            converted: List[CensorRegion] = []
            for x in self.regions:
                converted.append(
                    CensorRegion(
                        start_ms=int(x.get("start_ms", 0) or 0),
                        end_ms=int(x.get("end_ms", 0) or 0),
                        trigger_word=str(x.get("trigger_word", "") or ""),
                        source=str(x.get("source", "manual") or "manual"),
                        enabled=bool(x.get("enabled", True)),
                        mode=str(x.get("mode", "beep") or "beep"),
                    )
                )

            out = render_censored_video(
                self.video_path,
                self.output_path,
                converted,
                mode="mixed",
                render_device=self.render_device,
                beep_profile=self.beep_profile,
                beep_frequency=self.beep_frequency,
                beep_volume=self.beep_volume,
                beep_duck_level=self.beep_duck_level,
                progress_cb=lambda p, m: self.progress.emit(int(p), str(m)),
            )
            self.finished.emit(str(out))
        except Exception as e:
            logger.exception("AntiMate render failed: %s", e)
            self.error.emit(str(e))


class AntiMateBeepPreviewThread(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, output_wav: str, profile: str, frequency: int, volume: float):
        super().__init__()
        self.output_wav = str(output_wav or "")
        self.profile = str(profile or "classic")
        self.frequency = int(frequency or 1000)
        self.volume = float(volume or 0.9)

    def run(self):
        try:
            out = render_beep_preview(
                self.output_wav,
                profile=self.profile,
                beep_frequency=self.frequency,
                beep_volume=self.volume,
            )
            self.finished.emit(str(out))
        except Exception as e:
            logger.exception("AntiMate beep preview failed: %s", e)
            self.error.emit(str(e))
