# -*- coding: utf-8 -*-

import os
import shutil
import subprocess
import sys
import json
import tempfile
from types import SimpleNamespace
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QTextEdit,
)
from qfluentwidgets import (
    Action,
    BodyLabel,
    CardWidget,
    ComboBox,
    CommandBar,
    FluentIcon as FIF,
    InfoBar,
    InfoBarPosition,
    LineEdit,
    PrimaryPushButton,
    ProgressBar,
    PushButton,
)

from app.core.entities import (
    SupportedVideoFormats,
    VideoTranslateTask,
    language_value_to_ru,
    translator_service_to_ru,
)
from app.core.task_factory import TaskFactory
from app.core.bk_asr.transcribe import transcribe
from app.core.utils.video_utils import video2audio
from app.core.video_translate.bootstrap import VideoTranslateBootstrap
from app.core.video_translate.processor import VideoTranslationProcessor
from app.core.video_translate.rvc_model_registry import (
    default_rvc_model_root,
    save_rvc_model_params,
    scan_rvc_models,
)
from app.thread.video_translate_thread import VideoTranslateThread
from app.common.config import cfg
from app.config import PROJECT_ROOT


_SPEAKER_ANALYSIS_CACHE = PROJECT_ROOT / "AppData" / "cache" / "video_translate_last_speakers.json"
_LAST_TRANSLATION_CACHE = PROJECT_ROOT / "AppData" / "cache" / "video_translate_last_translation.json"
_LAST_ASR_CACHE = PROJECT_ROOT / "AppData" / "cache" / "video_translate_last_asr.json"
_STAGE_CACHE_DIR = PROJECT_ROOT / "AppData" / "cache" / "video_translate_stage"


class _DiagnosticsThread(QThread):
    finished_report = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

    def run(self):
        try:
            report = VideoTranslateBootstrap().readiness_report(
                model_name=self.model_name,
                auto_download=False,
            )
            self.finished_report.emit(report or {})
        except Exception as e:
            self.failed.emit(str(e))


class _PreviewTranslationThread(QThread):
    finished_rows = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, video_path: str, task: VideoTranslateTask):
        super().__init__()
        self.video_path = video_path
        self.task = task

    def run(self):
        try:
            tmp_dir = Path(tempfile.gettempdir()) / "video_translate_preview"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            src_audio = tmp_dir / "source_audio.wav"
            if not video2audio(self.video_path, str(src_audio)):
                raise RuntimeError("Не удалось извлечь аудио")

            asr_data = transcribe(str(src_audio), self.task.transcribe_config)
            translation_enabled = bool(getattr(self.task.video_translate_config, "translation_enabled", True))
            if translation_enabled:
                asr_data = VideoTranslationProcessor()._translate_asr(asr_data, self.task)
            else:
                for seg in asr_data.segments:
                    seg.translated_text = str(seg.text or "")

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
            self.finished_rows.emit(rows)
        except Exception as e:
            self.failed.emit(str(e))


class _RvcPreviewThread(QThread):
    finished_file = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, model):
        super().__init__()
        self.model = model

    def run(self):
        try:
            out = VideoTranslateInterface._render_rvc_preview_blocking(self.model)
            self.finished_file.emit(str(out))
        except Exception as e:
            self.failed.emit(str(e))


class _StageSeparationThread(QThread):
    finished_paths = pyqtSignal(str, str)
    failed = pyqtSignal(str)

    def __init__(self, video_path: str, task: VideoTranslateTask):
        super().__init__()
        self.video_path = video_path
        self.task = task

    def run(self):
        try:
            self.task.video_translate_config.source_separation_mode = "uvr_mdx_kim"
            try:
                if _STAGE_CACHE_DIR.exists():
                    shutil.rmtree(_STAGE_CACHE_DIR)
            except Exception:
                pass
            _STAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

            src_audio = _STAGE_CACHE_DIR / "source_audio.wav"
            if not video2audio(self.video_path, str(src_audio)):
                raise RuntimeError("Не удалось извлечь аудио из видео")

            processor = VideoTranslationProcessor()
            speech_audio, bg_audio = processor._prepare_audio_stems(self.task, src_audio, _STAGE_CACHE_DIR)

            speech_out = _STAGE_CACHE_DIR / "speech_audio.wav"
            bg_out = _STAGE_CACHE_DIR / "background_audio.wav"
            if speech_audio.exists() and speech_audio != speech_out:
                speech_out.write_bytes(speech_audio.read_bytes())
            if bg_audio.exists() and bg_audio != bg_out:
                bg_out.write_bytes(bg_audio.read_bytes())

            # Единый набор файлов для быстрой проверки качества этапа 1.
            # В одной папке: исходник, только голос, только фон.
            try:
                src_stage = _STAGE_CACHE_DIR / "source_audio.wav"
                voice_only = _STAGE_CACHE_DIR / "voice_only.wav"
                bg_only = _STAGE_CACHE_DIR / "background_only.wav"
                if src_stage.exists():
                    (_STAGE_CACHE_DIR / "original_source.wav").write_bytes(src_stage.read_bytes())
                if speech_out.exists():
                    voice_only.write_bytes(speech_out.read_bytes())
                if bg_out.exists():
                    bg_only.write_bytes(bg_out.read_bytes())
            except Exception:
                pass

            self.finished_paths.emit(str(speech_out), str(bg_out))
        except Exception as e:
            self.failed.emit(str(e))


class _StageAsrThread(QThread):
    finished_rows = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, video_path: str, task: VideoTranslateTask):
        super().__init__()
        self.video_path = video_path
        self.task = task

    def run(self):
        try:
            _STAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            speech_audio = _STAGE_CACHE_DIR / "speech_audio.wav"
            if not speech_audio.exists():
                src_audio = _STAGE_CACHE_DIR / "source_audio.wav"
                if not video2audio(self.video_path, str(src_audio)):
                    raise RuntimeError("Не удалось извлечь аудио из видео")
                speech_audio, _ = VideoTranslationProcessor()._prepare_audio_stems(self.task, src_audio, _STAGE_CACHE_DIR)

            asr_data = transcribe(str(speech_audio), self.task.transcribe_config)
            rows = []
            for seg in asr_data.segments:
                rows.append(
                    {
                        "start_ms": int(seg.start_time),
                        "end_ms": int(seg.end_time),
                        "text": str(seg.text or ""),
                    }
                )
            self.finished_rows.emit(rows)
        except Exception as e:
            self.failed.emit(str(e))


class VideoTranslateInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("VideoTranslateInterface")
        self.task: VideoTranslateTask | None = None
        self.translate_thread: VideoTranslateThread | None = None
        self.preview_thread: _PreviewTranslationThread | None = None
        self.rvc_preview_thread: _RvcPreviewThread | None = None
        self.diag_thread: _DiagnosticsThread | None = None
        self.stage_sep_thread: _StageSeparationThread | None = None
        self.stage_asr_thread: _StageAsrThread | None = None
        self.manual_translation_json_path: str = ""
        self.manual_transcription_json_path: str = ""
        self._rvc_models = []
        self._refresh_rvc_models()
        self._build_ui()
        self._connect_signals()

    def _refresh_rvc_models(self):
        root = Path(str(getattr(cfg, "video_translate_rvc_model_dir", None).value or "").strip()) if str(getattr(cfg, "video_translate_rvc_model_dir", None).value or "").strip() else default_rvc_model_root()
        self._rvc_models = scan_rvc_models(root)

    def _build_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(20)

        top_layout = QHBoxLayout()
        self.command_bar = CommandBar(self)
        self.command_bar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        top_layout.addWidget(self.command_bar, 1)

        self.start_button = PrimaryPushButton("Запустить полный пайплайн", self, icon=FIF.PLAY)
        self.start_button.setFixedHeight(34)
        top_layout.addWidget(self.start_button)
        self.main_layout.addLayout(top_layout)

        open_folder_action = Action(FIF.FOLDER, "Открыть папку", triggered=self.open_output_folder)
        choose_video_action = Action(FIF.FOLDER_ADD, "Выбрать видео", triggered=self.choose_video_file)
        self.command_bar.addAction(open_folder_action)
        self.command_bar.addAction(choose_video_action)
        manage_rvc_action = Action(FIF.SETTING, "Менеджер RVC моделей", triggered=self.open_rvc_model_manager)
        self.command_bar.addAction(manage_rvc_action)
        speaker_map_action = Action(FIF.PEOPLE, "Голоса спикеров", triggered=self.open_speaker_voice_map_dialog)
        self.command_bar.addAction(speaker_map_action)
        check_translate_action = Action(FIF.DICTIONARY, "Проверить перевод", triggered=self.preview_translation)
        self.command_bar.addAction(check_translate_action)
        open_translate_action = Action(FIF.DOCUMENT, "Открыть перевод (srt)", triggered=self.open_translation_file)
        self.command_bar.addAction(open_translate_action)
        open_stage_action = Action(FIF.FOLDER, "Открыть папку этапа 1", triggered=self.open_stage_cache_folder)
        self.command_bar.addAction(open_stage_action)
        edit_last_translate_action = Action(
            FIF.EDIT,
            "Редактировать последний перевод",
            triggered=self.edit_last_translation_and_prepare_redub,
        )
        self.command_bar.addAction(edit_last_translate_action)
        run_diag_action = Action(FIF.DEVELOPER_TOOLS, "GPU-диагностика", triggered=self.run_gpu_diagnostics)
        self.command_bar.addAction(run_diag_action)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.config_card = CardWidget(self.scroll_area)
        config_layout = QVBoxLayout(self.config_card)
        config_layout.setContentsMargins(12, 12, 12, 12)
        config_layout.setSpacing(10)

        file_row = QHBoxLayout()
        file_row.setSpacing(12)
        self.video_label = BodyLabel("Исходное видео", self)
        self.video_input = LineEdit(self)
        self.video_input.setPlaceholderText("Выберите или перетащите видеофайл")
        self.video_button = PushButton("Обзор", self)
        file_row.addWidget(self.video_label)
        file_row.addWidget(self.video_input)
        file_row.addWidget(self.video_button)
        config_layout.addLayout(file_row)

        hint = BodyLabel(
            "Модуль выполнит ASR → перевод текста → TTS/клонирование голоса → сборку нового видео. "
            "Провайдеры и качество настраиваются в Настройки → Перевод видео.",
            self,
        )
        hint.setWordWrap(True)
        config_layout.addWidget(hint)

        # Единая ширина label/combo, чтобы второй столбец не "плавал".
        form_label_w = 220
        form_field_min_w = 210

        row1 = QHBoxLayout()
        row1.setSpacing(10)
        self.device_label = BodyLabel("Устройство", self)
        self.device_label.setMinimumWidth(form_label_w)
        self.device_combo = ComboBox(self)
        self.device_combo.setMinimumWidth(form_field_min_w)
        self.device_combo.addItems(["GPU (CUDA)", "CPU"])
        current_dev = str(cfg.faster_whisper_device.value or "cuda").strip().lower()
        self.device_combo.setCurrentIndex(0 if current_dev == "cuda" else 1)
        row1.addWidget(self.device_label)
        row1.addWidget(self.device_combo, 1)

        self.quality_label = BodyLabel("Режим озвучки", self)
        self.quality_label.setMinimumWidth(form_label_w)
        self.quality_combo = ComboBox(self)
        self.quality_combo.setMinimumWidth(form_field_min_w)
        self.quality_combo.addItems(["Быстрый", "Сбалансированный", "Качество", "Студийный"])
        quality_to_idx = {"fast": 0, "balanced": 1, "high": 2, "studio": 3}
        current_quality = str(cfg.video_translate_voice_quality.value or "balanced").strip().lower()
        self.quality_combo.setCurrentIndex(quality_to_idx.get(current_quality, 1))
        row1.addWidget(self.quality_label)
        row1.addWidget(self.quality_combo, 1)

        row2 = QHBoxLayout()
        row2.setSpacing(10)
        self.overlap_label = BodyLabel("Разрешить overlap реплик (актуально при 2+ спикерах)", self)
        self.overlap_label.setMinimumWidth(form_label_w)
        self.overlap_combo = ComboBox(self)
        self.overlap_combo.setMinimumWidth(form_field_min_w)
        self.overlap_combo.addItems(["Да", "Нет"])
        self.overlap_combo.setCurrentIndex(
            0 if bool(getattr(cfg.video_translate_allow_speaker_overlap, "value", True)) else 1
        )
        row2.addWidget(self.overlap_label)
        row2.addWidget(self.overlap_combo, 1)

        self.mix_label = BodyLabel("Overlap-aware микс по спикерам (актуально при overlap)", self)
        self.mix_label.setMinimumWidth(form_label_w)
        self.mix_combo = ComboBox(self)
        self.mix_combo.setMinimumWidth(form_field_min_w)
        self.mix_combo.addItems(["Да", "Нет"])
        self.mix_combo.setCurrentIndex(
            0 if bool(getattr(cfg.video_translate_overlap_aware_mix, "value", True)) else 1
        )
        row2.addWidget(self.mix_label)
        row2.addWidget(self.mix_combo, 1)

        row3 = QHBoxLayout()
        row3.setSpacing(10)
        self.qa_label = BodyLabel("QA проверка TTS-сегментов (дольше, но чище)", self)
        self.qa_label.setMinimumWidth(form_label_w)
        self.qa_combo = ComboBox(self)
        self.qa_combo.setMinimumWidth(form_field_min_w)
        self.qa_combo.addItems(["Да", "Нет"])
        self.qa_combo.setCurrentIndex(
            0 if bool(getattr(cfg.video_translate_segment_qa_enabled, "value", True)) else 1
        )
        row3.addWidget(self.qa_label)
        row3.addWidget(self.qa_combo, 1)

        self.qa_retry_label = BodyLabel("Повторы QA", self)
        self.qa_retry_label.setMinimumWidth(form_label_w)
        self.qa_retry_combo = ComboBox(self)
        self.qa_retry_combo.setMinimumWidth(form_field_min_w)
        self.qa_retry_combo.addItems([str(i) for i in range(0, 5)])
        cur_qa_retry = int(getattr(cfg.video_translate_segment_qa_retry_count, "value", 1) or 1)
        self.qa_retry_combo.setCurrentText(str(max(0, min(4, cur_qa_retry))))
        row3.addWidget(self.qa_retry_label)
        row3.addWidget(self.qa_retry_combo, 1)

        row4 = QHBoxLayout()
        row4.setSpacing(10)
        self.duck_label = BodyLabel("Ducking фона под речь (не действует в режиме 1:1 фона)", self)
        self.duck_label.setMinimumWidth(form_label_w)
        self.duck_combo = ComboBox(self)
        self.duck_combo.setMinimumWidth(form_field_min_w)
        self.duck_combo.addItems(["Да", "Нет"])
        self.duck_combo.setCurrentIndex(
            0 if bool(getattr(cfg.video_translate_enable_background_ducking, "value", True)) else 1
        )
        row4.addWidget(self.duck_label)
        row4.addWidget(self.duck_combo, 1)

        self.preserve_bg_label = BodyLabel("Сохранить громкость музыки/SFX 1:1 (без loudnorm/ducking)", self)
        self.preserve_bg_label.setMinimumWidth(form_label_w)
        self.preserve_bg_combo = ComboBox(self)
        self.preserve_bg_combo.setMinimumWidth(form_field_min_w)
        self.preserve_bg_combo.addItems(["Да", "Нет"])
        self.preserve_bg_combo.setCurrentIndex(
            0 if bool(getattr(cfg.video_translate_preserve_background_loudness, "value", False)) else 1
        )
        row4.addWidget(self.preserve_bg_label)
        row4.addWidget(self.preserve_bg_combo, 1)

        self.provider_label = BodyLabel("Клонирование голоса", self)
        self.provider_label.setMinimumWidth(form_label_w)
        self.provider_combo = ComboBox(self)
        self.provider_combo.setMinimumWidth(form_field_min_w)
        self.provider_combo.addItems([
            "auto (рекомендуется)",
            "xtts (локально)",
            "rvc (локально, через модели)",
        ])
        current_provider = str(cfg.video_translate_voice_provider.value or "auto").strip().lower()
        provider_to_idx = {"auto": 0, "xtts": 1, "rvc": 2}
        self.provider_combo.setCurrentIndex(provider_to_idx.get(current_provider, 0))
        row4.addWidget(self.provider_label)
        row4.addWidget(self.provider_combo, 1)

        row4b = QHBoxLayout()
        row4b.setSpacing(10)
        self.sep_label = BodyLabel("Разделение источника", self)
        self.sep_label.setMinimumWidth(form_label_w)
        self.sep_combo = ComboBox(self)
        self.sep_combo.setMinimumWidth(form_field_min_w)
        self.sep_combo.addItems([
            "Demucs + UVR",
            "Demucs",
            "UVR MDX/Kim (рекомендуется)",
            "Auto",
        ])
        sep_mode = str(getattr(cfg.video_translate_source_separation_mode, "value", "demucs_plus_uvr") or "demucs_plus_uvr").strip().lower()
        sep_to_idx = {"demucs_plus_uvr": 0, "demucs": 1, "uvr_mdx_kim": 2, "auto": 3}
        self.sep_combo.setCurrentIndex(sep_to_idx.get(sep_mode, 0))
        row4b.addWidget(self.sep_label)
        row4b.addWidget(self.sep_combo, 1)
        self.vocal_kill_label = BodyLabel("Агрессивно убрать остатки оригинального голоса", self)
        self.vocal_kill_label.setMinimumWidth(form_label_w)
        self.vocal_kill_combo = ComboBox(self)
        self.vocal_kill_combo.setMinimumWidth(form_field_min_w)
        self.vocal_kill_combo.addItems(["Да", "Нет"])
        self.vocal_kill_combo.setCurrentIndex(
            0 if bool(getattr(cfg.video_translate_aggressive_vocal_suppression, "value", False)) else 1
        )
        row4b.addWidget(self.vocal_kill_label)
        row4b.addWidget(self.vocal_kill_combo, 1)

        row4c = QHBoxLayout()
        row4c.setSpacing(10)
        self.translation_mode_label = BodyLabel("Режим текста", self)
        self.translation_mode_label.setMinimumWidth(form_label_w)
        self.translation_mode_combo = ComboBox(self)
        self.translation_mode_combo.setMinimumWidth(form_field_min_w)
        self.translation_mode_combo.addItems([
            "Перевод + переозвучка",
            "Без перевода (только переозвучка)",
        ])
        self.translation_mode_combo.setCurrentIndex(0)
        row4c.addWidget(self.translation_mode_label)
        row4c.addWidget(self.translation_mode_combo, 1)
        row4c.addStretch(1)

        row5 = QHBoxLayout()
        row5.setSpacing(10)
        self.source_lang_label = BodyLabel("Исходный язык", self)
        self.source_lang_label.setMinimumWidth(form_label_w)
        self.source_lang_combo = ComboBox(self)
        self.source_lang_combo.setMinimumWidth(form_field_min_w)
        self._source_lang_options = list(cfg.video_translate_source_language.validator.options)
        for opt in self._source_lang_options:
            self.source_lang_combo.addItem(language_value_to_ru(opt.value))
        cur_source = str(cfg.video_translate_source_language.value.value)
        for i, opt in enumerate(self._source_lang_options):
            if str(opt.value) == cur_source:
                self.source_lang_combo.setCurrentIndex(i)
                break
        row5.addWidget(self.source_lang_label)
        row5.addWidget(self.source_lang_combo, 1)

        self.target_lang_label = BodyLabel("Целевой язык", self)
        self.target_lang_label.setMinimumWidth(form_label_w)
        self.target_lang_combo = ComboBox(self)
        self.target_lang_combo.setMinimumWidth(form_field_min_w)
        self._target_lang_options = list(cfg.video_translate_target_language.validator.options)
        for opt in self._target_lang_options:
            self.target_lang_combo.addItem(language_value_to_ru(opt.value))
        cur_target = str(cfg.video_translate_target_language.value.value)
        for i, opt in enumerate(self._target_lang_options):
            if str(opt.value) == cur_target:
                self.target_lang_combo.setCurrentIndex(i)
                break
        row5.addWidget(self.target_lang_label)
        row5.addWidget(self.target_lang_combo, 1)

        self.translator_label = BodyLabel("Сервис перевода", self)
        self.translator_label.setMinimumWidth(form_label_w)
        self.translator_combo = ComboBox(self)
        self.translator_combo.setMinimumWidth(form_field_min_w)
        self._translator_options = list(cfg.translator_service.validator.options)
        for opt in self._translator_options:
            self.translator_combo.addItem(translator_service_to_ru(opt.value))
        cur_translator = str(cfg.translator_service.value.value)
        for i, opt in enumerate(self._translator_options):
            if str(opt.value) == cur_translator:
                self.translator_combo.setCurrentIndex(i)
                break

        row6 = QHBoxLayout()
        row6.setSpacing(10)
        row6.addWidget(self.translator_label)
        row6.addWidget(self.translator_combo, 1)

        self.workers_label = BodyLabel("Параллельных TTS задач (ускорение этапа озвучки)", self)
        self.workers_label.setMinimumWidth(form_label_w)
        self.workers_combo = ComboBox(self)
        self.workers_combo.setMinimumWidth(form_field_min_w)
        self.workers_combo.addItems([str(i) for i in range(1, 9)])
        current_workers = int(getattr(cfg.video_translate_tts_parallel_workers, "value", 3) or 3)
        current_workers = max(1, min(8, current_workers))
        self.workers_combo.setCurrentText(str(current_workers))
        row6.addWidget(self.workers_label)
        row6.addWidget(self.workers_combo, 1)


        row7 = QHBoxLayout()
        row7.setSpacing(10)
        self.cache_label = BodyLabel("Кэшировать перевод", self)
        self.cache_label.setMinimumWidth(form_label_w)
        self.cache_combo = ComboBox(self)
        self.cache_combo.setMinimumWidth(form_field_min_w)
        self.cache_combo.addItems(["Да", "Нет"])
        use_cache = bool(getattr(cfg.video_translate_use_translation_cache, "value", True))
        self.cache_combo.setCurrentIndex(0 if use_cache else 1)
        row7.addWidget(self.cache_label)
        row7.addWidget(self.cache_combo, 1)

        self.asr_cache_label = BodyLabel("Кэшировать распознавание (ASR)", self)
        self.asr_cache_label.setMinimumWidth(form_label_w)
        self.asr_cache_combo = ComboBox(self)
        self.asr_cache_combo.setMinimumWidth(form_field_min_w)
        self.asr_cache_combo.addItems(["Да", "Нет"])
        use_asr_cache = bool(getattr(cfg.video_translate_use_asr_cache, "value", True))
        self.asr_cache_combo.setCurrentIndex(0 if use_asr_cache else 1)
        row7.addWidget(self.asr_cache_label)
        row7.addWidget(self.asr_cache_combo, 1)
        row7.addStretch(1)

        # ------------------------------
        # Этап 1: отделение голоса
        # ------------------------------
        stage1_card = CardWidget(self)
        stage1_layout = QVBoxLayout(stage1_card)
        stage1_layout.setContentsMargins(12, 12, 12, 12)
        stage1_layout.setSpacing(8)
        stage1_layout.addWidget(BodyLabel("Этап 1: Отделение голоса / фона", self))
        stage1_layout.addLayout(row4b)
        self.stage1_button = PrimaryPushButton("Запустить этап 1", self)
        self.stage1_check_button = PushButton("Проверка: открыть папку этапа 1", self)
        stage1_btns = QHBoxLayout()
        stage1_btns.addWidget(self.stage1_button)
        stage1_btns.addWidget(self.stage1_check_button)
        stage1_btns.addStretch(1)
        stage1_layout.addLayout(stage1_btns)
        self.stage1_progress = ProgressBar(self)
        self.stage1_progress.setValue(0)
        self.stage1_status = BodyLabel("Ожидание", self)
        stage1_layout.addWidget(self.stage1_progress)
        stage1_layout.addWidget(self.stage1_status)
        config_layout.addWidget(stage1_card)

        # ------------------------------
        # Этап 2: ASR
        # ------------------------------
        stage2_card = CardWidget(self)
        stage2_layout = QVBoxLayout(stage2_card)
        stage2_layout.setContentsMargins(12, 12, 12, 12)
        stage2_layout.setSpacing(8)
        stage2_layout.addWidget(BodyLabel("Этап 2: Распознавание речи", self))
        stage2_layout.addLayout(row1)
        stage2_layout.addLayout(row7)
        self.stage2_button = PushButton("Запустить этап 2", self)
        self.stage2_check_button = PushButton("Проверка: открыть распознавание (JSON)", self)
        stage2_btns = QHBoxLayout()
        stage2_btns.addWidget(self.stage2_button)
        stage2_btns.addWidget(self.stage2_check_button)
        stage2_btns.addStretch(1)
        stage2_layout.addLayout(stage2_btns)
        self.stage2_progress = ProgressBar(self)
        self.stage2_progress.setValue(0)
        self.stage2_status = BodyLabel("Ожидание", self)
        stage2_layout.addWidget(self.stage2_progress)
        stage2_layout.addWidget(self.stage2_status)
        config_layout.addWidget(stage2_card)

        # ------------------------------
        # Этап 3: перевод/правка
        # ------------------------------
        stage3_card = CardWidget(self)
        stage3_layout = QVBoxLayout(stage3_card)
        stage3_layout.setContentsMargins(12, 12, 12, 12)
        stage3_layout.setSpacing(8)
        stage3_layout.addWidget(BodyLabel("Этап 3: Перевод и ручная правка", self))
        stage3_layout.addLayout(row4c)
        stage3_layout.addLayout(row5)
        self.stage3_button = PushButton("Запустить этап 3 (предпросмотр)", self)
        self.stage3_check_button = PushButton("Проверка: открыть перевод", self)
        stage3_btns = QHBoxLayout()
        stage3_btns.addWidget(self.stage3_button)
        stage3_btns.addWidget(self.stage3_check_button)
        stage3_btns.addStretch(1)
        stage3_layout.addLayout(stage3_btns)
        self.stage3_progress = ProgressBar(self)
        self.stage3_progress.setValue(0)
        self.stage3_status = BodyLabel("Ожидание", self)
        stage3_layout.addWidget(self.stage3_progress)
        stage3_layout.addWidget(self.stage3_status)
        config_layout.addWidget(stage3_card)

        # ------------------------------
        # Этап 4: голоса/спикеры
        # ------------------------------
        stage4_card = CardWidget(self)
        stage4_layout = QVBoxLayout(stage4_card)
        stage4_layout.setContentsMargins(12, 12, 12, 12)
        stage4_layout.setSpacing(8)
        stage4_layout.addWidget(BodyLabel("Этап 4: Голоса спикеров и RVC", self))
        stage4_layout.addLayout(row6)
        self.stage4_button = PushButton("Запустить этап 4 (назначение голосов)", self)
        self.stage4_check_button = PushButton("Проверка: менеджер RVC моделей", self)
        stage4_btns = QHBoxLayout()
        stage4_btns.addWidget(self.stage4_button)
        stage4_btns.addWidget(self.stage4_check_button)
        stage4_btns.addStretch(1)
        stage4_layout.addLayout(stage4_btns)
        self.stage4_progress = ProgressBar(self)
        self.stage4_progress.setValue(0)
        self.stage4_status = BodyLabel("Ожидание", self)
        stage4_layout.addWidget(self.stage4_progress)
        stage4_layout.addWidget(self.stage4_status)
        config_layout.addWidget(stage4_card)

        # ------------------------------
        # Этап 5-6: сборка
        # ------------------------------
        stage56_card = CardWidget(self)
        stage56_layout = QVBoxLayout(stage56_card)
        stage56_layout.setContentsMargins(12, 12, 12, 12)
        stage56_layout.setSpacing(8)
        stage56_layout.addWidget(BodyLabel("Этап 5-6: Озвучка, микс и рендер", self))
        stage56_layout.addLayout(row2)
        stage56_layout.addLayout(row3)
        stage56_layout.addLayout(row4)
        self.stage56_button = PrimaryPushButton("Запустить этап 5-6 (финальный рендер)", self)
        self.stage56_check_button = PushButton("Проверка: открыть папку результата", self)
        stage56_btns = QHBoxLayout()
        stage56_btns.addWidget(self.stage56_button)
        stage56_btns.addWidget(self.stage56_check_button)
        stage56_btns.addStretch(1)
        stage56_layout.addLayout(stage56_btns)
        self.stage56_progress = ProgressBar(self)
        self.stage56_progress.setValue(0)
        self.stage56_status = BodyLabel("Ожидание", self)
        stage56_layout.addWidget(self.stage56_progress)
        stage56_layout.addWidget(self.stage56_status)
        config_layout.addWidget(stage56_card)

        self.diag_label = BodyLabel("Диагностика: не выполнялась", self)
        self.diag_label.setWordWrap(True)
        config_layout.addWidget(self.diag_label)

        diag_btn_row = QHBoxLayout()
        self.diag_button = PushButton("Проверить окружение/GPU", self)
        diag_btn_row.addWidget(self.diag_button)
        diag_btn_row.addStretch(1)
        config_layout.addLayout(diag_btn_row)

        self.stage_hint_label = BodyLabel(
            "Запускайте этапы сверху вниз. Для каждого этапа есть отдельная кнопка проверки.",
            self,
        )
        self.stage_hint_label.setWordWrap(True)
        config_layout.addWidget(self.stage_hint_label)

        self.scroll_area.setWidget(self.config_card)
        self.main_layout.addWidget(self.scroll_area, 1)

        bottom = QHBoxLayout()
        self.progress_bar = ProgressBar(self)
        self.status_label = BodyLabel("Готово", self)
        self.status_label.setMinimumWidth(160)
        self.status_label.setAlignment(Qt.AlignCenter)
        bottom.addWidget(self.progress_bar, 1)
        bottom.addWidget(self.status_label)
        self.main_layout.addLayout(bottom)

    def _connect_signals(self):
        self.video_button.clicked.connect(self.choose_video_file)
        self.start_button.clicked.connect(self.start_translate)
        self.stage1_button.clicked.connect(self.run_stage1_separation)
        self.stage1_check_button.clicked.connect(self.open_stage_cache_folder)
        self.stage2_button.clicked.connect(self.run_stage2_asr)
        self.stage2_check_button.clicked.connect(self.open_last_asr_cache_file)
        self.stage3_button.clicked.connect(self.preview_translation)
        self.stage3_check_button.clicked.connect(self.open_translation_file)
        self.stage4_button.clicked.connect(self.open_speaker_voice_map_dialog)
        self.stage4_check_button.clicked.connect(self.open_rvc_model_manager)
        self.stage56_button.clicked.connect(lambda: self.start_translate(prompt_rvc_map=False))
        self.stage56_check_button.clicked.connect(self.open_output_folder)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        self.quality_combo.currentIndexChanged.connect(self._on_quality_changed)
        self.workers_combo.currentIndexChanged.connect(self._on_workers_changed)
        self.cache_combo.currentIndexChanged.connect(self._on_cache_changed)
        self.asr_cache_combo.currentIndexChanged.connect(self._on_asr_cache_changed)
        self.translation_mode_combo.currentIndexChanged.connect(self._on_translation_mode_changed)
        self.source_lang_combo.currentIndexChanged.connect(self._on_source_language_changed)
        self.target_lang_combo.currentIndexChanged.connect(self._on_target_language_changed)
        self.translator_combo.currentIndexChanged.connect(self._on_translator_changed)
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        self.sep_combo.currentIndexChanged.connect(self._on_separation_mode_changed)
        self.overlap_combo.currentIndexChanged.connect(self._on_overlap_changed)
        self.mix_combo.currentIndexChanged.connect(self._on_overlap_mix_changed)
        self.qa_combo.currentIndexChanged.connect(self._on_qa_changed)
        self.qa_retry_combo.currentIndexChanged.connect(self._on_qa_retry_changed)
        self.duck_combo.currentIndexChanged.connect(self._on_ducking_changed)
        self.preserve_bg_combo.currentIndexChanged.connect(self._on_preserve_bg_changed)
        self.vocal_kill_combo.currentIndexChanged.connect(self._on_aggressive_vocal_suppression_changed)
        self.diag_button.clicked.connect(self.run_gpu_diagnostics)
        self._on_translation_mode_changed()
        self._refresh_filter_applicability_hints()

    def _translation_enabled(self) -> bool:
        return self.translation_mode_combo.currentIndex() == 0

    def _set_stage_progress(self, stage: int, value: int, status: str):
        value = max(0, min(100, int(value)))
        mapping = {
            1: (self.stage1_progress, self.stage1_status),
            2: (self.stage2_progress, self.stage2_status),
            3: (self.stage3_progress, self.stage3_status),
            4: (self.stage4_progress, self.stage4_status),
            56: (self.stage56_progress, self.stage56_status),
        }
        pair = mapping.get(stage)
        if not pair:
            return
        pb, lbl = pair
        pb.setValue(value)
        lbl.setText(status or "Выполняется")

    def _reset_all_stage_progress(self):
        self._set_stage_progress(1, 0, "Ожидание")
        self._set_stage_progress(2, 0, "Ожидание")
        self._set_stage_progress(3, 0, "Ожидание")
        self._set_stage_progress(4, 0, "Ожидание")
        self._set_stage_progress(56, 0, "Ожидание")

    def _route_full_pipeline_progress(self, value: int, status: str):
        # Маршрутизируем общий прогресс в прогресс-блоки этапов.
        if not self._translation_enabled():
            if value <= 11:
                self._set_stage_progress(1, int(value / 11 * 100) if value > 0 else 0, status)
                self._set_stage_progress(3, 0, "Не используется (без перевода)")
                return
            if value <= 34:
                self._set_stage_progress(1, 100, "Готово")
                self._set_stage_progress(2, int((value - 12) / 22 * 100), status)
                self._set_stage_progress(3, 0, "Не используется (без перевода)")
                return
            if value <= 84:
                self._set_stage_progress(1, 100, "Готово")
                self._set_stage_progress(2, 100, "Готово")
                self._set_stage_progress(3, 0, "Не используется (без перевода)")
                self._set_stage_progress(4, int((value - 35) / 49 * 100), status)
                return
            self._set_stage_progress(1, 100, "Готово")
            self._set_stage_progress(2, 100, "Готово")
            self._set_stage_progress(3, 0, "Не используется (без перевода)")
            self._set_stage_progress(4, 100, "Готово")
            self._set_stage_progress(56, int((value - 85) / 15 * 100), status)
            return

        if value <= 11:
            self._set_stage_progress(1, int(value / 11 * 100) if value > 0 else 0, status)
            return
        if value <= 34:
            self._set_stage_progress(1, 100, "Готово")
            self._set_stage_progress(2, int((value - 12) / 22 * 100), status)
            return
        if value <= 49:
            self._set_stage_progress(1, 100, "Готово")
            self._set_stage_progress(2, 100, "Готово")
            self._set_stage_progress(3, int((value - 35) / 14 * 100), status)
            return
        if value <= 84:
            self._set_stage_progress(1, 100, "Готово")
            self._set_stage_progress(2, 100, "Готово")
            self._set_stage_progress(3, 100, "Готово")
            self._set_stage_progress(4, int((value - 50) / 34 * 100), status)
            return
        self._set_stage_progress(1, 100, "Готово")
        self._set_stage_progress(2, 100, "Готово")
        self._set_stage_progress(3, 100, "Готово")
        self._set_stage_progress(4, 100, "Готово")
        self._set_stage_progress(56, int((value - 85) / 15 * 100), status)

    def _on_translation_mode_changed(self):
        enabled = self._translation_enabled()
        self.source_lang_combo.setEnabled(enabled)
        self.target_lang_combo.setEnabled(enabled)
        self.translator_combo.setEnabled(enabled)
        # Этап 3 нужен только когда включён перевод.
        self.stage3_button.setEnabled(enabled)
        self.stage3_check_button.setEnabled(enabled)
        self.stage3_progress.setEnabled(enabled)
        if not enabled:
            self._set_stage_progress(3, 0, "Не используется (без перевода)")
        else:
            self._set_stage_progress(3, 0, "Ожидание")
        if enabled:
            self.diag_label.setText("Режим: перевод + переозвучка")
        else:
            self.diag_label.setText("Режим: без перевода (переозвучка по оригинальному тексту)")

    def _has_complete_speaker_voice_map(self) -> bool:
        try:
            raw = str(getattr(cfg.video_translate_manual_voice_map_json, "value", "") or "").strip()
            if not raw:
                return False
            data = json.loads(raw)
            if not isinstance(data, dict):
                return False
            default_slot = str(data.get("default", "")).strip()
            if not default_slot:
                return False
            speaker_ids = self._load_recent_speaker_ids()
            for spk in speaker_ids:
                if not str(data.get(spk, "")).strip():
                    return False
            return True
        except Exception:
            return False

    def run_gpu_diagnostics(self):
        if self.diag_thread and self.diag_thread.isRunning():
            return
        self.diag_label.setText("Диагностика: проверка окружения...")
        self.diag_thread = _DiagnosticsThread(
            model_name=str(cfg.faster_whisper_model.value.value or "tiny")
        )
        self.diag_thread.finished_report.connect(self._on_diag_finished)
        self.diag_thread.failed.connect(self._on_diag_failed)
        self.diag_thread.start()

    def _on_diag_finished(self, report: dict):
        checks = report.get("checks", []) or []
        failed = [c for c in checks if not c.get("ok")]

        lines = [
            f"Диагностика: CUDA={'да' if report.get('torch_cuda') else 'нет'}, "
            f"ASR={report.get('asr_runtime')}, "
            f"Clone={report.get('clone_quality')}, "
            f"Diar={report.get('diarization_quality')}"
        ]
        if failed:
            lines.append("Проблемы:")
            for c in failed[:4]:
                lines.append(f"• {c.get('name')}: {c.get('hint')}")
        else:
            lines.append("Проблемы не обнаружены.")

        self.diag_label.setText("\n".join(lines))
        self.diag_thread = None

    def _on_diag_failed(self, err: str):
        self.diag_label.setText(f"Диагностика: ошибка запуска ({err})")
        self.diag_thread = None

    def _on_device_changed(self):
        selected = "cuda" if self.device_combo.currentIndex() == 0 else "cpu"
        cfg.set(cfg.faster_whisper_device, selected)

    def _on_quality_changed(self):
        idx = self.quality_combo.currentIndex()
        quality = {0: "fast", 1: "balanced", 2: "high", 3: "studio"}.get(idx, "balanced")
        cfg.set(cfg.video_translate_voice_quality, quality)
        self._apply_quality_preset(quality)

    def _on_workers_changed(self):
        try:
            workers = int(self.workers_combo.currentText())
        except Exception:
            workers = 3
        workers = max(1, min(8, workers))
        cfg.set(cfg.video_translate_tts_parallel_workers, workers)

    def _on_cache_changed(self):
        cfg.set(cfg.video_translate_use_translation_cache, self.cache_combo.currentIndex() == 0)

    def _on_asr_cache_changed(self):
        cfg.set(cfg.video_translate_use_asr_cache, self.asr_cache_combo.currentIndex() == 0)

    def _on_source_language_changed(self):
        idx = self.source_lang_combo.currentIndex()
        if 0 <= idx < len(self._source_lang_options):
            cfg.set(cfg.video_translate_source_language, self._source_lang_options[idx])

    def _on_target_language_changed(self):
        idx = self.target_lang_combo.currentIndex()
        if 0 <= idx < len(self._target_lang_options):
            cfg.set(cfg.video_translate_target_language, self._target_lang_options[idx])

    def _on_translator_changed(self):
        idx = self.translator_combo.currentIndex()
        if 0 <= idx < len(self._translator_options):
            cfg.set(cfg.translator_service, self._translator_options[idx])

    def _on_provider_changed(self):
        provider = {0: "auto", 1: "xtts", 2: "rvc"}.get(self.provider_combo.currentIndex(), "auto")
        cfg.set(cfg.video_translate_voice_provider, provider)

    def _on_separation_mode_changed(self):
        mode = {
            0: "demucs_plus_uvr",
            1: "demucs",
            2: "uvr_mdx_kim",
            3: "auto",
        }.get(self.sep_combo.currentIndex(), "demucs_plus_uvr")
        cfg.set(cfg.video_translate_source_separation_mode, mode)
        self._refresh_filter_applicability_hints()

    def _on_overlap_changed(self):
        cfg.set(cfg.video_translate_allow_speaker_overlap, self.overlap_combo.currentIndex() == 0)

    def _on_overlap_mix_changed(self):
        cfg.set(cfg.video_translate_overlap_aware_mix, self.mix_combo.currentIndex() == 0)

    def _on_qa_changed(self):
        cfg.set(cfg.video_translate_segment_qa_enabled, self.qa_combo.currentIndex() == 0)

    def _on_qa_retry_changed(self):
        try:
            retries = int(self.qa_retry_combo.currentText())
        except Exception:
            retries = 1
        cfg.set(cfg.video_translate_segment_qa_retry_count, max(0, min(4, retries)))

    def _on_ducking_changed(self):
        cfg.set(cfg.video_translate_enable_background_ducking, self.duck_combo.currentIndex() == 0)

    def _on_preserve_bg_changed(self):
        cfg.set(
            cfg.video_translate_preserve_background_loudness,
            self.preserve_bg_combo.currentIndex() == 0,
        )
        self._refresh_filter_applicability_hints()

    def _on_aggressive_vocal_suppression_changed(self):
        cfg.set(
            cfg.video_translate_aggressive_vocal_suppression,
            self.vocal_kill_combo.currentIndex() == 0,
        )
        self._refresh_filter_applicability_hints()

    def _refresh_filter_applicability_hints(self):
        mode = {
            0: "demucs_plus_uvr",
            1: "demucs",
            2: "uvr_mdx_kim",
            3: "auto",
        }.get(self.sep_combo.currentIndex(), "demucs_plus_uvr")

        # Для UVR MDX/Kim отдельный "агрессивный" cleanup обычно не нужен.
        if mode == "uvr_mdx_kim":
            self.vocal_kill_label.setText("Агрессивно убрать остатки оригинального голоса (обычно не нужно для UVR MDX/Kim)")
            self.vocal_kill_combo.setCurrentIndex(1)
            self.vocal_kill_combo.setEnabled(False)
        else:
            self.vocal_kill_label.setText("Агрессивно убрать остатки оригинального голоса (актуально для Demucs/Demucs+UVR)")
            self.vocal_kill_combo.setEnabled(True)

        preserve_bg = self.preserve_bg_combo.currentIndex() == 0
        aggressive = self.vocal_kill_combo.currentIndex() == 0 and self.vocal_kill_combo.isEnabled()
        ducking_effective = not preserve_bg and not aggressive
        self.duck_combo.setEnabled(ducking_effective)
        if preserve_bg:
            self.duck_label.setText("Ducking фона под речь (выключен: выбран режим 1:1 фона)")
        elif aggressive:
            self.duck_label.setText("Ducking фона под речь (выключен: включено агрессивное подавление вокала)")
        else:
            self.duck_label.setText("Ducking фона под речь (не действует в режиме 1:1 фона)")

    def _model_display_items(self):
        self._refresh_rvc_models()
        items = []
        for m in self._rvc_models:
            label = f"{m.slot}: {m.name}"
            items.append((label, m))
        return items

    @staticmethod
    def _find_model_preview_audio(model) -> Path | None:
        try:
            slot_dir = Path(getattr(model, "slot_dir", ""))
            if not slot_dir.exists():
                return None
            for ext in ("*.wav", "*.mp3", "*.ogg", "*.flac", "*.m4a"):
                files = sorted(slot_dir.glob(ext))
                if files:
                    return files[0]
        except Exception:
            return None
        return None

    @staticmethod
    def _resolve_rvc_runtime_python() -> Path | None:
        candidates = [
            PROJECT_ROOT / "runtime_rvc" / "Scripts" / "python.exe",
            PROJECT_ROOT / "runtime" / "python.exe",
        ]
        for p in candidates:
            if p.exists() and p.is_file():
                return p
        return None

    @staticmethod
    def _global_rvc_preview_audio() -> Path | None:
        root = PROJECT_ROOT / "AppData" / "models" / "rvc"
        if not root.exists():
            return None
        for ext in ("*.wav", "*.mp3", "*.ogg", "*.flac", "*.m4a"):
            files = sorted(root.glob(ext))
            if files:
                return files[0]
        return None

    @staticmethod
    def _render_rvc_preview_blocking(model) -> Path:
        preview_src = VideoTranslateInterface._global_rvc_preview_audio()
        if not preview_src:
            raise RuntimeError(
                "Не найден единый пример аудио в AppData/models/rvc. "
                "Положите туда 1 файл (например preview.wav)."
            )

        runtime_py = VideoTranslateInterface._resolve_rvc_runtime_python()
        if not runtime_py:
            raise RuntimeError("Не найден runtime Python для RVC (runtime_rvc/runtime)")

        script = PROJECT_ROOT / "scripts" / "rvc_native_infer.py"
        if not script.exists():
            raise RuntimeError(f"Не найден скрипт RVC: {script}")

        model_pth = Path(model.slot_dir) / str(model.model_file)
        idx_file = str(getattr(model, "index_file", "") or "").strip()
        idx_path = (Path(model.slot_dir) / idx_file) if idx_file else None

        out_wav = Path(tempfile.gettempdir()) / f"rvc_preview_{model.slot}.wav"
        cmd = [
            str(runtime_py),
            str(script),
            "--input", str(preview_src),
            "--output", str(out_wav),
            "--model", str(model_pth),
            "--f0-up-key", str(int(getattr(model, "default_tune", 0) or 0)),
            "--index-rate", f"{float(getattr(model, 'default_index_ratio', 0.0) or 0.0):.4f}",
            "--protect", f"{float(getattr(model, 'default_protect', 0.5) or 0.5):.4f}",
            "--filter-radius", "3",
            "--device", "cuda",
        ]
        if idx_path and idx_path.exists():
            cmd += ["--index", str(idx_path)]

        try:
            p = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
                creationflags=(subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0),
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("RVC preview timeout: генерация примера заняла слишком много времени (>120с)")
        if p.returncode != 0 or not out_wav.exists() or out_wav.stat().st_size <= 0:
            err = ((p.stderr or "") + "\n" + (p.stdout or "")).strip()
            raise RuntimeError(err[-1200:] if err else "RVC preview failed")
        return out_wav

    def _render_rvc_preview(self, model) -> Path:
        return self._render_rvc_preview_blocking(model)

    def _start_rvc_preview_async(self, *, model, owner_dialog: QDialog, trigger_button: PushButton):
        if self.rvc_preview_thread and self.rvc_preview_thread.isRunning():
            InfoBar.warning(
                "Подождите",
                "Уже идёт генерация примера голоса",
                duration=2200,
                parent=owner_dialog,
            )
            return

        old_text = trigger_button.text()
        try:
            trigger_button.setEnabled(False)
            trigger_button.setText("⏳ Генерация...")
        except Exception:
            pass
        self.diag_label.setText(f"Предпрослушка RVC: генерируем пример для модели '{getattr(model, 'name', getattr(model, 'slot', ''))}'...")

        th = _RvcPreviewThread(model=model)
        self.rvc_preview_thread = th

        def _restore_btn():
            try:
                trigger_button.setEnabled(True)
                trigger_button.setText(old_text)
            except Exception:
                pass

        def _on_finished(path: str):
            _restore_btn()
            self.rvc_preview_thread = None
            self.diag_label.setText("Предпрослушка RVC: готово")
            p = str(path or "").strip()
            if not p:
                return
            if sys.platform == "win32":
                os.startfile(p)
            elif sys.platform == "darwin":
                subprocess.run(["open", p])
            else:
                subprocess.run(["xdg-open", p])

        def _on_failed(err: str):
            _restore_btn()
            self.rvc_preview_thread = None
            self.diag_label.setText(f"Предпрослушка RVC: ошибка ({err})")
            self._show_copyable_error(
                title="Не удалось прослушать",
                message=str(err),
                log_file_name="rvc_preview_last_error.txt",
                parent_widget=owner_dialog,
                duration=12000,
            )

        th.finished_file.connect(_on_finished)
        th.failed.connect(_on_failed)
        th.start()

    def _show_copyable_error(
        self,
        *,
        title: str,
        message: str,
        log_file_name: str = "video_translate_last_error.txt",
        parent_widget=None,
        duration: int = 9000,
    ):
        msg = str(message or "").strip() or "Неизвестная ошибка"
        parent_ref = parent_widget or self

        # Clipboard
        try:
            QApplication.clipboard().setText(msg)
        except Exception:
            pass

        # Persist log
        err_file = PROJECT_ROOT / "AppData" / "logs" / str(log_file_name)
        try:
            err_file.parent.mkdir(parents=True, exist_ok=True)
            err_file.write_text(msg, encoding="utf-8")
        except Exception:
            pass

        InfoBar.error(
            title,
            f"{msg}\n(подробно: {err_file.as_posix()}, текст скопирован в буфер)",
            duration=duration,
            position=InfoBarPosition.TOP,
            parent=parent_ref,
        )

        # Copyable dialog
        try:
            dlg = QDialog(parent_ref)
            dlg.setWindowTitle(title)
            dlg.resize(980, 520)
            lay = QVBoxLayout(dlg)
            txt = QTextEdit(dlg)
            txt.setReadOnly(True)
            txt.setPlainText(msg)
            lay.addWidget(txt)
            btns = QHBoxLayout()
            copy_btn = PrimaryPushButton("Скопировать ошибку", dlg)
            open_log_btn = PushButton("Открыть лог", dlg)
            close_btn = PushButton("Закрыть", dlg)
            btns.addWidget(copy_btn)
            btns.addWidget(open_log_btn)
            btns.addWidget(close_btn)
            btns.addStretch(1)
            lay.addLayout(btns)

            def _copy_err():
                QApplication.clipboard().setText(txt.toPlainText())
                InfoBar.success("Скопировано", "Текст ошибки скопирован в буфер", duration=1800, parent=dlg)

            def _open_log():
                if err_file.exists():
                    if sys.platform == "win32":
                        os.startfile(str(err_file))
                    elif sys.platform == "darwin":
                        subprocess.run(["open", str(err_file)])
                    else:
                        subprocess.run(["xdg-open", str(err_file)])

            copy_btn.clicked.connect(_copy_err)
            open_log_btn.clicked.connect(_open_log)
            close_btn.clicked.connect(dlg.accept)
            dlg.exec_()
        except Exception:
            pass

    def open_rvc_model_manager(self):
        items = self._model_display_items()
        if not items:
            InfoBar.warning(
                "RVC модели не найдены",
                f"Проверьте папку: {default_rvc_model_root()}",
                duration=3500,
                parent=self,
            )
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Менеджер моделей RVC")
        dlg.resize(980, 620)
        lay = QVBoxLayout(dlg)

        table = QTableWidget(len(items), 11, dlg)
        table.setHorizontalHeaderLabels([
            "Слот",
            "Имя",
            "Модель (.pth)",
            "Индекс (.index)",
            "Иконка",
            "Пример",
            "Путь к иконке",
            "Тональность",
            "Сила индекса",
            "Защита",
            "Файл параметров",
        ])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(6, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(8, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(9, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(10, QHeaderView.Stretch)

        for i, (_, m) in enumerate(items):
            table.setItem(i, 0, QTableWidgetItem(str(m.slot)))
            table.setItem(i, 1, QTableWidgetItem(str(m.name)))
            table.setItem(i, 2, QTableWidgetItem(str(m.model_file)))
            table.setItem(i, 3, QTableWidgetItem(str(m.index_file or "")))
            preview_item = QTableWidgetItem()
            icon_path_text = str(m.icon_path or "")
            if icon_path_text and Path(icon_path_text).exists():
                pix = QPixmap(icon_path_text)
                if not pix.isNull():
                    preview_item.setIcon(QIcon(pix.scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
            table.setItem(i, 4, preview_item)

            preview_btn = PushButton("▶ Пример", dlg)

            def _bind_preview(row=i, model_ref=m):
                def _on_preview():
                    # Берём актуальные значения прямо из таблицы (даже если ещё не нажали "Сохранить").
                    try:
                        tune_val = int(float(table.item(row, 7).text() if table.item(row, 7) else getattr(model_ref, "default_tune", 0)))
                    except Exception:
                        tune_val = int(getattr(model_ref, "default_tune", 0) or 0)
                    try:
                        index_ratio_val = float(table.item(row, 8).text() if table.item(row, 8) else getattr(model_ref, "default_index_ratio", 0.0))
                    except Exception:
                        index_ratio_val = float(getattr(model_ref, "default_index_ratio", 0.0) or 0.0)
                    try:
                        protect_val = float(table.item(row, 9).text() if table.item(row, 9) else getattr(model_ref, "default_protect", 0.5))
                    except Exception:
                        protect_val = float(getattr(model_ref, "default_protect", 0.5) or 0.5)

                    model_for_preview = SimpleNamespace(
                        slot=getattr(model_ref, "slot", 0),
                        name=getattr(model_ref, "name", ""),
                        slot_dir=getattr(model_ref, "slot_dir", ""),
                        model_file=getattr(model_ref, "model_file", ""),
                        index_file=getattr(model_ref, "index_file", ""),
                        default_tune=tune_val,
                        default_index_ratio=index_ratio_val,
                        default_protect=protect_val,
                    )

                    self._start_rvc_preview_async(model=model_for_preview, owner_dialog=dlg, trigger_button=preview_btn)

                return _on_preview

            preview_btn.clicked.connect(_bind_preview())
            table.setCellWidget(i, 5, preview_btn)

            table.setItem(i, 6, QTableWidgetItem(icon_path_text))
            table.setItem(i, 7, QTableWidgetItem(str(m.default_tune)))
            table.setItem(i, 8, QTableWidgetItem(f"{m.default_index_ratio:.3f}"))
            table.setItem(i, 9, QTableWidgetItem(f"{m.default_protect:.3f}"))
            table.setItem(i, 10, QTableWidgetItem(str(m.params_path)))
            # read-only columns
            for col in (0, 2, 3, 4, 6, 10):
                item = table.item(i, col)
                if item:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            table.setRowHeight(i, 128)

        lay.addWidget(table)
        btn_row = QHBoxLayout()
        save_btn = PrimaryPushButton("Сохранить параметры", dlg)
        close_btn = PushButton("Закрыть", dlg)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(close_btn)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)

        def _save_models():
            ok_count = 0
            for i, (_, m) in enumerate(items):
                try:
                    name = str(table.item(i, 1).text() if table.item(i, 1) else m.name).strip() or m.name
                    tune = int(float(table.item(i, 7).text() if table.item(i, 7) else m.default_tune))
                    idx_ratio = float(table.item(i, 8).text() if table.item(i, 8) else m.default_index_ratio)
                    protect = float(table.item(i, 9).text() if table.item(i, 9) else m.default_protect)
                    save_rvc_model_params(
                        m,
                        name=name,
                        default_tune=tune,
                        default_index_ratio=max(0.0, min(1.0, idx_ratio)),
                        default_protect=max(0.0, min(0.5, protect)),
                    )
                    ok_count += 1
                except Exception:
                    continue
            self._refresh_rvc_models()
            InfoBar.success(
                "RVC модели обновлены",
                f"Сохранено файлов параметров: {ok_count}",
                duration=3000,
                parent=self,
            )

        save_btn.clicked.connect(_save_models)
        close_btn.clicked.connect(dlg.reject)
        dlg.exec_()

    def open_speaker_voice_map_dialog(self) -> bool:
        items = self._model_display_items()
        if not items:
            InfoBar.warning(
                "RVC модели не найдены",
                f"Проверьте папку: {default_rvc_model_root()}",
                duration=3500,
                parent=self,
            )
            return False

        speaker_ids = self._load_recent_speaker_ids()
        speaker_count = len(speaker_ids)

        # load existing map
        existing_raw = str(getattr(cfg.video_translate_manual_voice_map_json, "value", "") or "").strip()
        existing = {}
        if existing_raw:
            try:
                existing = json.loads(existing_raw)
            except Exception:
                existing = {}

        dlg = QDialog(self)
        dlg.setWindowTitle("Голоса спикеров (RVC)")
        dlg.resize(740, 520)
        lay = QVBoxLayout(dlg)

        hint = BodyLabel(
            "Выберите модель RVC для каждого спикера. Назначения будут сохранены автоматически.",
            dlg,
        )
        hint.setWordWrap(True)
        lay.addWidget(hint)

        table = QTableWidget(speaker_count, 4, dlg)
        table.setHorizontalHeaderLabels(["Спикер", "Модель RVC", "Тональность", "Пример голоса"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)

        combos = []
        for i, spk in enumerate(speaker_ids):
            table.setItem(i, 0, QTableWidgetItem(spk))
            table.item(i, 0).setFlags(table.item(i, 0).flags() & ~Qt.ItemIsEditable)

            combo = ComboBox(dlg)
            combo.addItem("(по умолчанию)")
            for label, _m in items:
                combo.addItem(label)
            saved_slot = str(existing.get(spk, "")).strip()
            if saved_slot:
                for idx, (_label, m) in enumerate(items, start=1):
                    if saved_slot == str(m.slot):
                        combo.setCurrentIndex(idx)
                        break
            table.setCellWidget(i, 1, combo)
            combos.append(combo)

            tune = "0"
            if combo.currentIndex() > 0:
                tune = str(items[combo.currentIndex() - 1][1].default_tune)
            table.setItem(i, 2, QTableWidgetItem(tune))
            table.item(i, 2).setFlags(table.item(i, 2).flags() & ~Qt.ItemIsEditable)

            preview_btn = PushButton("▶ Пример", dlg)

            def _bind_preview(c=combo):
                def _on_preview():
                    if c.currentIndex() <= 0:
                        InfoBar.warning(
                            "Не выбрана модель",
                            "Сначала выберите RVC-модель для этого спикера",
                            duration=2000,
                            parent=dlg,
                        )
                        return
                    model = items[c.currentIndex() - 1][1]
                    self._start_rvc_preview_async(model=model, owner_dialog=dlg, trigger_button=preview_btn)

                return _on_preview

            preview_btn.clicked.connect(_bind_preview())
            table.setCellWidget(i, 3, preview_btn)

            def _bind_update(row=i, c=combo):
                def _on_changed(_):
                    t = "0"
                    if c.currentIndex() > 0:
                        t = str(items[c.currentIndex() - 1][1].default_tune)
                    table.setItem(row, 2, QTableWidgetItem(t))
                    table.item(row, 2).setFlags(table.item(row, 2).flags() & ~Qt.ItemIsEditable)

                return _on_changed

            combo.currentIndexChanged.connect(_bind_update())

        lay.addWidget(table)

        btn_row = QHBoxLayout()
        save_btn = PrimaryPushButton("Сохранить назначение", dlg)
        close_btn = PushButton("Закрыть", dlg)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(close_btn)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)

        saved_ok = {"ok": False}

        def _save_map():
            out = {}
            for i, combo in enumerate(combos):
                if combo.currentIndex() <= 0:
                    InfoBar.warning(
                        "Нужно выбрать голоса",
                        "Назначьте RVC-модель каждому спикеру",
                        duration=2600,
                        parent=dlg,
                    )
                    return
                model = items[combo.currentIndex() - 1][1]
                spk = speaker_ids[i]
                out[spk] = str(model.slot)

            out["default"] = str(items[combos[0].currentIndex() - 1][1].slot)
            cfg.set(cfg.video_translate_manual_voice_map_json, json.dumps(out, ensure_ascii=False))
            saved_ok["ok"] = True
            InfoBar.success(
                "Сохранено",
                "Назначение голосов спикерам сохранено",
                duration=2500,
                parent=self,
            )
            dlg.accept()

        save_btn.clicked.connect(_save_map)
        close_btn.clicked.connect(dlg.reject)
        dlg.exec_()
        return bool(saved_ok["ok"])

    def _apply_quality_preset(self, quality: str):
        q = (quality or "balanced").strip().lower()
        if q == "fast":
            overlap, mix, qa, qa_retry, duck, preserve_bg, vocal_kill = False, False, False, 0, False, False, False
        elif q == "balanced":
            overlap, mix, qa, qa_retry, duck, preserve_bg, vocal_kill = False, True, False, 0, True, False, False
        elif q == "studio":
            overlap, mix, qa, qa_retry, duck, preserve_bg, vocal_kill = True, True, True, 2, True, True, True
        else:  # high
            overlap, mix, qa, qa_retry, duck, preserve_bg, vocal_kill = True, True, True, 1, True, False, True

        self.overlap_combo.setCurrentIndex(0 if overlap else 1)
        self.mix_combo.setCurrentIndex(0 if mix else 1)
        self.qa_combo.setCurrentIndex(0 if qa else 1)
        self.qa_retry_combo.setCurrentText(str(qa_retry))
        self.duck_combo.setCurrentIndex(0 if duck else 1)
        self.preserve_bg_combo.setCurrentIndex(0 if preserve_bg else 1)
        self.vocal_kill_combo.setCurrentIndex(0 if vocal_kill else 1)

        cfg.set(cfg.video_translate_allow_speaker_overlap, overlap)
        cfg.set(cfg.video_translate_overlap_aware_mix, mix)
        cfg.set(cfg.video_translate_segment_qa_enabled, qa)
        cfg.set(cfg.video_translate_segment_qa_retry_count, qa_retry)
        cfg.set(cfg.video_translate_enable_background_ducking, duck)
        cfg.set(cfg.video_translate_preserve_background_loudness, preserve_bg)
        cfg.set(cfg.video_translate_aggressive_vocal_suppression, vocal_kill)

    def choose_video_file(self):
        video_formats = " ".join(f"*.{fmt.value}" for fmt in SupportedVideoFormats)
        filter_str = f"Видеофайлы ({video_formats})"
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите видео", "", filter_str)
        if file_path:
            self.video_input.setText(file_path)

    def _ensure_video_path(self) -> str | None:
        video_path = (self.video_input.text() or "").strip()
        if not video_path or not Path(video_path).exists():
            InfoBar.warning("Внимание", "Сначала выберите видео", duration=2200, parent=self)
            return None
        return video_path

    def run_stage1_separation(self):
        video_path = self._ensure_video_path()
        if not video_path:
            return
        if self.stage_sep_thread and self.stage_sep_thread.isRunning():
            return
        # Используем рекомендуемый метод разделения для этапа 1.
        self.sep_combo.setCurrentIndex(2)  # UVR MDX/Kim
        cfg.set(cfg.video_translate_source_separation_mode, "uvr_mdx_kim")
        task = self._build_task_from_ui(video_path)
        self.stage_hint_label.setText("Этап 1: разделение (UVR MDX/Kim), подождите...")
        self._set_stage_progress(1, 5, "Этап 1: запуск")
        self.stage1_button.setEnabled(False)
        self.stage_sep_thread = _StageSeparationThread(video_path=video_path, task=task)

        def _ok(speech_path: str, bg_path: str):
            self.stage1_button.setEnabled(True)
            self.stage_sep_thread = None
            speech_out = Path(speech_path)
            bg_out = Path(bg_path)
            self.stage_hint_label.setText(
                f"Этап 1 завершён (UVR MDX/Kim). Голос: {speech_out.name}, Фон: {bg_out.name}. "
                f"Папка: {_STAGE_CACHE_DIR}"
            )
            self._set_stage_progress(1, 100, "Этап 1 завершён")
            if speech_out.exists() and sys.platform == "win32":
                os.startfile(str(speech_out))
            InfoBar.success("Этап 1", "Готово: можно прослушать чистый голос", duration=2600, parent=self)

        def _fail(err: str):
            self.stage1_button.setEnabled(True)
            self.stage_sep_thread = None
            self._set_stage_progress(1, 100, f"Ошибка этапа 1")
            self._show_copyable_error(
                title="Ошибка этапа 1",
                message=str(err),
                log_file_name="video_translate_stage1_error.txt",
                parent_widget=self,
                duration=12000,
            )

        self.stage_sep_thread.finished_paths.connect(_ok)
        self.stage_sep_thread.failed.connect(_fail)
        self.stage_sep_thread.start()

    def run_stage2_asr(self):
        video_path = self._ensure_video_path()
        if not video_path:
            return
        if self.stage_asr_thread and self.stage_asr_thread.isRunning():
            return
        task = self._build_task_from_ui(video_path)
        self.stage_hint_label.setText("Этап 2: распознавание речи, подождите...")
        self._set_stage_progress(2, 5, "Этап 2: запуск")
        self.stage2_button.setEnabled(False)
        self.stage_asr_thread = _StageAsrThread(video_path=video_path, task=task)

        def _ok(rows: list):
            self.stage2_button.setEnabled(True)
            self.stage_asr_thread = None
            _LAST_ASR_CACHE.parent.mkdir(parents=True, exist_ok=True)
            _LAST_ASR_CACHE.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            self.manual_transcription_json_path = str(_LAST_ASR_CACHE)
            self._open_asr_editor(rows)
            self.stage_hint_label.setText("Этап 2 завершён. Распознавание сохранено и доступно для правки.")
            self._set_stage_progress(2, 100, "Этап 2 завершён")

        def _fail(err: str):
            self.stage2_button.setEnabled(True)
            self.stage_asr_thread = None
            self._set_stage_progress(2, 100, "Ошибка этапа 2")
            self._show_copyable_error(
                title="Ошибка этапа 2",
                message=str(err),
                log_file_name="video_translate_stage2_error.txt",
                parent_widget=self,
                duration=12000,
            )

        self.stage_asr_thread.finished_rows.connect(_ok)
        self.stage_asr_thread.failed.connect(_fail)
        self.stage_asr_thread.start()

    def _open_asr_editor(self, rows: list):
        dlg = QDialog(self)
        dlg.setWindowTitle("Этап 2: Распознавание речи")
        dlg.resize(1100, 700)
        lay = QVBoxLayout(dlg)
        table = QTableWidget(len(rows), 3, dlg)
        table.setHorizontalHeaderLabels(["Start", "End", "Текст"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        for i, r in enumerate(rows):
            table.setItem(i, 0, QTableWidgetItem(str(r["start_ms"])))
            table.setItem(i, 1, QTableWidgetItem(str(r["end_ms"])))
            table.item(i, 0).setFlags(table.item(i, 0).flags() & ~Qt.ItemIsEditable)
            table.item(i, 1).setFlags(table.item(i, 1).flags() & ~Qt.ItemIsEditable)
            table.setItem(i, 2, QTableWidgetItem(r["text"]))
        lay.addWidget(table)

        btns = QHBoxLayout()
        apply_btn = PrimaryPushButton("Сохранить распознавание", dlg)
        cancel_btn = PushButton("Отмена", dlg)
        btns.addWidget(apply_btn)
        btns.addWidget(cancel_btn)
        btns.addStretch(1)
        lay.addLayout(btns)

        def _apply_and_close():
            out_rows = []
            for i, r in enumerate(rows):
                out_rows.append(
                    {
                        "start_ms": int(r["start_ms"]),
                        "end_ms": int(r["end_ms"]),
                        "text": str(table.item(i, 2).text() if table.item(i, 2) else r["text"]),
                    }
                )
            out_json = PROJECT_ROOT / "AppData" / "cache" / "video_translate_manual_transcription.json"
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
            self.manual_transcription_json_path = str(out_json)
            self.diag_label.setText(f"Этап 2: сохранено распознавание ({len(out_rows)} сегментов)")
            dlg.accept()

        apply_btn.clicked.connect(_apply_and_close)
        cancel_btn.clicked.connect(dlg.reject)
        dlg.exec_()

    def start_translate(self, *, prompt_rvc_map: bool = True):
        video_path = (self.video_input.text() or "").strip()
        if not video_path:
            InfoBar.warning(
                "Внимание",
                "Сначала выберите видеофайл",
                duration=2000,
                position=InfoBarPosition.TOP,
                parent=self,
            )
            return
        if not Path(video_path).exists():
            InfoBar.error(
                "Ошибка",
                "Файл не найден",
                duration=2500,
                position=InfoBarPosition.TOP,
                parent=self,
            )
            return

        self.task = TaskFactory.create_video_translate_task(video_path)
        # Берём ручной выбор устройства из UI
        selected_device = "cuda" if self.device_combo.currentIndex() == 0 else "cpu"
        self.task.transcribe_config.faster_whisper_device = selected_device
        selected_quality = {0: "fast", 1: "balanced", 2: "high", 3: "studio"}.get(
            self.quality_combo.currentIndex(), "balanced"
        )
        try:
            selected_workers = int(self.workers_combo.currentText())
        except Exception:
            selected_workers = 3
        selected_workers = max(1, min(8, selected_workers))
        selected_cache = self.cache_combo.currentIndex() == 0
        selected_asr_cache = self.asr_cache_combo.currentIndex() == 0
        sel_src_lang = self._source_lang_options[self.source_lang_combo.currentIndex()]
        sel_lang = self._target_lang_options[self.target_lang_combo.currentIndex()]
        sel_translator = self._translator_options[self.translator_combo.currentIndex()]
        sel_provider = {0: "auto", 1: "xtts", 2: "rvc"}.get(
            self.provider_combo.currentIndex(), "auto"
        )

        if sel_provider == "rvc" and prompt_rvc_map and not self._has_complete_speaker_voice_map():
            ok = self.open_speaker_voice_map_dialog()
            if not ok:
                InfoBar.warning(
                    "Запуск отменён",
                    "Для RVC нужно выбрать и сохранить голос для каждого спикера",
                    duration=3000,
                    parent=self,
                )
                return

        # Если этап 2 уже выполнялся — используем готовый ASR JSON и не делаем повторный ASR в этапе 5-6.
        if (not self.manual_transcription_json_path) and _LAST_ASR_CACHE.exists():
            self.manual_transcription_json_path = str(_LAST_ASR_CACHE)

        self.task.video_translate_config.voice_clone_quality = selected_quality
        self.task.video_translate_config.translation_enabled = self._translation_enabled()
        self.task.video_translate_config.voice_clone_provider = sel_provider
        self.task.video_translate_config.source_separation_mode = {
            0: "demucs_plus_uvr",
            1: "demucs",
            2: "uvr_mdx_kim",
            3: "auto",
        }.get(self.sep_combo.currentIndex(), "demucs_plus_uvr")
        self.task.video_translate_config.tts_parallel_workers = selected_workers
        self.task.video_translate_config.use_translation_cache = selected_cache
        self.task.video_translate_config.allow_speaker_overlap = self.overlap_combo.currentIndex() == 0
        self.task.video_translate_config.overlap_aware_mix = self.mix_combo.currentIndex() == 0
        self.task.video_translate_config.segment_qa_enabled = self.qa_combo.currentIndex() == 0
        self.task.video_translate_config.segment_qa_retry_count = int(self.qa_retry_combo.currentText() or 1)
        self.task.video_translate_config.enable_background_ducking = self.duck_combo.currentIndex() == 0
        self.task.video_translate_config.preserve_background_loudness = self.preserve_bg_combo.currentIndex() == 0
        self.task.video_translate_config.aggressive_vocal_suppression = self.vocal_kill_combo.currentIndex() == 0
        self.task.video_translate_config.manual_transcription_json = str(self.manual_transcription_json_path or "")
        self.task.video_translate_config.manual_translation_json = str(self.manual_translation_json_path or "")
        self.task.video_translate_config.source_language = sel_src_lang.value
        self.task.transcribe_config.use_asr_cache = selected_asr_cache
        from app.core.entities import LANGUAGES
        self.task.transcribe_config.transcribe_language = LANGUAGES.get(sel_src_lang.value, "auto")
        self.task.video_translate_config.target_language = sel_lang.value
        self.task.video_translate_config.translator_service = sel_translator
        self.task.subtitle_config.use_cache = selected_cache
        self.task.subtitle_config.target_language = sel_lang.value
        self.task.subtitle_config.translator_service = sel_translator

        # Важно: тяжёлая preflight-диагностика выполняется в worker-процессе.
        # Не блокируем UI при нажатии кнопки «Начать перевод видео».
        provider = str(self.task.video_translate_config.voice_clone_provider or "auto")
        self.diag_label.setText(
            f"Диагностика: запуск (provider={provider}, mode={selected_quality})"
        )

        self.translate_thread = VideoTranslateThread(self.task)
        self.translate_thread.progress.connect(self.on_progress)
        self.translate_thread.finished.connect(self.on_finished)
        self.translate_thread.error.connect(self.on_error)

        self._reset_all_stage_progress()
        self.progress_bar.setValue(0)
        self.status_label.setText("Запуск...")
        self._set_stage_progress(1, 1, "Запуск полного пайплайна")
        self.start_button.setEnabled(False)
        self.translate_thread.start()

    def on_progress(self, value: int, status: str):
        self.progress_bar.setValue(int(value))
        self.status_label.setText(status or "В обработке")
        self._route_full_pipeline_progress(int(value), status or "В обработке")

    def on_finished(self, task: VideoTranslateTask):
        self.start_button.setEnabled(True)
        self.progress_bar.setValue(100)
        self.status_label.setText("Готово")
        self._set_stage_progress(56, 100, "Этап 5-6 завершён")
        InfoBar.success(
            "Перевод завершён",
            f"Видео: {task.output_path}\nПеревод (srt): {task.output_subtitle_path}",
            duration=4000,
            position=InfoBarPosition.TOP,
            parent=self,
        )

    def _build_task_from_ui(self, video_path: str) -> VideoTranslateTask:
        task = TaskFactory.create_video_translate_task(video_path)
        selected_device = "cuda" if self.device_combo.currentIndex() == 0 else "cpu"
        task.transcribe_config.faster_whisper_device = selected_device
        selected_quality = {0: "fast", 1: "balanced", 2: "high", 3: "studio"}.get(
            self.quality_combo.currentIndex(), "balanced"
        )
        try:
            selected_workers = int(self.workers_combo.currentText())
        except Exception:
            selected_workers = 3
        selected_workers = max(1, min(8, selected_workers))
        selected_cache = self.cache_combo.currentIndex() == 0
        selected_asr_cache = self.asr_cache_combo.currentIndex() == 0
        sel_src_lang = self._source_lang_options[self.source_lang_combo.currentIndex()]
        sel_lang = self._target_lang_options[self.target_lang_combo.currentIndex()]
        sel_translator = self._translator_options[self.translator_combo.currentIndex()]
        sel_provider = {0: "auto", 1: "xtts", 2: "rvc"}.get(self.provider_combo.currentIndex(), "auto")
        task.video_translate_config.voice_clone_quality = selected_quality
        task.video_translate_config.translation_enabled = self._translation_enabled()
        task.video_translate_config.voice_clone_provider = sel_provider
        task.video_translate_config.source_separation_mode = {
            0: "demucs_plus_uvr",
            1: "demucs",
            2: "uvr_mdx_kim",
            3: "auto",
        }.get(self.sep_combo.currentIndex(), "demucs_plus_uvr")
        task.video_translate_config.tts_parallel_workers = selected_workers
        task.video_translate_config.use_translation_cache = selected_cache
        task.video_translate_config.allow_speaker_overlap = self.overlap_combo.currentIndex() == 0
        task.video_translate_config.overlap_aware_mix = self.mix_combo.currentIndex() == 0
        task.video_translate_config.segment_qa_enabled = self.qa_combo.currentIndex() == 0
        task.video_translate_config.segment_qa_retry_count = int(self.qa_retry_combo.currentText() or 1)
        task.video_translate_config.enable_background_ducking = self.duck_combo.currentIndex() == 0
        task.video_translate_config.preserve_background_loudness = self.preserve_bg_combo.currentIndex() == 0
        task.video_translate_config.aggressive_vocal_suppression = self.vocal_kill_combo.currentIndex() == 0
        manual_tx_path = str(self.manual_transcription_json_path or "").strip()
        if not manual_tx_path and _LAST_ASR_CACHE.exists():
            manual_tx_path = str(_LAST_ASR_CACHE)
        task.video_translate_config.manual_transcription_json = manual_tx_path
        task.video_translate_config.manual_translation_json = str(self.manual_translation_json_path or "")
        task.video_translate_config.source_language = sel_src_lang.value
        task.transcribe_config.use_asr_cache = selected_asr_cache
        from app.core.entities import LANGUAGES
        task.transcribe_config.transcribe_language = LANGUAGES.get(sel_src_lang.value, "auto")
        task.video_translate_config.target_language = sel_lang.value
        task.video_translate_config.translator_service = sel_translator
        task.subtitle_config.use_cache = selected_cache
        task.subtitle_config.target_language = sel_lang.value
        task.subtitle_config.translator_service = sel_translator
        return task

    def preview_translation(self):
        if not self._translation_enabled():
            InfoBar.warning(
                "Этап 3 отключён",
                "В режиме без перевода этап 3 не используется",
                duration=2200,
                parent=self,
            )
            return
        video_path = (self.video_input.text() or "").strip()
        if not video_path or not Path(video_path).exists():
            InfoBar.warning("Внимание", "Сначала выберите видео", duration=2000, parent=self)
            return
        if self.preview_thread and self.preview_thread.isRunning():
            return
        self._set_stage_progress(3, 5, "Этап 3: подготовка предпросмотра")
        self.diag_label.setText(
            "Проверка перевода: ASR + перевод..." if self._translation_enabled() else "Проверка режима без перевода: ASR + исходный текст"
        )
        task = self._build_task_from_ui(video_path)
        self.preview_thread = _PreviewTranslationThread(video_path=video_path, task=task)
        self.preview_thread.finished_rows.connect(self._on_preview_finished)
        self.preview_thread.failed.connect(self._on_preview_failed)
        self.preview_thread.start()

    def _on_preview_finished(self, rows: list):
        self.preview_thread = None
        self._set_stage_progress(3, 100, "Этап 3: предпросмотр готов")
        dlg = QDialog(self)
        dlg.setWindowTitle("Проверка перевода перед озвучкой")
        dlg.resize(1100, 700)
        lay = QVBoxLayout(dlg)
        table = QTableWidget(len(rows), 4, dlg)
        table.setHorizontalHeaderLabels(["Start", "End", "Оригинал", "Перевод"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        for i, r in enumerate(rows):
            table.setItem(i, 0, QTableWidgetItem(str(r["start_ms"])))
            table.setItem(i, 1, QTableWidgetItem(str(r["end_ms"])))
            table.item(i, 0).setFlags(table.item(i, 0).flags() & ~Qt.ItemIsEditable)
            table.item(i, 1).setFlags(table.item(i, 1).flags() & ~Qt.ItemIsEditable)
            table.setItem(i, 2, QTableWidgetItem(r["original"]))
            table.item(i, 2).setFlags(table.item(i, 2).flags() & ~Qt.ItemIsEditable)
            table.setItem(i, 3, QTableWidgetItem(r["translated"]))
        lay.addWidget(table)
        btns = QHBoxLayout()
        apply_btn = PrimaryPushButton("Применить перевод", dlg)
        cancel_btn = PushButton("Отмена", dlg)
        btns.addWidget(apply_btn)
        btns.addWidget(cancel_btn)
        btns.addStretch(1)
        lay.addLayout(btns)

        def _apply_and_close():
            out_rows = []
            for i, r in enumerate(rows):
                out_rows.append(
                    {
                        **r,
                        "translated": str(table.item(i, 3).text() if table.item(i, 3) else r["translated"]),
                    }
                )
            out_json = PROJECT_ROOT / "AppData" / "cache" / "video_translate_manual_translation.json"
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
            self.manual_translation_json_path = str(out_json)
            self.diag_label.setText(
                f"Проверка перевода: применено {len(out_rows)} сегментов. Файл: {out_json}"
            )
            dlg.accept()

        apply_btn.clicked.connect(_apply_and_close)
        cancel_btn.clicked.connect(dlg.reject)
        dlg.exec_()

    def _on_preview_failed(self, err: str):
        self.preview_thread = None
        self._set_stage_progress(3, 100, "Ошибка этапа 3")
        self.diag_label.setText(f"Проверка перевода: ошибка ({err})")
        self._show_copyable_error(
            title="Ошибка проверки перевода",
            message=str(err),
            log_file_name="video_translate_preview_last_error.txt",
            parent_widget=self,
            duration=12000,
        )

    def open_translation_file(self):
        target = None
        if self.task and self.task.output_subtitle_path and Path(self.task.output_subtitle_path).exists():
            target = Path(self.task.output_subtitle_path)
        elif self.manual_translation_json_path and Path(self.manual_translation_json_path).exists():
            target = Path(self.manual_translation_json_path)
        if not target:
            InfoBar.warning("Нет файла", "Сначала выполните проверку/перевод", duration=2000, parent=self)
            return
        if sys.platform == "win32":
            os.startfile(str(target))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(target)])
        else:
            subprocess.run(["xdg-open", str(target)])

    def open_last_asr_cache_file(self):
        target = None
        if _LAST_ASR_CACHE.exists():
            target = _LAST_ASR_CACHE
        elif self.manual_transcription_json_path and Path(self.manual_transcription_json_path).exists():
            target = Path(self.manual_transcription_json_path)
        if not target:
            InfoBar.warning("Нет файла", "Сначала выполните этап 2 (распознавание)", duration=2200, parent=self)
            return
        if sys.platform == "win32":
            os.startfile(str(target))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(target)])
        else:
            subprocess.run(["xdg-open", str(target)])

    def edit_last_translation_and_prepare_redub(self):
        """Открыть последний кэш перевода, отредактировать и подготовить перезапуск с этапа озвучки."""
        if not _LAST_TRANSLATION_CACHE.exists():
            InfoBar.warning(
                "Нет последнего перевода",
                "Сначала запустите перевод хотя бы один раз",
                duration=2500,
                parent=self,
            )
            return
        try:
            raw = json.loads(_LAST_TRANSLATION_CACHE.read_text(encoding="utf-8", errors="ignore") or "{}")
        except Exception:
            raw = {}
        rows = raw.get("rows", []) if isinstance(raw, dict) else []
        if not rows:
            InfoBar.warning(
                "Пустой кэш перевода",
                "Не удалось прочитать сегменты последнего перевода",
                duration=2500,
                parent=self,
            )
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Редактирование последнего перевода")
        dlg.resize(1160, 720)
        lay = QVBoxLayout(dlg)

        table = QTableWidget(len(rows), 4, dlg)
        table.setHorizontalHeaderLabels(["Start", "End", "Оригинал", "Перевод"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)

        for i, r in enumerate(rows):
            table.setItem(i, 0, QTableWidgetItem(str(int((r or {}).get("start_ms", 0) or 0))))
            table.setItem(i, 1, QTableWidgetItem(str(int((r or {}).get("end_ms", 0) or 0))))
            table.item(i, 0).setFlags(table.item(i, 0).flags() & ~Qt.ItemIsEditable)
            table.item(i, 1).setFlags(table.item(i, 1).flags() & ~Qt.ItemIsEditable)
            table.setItem(i, 2, QTableWidgetItem(str((r or {}).get("original", ""))))
            table.item(i, 2).setFlags(table.item(i, 2).flags() & ~Qt.ItemIsEditable)
            table.setItem(i, 3, QTableWidgetItem(str((r or {}).get("translated", ""))))

        lay.addWidget(table)

        btns = QHBoxLayout()
        apply_btn = PrimaryPushButton("Сохранить и использовать в следующем запуске", dlg)
        close_btn = PushButton("Закрыть", dlg)
        btns.addWidget(apply_btn)
        btns.addWidget(close_btn)
        btns.addStretch(1)
        lay.addLayout(btns)

        def _save_and_close():
            out_rows = []
            for i, r in enumerate(rows):
                out_rows.append(
                    {
                        "start_ms": int((r or {}).get("start_ms", 0) or 0),
                        "end_ms": int((r or {}).get("end_ms", 0) or 0),
                        "original": str((r or {}).get("original", "")),
                        "translated": str(table.item(i, 3).text() if table.item(i, 3) else (r or {}).get("translated", "")),
                    }
                )

            out_json = PROJECT_ROOT / "AppData" / "cache" / "video_translate_manual_translation.json"
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
            self.manual_translation_json_path = str(out_json)
            self.diag_label.setText(
                f"Перевод отредактирован: {len(out_rows)} сегментов. Следующий запуск возьмёт этот текст (без нового шага перевода)."
            )
            InfoBar.success(
                "Сохранено",
                "Исправленный перевод применится в следующем запуске",
                duration=2800,
                parent=self,
            )
            dlg.accept()

        apply_btn.clicked.connect(_save_and_close)
        close_btn.clicked.connect(dlg.reject)
        dlg.exec_()

    def on_error(self, message: str):
        self.start_button.setEnabled(True)
        self.status_label.setText("Ошибка")
        self._set_stage_progress(56, 100, "Ошибка на этапе 5-6")
        self.diag_label.setText(f"Ошибка: {message}")

        # Продлеваем показ ошибки, чтобы пользователь успел прочитать детали strict/XTTS фейла.
        is_xtts_fail = "xtts" in str(message).lower() or "voice clone" in str(message).lower()
        err_duration = 15000 if is_xtts_fail else 9000
        self._show_copyable_error(
            title="Ошибка перевода видео",
            message=str(message),
            log_file_name="video_translate_last_error.txt",
            parent_widget=self,
            duration=err_duration,
        )

    def _load_recent_speaker_ids(self) -> list[str]:
        """Берёт список спикеров из последнего анализа, затем из map, затем fallback."""
        ids: list[str] = []
        try:
            if _SPEAKER_ANALYSIS_CACHE.exists():
                raw = json.loads(_SPEAKER_ANALYSIS_CACHE.read_text(encoding="utf-8", errors="ignore") or "{}")
                rows = raw.get("speaker_ids", []) if isinstance(raw, dict) else []
                ids = [str(x).strip() for x in rows if str(x).strip()]
        except Exception:
            ids = []

        if not ids:
            existing_raw = str(getattr(cfg.video_translate_manual_voice_map_json, "value", "") or "").strip()
            if existing_raw:
                try:
                    data = json.loads(existing_raw)
                    if isinstance(data, dict):
                        ids = [
                            str(k).strip()
                            for k in data.keys()
                            if str(k).strip() and str(k).strip().lower() != "default"
                        ]
                except Exception:
                    ids = []

        if not ids:
            try:
                expected = int(getattr(cfg.video_translate_expected_speaker_count, "value", 0) or 0)
            except Exception:
                expected = 0
            speaker_count = max(2, expected if expected > 0 else 2)
            speaker_count = min(12, speaker_count)
            ids = [f"spk_{i}" for i in range(speaker_count)]

        # Стабильный порядок: spk_0, spk_1, ... затем прочие
        def _sort_key(v: str):
            t = str(v)
            if t.startswith("spk_"):
                try:
                    return (0, int(t.split("_", 1)[1]))
                except Exception:
                    return (1, t)
            return (2, t)

        ids = sorted(list(dict.fromkeys(ids)), key=_sort_key)
        return ids[:24]

    def open_output_folder(self):
        if not self.task or not self.task.output_path:
            return
        target_dir = str(Path(self.task.output_path).parent)
        if sys.platform == "win32":
            os.startfile(target_dir)
        elif sys.platform == "darwin":
            subprocess.run(["open", target_dir])
        else:
            subprocess.run(["xdg-open", target_dir])

    def open_stage_cache_folder(self):
        _STAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        target_dir = str(_STAGE_CACHE_DIR)
        if sys.platform == "win32":
            os.startfile(target_dir)
        elif sys.platform == "darwin":
            subprocess.run(["open", target_dir])
        else:
            subprocess.run(["xdg-open", target_dir])
