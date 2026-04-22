# -*- coding: utf-8 -*-

import os
import subprocess
import sys
import json
import tempfile
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
            asr_data = VideoTranslationProcessor()._translate_asr(asr_data, self.task)

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


class VideoTranslateInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("VideoTranslateInterface")
        self.task: VideoTranslateTask | None = None
        self.translate_thread: VideoTranslateThread | None = None
        self.preview_thread: _PreviewTranslationThread | None = None
        self.diag_thread: _DiagnosticsThread | None = None
        self.manual_translation_json_path: str = ""
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

        self.start_button = PrimaryPushButton("Начать перевод видео", self, icon=FIF.PLAY)
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

        # Компактная сетка настроек (в несколько колонок)
        settings_grid = QVBoxLayout()
        settings_grid.setSpacing(8)

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
        settings_grid.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(10)
        self.overlap_label = BodyLabel("Разрешить overlap реплик", self)
        self.overlap_label.setMinimumWidth(form_label_w)
        self.overlap_combo = ComboBox(self)
        self.overlap_combo.setMinimumWidth(form_field_min_w)
        self.overlap_combo.addItems(["Да", "Нет"])
        self.overlap_combo.setCurrentIndex(
            0 if bool(getattr(cfg.video_translate_allow_speaker_overlap, "value", True)) else 1
        )
        row2.addWidget(self.overlap_label)
        row2.addWidget(self.overlap_combo, 1)

        self.mix_label = BodyLabel("Overlap-aware микс по спикерам", self)
        self.mix_label.setMinimumWidth(form_label_w)
        self.mix_combo = ComboBox(self)
        self.mix_combo.setMinimumWidth(form_field_min_w)
        self.mix_combo.addItems(["Да", "Нет"])
        self.mix_combo.setCurrentIndex(
            0 if bool(getattr(cfg.video_translate_overlap_aware_mix, "value", True)) else 1
        )
        row2.addWidget(self.mix_label)
        row2.addWidget(self.mix_combo, 1)
        settings_grid.addLayout(row2)

        row3 = QHBoxLayout()
        row3.setSpacing(10)
        self.qa_label = BodyLabel("QA проверка TTS-сегментов", self)
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
        settings_grid.addLayout(row3)

        row4 = QHBoxLayout()
        row4.setSpacing(10)
        self.duck_label = BodyLabel("Ducking фона под речь", self)
        self.duck_label.setMinimumWidth(form_label_w)
        self.duck_combo = ComboBox(self)
        self.duck_combo.setMinimumWidth(form_field_min_w)
        self.duck_combo.addItems(["Да", "Нет"])
        self.duck_combo.setCurrentIndex(
            0 if bool(getattr(cfg.video_translate_enable_background_ducking, "value", True)) else 1
        )
        row4.addWidget(self.duck_label)
        row4.addWidget(self.duck_combo, 1)

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
        settings_grid.addLayout(row4)

        row4b = QHBoxLayout()
        row4b.setSpacing(10)
        self.sep_label = BodyLabel("Разделение источника", self)
        self.sep_label.setMinimumWidth(form_label_w)
        self.sep_combo = ComboBox(self)
        self.sep_combo.setMinimumWidth(form_field_min_w)
        self.sep_combo.addItems([
            "Demucs + UVR (рекомендуется)",
            "Demucs",
            "UVR MDX/Kim",
            "Auto",
        ])
        sep_mode = str(getattr(cfg.video_translate_source_separation_mode, "value", "demucs_plus_uvr") or "demucs_plus_uvr").strip().lower()
        sep_to_idx = {"demucs_plus_uvr": 0, "demucs": 1, "uvr_mdx_kim": 2, "auto": 3}
        self.sep_combo.setCurrentIndex(sep_to_idx.get(sep_mode, 0))
        row4b.addWidget(self.sep_label)
        row4b.addWidget(self.sep_combo, 1)
        self.vocal_kill_label = BodyLabel("Агрессивно убрать оригинальный голос", self)
        self.vocal_kill_label.setMinimumWidth(form_label_w)
        self.vocal_kill_combo = ComboBox(self)
        self.vocal_kill_combo.setMinimumWidth(form_field_min_w)
        self.vocal_kill_combo.addItems(["Да", "Нет"])
        self.vocal_kill_combo.setCurrentIndex(
            0 if bool(getattr(cfg.video_translate_aggressive_vocal_suppression, "value", False)) else 1
        )
        row4b.addWidget(self.vocal_kill_label)
        row4b.addWidget(self.vocal_kill_combo, 1)
        settings_grid.addLayout(row4b)

        row5 = QHBoxLayout()
        row5.setSpacing(10)
        self.target_lang_label = BodyLabel("Язык перевода", self)
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
        settings_grid.addLayout(row5)

        row6 = QHBoxLayout()
        row6.setSpacing(10)
        row6.addWidget(self.translator_label)
        row6.addWidget(self.translator_combo, 1)

        self.workers_label = BodyLabel("Параллельных TTS задач", self)
        self.workers_label.setMinimumWidth(form_label_w)
        self.workers_combo = ComboBox(self)
        self.workers_combo.setMinimumWidth(form_field_min_w)
        self.workers_combo.addItems([str(i) for i in range(1, 9)])
        current_workers = int(getattr(cfg.video_translate_tts_parallel_workers, "value", 3) or 3)
        current_workers = max(1, min(8, current_workers))
        self.workers_combo.setCurrentText(str(current_workers))
        row6.addWidget(self.workers_label)
        row6.addWidget(self.workers_combo, 1)

        settings_grid.addLayout(row6)

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
        settings_grid.addLayout(row7)

        config_layout.addLayout(settings_grid)

        self.diag_label = BodyLabel("Диагностика: не выполнялась", self)
        self.diag_label.setWordWrap(True)
        config_layout.addWidget(self.diag_label)

        diag_btn_row = QHBoxLayout()
        self.diag_button = PushButton("Проверить окружение/GPU", self)
        diag_btn_row.addWidget(self.diag_button)
        diag_btn_row.addStretch(1)
        config_layout.addLayout(diag_btn_row)

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
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        self.quality_combo.currentIndexChanged.connect(self._on_quality_changed)
        self.workers_combo.currentIndexChanged.connect(self._on_workers_changed)
        self.cache_combo.currentIndexChanged.connect(self._on_cache_changed)
        self.asr_cache_combo.currentIndexChanged.connect(self._on_asr_cache_changed)
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
        self.vocal_kill_combo.currentIndexChanged.connect(self._on_aggressive_vocal_suppression_changed)
        self.diag_button.clicked.connect(self.run_gpu_diagnostics)

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

    def _on_aggressive_vocal_suppression_changed(self):
        cfg.set(
            cfg.video_translate_aggressive_vocal_suppression,
            self.vocal_kill_combo.currentIndex() == 0,
        )

    def _model_display_items(self):
        self._refresh_rvc_models()
        items = []
        for m in self._rvc_models:
            label = f"{m.slot}: {m.name}"
            items.append((label, m))
        return items

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
        dlg.setWindowTitle("Менеджер RVC моделей (params.json)")
        dlg.resize(980, 620)
        lay = QVBoxLayout(dlg)

        table = QTableWidget(len(items), 10, dlg)
        table.setHorizontalHeaderLabels([
            "Slot",
            "Name",
            "Model",
            "Index",
            "Preview",
            "Icon",
            "Tune",
            "IndexRatio",
            "Protect",
            "Params",
        ])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(8, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(9, QHeaderView.Stretch)

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
                    preview_item.setIcon(QIcon(pix.scaled(56, 56, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
            table.setItem(i, 4, preview_item)
            table.setItem(i, 5, QTableWidgetItem(icon_path_text))
            table.setItem(i, 6, QTableWidgetItem(str(m.default_tune)))
            table.setItem(i, 7, QTableWidgetItem(f"{m.default_index_ratio:.3f}"))
            table.setItem(i, 8, QTableWidgetItem(f"{m.default_protect:.3f}"))
            table.setItem(i, 9, QTableWidgetItem(str(m.params_path)))
            # read-only columns
            for col in (0, 2, 3, 4, 5, 9):
                item = table.item(i, col)
                if item:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            table.setRowHeight(i, 64)

        lay.addWidget(table)
        btn_row = QHBoxLayout()
        save_btn = PrimaryPushButton("Сохранить в params.json", dlg)
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
                    tune = int(float(table.item(i, 6).text() if table.item(i, 6) else m.default_tune))
                    idx_ratio = float(table.item(i, 7).text() if table.item(i, 7) else m.default_index_ratio)
                    protect = float(table.item(i, 8).text() if table.item(i, 8) else m.default_protect)
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
                f"Сохранено params.json: {ok_count}",
                duration=3000,
                parent=self,
            )

        save_btn.clicked.connect(_save_models)
        close_btn.clicked.connect(dlg.reject)
        dlg.exec_()

    def open_speaker_voice_map_dialog(self):
        items = self._model_display_items()
        if not items:
            InfoBar.warning(
                "RVC модели не найдены",
                f"Проверьте папку: {default_rvc_model_root()}",
                duration=3500,
                parent=self,
            )
            return

        try:
            expected = int(getattr(cfg.video_translate_expected_speaker_count, "value", 0) or 0)
        except Exception:
            expected = 0
        speaker_count = max(2, expected if expected > 0 else 2)
        speaker_count = min(12, speaker_count)

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
            "Выберите RVC-голос для каждого спикера. Сохраняется в ManualVoiceMapJson.",
            dlg,
        )
        hint.setWordWrap(True)
        lay.addWidget(hint)

        table = QTableWidget(speaker_count, 3, dlg)
        table.setHorizontalHeaderLabels(["Спикер", "RVC модель", "Tune (из params)"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)

        combos = []
        for i in range(speaker_count):
            spk = f"spk_{i}"
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

        def _save_map():
            out = {}
            default_slot = ""
            for i, combo in enumerate(combos):
                if combo.currentIndex() <= 0:
                    continue
                model = items[combo.currentIndex() - 1][1]
                spk = f"spk_{i}"
                out[spk] = str(model.slot)
                if not default_slot:
                    default_slot = str(model.slot)
            if default_slot:
                out["default"] = default_slot
            cfg.set(cfg.video_translate_manual_voice_map_json, json.dumps(out, ensure_ascii=False))
            InfoBar.success(
                "Сохранено",
                "Назначение голосов спикерам сохранено",
                duration=2500,
                parent=self,
            )

        save_btn.clicked.connect(_save_map)
        close_btn.clicked.connect(dlg.reject)
        dlg.exec_()

    def _apply_quality_preset(self, quality: str):
        q = (quality or "balanced").strip().lower()
        if q == "fast":
            overlap, mix, qa, qa_retry, duck, vocal_kill = False, False, False, 0, False, False
        elif q == "balanced":
            overlap, mix, qa, qa_retry, duck, vocal_kill = False, True, False, 0, True, False
        elif q == "studio":
            overlap, mix, qa, qa_retry, duck, vocal_kill = True, True, True, 2, True, True
        else:  # high
            overlap, mix, qa, qa_retry, duck, vocal_kill = True, True, True, 1, True, True

        self.overlap_combo.setCurrentIndex(0 if overlap else 1)
        self.mix_combo.setCurrentIndex(0 if mix else 1)
        self.qa_combo.setCurrentIndex(0 if qa else 1)
        self.qa_retry_combo.setCurrentText(str(qa_retry))
        self.duck_combo.setCurrentIndex(0 if duck else 1)
        self.vocal_kill_combo.setCurrentIndex(0 if vocal_kill else 1)

        cfg.set(cfg.video_translate_allow_speaker_overlap, overlap)
        cfg.set(cfg.video_translate_overlap_aware_mix, mix)
        cfg.set(cfg.video_translate_segment_qa_enabled, qa)
        cfg.set(cfg.video_translate_segment_qa_retry_count, qa_retry)
        cfg.set(cfg.video_translate_enable_background_ducking, duck)
        cfg.set(cfg.video_translate_aggressive_vocal_suppression, vocal_kill)

    def choose_video_file(self):
        video_formats = " ".join(f"*.{fmt.value}" for fmt in SupportedVideoFormats)
        filter_str = f"Видеофайлы ({video_formats})"
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите видео", "", filter_str)
        if file_path:
            self.video_input.setText(file_path)

    def start_translate(self):
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
        self.task.video_translate_config.voice_clone_quality = selected_quality
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
        self.task.video_translate_config.aggressive_vocal_suppression = self.vocal_kill_combo.currentIndex() == 0
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

        self.progress_bar.setValue(0)
        self.status_label.setText("Запуск...")
        self.start_button.setEnabled(False)
        self.translate_thread.start()

    def on_progress(self, value: int, status: str):
        self.progress_bar.setValue(int(value))
        self.status_label.setText(status or "В обработке")

    def on_finished(self, task: VideoTranslateTask):
        self.start_button.setEnabled(True)
        self.progress_bar.setValue(100)
        self.status_label.setText("Готово")
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
        task.video_translate_config.aggressive_vocal_suppression = self.vocal_kill_combo.currentIndex() == 0
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
        video_path = (self.video_input.text() or "").strip()
        if not video_path or not Path(video_path).exists():
            InfoBar.warning("Внимание", "Сначала выберите видео", duration=2000, parent=self)
            return
        if self.preview_thread and self.preview_thread.isRunning():
            return
        self.diag_label.setText("Проверка перевода: ASR + перевод...")
        task = self._build_task_from_ui(video_path)
        self.preview_thread = _PreviewTranslationThread(video_path=video_path, task=task)
        self.preview_thread.finished_rows.connect(self._on_preview_finished)
        self.preview_thread.failed.connect(self._on_preview_failed)
        self.preview_thread.start()

    def _on_preview_finished(self, rows: list):
        self.preview_thread = None
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
        self.diag_label.setText(f"Проверка перевода: ошибка ({err})")

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

    def on_error(self, message: str):
        self.start_button.setEnabled(True)
        self.status_label.setText("Ошибка")
        self.diag_label.setText(f"Ошибка: {message}")

        # Чтобы ошибку можно было сразу вставить в чат/issue.
        try:
            QApplication.clipboard().setText(str(message or ""))
        except Exception:
            pass

        # Всегда сохраняем последнюю ошибку в файл, чтобы можно было открыть/скопировать.
        try:
            err_file = PROJECT_ROOT / "AppData" / "logs" / "video_translate_last_error.txt"
            err_file.parent.mkdir(parents=True, exist_ok=True)
            err_file.write_text(str(message or ""), encoding="utf-8")
        except Exception:
            pass

        # Продлеваем показ ошибки, чтобы пользователь успел прочитать детали strict/XTTS фейла.
        is_xtts_fail = "xtts" in str(message).lower() or "voice clone" in str(message).lower()
        err_duration = 15000 if is_xtts_fail else 8000
        InfoBar.error(
            "Ошибка перевода видео",
            f"{message}\n(подробно: AppData/logs/video_translate_last_error.txt, текст скопирован в буфер)",
            duration=err_duration,
            position=InfoBarPosition.TOP,
            parent=self,
        )

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
