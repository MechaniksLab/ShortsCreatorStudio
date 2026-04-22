# -*- coding: utf-8 -*-

import os
import subprocess
import sys
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QHBoxLayout, QVBoxLayout, QWidget
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
from app.core.video_translate.bootstrap import VideoTranslateBootstrap
from app.thread.video_translate_thread import VideoTranslateThread
from app.common.config import cfg
from app.config import PROJECT_ROOT


class VideoTranslateInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("VideoTranslateInterface")
        self.task: VideoTranslateTask | None = None
        self.translate_thread: VideoTranslateThread | None = None
        self._build_ui()
        self._connect_signals()

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

        self.config_card = CardWidget(self)
        config_layout = QVBoxLayout(self.config_card)
        config_layout.setContentsMargins(20, 20, 20, 20)
        config_layout.setSpacing(18)

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

        # Простой ручной выбор CPU/GPU для ASR/части пайплайна
        device_row = QHBoxLayout()
        device_row.setSpacing(12)
        self.device_label = BodyLabel("Устройство", self)
        self.device_combo = ComboBox(self)
        self.device_combo.addItems(["GPU (CUDA)", "CPU"])
        current_dev = str(cfg.faster_whisper_device.value or "cuda").strip().lower()
        self.device_combo.setCurrentIndex(0 if current_dev == "cuda" else 1)
        device_row.addWidget(self.device_label)
        device_row.addWidget(self.device_combo)
        device_row.addStretch(1)
        config_layout.addLayout(device_row)

        quality_row = QHBoxLayout()
        quality_row.setSpacing(12)
        self.quality_label = BodyLabel("Режим озвучки", self)
        self.quality_combo = ComboBox(self)
        self.quality_combo.addItems(["Быстрый", "Сбалансированный", "Качество", "Студийный"])
        quality_to_idx = {"fast": 0, "balanced": 1, "high": 2, "studio": 3}
        current_quality = str(cfg.video_translate_voice_quality.value or "balanced").strip().lower()
        self.quality_combo.setCurrentIndex(quality_to_idx.get(current_quality, 1))
        quality_row.addWidget(self.quality_label)
        quality_row.addWidget(self.quality_combo)
        quality_row.addStretch(1)
        config_layout.addLayout(quality_row)

        provider_row = QHBoxLayout()
        provider_row.setSpacing(12)
        self.provider_label = BodyLabel("Клонирование голоса", self)
        self.provider_combo = ComboBox(self)
        self.provider_combo.addItems([
            "auto (рекомендуется)",
            "xtts (локально)",
        ])
        current_provider = str(cfg.video_translate_voice_provider.value or "auto").strip().lower()
        provider_to_idx = {"auto": 0, "xtts": 1}
        self.provider_combo.setCurrentIndex(provider_to_idx.get(current_provider, 0))
        provider_row.addWidget(self.provider_label)
        provider_row.addWidget(self.provider_combo)
        provider_row.addStretch(1)
        config_layout.addLayout(provider_row)

        lang_row = QHBoxLayout()
        lang_row.setSpacing(12)
        self.target_lang_label = BodyLabel("Язык перевода", self)
        self.target_lang_combo = ComboBox(self)
        self._target_lang_options = list(cfg.video_translate_target_language.validator.options)
        for opt in self._target_lang_options:
            self.target_lang_combo.addItem(language_value_to_ru(opt.value))
        cur_target = str(cfg.video_translate_target_language.value.value)
        for i, opt in enumerate(self._target_lang_options):
            if str(opt.value) == cur_target:
                self.target_lang_combo.setCurrentIndex(i)
                break
        lang_row.addWidget(self.target_lang_label)
        lang_row.addWidget(self.target_lang_combo)
        lang_row.addStretch(1)
        config_layout.addLayout(lang_row)

        translator_row = QHBoxLayout()
        translator_row.setSpacing(12)
        self.translator_label = BodyLabel("Сервис перевода", self)
        self.translator_combo = ComboBox(self)
        self._translator_options = list(cfg.translator_service.validator.options)
        for opt in self._translator_options:
            self.translator_combo.addItem(translator_service_to_ru(opt.value))
        cur_translator = str(cfg.translator_service.value.value)
        for i, opt in enumerate(self._translator_options):
            if str(opt.value) == cur_translator:
                self.translator_combo.setCurrentIndex(i)
                break
        translator_row.addWidget(self.translator_label)
        translator_row.addWidget(self.translator_combo)
        translator_row.addStretch(1)
        config_layout.addLayout(translator_row)

        workers_row = QHBoxLayout()
        workers_row.setSpacing(12)
        self.workers_label = BodyLabel("Параллельных TTS задач", self)
        self.workers_combo = ComboBox(self)
        self.workers_combo.addItems([str(i) for i in range(1, 9)])
        current_workers = int(getattr(cfg.video_translate_tts_parallel_workers, "value", 3) or 3)
        current_workers = max(1, min(8, current_workers))
        self.workers_combo.setCurrentText(str(current_workers))
        workers_row.addWidget(self.workers_label)
        workers_row.addWidget(self.workers_combo)
        workers_row.addStretch(1)
        config_layout.addLayout(workers_row)

        cache_row = QHBoxLayout()
        cache_row.setSpacing(12)
        self.cache_label = BodyLabel("Кэшировать перевод", self)
        self.cache_combo = ComboBox(self)
        self.cache_combo.addItems(["Да", "Нет"])
        use_cache = bool(getattr(cfg.video_translate_use_translation_cache, "value", True))
        self.cache_combo.setCurrentIndex(0 if use_cache else 1)
        cache_row.addWidget(self.cache_label)
        cache_row.addWidget(self.cache_combo)
        cache_row.addStretch(1)
        config_layout.addLayout(cache_row)

        asr_cache_row = QHBoxLayout()
        asr_cache_row.setSpacing(12)
        self.asr_cache_label = BodyLabel("Кэшировать распознавание (ASR)", self)
        self.asr_cache_combo = ComboBox(self)
        self.asr_cache_combo.addItems(["Да", "Нет"])
        use_asr_cache = bool(getattr(cfg.video_translate_use_asr_cache, "value", True))
        self.asr_cache_combo.setCurrentIndex(0 if use_asr_cache else 1)
        asr_cache_row.addWidget(self.asr_cache_label)
        asr_cache_row.addWidget(self.asr_cache_combo)
        asr_cache_row.addStretch(1)
        config_layout.addLayout(asr_cache_row)

        self.diag_label = BodyLabel("Диагностика: не выполнялась", self)
        self.diag_label.setWordWrap(True)
        config_layout.addWidget(self.diag_label)

        self.main_layout.addWidget(self.config_card)
        self.main_layout.addStretch(1)

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
        self.target_lang_combo.currentIndexChanged.connect(self._on_target_language_changed)
        self.translator_combo.currentIndexChanged.connect(self._on_translator_changed)
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)

    def _on_device_changed(self):
        selected = "cuda" if self.device_combo.currentIndex() == 0 else "cpu"
        cfg.set(cfg.faster_whisper_device, selected)

    def _on_quality_changed(self):
        idx = self.quality_combo.currentIndex()
        quality = {0: "fast", 1: "balanced", 2: "high", 3: "studio"}.get(idx, "balanced")
        cfg.set(cfg.video_translate_voice_quality, quality)

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

    def _on_target_language_changed(self):
        idx = self.target_lang_combo.currentIndex()
        if 0 <= idx < len(self._target_lang_options):
            cfg.set(cfg.video_translate_target_language, self._target_lang_options[idx])

    def _on_translator_changed(self):
        idx = self.translator_combo.currentIndex()
        if 0 <= idx < len(self._translator_options):
            cfg.set(cfg.translator_service, self._translator_options[idx])

    def _on_provider_changed(self):
        provider = {0: "auto", 1: "xtts"}.get(self.provider_combo.currentIndex(), "auto")
        cfg.set(cfg.video_translate_voice_provider, provider)

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
        sel_lang = self._target_lang_options[self.target_lang_combo.currentIndex()]
        sel_translator = self._translator_options[self.translator_combo.currentIndex()]
        sel_provider = {0: "auto", 1: "xtts"}.get(
            self.provider_combo.currentIndex(), "auto"
        )
        self.task.video_translate_config.voice_clone_quality = selected_quality
        self.task.video_translate_config.voice_clone_provider = sel_provider
        self.task.video_translate_config.tts_parallel_workers = selected_workers
        self.task.video_translate_config.use_translation_cache = selected_cache
        self.task.transcribe_config.use_asr_cache = selected_asr_cache
        self.task.video_translate_config.target_language = sel_lang.value
        self.task.video_translate_config.translator_service = sel_translator
        self.task.subtitle_config.use_cache = selected_cache
        self.task.subtitle_config.target_language = sel_lang.value
        self.task.subtitle_config.translator_service = sel_translator

        # Preflight-диагностика перед запуском
        try:
            model_name = str(getattr(self.task.transcribe_config, "faster_whisper_model", "tiny") or "tiny")
            auto_download = bool(getattr(self.task.video_translate_config, "auto_download_models", True))
            report = VideoTranslateBootstrap().readiness_report(model_name=model_name, auto_download=auto_download)
            failed = report.get("critical_failed", [])
            if failed:
                details = "\n".join(f"• {c.get('name')}: {c.get('hint')}" for c in failed)
                self.diag_label.setText(f"Диагностика: ошибки\n{details}")
                InfoBar.error(
                    "Preflight не пройден",
                    details,
                    duration=7000,
                    position=InfoBarPosition.TOP,
                    parent=self,
                )
                return

            clone_quality = report.get("clone_quality")
            diar_quality = report.get("diarization_quality")
            asr_runtime = report.get("asr_runtime", "unknown")
            torch_cuda = bool(report.get("torch_cuda", False))
            provider = str(self.task.video_translate_config.voice_clone_provider or "auto")
            self.diag_label.setText(
                f"Диагностика: asr={asr_runtime}, gpu={torch_cuda}, diarization={diar_quality}, "
                f"clone={clone_quality}, provider={provider}, mode={selected_quality}"
            )
            if clone_quality != "high":
                InfoBar.warning(
                    "Режим клонирования: fallback (это не ошибка)",
                    "XTTS сейчас недоступен, поэтому включена локальная fallback-озвучка. Перевод продолжится; для точного тембра установите Coqui TTS.",
                    duration=6000,
                    position=InfoBarPosition.TOP,
                    parent=self,
                )
            if not torch_cuda:
                InfoBar.warning(
                    "GPU не обнаружен",
                    "Сейчас используется CPU-режим. Для ускорения установите CUDA-версию PyTorch в runtime и драйвер NVIDIA.",
                    duration=5000,
                    position=InfoBarPosition.TOP,
                    parent=self,
                )
        except Exception as e:
            self.diag_label.setText(f"Диагностика: ошибка ({e})")
            InfoBar.error(
                "Ошибка диагностики",
                str(e),
                duration=5000,
                position=InfoBarPosition.TOP,
                parent=self,
            )
            return

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
            f"Файл: {task.output_path}",
            duration=4000,
            position=InfoBarPosition.TOP,
            parent=self,
        )

    def on_error(self, message: str):
        self.start_button.setEnabled(True)
        self.status_label.setText("Ошибка")
        self.diag_label.setText(f"Ошибка: {message}")

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
            f"{message}\n(подробно: AppData/logs/video_translate_last_error.txt)",
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
