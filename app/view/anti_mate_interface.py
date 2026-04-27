import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtCore import Qt, QStandardPaths
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    Action,
    BodyLabel,
    CardWidget,
    CheckBox,
    ComboBox,
    CommandBar,
    FluentIcon,
    InfoBar,
    InfoBarPosition,
    MessageBox,
    PrimaryPushButton,
    ProgressBar,
    PushButton,
    StrongBodyLabel,
)

try:
    from qfluentwidgets import PlainTextEdit
except Exception:  # pragma: no cover
    from qfluentwidgets import TextEdit as PlainTextEdit

from app.common.config import cfg
from app.core.anti_mate import is_llm_runtime_available
from app.core.entities import LLMServiceEnum, SupportedAudioFormats, SupportedVideoFormats
from app.thread.anti_mate_thread import (
    AntiMateAnalyzeThread,
    AntiMateBeepPreviewThread,
    AntiMateRenderThread,
)


class AntiMateInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_path: str = ""
        self.asr_json: Dict = {}
        self.regions: List[Dict] = []
        self.last_output_path: str = ""
        self.last_output_dir: str = ""
        self.analyze_thread: Optional[AntiMateAnalyzeThread] = None
        self.render_thread: Optional[AntiMateRenderThread] = None
        self.preview_thread: Optional[AntiMateBeepPreviewThread] = None
        self._preview_temp_wav: str = ""
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(12)

        self.command_bar = CommandBar(self)
        self.command_bar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.command_bar.setFixedHeight(40)
        choose_action = Action(FluentIcon.FOLDER, "Выбрать видео/аудио")
        choose_action.triggered.connect(self._on_choose_file)
        self.command_bar.addAction(choose_action)
        root.addWidget(self.command_bar)

        self.video_card = CardWidget(self)
        video_layout = QHBoxLayout(self.video_card)
        self.video_label = BodyLabel("Файл не выбран")
        self.open_folder_btn = PushButton("Открыть папку результата")
        self.open_folder_btn.clicked.connect(self._open_output_folder)
        video_layout.addWidget(self.video_label)
        video_layout.addStretch(1)
        video_layout.addWidget(self.open_folder_btn)
        root.addWidget(self.video_card)

        self.settings_card = CardWidget(self)
        settings_layout = QVBoxLayout(self.settings_card)
        settings_layout.addWidget(StrongBodyLabel("Система Антимат"))
        settings_layout.addWidget(
            BodyLabel(
                "Пайплайн: 1) Whisper распознаёт речь с таймингами слов; "
                "2) модель находит мат; 3) вы проверяете области; 4) рендер с beep/mute."
            )
        )

        mode_row = QHBoxLayout()
        mode_row.addWidget(BodyLabel("Режим по умолчанию:"))
        self.default_mode_combo = ComboBox(self)
        self.default_mode_combo.addItems(["beep", "mute"])
        self.default_mode_combo.setCurrentText("beep")
        mode_row.addWidget(self.default_mode_combo)
        mode_row.addWidget(BodyLabel("Устройство ASR:"))
        self.device_combo = ComboBox(self)
        self.device_combo.addItems(["GPU (CUDA)", "CPU"])
        current_dev = str(cfg.faster_whisper_device.value or "cuda").strip().lower()
        self.device_combo.setCurrentIndex(0 if current_dev == "cuda" else 1)
        mode_row.addWidget(self.device_combo)
        self.asr_cache_check = CheckBox("Кэшировать ASR", self)
        use_asr_cache = bool(getattr(cfg, "use_asr_cache").value)
        self.asr_cache_check.setChecked(use_asr_cache)
        mode_row.addWidget(self.asr_cache_check)
        settings_layout.addLayout(mode_row)

        llm_row = QHBoxLayout()
        llm_row.addWidget(BodyLabel("Режим детекции:"))
        self.llm_mode_combo = ComboBox(self)
        self.llm_mode_combo.addItems([
            "Гибрид: LLM + списки",
            "Только списки (без LLM)",
        ])
        self.llm_mode_combo.setCurrentIndex(0)
        llm_row.addWidget(self.llm_mode_combo)
        llm_row.addStretch(1)
        settings_layout.addLayout(llm_row)

        timing_row_1 = QHBoxLayout()
        timing_row_1.addWidget(BodyLabel("Захват до слова (мс):"))
        self.pad_before_spin = QSpinBox(self)
        self.pad_before_spin.setRange(0, 2000)
        self.pad_before_spin.setValue(120)
        timing_row_1.addWidget(self.pad_before_spin)

        timing_row_1.addWidget(BodyLabel("Захват после слова (мс):"))
        self.pad_after_spin = QSpinBox(self)
        self.pad_after_spin.setRange(0, 3000)
        self.pad_after_spin.setValue(180)
        timing_row_1.addWidget(self.pad_after_spin)

        timing_row_1.addWidget(BodyLabel("Склейка пауз (мс):"))
        self.merge_gap_spin = QSpinBox(self)
        self.merge_gap_spin.setRange(0, 3000)
        self.merge_gap_spin.setValue(220)
        timing_row_1.addWidget(self.merge_gap_spin)
        timing_row_1.addStretch(1)
        settings_layout.addLayout(timing_row_1)

        timing_row_2 = QHBoxLayout()
        timing_row_2.addWidget(BodyLabel("Мин. длительность области (мс):"))
        self.min_region_spin = QSpinBox(self)
        self.min_region_spin.setRange(40, 5000)
        self.min_region_spin.setValue(120)
        timing_row_2.addWidget(self.min_region_spin)

        timing_row_2.addWidget(BodyLabel("Макс. длительность области (мс):"))
        self.max_region_spin = QSpinBox(self)
        self.max_region_spin.setRange(120, 20000)
        self.max_region_spin.setValue(2200)
        timing_row_2.addWidget(self.max_region_spin)
        timing_row_2.addStretch(1)
        settings_layout.addLayout(timing_row_2)

        beep_row = QHBoxLayout()
        beep_row.addWidget(BodyLabel("Профиль beep:"))
        self.beep_profile_combo = ComboBox(self)
        self.beep_profile_combo.addItems([
            "classic",
            "dual",
            "soft",
            "radio",
            "noise",
        ])
        self.beep_profile_combo.setCurrentText("classic")
        beep_row.addWidget(self.beep_profile_combo)

        beep_row.addWidget(BodyLabel("Частота (Hz):"))
        self.beep_freq_spin = QSpinBox(self)
        self.beep_freq_spin.setRange(120, 4000)
        self.beep_freq_spin.setValue(1000)
        beep_row.addWidget(self.beep_freq_spin)

        beep_row.addWidget(BodyLabel("Громкость (%):"))
        self.beep_volume_spin = QSpinBox(self)
        self.beep_volume_spin.setRange(5, 200)
        self.beep_volume_spin.setValue(90)
        beep_row.addWidget(self.beep_volume_spin)

        beep_row.addWidget(BodyLabel("Подмешивание речи (%):"))
        self.beep_duck_spin = QSpinBox(self)
        self.beep_duck_spin.setRange(0, 100)
        self.beep_duck_spin.setValue(8)
        beep_row.addWidget(self.beep_duck_spin)

        self.preview_beep_btn = PushButton("Предпрослушать beep")
        self.preview_beep_btn.clicked.connect(self._preview_beep)
        beep_row.addWidget(self.preview_beep_btn)
        beep_row.addStretch(1)
        settings_layout.addLayout(beep_row)

        lists_row = QHBoxLayout()
        lists_row.setSpacing(12)

        forbidden_col = QVBoxLayout()
        forbidden_col.addWidget(BodyLabel("Принудительно запрещённые слова (по одному или через запятую):"))
        self.forbidden_words_edit = PlainTextEdit(self)
        self.forbidden_words_edit.setPlaceholderText("пример: бля, нахуй, пиздец")
        self.forbidden_words_edit.setFixedHeight(90)
        forbidden_col.addWidget(self.forbidden_words_edit)

        allowed_col = QVBoxLayout()
        allowed_col.addWidget(BodyLabel("Принудительно разрешённые слова (исключения):"))
        self.allowed_words_edit = PlainTextEdit(self)
        self.allowed_words_edit.setPlaceholderText("пример: бляха, хулиган")
        self.allowed_words_edit.setFixedHeight(90)
        allowed_col.addWidget(self.allowed_words_edit)

        lists_row.addLayout(forbidden_col, 1)
        lists_row.addLayout(allowed_col, 1)
        settings_layout.addLayout(lists_row)

        actions_row = QHBoxLayout()
        self.analyze_btn = PrimaryPushButton("1) Анализ речи и поиск мата")
        self.analyze_btn.clicked.connect(lambda: self._start_analyze(auto_render=False))
        self.auto_run_btn = PushButton("Авто режим: анализ + рендер")
        self.auto_run_btn.clicked.connect(lambda: self._start_analyze(auto_render=True))
        self.render_btn = PrimaryPushButton("2) Применить цензуру")
        self.render_btn.clicked.connect(self._start_render)
        actions_row.addWidget(self.analyze_btn)
        actions_row.addWidget(self.auto_run_btn)
        actions_row.addWidget(self.render_btn)
        actions_row.addStretch(1)
        settings_layout.addLayout(actions_row)

        def _set_pair_tooltip(label_widget, value_widget, text: str):
            label_widget.setToolTip(text)
            value_widget.setToolTip(text)

        _set_pair_tooltip(
            mode_row.itemAt(0).widget(),
            self.default_mode_combo,
            "Режим, который будет предустановлен для найденных областей: beep (запикать) или mute (заглушить).",
        )
        _set_pair_tooltip(
            mode_row.itemAt(2).widget(),
            self.device_combo,
            "Устройство для распознавания речи Whisper: GPU быстрее, CPU стабильнее на слабых ПК.",
        )
        _set_pair_tooltip(
            self.asr_cache_check,
            self.asr_cache_check,
            "Повторно использовать результаты распознавания для этого режима, чтобы ускорить повторные прогоны.",
        )
        _set_pair_tooltip(
            llm_row.itemAt(0).widget(),
            self.llm_mode_combo,
            "Гибридный режим использует LLM + чёрный/белый списки. В режиме только списков LLM-этап отключён.",
        )
        _set_pair_tooltip(
            timing_row_1.itemAt(0).widget(),
            self.pad_before_spin,
            "Сколько миллисекунд захватывать до начала матного слова.",
        )
        _set_pair_tooltip(
            timing_row_1.itemAt(2).widget(),
            self.pad_after_spin,
            "Сколько миллисекунд захватывать после окончания матного слова.",
        )
        _set_pair_tooltip(
            timing_row_1.itemAt(4).widget(),
            self.merge_gap_spin,
            "Если между соседними матными словами пауза меньше порога, области объединяются.",
        )
        _set_pair_tooltip(
            timing_row_2.itemAt(0).widget(),
            self.min_region_spin,
            "Минимальная длина одной области цензуры после всех расчётов.",
        )
        _set_pair_tooltip(
            timing_row_2.itemAt(2).widget(),
            self.max_region_spin,
            "Максимальная длина одной области цензуры (защита от слишком длинных перекрытий).",
        )
        _set_pair_tooltip(
            beep_row.itemAt(0).widget(),
            self.beep_profile_combo,
            "Тип сигнала для запикивания: classic, dual, soft, radio, noise.",
        )
        _set_pair_tooltip(
            beep_row.itemAt(2).widget(),
            self.beep_freq_spin,
            "Базовая частота beep-сигнала в герцах.",
        )
        _set_pair_tooltip(
            beep_row.itemAt(4).widget(),
            self.beep_volume_spin,
            "Громкость beep-сигнала в процентах.",
        )
        _set_pair_tooltip(
            beep_row.itemAt(6).widget(),
            self.beep_duck_spin,
            "Сколько оставить оригинального звука под beep (0% = полный duck).",
        )
        self.forbidden_words_edit.setToolTip("Эти слова всегда считаются запрещёнными. Можно перечислять через запятую или с новой строки.")
        self.allowed_words_edit.setToolTip("Эти слова всегда считаются разрешёнными и исключаются из цензуры.")
        self.analyze_btn.setToolTip("Запустить распознавание речи и поиск мата с текущими настройками.")
        self.auto_run_btn.setToolTip("Полный автомат: анализ + немедленный рендер результата.")
        self.render_btn.setToolTip("Применить beep/mute к выбранным областям из монтажной таблицы.")

        self.progress = ProgressBar(self)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress_label = BodyLabel("Ожидание")
        settings_layout.addWidget(self.progress)
        settings_layout.addWidget(self.progress_label)

        root.addWidget(self.settings_card)

        self.table_card = CardWidget(self)
        table_layout = QVBoxLayout(self.table_card)
        table_layout.addWidget(StrongBodyLabel("Области цензуры (монтажная зона)"))
        table_layout.addWidget(BodyLabel("Снимайте/ставьте галочки и выбирайте режим beep/mute для каждой области."))

        self.table = QTableWidget(0, 6, self)
        self.table.setHorizontalHeaderLabels([
            "✓",
            "Старт",
            "Финиш",
            "Триггер",
            "Источник",
            "Режим",
        ])
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        table_layout.addWidget(self.table)

        table_actions = QHBoxLayout()
        self.select_all_btn = PushButton("Выбрать все")
        self.select_all_btn.clicked.connect(lambda: self._set_all_region_checks(True))
        self.clear_all_btn = PushButton("Снять всё")
        self.clear_all_btn.clicked.connect(lambda: self._set_all_region_checks(False))
        table_actions.addWidget(self.select_all_btn)
        table_actions.addWidget(self.clear_all_btn)
        table_actions.addStretch(1)
        table_layout.addLayout(table_actions)

        root.addWidget(self.table_card)

    @staticmethod
    def _fmt_ms(ms: int) -> str:
        total = max(0, int(ms or 0)) // 1000
        hh = total // 3600
        mm = (total % 3600) // 60
        ss = total % 60
        return f"{hh:02}:{mm:02}:{ss:02}"

    @staticmethod
    def _split_words_text(text: str) -> List[str]:
        parts = re.split(r"[\n,;]+", str(text or ""))
        return [p.strip() for p in parts if p and p.strip()]

    def _set_controls_enabled(self, enabled: bool):
        self.analyze_btn.setEnabled(enabled)
        self.auto_run_btn.setEnabled(enabled)
        self.render_btn.setEnabled(enabled)

    def _set_progress(self, value: int, message: str):
        self.progress.setValue(max(0, min(100, int(value or 0))))
        self.progress_label.setText(str(message or ""))

    def _on_choose_file(self):
        desktop_path = QStandardPaths.writableLocation(QStandardPaths.DesktopLocation)
        video_formats = " ".join(f"*.{fmt.value}" for fmt in SupportedVideoFormats)
        audio_formats = " ".join(f"*.{fmt.value}" for fmt in SupportedAudioFormats)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите видео/аудио",
            desktop_path,
            f"Медиафайлы ({video_formats} {audio_formats})",
        )
        if not file_path:
            return
        self.video_path = file_path
        self.video_label.setText(file_path)
        self.asr_json = {}
        self.regions = []
        self.table.setRowCount(0)
        self.progress.setValue(0)
        self.progress_label.setText("Файл выбран")

    def _resolve_llm_config(self) -> Dict[str, str]:
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

    def _start_analyze(self, auto_render: bool = False):
        if not self.video_path:
            InfoBar.warning(
                "Внимание",
                "Сначала выберите видео/аудио файл",
                duration=2200,
                position=InfoBarPosition.TOP,
                parent=self,
            )
            return

        self._set_controls_enabled(False)
        self._set_progress(0, "Подготовка анализа")

        selected_device = "cuda" if self.device_combo.currentIndex() == 0 else "cpu"
        cfg.set(cfg.faster_whisper_device, selected_device)
        use_asr_cache = bool(self.asr_cache_check.isChecked())
        cfg.set(cfg.use_asr_cache, use_asr_cache)

        min_region_ms = int(self.min_region_spin.value())
        max_region_ms = int(self.max_region_spin.value())
        if min_region_ms > max_region_ms:
            InfoBar.warning(
                "Некорректные тайминги",
                "Минимальная длительность области не может быть больше максимальной.",
                duration=2600,
                position=InfoBarPosition.TOP,
                parent=self,
            )
            self._set_controls_enabled(True)
            return

        use_llm = self.llm_mode_combo.currentIndex() == 0
        if use_llm:
            llm_cfg = self._resolve_llm_config()
            if not all(str(llm_cfg.get(k, "") or "").strip() for k in ("base_url", "api_key", "model")) or not is_llm_runtime_available():
                reason = (
                    "LLM недоступен: проверьте заполнение Base URL / API key / Model в настройках LLM"
                    if is_llm_runtime_available()
                    else "LLM недоступен: пакет openai не установлен в runtime"
                )
                box = MessageBox(
                    "LLM недоступен",
                    f"{reason}.\n\nПродолжить в режиме ТОЛЬКО чёрного/белого списков?",
                    self,
                )
                box.yesButton.setText("Продолжить без LLM")
                box.cancelButton.setText("Отмена")
                if not box.exec():
                    self._set_controls_enabled(True)
                    self._set_progress(0, "Анализ отменён")
                    return
                use_llm = False
                InfoBar.warning(
                    "Режим изменён",
                    "Продолжение без LLM: используется только чёрный/белый список.",
                    duration=3200,
                    position=InfoBarPosition.TOP,
                    parent=self,
                )
        else:
            box = MessageBox(
                "Подтверждение режима",
                "Вы выбрали режим без LLM. Использовать только чёрный/белый списки?",
                self,
            )
            box.yesButton.setText("Да, только списки")
            box.cancelButton.setText("Отмена")
            if not box.exec():
                self._set_controls_enabled(True)
                self._set_progress(0, "Анализ отменён")
                return
            InfoBar.warning(
                "Режим списков",
                "LLM-этап отключён пользователем. Детекция выполняется только по словарям.",
                duration=3000,
                position=InfoBarPosition.TOP,
                parent=self,
            )

        self.analyze_thread = AntiMateAnalyzeThread(
            self.video_path,
            forced_forbidden_words=self._split_words_text(self.forbidden_words_edit.toPlainText()),
            forced_allowed_words=self._split_words_text(self.allowed_words_edit.toPlainText()),
            use_llm=use_llm,
            llm_config=self._resolve_llm_config(),
            use_asr_cache=use_asr_cache,
            asr_device=selected_device,
            asr_cache_tag="anti_mate:word_ts",
            default_mode=self.default_mode_combo.currentText().strip().lower() or "beep",
            pad_before_ms=int(self.pad_before_spin.value()),
            pad_after_ms=int(self.pad_after_spin.value()),
            merge_gap_ms=int(self.merge_gap_spin.value()),
            min_region_ms=min_region_ms,
            max_region_ms=max_region_ms,
        )
        self.analyze_thread.progress.connect(self._set_progress)
        self.analyze_thread.notice.connect(self._on_notice)
        self.analyze_thread.finished.connect(
            lambda payload: self._on_analyze_finished(payload, auto_render=auto_render)
        )
        self.analyze_thread.error.connect(self._on_error)
        self.analyze_thread.start()

    def _on_analyze_finished(self, payload: Dict, auto_render: bool = False):
        self._set_controls_enabled(True)
        self.asr_json = dict((payload or {}).get("asr_json") or {})
        self.regions = list((payload or {}).get("regions") or [])
        self._fill_regions_table(self.regions)
        self._set_progress(100, f"Анализ завершён. Областей: {len(self.regions)}")
        InfoBar.success(
            "Анализ завершён",
            f"Найдено потенциальных фрагментов: {len(self.regions)}",
            duration=3000,
            position=InfoBarPosition.TOP,
            parent=self,
        )
        if auto_render and self.regions:
            self._start_render()

    def _fill_regions_table(self, regions: List[Dict]):
        self.table.setRowCount(0)
        for row, r in enumerate(regions):
            self.table.insertRow(row)

            enabled_box = CheckBox("", self)
            enabled_box.setChecked(bool(r.get("enabled", True)))
            self.table.setCellWidget(row, 0, enabled_box)

            self.table.setItem(row, 1, QTableWidgetItem(self._fmt_ms(int(r.get("start_ms", 0) or 0))))
            self.table.setItem(row, 2, QTableWidgetItem(self._fmt_ms(int(r.get("end_ms", 0) or 0))))
            self.table.setItem(row, 3, QTableWidgetItem(str(r.get("trigger_word", "") or "")))
            self.table.setItem(row, 4, QTableWidgetItem(str(r.get("source", "") or "")))

            mode_combo = ComboBox(self)
            mode_combo.addItems(["beep", "mute"])
            mode_combo.setCurrentText(str(r.get("mode", "beep") or "beep"))
            self.table.setCellWidget(row, 5, mode_combo)

        self.table.resizeColumnsToContents()

    def _set_all_region_checks(self, enabled: bool):
        for row in range(self.table.rowCount()):
            cb = self.table.cellWidget(row, 0)
            if cb:
                cb.setChecked(bool(enabled))

    def _collect_regions_from_table(self) -> List[Dict]:
        collected: List[Dict] = []
        for row in range(self.table.rowCount()):
            src = self.regions[row] if row < len(self.regions) else {}
            enabled_box = self.table.cellWidget(row, 0)
            mode_combo = self.table.cellWidget(row, 5)
            collected.append(
                {
                    "start_ms": int(src.get("start_ms", 0) or 0),
                    "end_ms": int(src.get("end_ms", 0) or 0),
                    "trigger_word": str(src.get("trigger_word", "") or ""),
                    "source": str(src.get("source", "manual") or "manual"),
                    "enabled": bool(enabled_box.isChecked()) if enabled_box else True,
                    "mode": str(mode_combo.currentText() or "beep") if mode_combo else "beep",
                }
            )
        return collected

    def _build_output_path(self) -> str:
        src = Path(self.video_path)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = src.suffix or ".mp4"
        return str(src.parent / f"антимат_{src.stem}_{stamp}{ext}")

    def _start_render(self):
        if not self.video_path:
            InfoBar.warning("Внимание", "Сначала выберите файл", duration=2200, parent=self)
            return

        regions = self._collect_regions_from_table()
        enabled_regions = [r for r in regions if bool(r.get("enabled", True))]
        if not enabled_regions:
            InfoBar.warning(
                "Внимание",
                "Не выбраны области цензуры",
                duration=2200,
                position=InfoBarPosition.TOP,
                parent=self,
            )
            return

        out_path = self._build_output_path()
        self._set_controls_enabled(False)
        self._set_progress(0, "Подготовка рендера")
        self.render_thread = AntiMateRenderThread(self.video_path, out_path, enabled_regions)
        self.render_thread.beep_profile = str(self.beep_profile_combo.currentText() or "classic")
        self.render_thread.beep_frequency = int(self.beep_freq_spin.value())
        self.render_thread.beep_volume = float(self.beep_volume_spin.value()) / 100.0
        self.render_thread.beep_duck_level = float(self.beep_duck_spin.value()) / 100.0
        self.render_thread.progress.connect(self._set_progress)
        self.render_thread.finished.connect(self._on_render_finished)
        self.render_thread.error.connect(self._on_error)
        self.render_thread.start()

    def _on_render_finished(self, out_path: str):
        self._set_controls_enabled(True)
        self.last_output_path = str(out_path or "")
        self.last_output_dir = str(Path(self.last_output_path).parent) if self.last_output_path else ""
        self._set_progress(100, f"Готово: {self.last_output_path}")
        InfoBar.success(
            "Антимат завершён",
            f"Файл сохранён:\n{self.last_output_path}",
            duration=3500,
            position=InfoBarPosition.TOP,
            parent=self,
        )

    def _on_error(self, message: str):
        self._set_controls_enabled(True)
        InfoBar.error(
            "Ошибка",
            str(message or "Неизвестная ошибка"),
            duration=3800,
            position=InfoBarPosition.TOP,
            parent=self,
        )

    def _on_notice(self, message: str):
        if not message:
            return
        InfoBar.warning(
            "Уведомление",
            str(message),
            duration=3800,
            position=InfoBarPosition.TOP,
            parent=self,
        )

    def _open_output_folder(self):
        target = None
        if self.last_output_dir and Path(self.last_output_dir).exists():
            target = Path(self.last_output_dir)
        elif self.video_path:
            p = Path(self.video_path).parent
            if p.exists():
                target = p
        if not target:
            return
        if os.name == "nt":
            os.startfile(str(target))

    def _preview_beep(self):
        if self.preview_thread and self.preview_thread.isRunning():
            return
        fd, wav_path = tempfile.mkstemp(prefix="anti_mate_beep_", suffix=".wav")
        os.close(fd)
        self._preview_temp_wav = wav_path
        self.preview_beep_btn.setEnabled(False)
        self.preview_thread = AntiMateBeepPreviewThread(
            wav_path,
            profile=str(self.beep_profile_combo.currentText() or "classic"),
            frequency=int(self.beep_freq_spin.value()),
            volume=float(self.beep_volume_spin.value()) / 100.0,
        )
        self.preview_thread.finished.connect(self._on_beep_preview_ready)
        self.preview_thread.error.connect(self._on_beep_preview_error)
        self.preview_thread.start()

    def _on_beep_preview_ready(self, path: str):
        self.preview_beep_btn.setEnabled(True)
        try:
            if os.name == "nt" and path:
                os.startfile(str(path))
            InfoBar.success(
                "Preview готов",
                "Открылся WAV с тестовым beep-профилем.",
                duration=2600,
                position=InfoBarPosition.TOP,
                parent=self,
            )
        except Exception as e:
            self._on_beep_preview_error(str(e))

    def _on_beep_preview_error(self, message: str):
        self.preview_beep_btn.setEnabled(True)
        InfoBar.error(
            "Preview beep: ошибка",
            str(message or "Не удалось создать preview"),
            duration=3200,
            position=InfoBarPosition.TOP,
            parent=self,
        )
