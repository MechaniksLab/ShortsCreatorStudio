import json
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
    QLineEdit,
    QSizePolicy,
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
    MessageBoxBase,
    PrimaryPushButton,
    ProgressBar,
    PushButton,
    ScrollArea,
    StrongBodyLabel,
)

from app.common.config import cfg
from app.common.theme_manager import get_theme_palette
from app.config import APPDATA_PATH
from app.core.anti_mate import is_llm_runtime_available
from app.core.anti_mate.processor import DEFAULT_FORBIDDEN_WORDS
from app.core.entities import LLMServiceEnum, SupportedAudioFormats, SupportedVideoFormats
from app.thread.anti_mate_thread import (
    AntiMateAnalyzeThread,
    AntiMateBeepPreviewThread,
    AntiMateRenderThread,
)

ANTIMATE_PRESETS_PATH = APPDATA_PATH / "anti_mate_presets.json"


class WordsEditDialog(MessageBoxBase):
    def __init__(self, title: str, words: List[str], placeholder: str, parent=None):
        super().__init__(parent)
        self.titleLabel = BodyLabel(title, self)
        self.contentLabel = BodyLabel("По одному слову на строку или через запятую/точку с запятой", self)
        self.edit = QLineEdit(self)
        self.edit.setPlaceholderText(placeholder)
        self.edit.setText(", ".join(words or []))
        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.contentLabel)
        self.viewLayout.addWidget(self.edit)
        self.yesButton.setText("Сохранить")
        self.cancelButton.setText("Отмена")

    def get_words(self) -> List[str]:
        parts = re.split(r"[\n,;]+", self.edit.text() or "")
        return [p.strip() for p in parts if p and p.strip()]


class PresetNameDialog(MessageBoxBase):
    def __init__(self, title: str, default_name: str = "", parent=None):
        super().__init__(parent)
        self.titleLabel = BodyLabel(title, self)
        self.contentLabel = BodyLabel("Введите имя пресета", self)
        self.name_edit = QLineEdit(self)
        self.name_edit.setPlaceholderText("например: YouTube Strict")
        self.name_edit.setText(default_name or "")
        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.contentLabel)
        self.viewLayout.addWidget(self.name_edit)
        self.yesButton.setText("Сохранить")
        self.cancelButton.setText("Отмена")
        self.yesButton.setEnabled(bool((default_name or "").strip()))
        self.name_edit.textChanged.connect(
            lambda txt: self.yesButton.setEnabled(bool((txt or "").strip()))
        )
        p = get_theme_palette()
        self.widget.setStyleSheet(
            f"""
            QWidget {{
                background: {p['card_bg']};
                color: {p['text']};
            }}
            QLineEdit {{
                background: {p['panel_bg']};
                border: 1px solid {p['border']};
                border-radius: 6px;
                padding: 5px;
            }}
            """
        )

    def get_name(self) -> str:
        return (self.name_edit.text() or "").strip()


class AntiMateInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("AntiMateInterface")
        self.video_path: str = ""
        self.asr_json: Dict = {}
        self.regions: List[Dict] = []
        self.last_output_path: str = ""
        self.last_output_dir: str = ""

        self.forbidden_words: List[str] = []
        self.allowed_words: List[str] = []

        self.analyze_thread: Optional[AntiMateAnalyzeThread] = None
        self.render_thread: Optional[AntiMateRenderThread] = None
        self.preview_thread: Optional[AntiMateBeepPreviewThread] = None

        self._build_ui()
        self._load_words_and_presets_defaults()
        self._refresh_presets_combo()
        self._setup_tooltips()
        self._apply_theme_style()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

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

        self.scroll_area = ScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget(self)
        self.scroll_widget.setObjectName("AntiMateScrollWidget")
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_layout.setSpacing(12)

        self._build_presets_card()
        self._build_stage1_card()
        self._build_stage2_card()
        self._build_timeline_card()

        self.scroll_area.setWidget(self.scroll_widget)
        root.addWidget(self.scroll_area, 1)

    def _build_presets_card(self):
        card = CardWidget(self)
        lay = QVBoxLayout(card)
        lay.addWidget(StrongBodyLabel("Пресеты Антимата"))
        row = QHBoxLayout()
        row.addWidget(BodyLabel("Пресет:"))
        self.preset_combo = ComboBox(self)
        self.preset_combo.currentTextChanged.connect(self._on_preset_selected)
        row.addWidget(self.preset_combo)
        self.save_preset_btn = PushButton("Сохранить/обновить")
        self.save_preset_btn.clicked.connect(self._save_or_update_preset)
        self.delete_preset_btn = PushButton("Удалить")
        self.delete_preset_btn.clicked.connect(self._delete_preset)
        row.addWidget(self.save_preset_btn)
        row.addWidget(self.delete_preset_btn)
        row.addStretch(1)
        lay.addLayout(row)
        self.scroll_layout.addWidget(card)

    def _build_stage1_card(self):
        self.stage1_card = CardWidget(self)
        lay = QVBoxLayout(self.stage1_card)
        lay.addWidget(StrongBodyLabel("Этап 1: Анализ речи и поиск мата"))

        row1 = QHBoxLayout()
        row1.addWidget(BodyLabel("Режим детекции:"))
        self.llm_mode_combo = ComboBox(self)
        self.llm_mode_combo.addItems(["Гибрид: LLM + списки", "Только списки (без LLM)"])
        row1.addWidget(self.llm_mode_combo)
        row1.addWidget(BodyLabel("Устройство ASR:"))
        self.device_combo = ComboBox(self)
        self.device_combo.addItems(["GPU (CUDA)", "CPU"])
        current_dev = str(cfg.faster_whisper_device.value or "cuda").strip().lower()
        self.device_combo.setCurrentIndex(0 if current_dev == "cuda" else 1)
        row1.addWidget(self.device_combo)
        self.asr_cache_check = CheckBox("Кэшировать ASR", self)
        self.asr_cache_check.setChecked(bool(cfg.use_asr_cache.value))
        row1.addWidget(self.asr_cache_check)
        row1.addStretch(1)
        lay.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(BodyLabel("Захват до слова (мс):"))
        self.pad_before_spin = QSpinBox(self)
        self.pad_before_spin.setRange(0, 2000)
        self.pad_before_spin.setValue(120)
        row2.addWidget(self.pad_before_spin)
        row2.addWidget(BodyLabel("Захват после слова (мс):"))
        self.pad_after_spin = QSpinBox(self)
        self.pad_after_spin.setRange(0, 3000)
        self.pad_after_spin.setValue(180)
        row2.addWidget(self.pad_after_spin)
        row2.addWidget(BodyLabel("Склейка пауз (мс):"))
        self.merge_gap_spin = QSpinBox(self)
        self.merge_gap_spin.setRange(0, 3000)
        self.merge_gap_spin.setValue(220)
        row2.addWidget(self.merge_gap_spin)
        row2.addStretch(1)
        lay.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(BodyLabel("Мин. длительность области (мс):"))
        self.min_region_spin = QSpinBox(self)
        self.min_region_spin.setRange(40, 5000)
        self.min_region_spin.setValue(120)
        row3.addWidget(self.min_region_spin)
        row3.addWidget(BodyLabel("Макс. длительность области (мс):"))
        self.max_region_spin = QSpinBox(self)
        self.max_region_spin.setRange(120, 20000)
        self.max_region_spin.setValue(2200)
        row3.addWidget(self.max_region_spin)
        row3.addStretch(1)
        lay.addLayout(row3)

        words_row = QHBoxLayout()
        self.forbidden_btn = PushButton("Запрещённые слова")
        self.forbidden_btn.clicked.connect(self._edit_forbidden_words)
        self.allowed_btn = PushButton("Разрешённые слова")
        self.allowed_btn.clicked.connect(self._edit_allowed_words)
        self.words_state_label = BodyLabel("")
        words_row.addWidget(self.forbidden_btn)
        words_row.addWidget(self.allowed_btn)
        words_row.addWidget(self.words_state_label)
        words_row.addStretch(1)
        lay.addLayout(words_row)

        self.analyze_btn = PrimaryPushButton("1) Анализ речи и поиск мата")
        self.analyze_btn.setMinimumHeight(38)
        self.analyze_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.analyze_btn.clicked.connect(lambda: self._start_analyze(auto_render=False))
        lay.addWidget(self.analyze_btn)

        self.progress = ProgressBar(self)
        self.progress.setRange(0, 100)
        self.progress_label = BodyLabel("Ожидание")
        lay.addWidget(self.progress)
        lay.addWidget(self.progress_label)

        self.scroll_layout.addWidget(self.stage1_card)

    def _build_stage2_card(self):
        self.stage2_card = CardWidget(self)
        lay = QVBoxLayout(self.stage2_card)
        lay.addWidget(StrongBodyLabel("Этап 2: Фильтры цензуры и рендер"))

        row1 = QHBoxLayout()
        row1.addWidget(BodyLabel("Режим по умолчанию:"))
        self.default_mode_combo = ComboBox(self)
        self.default_mode_combo.addItems(["beep", "mute"])
        self.default_mode_combo.setCurrentText("beep")
        row1.addWidget(self.default_mode_combo)
        row1.addWidget(BodyLabel("Профиль beep:"))
        self.beep_profile_combo = ComboBox(self)
        self.beep_profile_combo.addItems(["classic", "dual", "soft", "radio", "noise"])
        row1.addWidget(self.beep_profile_combo)
        row1.addWidget(BodyLabel("Частота (Hz):"))
        self.beep_freq_spin = QSpinBox(self)
        self.beep_freq_spin.setRange(120, 4000)
        self.beep_freq_spin.setValue(1000)
        row1.addWidget(self.beep_freq_spin)
        row1.addStretch(1)
        lay.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(BodyLabel("Громкость beep (%):"))
        self.beep_volume_spin = QSpinBox(self)
        self.beep_volume_spin.setRange(5, 200)
        self.beep_volume_spin.setValue(90)
        row2.addWidget(self.beep_volume_spin)
        row2.addWidget(BodyLabel("Подмешивание речи (%):"))
        self.beep_duck_spin = QSpinBox(self)
        self.beep_duck_spin.setRange(0, 100)
        self.beep_duck_spin.setValue(8)
        row2.addWidget(self.beep_duck_spin)
        self.preview_beep_btn = PushButton("Предпрослушать beep")
        self.preview_beep_btn.clicked.connect(self._preview_beep)
        row2.addWidget(self.preview_beep_btn)
        row2.addStretch(1)
        lay.addLayout(row2)

        btn_row = QVBoxLayout()
        self.auto_run_btn = PushButton("Авто режим: анализ + рендер")
        self.auto_run_btn.setMinimumHeight(38)
        self.auto_run_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.auto_run_btn.clicked.connect(lambda: self._start_analyze(auto_render=True))
        self.render_btn = PrimaryPushButton("2) Применить цензуру")
        self.render_btn.setMinimumHeight(38)
        self.render_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.render_btn.clicked.connect(self._start_render)
        btn_row.addWidget(self.auto_run_btn)
        btn_row.addWidget(self.render_btn)
        lay.addLayout(btn_row)

        self.scroll_layout.addWidget(self.stage2_card)

    def _build_timeline_card(self):
        self.table_card = CardWidget(self)
        lay = QVBoxLayout(self.table_card)
        lay.addWidget(StrongBodyLabel("Монтажная зона цензуры"))
        self.table = QTableWidget(0, 6, self)
        self.table.setHorizontalHeaderLabels(["✓", "Старт", "Финиш", "Триггер", "Источник", "Режим"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self.table)
        row = QHBoxLayout()
        self.select_all_btn = PushButton("Выбрать все")
        self.select_all_btn.clicked.connect(lambda: self._set_all_region_checks(True))
        self.clear_all_btn = PushButton("Снять всё")
        self.clear_all_btn.clicked.connect(lambda: self._set_all_region_checks(False))
        row.addWidget(self.select_all_btn)
        row.addWidget(self.clear_all_btn)
        row.addStretch(1)
        lay.addLayout(row)
        self.scroll_layout.addWidget(self.table_card)

    def _apply_theme_style(self):
        p = get_theme_palette()
        self.setStyleSheet(
            f"""
            QWidget#AntiMateInterface {{
                background: {p['window_bg']};
            }}
            QWidget#AntiMateScrollWidget {{
                background: transparent;
            }}
            CardWidget {{
                background: {p['card_bg']};
                border: 1px solid {p['border']};
                border-radius: 10px;
            }}
            QTableWidget {{
                background: {p['panel_bg']};
                border: 1px solid {p['border']};
                gridline-color: {p['border']};
                border-radius: 8px;
            }}
            QHeaderView::section {{
                background: {p['card_bg']};
                border: 1px solid {p['border']};
                padding: 4px;
            }}
            QScrollBar:vertical {{ width: 9px; background: transparent; margin: 2px 0; }}
            QScrollBar::handle:vertical {{ background: {p['border']}; border-radius: 4px; min-height: 24px; }}
            QScrollBar::handle:vertical:hover {{ background: {p['accent']}; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; height: 0; }}
            QScrollBar:horizontal {{ height: 9px; background: transparent; margin: 0 2px; }}
            QScrollBar::handle:horizontal {{ background: {p['border']}; border-radius: 4px; min-width: 24px; }}
            QScrollBar::handle:horizontal:hover {{ background: {p['accent']}; }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: transparent; width: 0; }}
            """
        )

    def _setup_tooltips(self):
        self.llm_mode_combo.setToolTip("Гибрид: LLM + словари. Только списки: без обращения к LLM.")
        self.device_combo.setToolTip("Выбор устройства для Whisper ASR: GPU быстрее, CPU стабильнее.")
        self.asr_cache_check.setToolTip("Повторно использовать распознавание, чтобы ускорить новые прогоны.")
        self.pad_before_spin.setToolTip("Добавить этот запас до матного слова (мс).")
        self.pad_after_spin.setToolTip("Добавить этот запас после матного слова (мс).")
        self.merge_gap_spin.setToolTip("Если пауза между матами меньше значения, зоны объединяются.")
        self.min_region_spin.setToolTip("Минимальная длительность цензурной зоны (мс).")
        self.max_region_spin.setToolTip("Максимальная длительность цензурной зоны (мс).")
        self.forbidden_btn.setToolTip("Открыть редактирование принудительно запрещённых слов.")
        self.allowed_btn.setToolTip("Открыть редактирование принудительно разрешённых слов-исключений.")
        self.default_mode_combo.setToolTip("Режим для новых найденных областей: beep или mute.")
        self.beep_profile_combo.setToolTip("Тип beep-сигнала: classic/dual/soft/radio/noise.")
        self.beep_freq_spin.setToolTip("Базовая частота beep в герцах.")
        self.beep_volume_spin.setToolTip("Громкость beep в процентах.")
        self.beep_duck_spin.setToolTip("Сколько исходной речи оставить под beep (ducking).")
        self.preview_beep_btn.setToolTip("Сгенерировать и открыть тестовый WAV для выбранного профиля beep.")
        self.analyze_btn.setToolTip("Запустить только этап анализа речи и поиска нецензурных зон.")
        self.auto_run_btn.setToolTip("Запустить полный авто-режим: анализ + рендер цензуры.")
        self.render_btn.setToolTip("Применить цензуру к выбранным зонам из таблицы.")

    @staticmethod
    def _fmt_ms(ms: int) -> str:
        total = max(0, int(ms or 0)) // 1000
        hh = total // 3600
        mm = (total % 3600) // 60
        ss = total % 60
        return f"{hh:02}:{mm:02}:{ss:02}"

    def _read_presets_data(self) -> Dict:
        try:
            if not ANTIMATE_PRESETS_PATH.exists():
                return {}
            data = json.loads(ANTIMATE_PRESETS_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _write_presets_data(self, data: Dict):
        ANTIMATE_PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
        ANTIMATE_PRESETS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_words_and_presets_defaults(self):
        data = self._read_presets_data()
        self.forbidden_words = list(data.get("forbidden_words") or sorted(DEFAULT_FORBIDDEN_WORDS))
        self.allowed_words = list(data.get("allowed_words") or ["бляха", "хулиган"])
        self._update_words_state()
        last = data.get("last_state") or {}
        if isinstance(last, dict):
            self._apply_state(last)
        if not ANTIMATE_PRESETS_PATH.exists():
            self._persist_last_state()

    @staticmethod
    def _sanitize_preset_name(name: str) -> str:
        n = (name or "").strip()
        n = re.sub(r"[\\/:*?\"<>|]+", "_", n)
        return n.strip()

    def _next_default_preset_name(self) -> str:
        data = self._read_presets_data()
        presets = data.get("presets") if isinstance(data.get("presets"), dict) else {}
        idx = 1
        while f"Preset {idx}" in presets:
            idx += 1
        return f"Preset {idx}"

    def _update_words_state(self):
        self.words_state_label.setText(
            f"Запрещённые: {len(self.forbidden_words)} | Разрешённые: {len(self.allowed_words)}"
        )

    def _collect_state(self) -> Dict:
        return {
            "llm_mode": int(self.llm_mode_combo.currentIndex()),
            "device": int(self.device_combo.currentIndex()),
            "asr_cache": bool(self.asr_cache_check.isChecked()),
            "pad_before": int(self.pad_before_spin.value()),
            "pad_after": int(self.pad_after_spin.value()),
            "merge_gap": int(self.merge_gap_spin.value()),
            "min_region": int(self.min_region_spin.value()),
            "max_region": int(self.max_region_spin.value()),
            "default_mode": str(self.default_mode_combo.currentText() or "beep"),
            "beep_profile": str(self.beep_profile_combo.currentText() or "classic"),
            "beep_freq": int(self.beep_freq_spin.value()),
            "beep_volume": int(self.beep_volume_spin.value()),
            "beep_duck": int(self.beep_duck_spin.value()),
            "forbidden_words": list(self.forbidden_words),
            "allowed_words": list(self.allowed_words),
        }

    def _apply_state(self, state: Dict):
        try:
            self.llm_mode_combo.setCurrentIndex(int(state.get("llm_mode", self.llm_mode_combo.currentIndex())))
            self.device_combo.setCurrentIndex(int(state.get("device", self.device_combo.currentIndex())))
            self.asr_cache_check.setChecked(bool(state.get("asr_cache", self.asr_cache_check.isChecked())))
            self.pad_before_spin.setValue(int(state.get("pad_before", self.pad_before_spin.value())))
            self.pad_after_spin.setValue(int(state.get("pad_after", self.pad_after_spin.value())))
            self.merge_gap_spin.setValue(int(state.get("merge_gap", self.merge_gap_spin.value())))
            self.min_region_spin.setValue(int(state.get("min_region", self.min_region_spin.value())))
            self.max_region_spin.setValue(int(state.get("max_region", self.max_region_spin.value())))
            self.default_mode_combo.setCurrentText(str(state.get("default_mode", self.default_mode_combo.currentText())))
            self.beep_profile_combo.setCurrentText(str(state.get("beep_profile", self.beep_profile_combo.currentText())))
            self.beep_freq_spin.setValue(int(state.get("beep_freq", self.beep_freq_spin.value())))
            self.beep_volume_spin.setValue(int(state.get("beep_volume", self.beep_volume_spin.value())))
            self.beep_duck_spin.setValue(int(state.get("beep_duck", self.beep_duck_spin.value())))
            fw = state.get("forbidden_words")
            aw = state.get("allowed_words")
            if isinstance(fw, list):
                self.forbidden_words = [str(x).strip() for x in fw if str(x).strip()]
            if isinstance(aw, list):
                self.allowed_words = [str(x).strip() for x in aw if str(x).strip()]
            self._update_words_state()
        except Exception:
            pass

    def _refresh_presets_combo(self):
        data = self._read_presets_data()
        presets = data.get("presets") if isinstance(data.get("presets"), dict) else {}
        names = sorted(presets.keys(), key=lambda x: x.lower())
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem("—")
        for name in names:
            self.preset_combo.addItem(name)
        last_name = str(data.get("last_selected_preset", "") or "").strip()
        if last_name and last_name in names:
            self.preset_combo.setCurrentText(last_name)
        else:
            self.preset_combo.setCurrentText("—")
        self.preset_combo.blockSignals(False)

    def _on_preset_selected(self, name: str):
        n = (name or "").strip()
        if not n or n == "—":
            return
        data = self._read_presets_data()
        presets = data.get("presets") if isinstance(data.get("presets"), dict) else {}
        state = presets.get(n)
        if not isinstance(state, dict):
            return
        self._apply_state(state)
        data["last_selected_preset"] = n
        self._write_presets_data(data)

    def _save_or_update_preset(self):
        try:
            current_name = (self.preset_combo.currentText() or "").strip()
            if current_name and current_name != "—":
                target_name = current_name
            else:
                default_name = self._next_default_preset_name()
                dlg = PresetNameDialog(
                    "Сохранение пресета антимата",
                    default_name=default_name,
                    parent=self,
                )
                if not dlg.exec():
                    return
                target_name = self._sanitize_preset_name(dlg.get_name())
                if not target_name:
                    InfoBar.warning(
                        "Пресет",
                        "Имя пресета не может быть пустым.",
                        duration=2200,
                        position=InfoBarPosition.TOP,
                        parent=self,
                    )
                    return

            data = self._read_presets_data()
            presets = data.get("presets") if isinstance(data.get("presets"), dict) else {}
            presets[target_name] = self._collect_state()
            data["presets"] = presets
            data["last_selected_preset"] = target_name
            data["forbidden_words"] = list(self.forbidden_words)
            data["allowed_words"] = list(self.allowed_words)
            data["last_state"] = self._collect_state()
            self._write_presets_data(data)
            self._refresh_presets_combo()
            self.preset_combo.setCurrentText(target_name)
            InfoBar.success(
                "Пресет сохранён",
                f"Антимат пресет: {target_name}",
                duration=2200,
                position=InfoBarPosition.TOP,
                parent=self,
            )
        except Exception as e:
            InfoBar.error(
                "Ошибка сохранения пресета",
                str(e),
                duration=3200,
                position=InfoBarPosition.TOP,
                parent=self,
            )

    def _delete_preset(self):
        name = (self.preset_combo.currentText() or "").strip()
        if not name or name == "—":
            return
        data = self._read_presets_data()
        presets = data.get("presets") if isinstance(data.get("presets"), dict) else {}
        if name in presets:
            presets.pop(name, None)
            data["presets"] = presets
            if str(data.get("last_selected_preset", "")) == name:
                data["last_selected_preset"] = ""
            self._write_presets_data(data)
            self._refresh_presets_combo()
            InfoBar.success(
                "Пресет удалён",
                f"Удалён: {name}",
                duration=1800,
                position=InfoBarPosition.TOP,
                parent=self,
            )

    def _persist_last_state(self):
        data = self._read_presets_data()
        data["forbidden_words"] = list(self.forbidden_words)
        data["allowed_words"] = list(self.allowed_words)
        data["last_state"] = self._collect_state()
        self._write_presets_data(data)

    def _edit_forbidden_words(self):
        dlg = WordsEditDialog(
            "Запрещённые слова",
            self.forbidden_words,
            "пример: бля, нахуй, ...",
            self,
        )
        if not dlg.exec():
            return
        self.forbidden_words = dlg.get_words()
        self._update_words_state()
        self._persist_last_state()

    def _edit_allowed_words(self):
        dlg = WordsEditDialog(
            "Разрешённые слова",
            self.allowed_words,
            "пример: бляха, хулиган, ...",
            self,
        )
        if not dlg.exec():
            return
        self.allowed_words = dlg.get_words()
        self._update_words_state()
        self._persist_last_state()

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
        self._set_progress(0, "Файл выбран")

    def _resolve_llm_config(self) -> Dict[str, str]:
        service = cfg.llm_service.value
        if service == LLMServiceEnum.OPENAI:
            return {"base_url": cfg.openai_api_base.value, "api_key": cfg.openai_api_key.value, "model": cfg.openai_model.value}
        if service == LLMServiceEnum.SILICON_CLOUD:
            return {"base_url": cfg.silicon_cloud_api_base.value, "api_key": cfg.silicon_cloud_api_key.value, "model": cfg.silicon_cloud_model.value}
        if service == LLMServiceEnum.DEEPSEEK:
            return {"base_url": cfg.deepseek_api_base.value, "api_key": cfg.deepseek_api_key.value, "model": cfg.deepseek_model.value}
        if service == LLMServiceEnum.OLLAMA:
            return {"base_url": cfg.ollama_api_base.value, "api_key": cfg.ollama_api_key.value, "model": cfg.ollama_model.value}
        if service == LLMServiceEnum.LM_STUDIO:
            return {"base_url": cfg.lm_studio_api_base.value, "api_key": cfg.lm_studio_api_key.value, "model": cfg.lm_studio_model.value}
        if service == LLMServiceEnum.GEMINI:
            return {"base_url": cfg.gemini_api_base.value, "api_key": cfg.gemini_api_key.value, "model": cfg.gemini_model.value}
        if service == LLMServiceEnum.CHATGLM:
            return {"base_url": cfg.chatglm_api_base.value, "api_key": cfg.chatglm_api_key.value, "model": cfg.chatglm_model.value}
        return {"base_url": cfg.public_api_base.value, "api_key": cfg.public_api_key.value, "model": cfg.public_model.value}

    def _start_analyze(self, auto_render: bool = False):
        if not self.video_path:
            InfoBar.warning("Внимание", "Сначала выберите видео/аудио файл", duration=2200, position=InfoBarPosition.TOP, parent=self)
            return

        min_region_ms = int(self.min_region_spin.value())
        max_region_ms = int(self.max_region_spin.value())
        if min_region_ms > max_region_ms:
            InfoBar.warning("Некорректные тайминги", "Минимальная длительность не может быть больше максимальной", duration=2200, position=InfoBarPosition.TOP, parent=self)
            return

        self._set_controls_enabled(False)
        self._set_progress(0, "Подготовка анализа")

        selected_device = "cuda" if self.device_combo.currentIndex() == 0 else "cpu"
        use_asr_cache = bool(self.asr_cache_check.isChecked())
        cfg.set(cfg.faster_whisper_device, selected_device)
        cfg.set(cfg.use_asr_cache, use_asr_cache)

        use_llm = self.llm_mode_combo.currentIndex() == 0
        if use_llm:
            llm_cfg = self._resolve_llm_config()
            if not all(str(llm_cfg.get(k, "") or "").strip() for k in ("base_url", "api_key", "model")) or not is_llm_runtime_available():
                box = MessageBox("LLM недоступен", "Продолжить только по словарным спискам?", self)
                box.yesButton.setText("Да")
                box.cancelButton.setText("Отмена")
                if not box.exec():
                    self._set_controls_enabled(True)
                    return
                use_llm = False

        self.analyze_thread = AntiMateAnalyzeThread(
            self.video_path,
            forced_forbidden_words=list(self.forbidden_words),
            forced_allowed_words=list(self.allowed_words),
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
        self.analyze_thread.finished.connect(lambda payload: self._on_analyze_finished(payload, auto_render=auto_render))
        self.analyze_thread.error.connect(self._on_error)
        self.analyze_thread.start()
        self._persist_last_state()

    def _on_analyze_finished(self, payload: Dict, auto_render: bool = False):
        self._set_controls_enabled(True)
        self.asr_json = dict((payload or {}).get("asr_json") or {})
        self.regions = list((payload or {}).get("regions") or [])
        self._fill_regions_table(self.regions)
        self._set_progress(100, f"Анализ завершён. Областей: {len(self.regions)}")
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
            InfoBar.warning("Внимание", "Не выбраны области цензуры", duration=2200, position=InfoBarPosition.TOP, parent=self)
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
        self._persist_last_state()

    def _on_render_finished(self, out_path: str):
        self._set_controls_enabled(True)
        self.last_output_path = str(out_path or "")
        self.last_output_dir = str(Path(self.last_output_path).parent) if self.last_output_path else ""
        self._set_progress(100, f"Готово: {self.last_output_path}")
        InfoBar.success("Антимат завершён", f"Файл сохранён:\n{self.last_output_path}", duration=3200, position=InfoBarPosition.TOP, parent=self)

    def _on_error(self, message: str):
        self._set_controls_enabled(True)
        InfoBar.error("Ошибка", str(message or "Неизвестная ошибка"), duration=3600, position=InfoBarPosition.TOP, parent=self)

    def _on_notice(self, message: str):
        if not message:
            return
        InfoBar.warning("Уведомление", str(message), duration=3000, position=InfoBarPosition.TOP, parent=self)

    def _open_output_folder(self):
        target = None
        if self.last_output_dir and Path(self.last_output_dir).exists():
            target = Path(self.last_output_dir)
        elif self.video_path:
            p = Path(self.video_path).parent
            if p.exists():
                target = p
        if target and os.name == "nt":
            os.startfile(str(target))

    def _preview_beep(self):
        if self.preview_thread and self.preview_thread.isRunning():
            return
        fd, wav_path = tempfile.mkstemp(prefix="anti_mate_beep_", suffix=".wav")
        os.close(fd)
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
            InfoBar.success("Preview готов", "Открылся WAV с тестовым beep-профилем.", duration=2200, position=InfoBarPosition.TOP, parent=self)
        except Exception as e:
            self._on_beep_preview_error(str(e))

    def _on_beep_preview_error(self, message: str):
        self.preview_beep_btn.setEnabled(True)
        InfoBar.error("Preview beep: ошибка", str(message or "Не удалось создать preview"), duration=3200, position=InfoBarPosition.TOP, parent=self)
