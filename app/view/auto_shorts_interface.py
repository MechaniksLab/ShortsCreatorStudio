import json
import os
import subprocess
import sys
import tempfile
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtCore import QPointF, QRect, QRectF, Qt, QStandardPaths, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import Action, BodyLabel, CardWidget, CheckBox, ComboBox, CommandBar, FluentIcon, InfoBar, InfoBarPosition, LineEdit, MessageBoxBase, PrimaryPushButton, ProgressBar, PushButton, ScrollArea, StrongBodyLabel, SpinBox, isDarkTheme

from app.common.config import cfg
from app.common.theme_manager import get_theme_palette
from app.config import APPDATA_PATH, WORK_PATH
from app.core.entities import BatchTaskType, SupportedAudioFormats, SupportedVideoFormats
from app.core.task_factory import TaskFactory


class LayerPreviewWidget(QWidget):
    changed = pyqtSignal(dict)

    def __init__(self, canvas_w: int, canvas_h: int, parent=None):
        super().__init__(parent)
        self.canvas_w = max(2, int(canvas_w))
        self.canvas_h = max(2, int(canvas_h))
        self.background = QPixmap()
        self.setMinimumHeight(240)
        self.dark_theme = True

        self.layers = {
            "webcam": QRectF(0, 0, self.canvas_w * 0.45, self.canvas_h * 0.33),
            "game": QRectF(0, self.canvas_h * 0.33, self.canvas_w, self.canvas_h * 0.67),
        }
        self.colors = {
            "webcam": QColor(78, 201, 176, 180),
            "game": QColor(86, 156, 214, 180),
        }
        self.active_layer = "webcam"
        self.drag_mode = "none"  # none|move|resize
        self.drag_offset = QPointF(0, 0)
        self.resize_anchor = QPointF(0, 0)
        self.keep_aspect = False
        self.resize_aspect_ratio = 1.0
        self.interactive = True
        self.show_layer_overlay = True
        self.webcam_enabled = True

    def set_interactive(self, value: bool):
        self.interactive = bool(value)

    def set_show_layer_overlay(self, value: bool):
        self.show_layer_overlay = bool(value)
        self.update()

    def set_webcam_enabled(self, value: bool):
        self.webcam_enabled = bool(value)
        if not self.webcam_enabled and self.active_layer == "webcam":
            self.active_layer = "game"
            self.drag_mode = "none"
        self.update()

    def set_keep_aspect(self, value: bool):
        self.keep_aspect = bool(value)

    def set_theme(self, dark_theme: bool):
        self.dark_theme = bool(dark_theme)
        if self.dark_theme:
            self.colors = {
                "webcam": QColor(110, 231, 183, 210),
                "game": QColor(96, 165, 250, 210),
            }
        else:
            self.colors = {
                "webcam": QColor(16, 185, 129, 210),
                "game": QColor(37, 99, 235, 210),
            }
        self.update()

    def set_canvas_size(self, w: int, h: int):
        old_w, old_h = self.canvas_w, self.canvas_h
        self.canvas_w, self.canvas_h = max(2, int(w)), max(2, int(h))
        if old_w > 0 and old_h > 0:
            sx = self.canvas_w / old_w
            sy = self.canvas_h / old_h
            for k in self.layers:
                r = self.layers[k]
                self.layers[k] = QRectF(r.x() * sx, r.y() * sy, r.width() * sx, r.height() * sy)
        self._clamp_all()
        self.update()

    def set_background(self, pixmap: QPixmap):
        self.background = pixmap if pixmap and not pixmap.isNull() else QPixmap()
        self.update()

    def set_layers(self, webcam_rect: QRectF, game_rect: QRectF):
        self.layers["webcam"] = QRectF(webcam_rect)
        self.layers["game"] = QRectF(game_rect)
        self._clamp_all()
        self.update()

    def get_layers(self) -> Dict[str, Dict[str, int]]:
        return {
            name: {
                "x": int(rect.x()),
                "y": int(rect.y()),
                "w": int(rect.width()),
                "h": int(rect.height()),
            }
            for name, rect in self.layers.items()
        }

    def _draw_area(self) -> QRectF:
        margin = 10
        avail_w = max(1, self.width() - 2 * margin)
        avail_h = max(1, self.height() - 2 * margin)
        scale = min(avail_w / self.canvas_w, avail_h / self.canvas_h)
        draw_w = self.canvas_w * scale
        draw_h = self.canvas_h * scale
        x = (self.width() - draw_w) / 2
        y = (self.height() - draw_h) / 2
        return QRectF(x, y, draw_w, draw_h)

    def _canvas_to_view(self, p: QPointF) -> QPointF:
        area = self._draw_area()
        return QPointF(area.x() + (p.x() / self.canvas_w) * area.width(), area.y() + (p.y() / self.canvas_h) * area.height())

    def _view_to_canvas(self, p: QPointF) -> QPointF:
        area = self._draw_area()
        if area.width() <= 0 or area.height() <= 0:
            return QPointF(0, 0)
        x = ((p.x() - area.x()) / area.width()) * self.canvas_w
        y = ((p.y() - area.y()) / area.height()) * self.canvas_h
        return QPointF(max(0, min(self.canvas_w, x)), max(0, min(self.canvas_h, y)))

    def _layer_hit_test(self, canvas_pos: QPointF):
        for name in ["webcam", "game"]:
            if name == "webcam" and not self.webcam_enabled:
                continue
            if self.layers[name].contains(canvas_pos):
                return name
        return None

    def _layer_rect_view(self, layer: str) -> QRectF:
        r = self.layers[layer]
        tl = self._canvas_to_view(r.topLeft())
        br = self._canvas_to_view(r.bottomRight())
        return QRectF(tl, br)

    def _handle_hit_test(self, layer: str, view_pos: QPointF) -> bool:
        vr = self._layer_rect_view(layer)
        hs = 16
        handle = QRectF(vr.right() - hs, vr.bottom() - hs, hs, hs)
        return handle.contains(view_pos)

    def mousePressEvent(self, event):
        if not self.interactive:
            return
        if event.button() != Qt.LeftButton:
            return
        vp = QPointF(event.pos())
        cp = self._view_to_canvas(vp)
        hit = self._layer_hit_test(cp)
        if not hit:
            self.drag_mode = "none"
            return
        self.active_layer = hit
        if self._handle_hit_test(hit, vp):
            self.drag_mode = "resize"
            self.resize_anchor = self.layers[hit].topLeft()
            rect = self.layers[hit]
            self.resize_aspect_ratio = max(0.01, rect.width() / max(1.0, rect.height()))
        else:
            self.drag_mode = "move"
            self.drag_offset = cp - self.layers[hit].topLeft()
        self.update()

    def mouseMoveEvent(self, event):
        if not self.interactive:
            return
        if self.drag_mode == "none":
            return
        cp = self._view_to_canvas(QPointF(event.pos()))
        r = QRectF(self.layers[self.active_layer])
        min_size = 40
        if self.drag_mode == "move":
            new_tl = cp - self.drag_offset
            r.moveTo(new_tl)
        elif self.drag_mode == "resize":
            anchor = self.resize_anchor
            new_w = max(min_size, cp.x() - anchor.x())
            new_h = max(min_size, cp.y() - anchor.y())
            if self.keep_aspect:
                ratio = max(0.01, self.resize_aspect_ratio)
                # подгоняем размер с сохранением пропорций текущего слоя
                if abs(new_w / ratio - new_h) > abs(new_h * ratio - new_w):
                    new_h = max(min_size, new_w / ratio)
                else:
                    new_w = max(min_size, new_h * ratio)
            r = QRectF(anchor.x(), anchor.y(), new_w, new_h)

        if r.x() < 0:
            r.moveLeft(0)
        if r.y() < 0:
            r.moveTop(0)
        if r.right() > self.canvas_w:
            r.moveRight(self.canvas_w)
        if r.bottom() > self.canvas_h:
            r.moveBottom(self.canvas_h)

        self.layers[self.active_layer] = r
        self.changed.emit(self.get_layers())
        self.update()

    def mouseReleaseEvent(self, event):
        if not self.interactive:
            return
        self.drag_mode = "none"

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(32, 32, 32) if self.dark_theme else QColor(245, 247, 250))
        area = self._draw_area()

        p.fillRect(area, QColor(20, 20, 20) if self.dark_theme else QColor(255, 255, 255))
        if self.background and not self.background.isNull():
            bg = self.background.scaled(int(area.width()), int(area.height()), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            p.drawPixmap(int(area.x()), int(area.y()), bg)

        p.setPen(QPen(QColor(120, 120, 120) if self.dark_theme else QColor(210, 214, 220), 1))
        p.drawRect(area)

        if self.show_layer_overlay:
            layer_order = ["game", "webcam"] if self.webcam_enabled else ["game"]
            for name in layer_order:
                r = self.layers[name]
                tl = self._canvas_to_view(r.topLeft())
                br = self._canvas_to_view(r.bottomRight())
                vr = QRectF(tl, br)

                color = self.colors[name]
                p.setPen(QPen(color, 2 if name == self.active_layer else 1, Qt.SolidLine))
                fill = QColor(color)
                fill.setAlpha(50)
                p.fillRect(vr, fill)
                p.drawRect(vr)
                p.drawText(vr.adjusted(6, 6, -6, -6), Qt.AlignLeft | Qt.AlignTop, name.upper())

                hs = 12
                handle = QRectF(vr.right() - hs, vr.bottom() - hs, hs, hs)
                p.fillRect(handle, color)

    def _clamp_all(self):
        for k in self.layers:
            r = self.layers[k]
            if r.width() < 20:
                r.setWidth(20)
            if r.height() < 20:
                r.setHeight(20)
            if r.x() < 0:
                r.moveLeft(0)
            if r.y() < 0:
                r.moveTop(0)
            if r.right() > self.canvas_w:
                r.moveRight(self.canvas_w)
            if r.bottom() > self.canvas_h:
                r.moveBottom(self.canvas_h)
            self.layers[k] = r


class TimeRangeSlider(QWidget):
    rangeChanged = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.minimum = 0
        self.maximum = 100
        self.start_value = 0
        self.end_value = 100
        self._active_handle = None  # "start" | "end"
        self.setMinimumHeight(36)

    def set_bounds(self, minimum: int, maximum: int):
        self.minimum = max(0, int(minimum))
        self.maximum = max(self.minimum + 1, int(maximum))
        self.set_values(self.start_value, self.end_value, emit_signal=False)
        self.update()

    def set_values(self, start_value: int, end_value: int, emit_signal: bool = True):
        s = max(self.minimum, min(int(start_value), self.maximum - 1))
        e = max(s + 1, min(int(end_value), self.maximum))
        changed = (s != self.start_value) or (e != self.end_value)
        self.start_value, self.end_value = s, e
        if changed:
            self.update()
            if emit_signal:
                self.rangeChanged.emit(self.start_value, self.end_value)

    def _track_rect(self) -> QRect:
        margin = 12
        h = 8
        y = (self.height() - h) // 2
        return QRect(margin, y, max(30, self.width() - margin * 2), h)

    def _value_to_x(self, value: int) -> int:
        tr = self._track_rect()
        ratio = (value - self.minimum) / max(1, self.maximum - self.minimum)
        return tr.x() + int(round(ratio * tr.width()))

    def _x_to_value(self, x: int) -> int:
        tr = self._track_rect()
        clamped_x = max(tr.left(), min(tr.right(), x))
        ratio = (clamped_x - tr.x()) / max(1, tr.width())
        return self.minimum + int(round(ratio * (self.maximum - self.minimum)))

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        x = event.pos().x()
        sx = self._value_to_x(self.start_value)
        ex = self._value_to_x(self.end_value)
        self._active_handle = "start" if abs(x - sx) <= abs(x - ex) else "end"
        self._apply_drag_x(x)

    def mouseMoveEvent(self, event):
        if not self._active_handle:
            return
        self._apply_drag_x(event.pos().x())

    def mouseReleaseEvent(self, event):
        self._active_handle = None

    def _apply_drag_x(self, x: int):
        v = self._x_to_value(x)
        if self._active_handle == "start":
            self.set_values(v, self.end_value)
        elif self._active_handle == "end":
            self.set_values(self.start_value, v)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        tr = self._track_rect()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(120, 120, 120, 90))
        p.drawRoundedRect(tr, 4, 4)

        sx = self._value_to_x(self.start_value)
        ex = self._value_to_x(self.end_value)
        sel = QRect(min(sx, ex), tr.y(), max(2, abs(ex - sx)), tr.height())
        p.setBrush(QColor(59, 130, 246, 190))
        p.drawRoundedRect(sel, 4, 4)

        p.setBrush(QColor(245, 245, 245))
        p.setPen(QPen(QColor(70, 70, 70), 1))
        r = 7
        p.drawEllipse(QPointF(sx, tr.center().y()), r, r)
        p.drawEllipse(QPointF(ex, tr.center().y()), r, r)


class AutoShortsInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAcceptDrops(True)

        self.video_path: str = ""
        self.candidates: List[Dict] = []
        self.rendered_files: List[str] = []
        self.last_output_dir: str = ""
        self.templates_dir = APPDATA_PATH / "shorts_templates"
        self.template_state_path = APPDATA_PATH / "shorts_template_state.json"
        self.template_path = self._resolve_startup_template_path()
        self.source_width = 1920
        self.source_height = 1080
        self.source_frame_pixmap = QPixmap()
        self.video_duration_s = 0
        self.original_video_duration_s = 0
        self.selected_start_s = 0
        self.selected_end_s = 1
        self.asr_payload: Dict = {}
        self._autonomous_run = False
        self._active_progress_stage = 0
        self._fx_layer_cache: Dict[tuple, QPixmap] = {}

        self._fx_preview_timer = QTimer(self)
        self._fx_preview_timer.setSingleShot(True)
        self._fx_preview_timer.setInterval(24)
        self._fx_preview_timer.timeout.connect(self._render_fx_preview_now)

        self._init_ui()
        self._apply_theme_style()

    def _init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_area = ScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.scroll_content = QWidget(self)
        self.main_layout = QVBoxLayout(self.scroll_content)
        self.main_layout.setSpacing(14)
        self.main_layout.setContentsMargins(10, 10, 10, 10)

        self.command_bar = CommandBar(self)
        self.command_bar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.command_bar.setFixedHeight(40)
        self.open_file_action = Action(FluentIcon.FOLDER, "Выбрать видео")
        self.open_file_action.triggered.connect(self._on_file_select)
        self.command_bar.addAction(self.open_file_action)
        self.main_layout.addWidget(self.command_bar)

        self.video_card = CardWidget(self)
        self.video_card.setObjectName("shortsVideoCard")
        video_layout = QHBoxLayout(self.video_card)
        self.video_label = BodyLabel("Файл не выбран")
        self.open_folder_btn = PushButton("Открыть папку")
        self.open_folder_btn.clicked.connect(self._open_output_folder)
        video_layout.addWidget(self.video_label)
        video_layout.addStretch(1)
        video_layout.addWidget(self.open_folder_btn)
        self.main_layout.addWidget(self.video_card)

        self.stage_card = CardWidget(self)
        self.stage_card.setObjectName("shortsStageCard")
        stage_layout = QVBoxLayout(self.stage_card)
        stage_layout.addWidget(BodyLabel("Этапы:"))
        self.stage_1 = QLabel("1) Whisper: разбор речи")
        self.stage_2 = QLabel("2) LLM: отбор кандидатов")
        self.stage_3 = QLabel("3) Проверка и выбор кандидатов")
        self.stage_4 = QLabel("4) Рендер шортсов")
        for w in [self.stage_1, self.stage_2, self.stage_3, self.stage_4]:
            stage_layout.addWidget(w)
        self.main_layout.addWidget(self.stage_card)

        self.control_card = QWidget(self)
        control_layout = QVBoxLayout(self.control_card)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(10)

        title_row = QHBoxLayout()
        title_row.addWidget(StrongBodyLabel("Пайплайн генерации (строго по этапам)"))
        title_row.addStretch(1)
        control_layout.addLayout(title_row)

        template_actions_top = QHBoxLayout()
        template_actions_top.addWidget(BodyLabel("Пресет:"))
        self.template_preset_combo = ComboBox(self)
        self.template_preset_combo.setMinimumWidth(260)
        self.template_preset_combo.currentTextChanged.connect(self._on_template_preset_selected)
        template_actions_top.addWidget(self.template_preset_combo)
        self.save_preset_quick_btn = PushButton("Сохранить/обновить пресет")
        self.save_preset_quick_btn.clicked.connect(self._save_template_as_named_preset)
        template_actions_top.addWidget(self.save_preset_quick_btn)
        self.reset_template_btn = PushButton("Сбросить пресет")
        self.reset_template_btn.clicked.connect(self._reset_layout_template)
        template_actions_top.addWidget(self.reset_template_btn)
        template_actions_top.addStretch(1)
        control_layout.addLayout(template_actions_top)

        template_scope_hint_top = BodyLabel(
            "Пресет сохраняет: кроп/позиции WEBCAM+GAME, 2-слойный режим, цветокоррекцию, "
            "параметры этапов (диапазон, длительность, лимиты кандидатов, анти-дубль, склейку речи) "
            "и настройки рендера. Не сохраняет: выбранные кандидаты и текущий список шортсов."
        )
        template_scope_hint_top.setWordWrap(True)
        control_layout.addWidget(template_scope_hint_top)

        stage1_card = CardWidget(self)
        stage1_card.setObjectName("shortsStage1Card")
        stage1_layout = QVBoxLayout(stage1_card)
        stage1_layout.setContentsMargins(12, 12, 12, 12)
        stage1_layout.setSpacing(8)

        stage1_layout.addWidget(StrongBodyLabel("Этап 1: Whisper (распознавание речи)"))
        stage1_hint = BodyLabel("Применяется на этапе 1: параметры ниже влияют только на распознавание и диапазон анализа.")
        stage1_hint.setWordWrap(True)
        stage1_layout.addWidget(stage1_hint)

        stage1_layout.addWidget(BodyLabel("Диапазон анализа (всегда активен):"))
        self.range_slider = TimeRangeSlider(self)
        self.range_slider.set_bounds(0, 1)
        self.range_slider.set_values(0, 1, emit_signal=False)
        self.range_slider.rangeChanged.connect(self._on_range_slider_changed)
        stage1_layout.addWidget(self.range_slider)

        range_info_row = QHBoxLayout()
        self.range_start_label = BodyLabel("С: 00:00")
        self.range_end_label = BodyLabel("До: 00:01")
        self.range_span_label = BodyLabel("Длительность: 00:01")
        range_info_row.addWidget(self.range_start_label)
        range_info_row.addStretch(1)
        range_info_row.addWidget(self.range_end_label)
        range_info_row.addStretch(1)
        range_info_row.addWidget(self.range_span_label)
        stage1_layout.addLayout(range_info_row)

        hint = BodyLabel(
            "По умолчанию выбран весь ролик. Перемещайте левый и правый маркеры, чтобы ограничить промежуток анализа."
        )
        hint.setWordWrap(True)
        stage1_layout.addWidget(hint)

        stage1_actions = QHBoxLayout()
        self.transcribe_btn = PrimaryPushButton("1) Запустить Whisper")
        self.transcribe_btn.clicked.connect(self._start_transcribe)
        self.autonomous_checkbox = CheckBox("Полностью автономно")
        self.run_all_btn = PushButton("Запустить все этапы подряд")
        self.run_all_btn.clicked.connect(self._start_full_pipeline)
        stage1_actions.addWidget(self.transcribe_btn)
        stage1_actions.addWidget(self.autonomous_checkbox)
        stage1_actions.addStretch(1)
        stage1_actions.addWidget(self.run_all_btn)
        stage1_layout.addLayout(stage1_actions)

        self.stage1_progress_wrap = QWidget(self)
        stage1_progress_layout = QVBoxLayout(self.stage1_progress_wrap)
        stage1_progress_layout.setContentsMargins(0, 0, 0, 0)
        self.stage1_progress_bar = ProgressBar(self)
        self.stage1_progress_bar.setRange(0, 100)
        self.stage1_progress_bar.setValue(0)
        self.stage1_progress_label = BodyLabel("Этап 1: ожидание")
        stage1_progress_layout.addWidget(self.stage1_progress_bar)
        stage1_progress_layout.addWidget(self.stage1_progress_label)
        self.stage1_progress_wrap.setVisible(False)
        stage1_layout.addWidget(self.stage1_progress_wrap)

        control_layout.addWidget(stage1_card)
        control_layout.addSpacing(10)

        stage2_card = CardWidget(self)
        stage2_card.setObjectName("shortsStage2Card")
        stage2_layout = QVBoxLayout(stage2_card)
        stage2_layout.setContentsMargins(12, 12, 12, 12)
        stage2_layout.setSpacing(8)

        stage2_layout.addWidget(StrongBodyLabel("Этап 2: Отбор кандидатов (LLM/эвристика)"))
        self.llm_tokens_hint_label = BodyLabel("Оценка токенов LLM: появится после этапа 1 (Whisper)")
        self.llm_tokens_hint_label.setWordWrap(True)
        stage2_layout.addWidget(self.llm_tokens_hint_label)
        stage2_layout.addSpacing(8)

        duration_row = QHBoxLayout()
        min_duration_label = BodyLabel("Мин. длительность шортса (сек):")
        min_duration_label.setToolTip("Минимальная длина одного шортса. Слишком малые значения чаще дают обрывки фраз.")
        duration_row.addWidget(min_duration_label)
        self.min_duration = SpinBox(self)
        self.min_duration.setRange(8, 120)
        self.min_duration.setValue(22)
        self.min_duration.setToolTip("Минимальная длина шортса в секундах")
        duration_row.addWidget(self.min_duration)

        max_duration_label = BodyLabel("Макс. длительность шортса (сек):")
        max_duration_label.setToolTip("Максимальная длина одного шортса. Большие значения дают меньше, но длиннее клипов.")
        duration_row.addWidget(max_duration_label)
        self.max_duration = SpinBox(self)
        self.max_duration.setRange(20, 300)
        self.max_duration.setValue(110)
        self.max_duration.setToolTip("Максимальная длина шортса в секундах")
        duration_row.addWidget(self.max_duration)
        duration_row.addStretch(1)
        stage2_layout.addLayout(duration_row)

        stage2_layout.addSpacing(6)
        tune_title = StrongBodyLabel("Тонкая настройка монтажа")
        stage2_layout.addWidget(tune_title)

        anti_repeat_title = BodyLabel("1) Повторы и разнообразие")
        anti_repeat_title.setToolTip("Группа фильтров, которые уменьшают повторы и повышают разнообразие найденных моментов")
        stage2_layout.addWidget(anti_repeat_title)

        anti_repeat_row = QHBoxLayout()
        repeat_similarity_label = BodyLabel("Анти-дубль (%):")
        repeat_similarity_label.setToolTip(
            "Порог схожести кандидатов.\n"
            "Ниже — больше похожих моментов, выше — больше разнообразия."
        )
        anti_repeat_row.addWidget(repeat_similarity_label)
        self.repeat_similarity_spin = SpinBox(self)
        self.repeat_similarity_spin.setRange(40, 100)
        self.repeat_similarity_spin.setValue(int(cfg.get(cfg.auto_shorts_repeat_similarity_percent)))
        self.repeat_similarity_spin.setToolTip(
            "Порог схожести кандидатов.\n"
            "Ниже — больше похожих моментов, выше — больше разнообразия."
        )
        self.repeat_similarity_spin.valueChanged.connect(
            lambda v: cfg.set(cfg.auto_shorts_repeat_similarity_percent, int(v))
        )
        anti_repeat_row.addWidget(self.repeat_similarity_spin)
        anti_repeat_row.addStretch(1)
        stage2_layout.addLayout(anti_repeat_row)

        candidates_limit_title = BodyLabel("1.1) Количество кандидатов")
        candidates_limit_title.setToolTip("Ограничение числа найденных фрагментов после этапа отбора")
        stage2_layout.addWidget(candidates_limit_title)

        candidates_limit_row = QHBoxLayout()
        min_candidates_label = BodyLabel("Мин. кандидатов:")
        min_candidates_label.setToolTip("Минимальное число кандидатов после отбора")
        candidates_limit_row.addWidget(min_candidates_label)
        self.min_candidates_spin = SpinBox(self)
        self.min_candidates_spin.setRange(1, 300)
        self.min_candidates_spin.setValue(8)
        self.min_candidates_spin.setToolTip("Минимальное число кандидатов после отбора")
        candidates_limit_row.addWidget(self.min_candidates_spin)

        max_candidates_label = BodyLabel("Макс. кандидатов:")
        max_candidates_label.setToolTip("Максимальное число кандидатов после отбора")
        candidates_limit_row.addWidget(max_candidates_label)
        self.max_candidates_spin = SpinBox(self)
        self.max_candidates_spin.setRange(2, 500)
        self.max_candidates_spin.setValue(40)
        self.max_candidates_spin.setToolTip("Максимальное число кандидатов после отбора")
        candidates_limit_row.addWidget(self.max_candidates_spin)

        self.recommend_candidates_btn = PushButton("Рекомендовать по длине")
        self.recommend_candidates_btn.clicked.connect(self._apply_recommended_candidate_limits)
        candidates_limit_row.addWidget(self.recommend_candidates_btn)
        candidates_limit_row.addStretch(1)
        stage2_layout.addLayout(candidates_limit_row)

        filter_row = QHBoxLayout()
        self.auto_filter_weak_checkbox = CheckBox("Авто-фильтр слабых кандидатов")
        self.auto_filter_weak_checkbox.setChecked(True)
        self.auto_filter_weak_checkbox.setToolTip(
            "Автоматически убирает слабые моменты с низким качеством/хуком и длинными паузами"
        )
        filter_row.addWidget(self.auto_filter_weak_checkbox)
        filter_row.addStretch(1)
        stage2_layout.addLayout(filter_row)

        filter_profile_row = QHBoxLayout()
        filter_profile_label = BodyLabel("Профиль авто-фильтра:")
        filter_profile_label.setToolTip(
            "Мягкий — больше кандидатов, но больше слабых.\n"
            "Сбалансированный — стандартный режим.\n"
            "Строгий — максимум качества, но кандидатов будет меньше."
        )
        filter_profile_row.addWidget(filter_profile_label)
        self.auto_filter_profile_combo = ComboBox(self)
        self.auto_filter_profile_combo.addItems(["Мягкий", "Сбалансированный", "Строгий"])
        self.auto_filter_profile_combo.setCurrentText("Сбалансированный")
        self.auto_filter_profile_combo.setToolTip(
            "Профиль строгости авто-фильтра слабых кандидатов"
        )
        filter_profile_row.addWidget(self.auto_filter_profile_combo)
        filter_profile_row.addStretch(1)
        stage2_layout.addLayout(filter_profile_row)

        llm_density_title = BodyLabel("1.2) Плотность отбора LLM")
        llm_density_title.setToolTip("Настройки, которые увеличивают/уменьшают число кандидатов без ослабления анти-дубля")
        stage2_layout.addWidget(llm_density_title)

        llm_density_row = QHBoxLayout()
        llm_intensity_label = BodyLabel("Интенсивность LLM-поиска:")
        llm_intensity_label.setToolTip(
            "1 — экономный режим (крупные пакеты, меньше вариантов).\n"
            "5 — плотный режим (больше пакетов и вариантов на пакет)."
        )
        llm_density_row.addWidget(llm_intensity_label)
        self.llm_search_intensity_spin = SpinBox(self)
        self.llm_search_intensity_spin.setRange(1, 5)
        self.llm_search_intensity_spin.setValue(3)
        self.llm_search_intensity_spin.setToolTip(
            "Управляет количеством анализируемых пакетов и целевым числом кандидатов от LLM"
        )
        self.llm_search_intensity_spin.valueChanged.connect(lambda _v: self._update_llm_token_estimate())
        llm_density_row.addWidget(self.llm_search_intensity_spin)

        interest_label = BodyLabel("Порог интереса (quality %):")
        interest_label.setToolTip(
            "Минимальный quality_score кандидата после ранжирования.\n"
            "Ниже значение — больше кандидатов, выше — строже качество."
        )
        llm_density_row.addWidget(interest_label)
        self.interest_threshold_spin = SpinBox(self)
        self.interest_threshold_spin.setRange(30, 95)
        self.interest_threshold_spin.setValue(52)
        self.interest_threshold_spin.setToolTip(
            "Порог интереса для финального списка кандидатов"
        )
        llm_density_row.addWidget(self.interest_threshold_spin)
        llm_density_row.addStretch(1)
        stage2_layout.addLayout(llm_density_row)

        llm_density_hint = BodyLabel(
            "Рекомендуемо для длинных роликов (40–120 мин): интенсивность 4–5, "
            "порог интереса 45–58, профиль фильтра «Сбалансированный» или «Мягкий»."
        )
        llm_density_hint.setWordWrap(True)
        stage2_layout.addWidget(llm_density_hint)

        self.min_candidates_spin.valueChanged.connect(self._on_candidate_limits_changed)
        self.max_candidates_spin.valueChanged.connect(self._on_candidate_limits_changed)

        stage2_layout.addSpacing(8)
        anti_cut_title = BodyLabel("2) Границы фраз и паузы (чтобы не резало слова)")
        anti_cut_title.setToolTip("Параметры склейки речи и отступов, чтобы фразы не обрезались на стыках")
        stage2_layout.addWidget(anti_cut_title)

        tune_row_1 = QHBoxLayout()

        speech_coverage_label = BodyLabel("Покрытие речи (%):")
        speech_coverage_label.setToolTip(
            "Минимальная доля времени клипа, где есть речь.\n"
            "Пример: 70% = минимум ~70% длительности занимают реплики, а не паузы/тишина."
        )
        tune_row_1.addWidget(speech_coverage_label)
        self.speech_min_coverage_spin = SpinBox(self)
        self.speech_min_coverage_spin.setRange(30, 100)
        self.speech_min_coverage_spin.setValue(int(cfg.get(cfg.auto_shorts_speech_min_coverage_percent)))
        self.speech_min_coverage_spin.setToolTip(
            "Минимальная доля речи в клипе.\n"
            "Если речи меньше этого порога, алгоритм избегает агрессивной склейки и берёт более цельный фрагмент."
        )
        self.speech_min_coverage_spin.valueChanged.connect(
            lambda v: cfg.set(cfg.auto_shorts_speech_min_coverage_percent, int(v))
        )
        tune_row_1.addWidget(self.speech_min_coverage_spin)

        speech_merge_label = BodyLabel("Склейка пауз (мс):")
        speech_merge_label.setToolTip("Максимальная пауза между репликами для их склейки в один кусок")
        tune_row_1.addWidget(speech_merge_label)
        self.speech_merge_gap_spin = SpinBox(self)
        self.speech_merge_gap_spin.setRange(60, 2000)
        self.speech_merge_gap_spin.setValue(int(cfg.get(cfg.auto_shorts_speech_merge_gap_ms)))
        self.speech_merge_gap_spin.setToolTip("Максимальная пауза между репликами для их склейки в один кусок")
        self.speech_merge_gap_spin.valueChanged.connect(
            lambda v: cfg.set(cfg.auto_shorts_speech_merge_gap_ms, int(v))
        )
        tune_row_1.addWidget(self.speech_merge_gap_spin)
        tune_row_1.addStretch(1)
        stage2_layout.addLayout(tune_row_1)

        tune_row_2 = QHBoxLayout()
        speech_pre_pad_label = BodyLabel("Перед словом (мс):")
        speech_pre_pad_label.setToolTip("Сколько миллисекунд добавить перед началом речи")
        tune_row_2.addWidget(speech_pre_pad_label)
        self.speech_pre_pad_spin = SpinBox(self)
        self.speech_pre_pad_spin.setRange(0, 1500)
        self.speech_pre_pad_spin.setValue(int(cfg.get(cfg.auto_shorts_speech_pre_pad_ms)))
        self.speech_pre_pad_spin.setToolTip("Сколько миллисекунд добавить перед началом речи")
        self.speech_pre_pad_spin.valueChanged.connect(
            lambda v: cfg.set(cfg.auto_shorts_speech_pre_pad_ms, int(v))
        )
        tune_row_2.addWidget(self.speech_pre_pad_spin)

        speech_post_pad_label = BodyLabel("После слова (мс):")
        speech_post_pad_label.setToolTip("Сколько миллисекунд добавить после окончания речи")
        tune_row_2.addWidget(speech_post_pad_label)
        self.speech_post_pad_spin = SpinBox(self)
        self.speech_post_pad_spin.setRange(0, 2000)
        self.speech_post_pad_spin.setValue(int(cfg.get(cfg.auto_shorts_speech_post_pad_ms)))
        self.speech_post_pad_spin.setToolTip("Сколько миллисекунд добавить после окончания речи")
        self.speech_post_pad_spin.valueChanged.connect(
            lambda v: cfg.set(cfg.auto_shorts_speech_post_pad_ms, int(v))
        )
        tune_row_2.addWidget(self.speech_post_pad_spin)

        clip_head_pad_label = BodyLabel("До клипа (мс):")
        clip_head_pad_label.setToolTip("Общий отступ в начале готового клипа")
        tune_row_2.addWidget(clip_head_pad_label)
        self.clip_head_pad_spin = SpinBox(self)
        self.clip_head_pad_spin.setRange(0, 2000)
        self.clip_head_pad_spin.setValue(int(cfg.get(cfg.auto_shorts_clip_head_pad_ms)))
        self.clip_head_pad_spin.setToolTip("Общий отступ в начале готового клипа")
        self.clip_head_pad_spin.valueChanged.connect(
            lambda v: cfg.set(cfg.auto_shorts_clip_head_pad_ms, int(v))
        )
        tune_row_2.addWidget(self.clip_head_pad_spin)

        clip_tail_pad_label = BodyLabel("После клипа (мс):")
        clip_tail_pad_label.setToolTip("Общий отступ в конце готового клипа")
        tune_row_2.addWidget(clip_tail_pad_label)
        self.clip_tail_pad_spin = SpinBox(self)
        self.clip_tail_pad_spin.setRange(0, 3000)
        self.clip_tail_pad_spin.setValue(int(cfg.get(cfg.auto_shorts_clip_tail_pad_ms)))
        self.clip_tail_pad_spin.setToolTip("Общий отступ в конце готового клипа")
        self.clip_tail_pad_spin.valueChanged.connect(
            lambda v: cfg.set(cfg.auto_shorts_clip_tail_pad_ms, int(v))
        )
        tune_row_2.addWidget(self.clip_tail_pad_spin)
        tune_row_2.addStretch(1)
        stage2_layout.addLayout(tune_row_2)

        stage2_layout.addSpacing(8)
        stage2_actions = QHBoxLayout()
        self.select_candidates_btn = PrimaryPushButton("2) Отобрать кандидатов")
        self.select_candidates_btn.clicked.connect(self._start_candidate_selection)
        stage2_actions.addWidget(self.select_candidates_btn)
        stage2_actions.addStretch(1)
        stage2_layout.addLayout(stage2_actions)

        self.stage2_progress_wrap = QWidget(self)
        stage2_progress_layout = QVBoxLayout(self.stage2_progress_wrap)
        stage2_progress_layout.setContentsMargins(0, 0, 0, 0)
        self.stage2_progress_bar = ProgressBar(self)
        self.stage2_progress_bar.setRange(0, 100)
        self.stage2_progress_bar.setValue(0)
        self.stage2_progress_label = BodyLabel("Этап 2: ожидание")
        stage2_progress_layout.addWidget(self.stage2_progress_bar)
        stage2_progress_layout.addWidget(self.stage2_progress_label)
        self.stage2_progress_wrap.setVisible(False)
        stage2_layout.addWidget(self.stage2_progress_wrap)

        control_layout.addWidget(stage2_card)

        self.main_layout.addWidget(self.control_card)

        self.template_card = CardWidget(self)
        self.template_card.setObjectName("shortsTemplateCard")
        template_layout = QVBoxLayout(self.template_card)
        template_layout.addWidget(StrongBodyLabel("Наглядный пресет монтажа"))
        frame_row = QHBoxLayout()
        frame_row.addWidget(BodyLabel("Кадр предпросмотра (сек):"))
        self.preview_time_s = SpinBox(self)
        self.preview_time_s.setRange(0, 24 * 3600)
        self.preview_time_s.setValue(2)
        frame_row.addWidget(self.preview_time_s)
        self.refresh_preview_btn = PushButton("Обновить кадр")
        self.refresh_preview_btn.clicked.connect(self._reload_preview_frame)
        frame_row.addWidget(self.refresh_preview_btn)
        self.exact_fx_preview_btn = PushButton("Точный FX-кадр (FFmpeg)")
        self.exact_fx_preview_btn.clicked.connect(self._render_exact_fx_preview_frame)
        frame_row.addWidget(self.exact_fx_preview_btn)
        frame_row.addStretch(1)
        self.keep_aspect_checkbox = CheckBox("Сохранять пропорции")
        self.keep_aspect_checkbox.setChecked(True)
        self.keep_aspect_checkbox.stateChanged.connect(self._on_keep_aspect_changed)
        frame_row.addWidget(self.keep_aspect_checkbox)
        template_layout.addLayout(frame_row)

        row_tpl_actions = QHBoxLayout()
        self.dual_layer_enabled = CheckBox("Включить область веб-камеры")
        self.dual_layer_enabled.setChecked(True)
        self.dual_layer_enabled.stateChanged.connect(self._on_webcam_area_toggled)
        row_tpl_actions.addWidget(self.dual_layer_enabled)
        row_tpl_actions.addStretch(1)
        template_layout.addLayout(row_tpl_actions)

        template_layout.addWidget(BodyLabel("1) На исходном кадре перетяните и растяните области WEBCAM и GAME (кроп)."))
        self.source_preview = LayerPreviewWidget(1920, 1080, self)
        self.source_preview.set_keep_aspect(True)
        self.source_preview.setMinimumHeight(420)
        template_layout.addWidget(self.source_preview)

        template_layout.addWidget(BodyLabel("2) На вертикальном кадре 1080x1920 настройте монтаж слоёв."))
        montage_row = QHBoxLayout()
        montage_row.setSpacing(14)

        self.output_preview = LayerPreviewWidget(1080, 1920, self)
        self.output_preview.set_keep_aspect(True)
        self.output_preview.set_background(QPixmap())
        self.output_preview.setMinimumHeight(620)
        montage_row.addWidget(self.output_preview, 1)

        self.effects_preview = LayerPreviewWidget(1080, 1920, self)
        self.effects_preview.set_keep_aspect(True)
        self.effects_preview.set_background(QPixmap())
        self.effects_preview.setMinimumHeight(620)
        self.effects_preview.set_interactive(False)
        self.effects_preview.set_show_layer_overlay(False)
        montage_row.addWidget(self.effects_preview, 1)
        template_layout.addLayout(montage_row)

        fx_hint = BodyLabel("3) Предпросмотр эффектов справа: без выделяемых рамок, обновляется мгновенно при движении ползунков.")
        fx_hint.setWordWrap(True)
        template_layout.addWidget(fx_hint)
        fx_exact_hint = BodyLabel(
            "Кнопка «Точный FX-кадр (FFmpeg)» делает референсный кадр максимально близко к финальному рендеру "
            "(полезно перед массовым запуском)."
        )
        fx_exact_hint.setWordWrap(True)
        template_layout.addWidget(fx_exact_hint)

        self.source_preview.changed.connect(lambda _: self._on_layout_changed())
        self.output_preview.changed.connect(lambda _: self._on_layout_changed())
        self._on_webcam_area_toggled()

        fx_title_row = QHBoxLayout()
        fx_title = StrongBodyLabel("Цветокоррекция слоёв")
        fx_title_row.addWidget(fx_title)
        fx_title_row.addStretch(1)
        self.reset_fx_btn = PushButton("Сбросить FX")
        self.reset_fx_btn.clicked.connect(self._reset_fx_controls)
        fx_title_row.addWidget(self.reset_fx_btn)
        template_layout.addLayout(fx_title_row)

        webcam_title = StrongBodyLabel("WEBCAM")
        template_layout.addWidget(webcam_title)

        self.wc_brightness = SpinBox(self)
        self.wc_brightness.setRange(-50, 50)
        self.wc_brightness.setValue(0)
        self.wc_brightness.setAccelerated(True)
        self.wc_contrast = SpinBox(self)
        self.wc_contrast.setRange(50, 180)
        self.wc_contrast.setValue(100)
        self.wc_contrast.setAccelerated(True)
        self.wc_saturation = SpinBox(self)
        self.wc_saturation.setRange(50, 180)
        self.wc_saturation.setValue(100)
        self.wc_saturation.setAccelerated(True)
        self.wc_gamma = SpinBox(self)
        self.wc_gamma.setRange(70, 160)
        self.wc_gamma.setValue(100)
        self.wc_gamma.setAccelerated(True)
        self.wc_temperature = SpinBox(self)
        self.wc_temperature.setRange(-40, 40)
        self.wc_temperature.setValue(0)
        self.wc_temperature.setAccelerated(True)
        self.wc_brightness_slider = QSlider(Qt.Horizontal, self)
        self.wc_brightness_slider.setRange(-50, 50)
        self.wc_brightness_slider.setValue(0)
        self.wc_contrast_slider = QSlider(Qt.Horizontal, self)
        self.wc_contrast_slider.setRange(50, 180)
        self.wc_contrast_slider.setValue(100)
        self.wc_saturation_slider = QSlider(Qt.Horizontal, self)
        self.wc_saturation_slider.setRange(50, 180)
        self.wc_saturation_slider.setValue(100)
        self.wc_gamma_slider = QSlider(Qt.Horizontal, self)
        self.wc_gamma_slider.setRange(70, 160)
        self.wc_gamma_slider.setValue(100)
        self.wc_temperature_slider = QSlider(Qt.Horizontal, self)
        self.wc_temperature_slider.setRange(-40, 40)
        self.wc_temperature_slider.setValue(0)

        webcam_controls_row = QHBoxLayout()
        webcam_controls_row.setSpacing(16)

        def _add_fx_column(parent_row: QHBoxLayout, title: str, spin: SpinBox, slider: QSlider):
            container = QWidget(self)
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.setSpacing(6)

            head_row = QHBoxLayout()
            head_row.setContentsMargins(0, 0, 0, 0)
            head_row.setSpacing(6)
            head_row.addStretch(1)
            head_row.addWidget(BodyLabel(title))
            head_row.addWidget(spin)
            head_row.addStretch(1)

            slider.setMinimumWidth(200)

            container_layout.addLayout(head_row)
            container_layout.addWidget(slider)
            parent_row.addWidget(container)

        webcam_controls_row.addStretch(1)
        _add_fx_column(webcam_controls_row, "Ярк.", self.wc_brightness, self.wc_brightness_slider)
        _add_fx_column(webcam_controls_row, "Контр.", self.wc_contrast, self.wc_contrast_slider)
        _add_fx_column(webcam_controls_row, "Насыщ.", self.wc_saturation, self.wc_saturation_slider)
        _add_fx_column(webcam_controls_row, "Гамма", self.wc_gamma, self.wc_gamma_slider)
        _add_fx_column(webcam_controls_row, "Темп.", self.wc_temperature, self.wc_temperature_slider)
        webcam_controls_row.addStretch(1)
        template_layout.addLayout(webcam_controls_row)

        game_title = StrongBodyLabel("GAME")
        template_layout.addWidget(game_title)

        self.gm_brightness = SpinBox(self)
        self.gm_brightness.setRange(-50, 50)
        self.gm_brightness.setValue(0)
        self.gm_brightness.setAccelerated(True)
        self.gm_contrast = SpinBox(self)
        self.gm_contrast.setRange(50, 180)
        self.gm_contrast.setValue(100)
        self.gm_contrast.setAccelerated(True)
        self.gm_saturation = SpinBox(self)
        self.gm_saturation.setRange(50, 180)
        self.gm_saturation.setValue(100)
        self.gm_saturation.setAccelerated(True)
        self.gm_gamma = SpinBox(self)
        self.gm_gamma.setRange(70, 160)
        self.gm_gamma.setValue(100)
        self.gm_gamma.setAccelerated(True)
        self.gm_temperature = SpinBox(self)
        self.gm_temperature.setRange(-40, 40)
        self.gm_temperature.setValue(0)
        self.gm_temperature.setAccelerated(True)

        self.gm_brightness_slider = QSlider(Qt.Horizontal, self)
        self.gm_brightness_slider.setRange(-50, 50)
        self.gm_brightness_slider.setValue(0)
        self.gm_contrast_slider = QSlider(Qt.Horizontal, self)
        self.gm_contrast_slider.setRange(50, 180)
        self.gm_contrast_slider.setValue(100)
        self.gm_saturation_slider = QSlider(Qt.Horizontal, self)
        self.gm_saturation_slider.setRange(50, 180)
        self.gm_saturation_slider.setValue(100)
        self.gm_gamma_slider = QSlider(Qt.Horizontal, self)
        self.gm_gamma_slider.setRange(70, 160)
        self.gm_gamma_slider.setValue(100)
        self.gm_temperature_slider = QSlider(Qt.Horizontal, self)
        self.gm_temperature_slider.setRange(-40, 40)
        self.gm_temperature_slider.setValue(0)

        game_controls_row = QHBoxLayout()
        game_controls_row.setSpacing(16)
        game_controls_row.addStretch(1)
        _add_fx_column(game_controls_row, "Ярк.", self.gm_brightness, self.gm_brightness_slider)
        _add_fx_column(game_controls_row, "Контр.", self.gm_contrast, self.gm_contrast_slider)
        _add_fx_column(game_controls_row, "Насыщ.", self.gm_saturation, self.gm_saturation_slider)
        _add_fx_column(game_controls_row, "Гамма", self.gm_gamma, self.gm_gamma_slider)
        _add_fx_column(game_controls_row, "Темп.", self.gm_temperature, self.gm_temperature_slider)
        game_controls_row.addStretch(1)
        template_layout.addLayout(game_controls_row)

        self._link_fx_control_pair(self.wc_brightness, self.wc_brightness_slider)
        self._link_fx_control_pair(self.wc_contrast, self.wc_contrast_slider)
        self._link_fx_control_pair(self.wc_saturation, self.wc_saturation_slider)
        self._link_fx_control_pair(self.wc_gamma, self.wc_gamma_slider)
        self._link_fx_control_pair(self.wc_temperature, self.wc_temperature_slider)
        self._link_fx_control_pair(self.gm_brightness, self.gm_brightness_slider)
        self._link_fx_control_pair(self.gm_contrast, self.gm_contrast_slider)
        self._link_fx_control_pair(self.gm_saturation, self.gm_saturation_slider)
        self._link_fx_control_pair(self.gm_gamma, self.gm_gamma_slider)
        self._link_fx_control_pair(self.gm_temperature, self.gm_temperature_slider)

        self.stage3_card = CardWidget(self)
        self.stage3_card.setObjectName("shortsStage3Card")
        stage3_layout = QVBoxLayout(self.stage3_card)
        stage3_layout.addWidget(StrongBodyLabel("Этап 3: Проверка и ручной выбор кандидатов"))

        self.table = QTableWidget(self)
        self.table.setColumnCount(11)
        self.table.setHorizontalHeaderLabels(
            [
                "Выбрать",
                "Таймкод",
                "Длит.",
                "Score",
                "Q",
                "Grade",
                "Hook",
                "Pause",
                "Заголовок",
                "Вирусный заголовок",
                "Почему выбран",
            ]
        )
        self._apply_candidates_table_tooltips()
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(False)
        self.table.verticalHeader().setDefaultSectionSize(64)
        self.table.setSortingEnabled(True)
        stage3_layout.addWidget(self.table)
        self.main_layout.addWidget(self.stage3_card)

        # Монтаж применяется именно на этапе рендера,
        # поэтому блок наглядного пресета должен быть расположен
        # непосредственно перед этапом 4.
        self.main_layout.addWidget(self.template_card)

        self.stage4_card = CardWidget(self)
        self.stage4_card.setObjectName("shortsStage4Card")
        stage4_layout = QVBoxLayout(self.stage4_card)
        stage4_layout.addWidget(StrongBodyLabel("Этап 4: Рендер выбранных шортсов"))
        stage4_hint = BodyLabel("Параметры рендера ниже применяются только на этапе 4.")
        stage4_hint.setWordWrap(True)
        stage4_layout.addWidget(stage4_hint)

        sort_row = QHBoxLayout()
        sort_row.addWidget(BodyLabel("Порядок нумерации шортсов:"))
        self.render_sort_combo = ComboBox(self)
        self.render_sort_combo.addItems(
            [
                "По времени (раньше → позже)",
                "По времени (позже → раньше)",
                "По интересу (Q ↓)",
                "По общему score (↓)",
                "По hook (↓)",
                "По длительности (длинные сначала)",
                "По длительности (короткие сначала)",
            ]
        )
        self.render_sort_combo.setCurrentIndex(0)
        self.render_sort_combo.setToolTip(
            "Определяет порядок, в котором выбранные кандидаты будут пронумерованы как 001..N.\n"
            "Этот же порядок влияет на итоговые имена файлов шортсов."
        )
        sort_row.addWidget(self.render_sort_combo)
        sort_row.addStretch(1)
        stage4_layout.addLayout(sort_row)

        filename_hint = BodyLabel("Что включать в имя файла шортса:")
        filename_hint.setWordWrap(True)
        stage4_layout.addWidget(filename_hint)

        filename_row = QHBoxLayout()
        self.filename_include_title_checkbox = CheckBox("Заголовок", self)
        self.filename_include_title_checkbox.setChecked(True)
        self.filename_include_title_checkbox.setToolTip(
            "Добавлять в имя файла краткий вирусный заголовок (YouTube-стиль)."
        )
        self.filename_include_time_checkbox = CheckBox("Таймкод", self)
        self.filename_include_time_checkbox.setChecked(True)
        self.filename_include_time_checkbox.setToolTip(
            "Добавлять в имя файла диапазон времени фрагмента (в секундах)."
        )
        self.filename_include_quality_checkbox = CheckBox("Q (интерес)", self)
        self.filename_include_quality_checkbox.setChecked(False)
        self.filename_include_quality_checkbox.setToolTip(
            "Q = интегральная оценка интересности фрагмента (0..100)."
        )
        self.filename_include_score_checkbox = CheckBox("Score", self)
        self.filename_include_score_checkbox.setChecked(False)
        self.filename_include_score_checkbox.setToolTip(
            "Score = исходный базовый балл кандидата до финальных quality-правок."
        )
        self.filename_include_hook_checkbox = CheckBox("Hook", self)
        self.filename_include_hook_checkbox.setChecked(False)
        self.filename_include_hook_checkbox.setToolTip(
            "Hook = сила захвата внимания в начале фрагмента (0..10)."
        )
        self.filename_include_grade_checkbox = CheckBox("Grade", self)
        self.filename_include_grade_checkbox.setChecked(False)
        self.filename_include_grade_checkbox.setToolTip(
            "Grade = буквенная категория качества (A/B/C/D)."
        )
        filename_row.addWidget(self.filename_include_title_checkbox)
        filename_row.addWidget(self.filename_include_time_checkbox)
        filename_row.addWidget(self.filename_include_quality_checkbox)
        filename_row.addWidget(self.filename_include_score_checkbox)
        filename_row.addWidget(self.filename_include_hook_checkbox)
        filename_row.addWidget(self.filename_include_grade_checkbox)
        filename_row.addStretch(1)
        stage4_layout.addLayout(filename_row)

        self.bottom_card = QWidget(self)
        bottom_layout = QHBoxLayout(self.bottom_card)
        self.select_all_btn = PushButton("Выбрать все")
        self.select_all_btn.clicked.connect(self._select_all)
        self.clear_select_btn = PushButton("Снять выбор")
        self.clear_select_btn.clicked.connect(self._clear_select)
        self.render_backend_label = BodyLabel("Рендер:")
        self.render_backend_combo = ComboBox(self)
        self.render_backend_combo.addItems(["Auto", "CPU", "GPU", "CUDA"])
        self.render_backend_combo.setCurrentIndex(self._backend_to_index(str(cfg.auto_shorts_render_backend.value or "auto")))
        self.render_backend_combo.currentIndexChanged.connect(self._on_render_backend_changed)

        self.render_fps_label = BodyLabel("FPS:")
        self.render_fps_combo = ComboBox(self)
        self.render_fps_combo.addItems(["Исходный", "30", "60"])
        self.render_fps_combo.setCurrentIndex(1)

        self.render_resolution_label = BodyLabel("Разрешение:")
        self.render_resolution_combo = ComboBox(self)
        self.render_resolution_combo.addItems(["1080x1920", "720x1280", "1440x2560", "Исходное"])
        self.render_resolution_combo.setCurrentIndex(0)

        self.render_quality_label = BodyLabel("Качество:")
        self.render_quality_combo = ComboBox(self)
        self.render_quality_combo.addItems(["Высокое", "Сбалансированное", "Быстрое"])
        self.render_quality_combo.setCurrentIndex(1)


        self.render_btn = PrimaryPushButton("Сделать шортсы из выбранных")
        self.render_btn.clicked.connect(self._start_render)
        self.stop_render_btn = PushButton("Стоп")
        self.stop_render_btn.setEnabled(False)
        self.stop_render_btn.clicked.connect(self._stop_render)
        bottom_layout.addWidget(self.select_all_btn)
        bottom_layout.addWidget(self.clear_select_btn)
        bottom_layout.addWidget(self.render_backend_label)
        bottom_layout.addWidget(self.render_backend_combo)
        bottom_layout.addWidget(self.render_fps_label)
        bottom_layout.addWidget(self.render_fps_combo)
        bottom_layout.addWidget(self.render_resolution_label)
        bottom_layout.addWidget(self.render_resolution_combo)
        bottom_layout.addWidget(self.render_quality_label)
        bottom_layout.addWidget(self.render_quality_combo)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(self.stop_render_btn)
        bottom_layout.addWidget(self.render_btn)
        stage4_layout.addWidget(self.bottom_card)

        self.stage4_progress_wrap = QWidget(self)
        stage4_progress_layout = QVBoxLayout(self.stage4_progress_wrap)
        stage4_progress_layout.setContentsMargins(0, 0, 0, 0)
        self.stage4_progress_bar = ProgressBar(self)
        self.stage4_progress_bar.setRange(0, 100)
        self.stage4_progress_bar.setValue(0)
        self.stage4_progress_label = BodyLabel("Этап 4: ожидание")
        stage4_progress_layout.addWidget(self.stage4_progress_bar)
        stage4_progress_layout.addWidget(self.stage4_progress_label)
        self.stage4_progress_wrap.setVisible(False)
        stage4_layout.addWidget(self.stage4_progress_wrap)

        output_row = QHBoxLayout()
        self.output_hint_label = BodyLabel("Папка результата: пока нет")
        self.output_hint_label.setWordWrap(True)
        self.open_output_folder_btn = PushButton("Открыть папку с шортсами")
        self.open_output_folder_btn.clicked.connect(self._open_output_folder)
        output_row.addWidget(self.output_hint_label, 1)
        output_row.addWidget(self.open_output_folder_btn)
        stage4_layout.addLayout(output_row)

        self.rendered_card = QWidget(self)
        rendered_layout = QVBoxLayout(self.rendered_card)
        rendered_layout.addWidget(StrongBodyLabel("Готовые шортсы"))
        self.rendered_list = QListWidget(self)
        self.rendered_list.setSelectionMode(self.rendered_list.ExtendedSelection)
        rendered_layout.addWidget(self.rendered_list)

        rendered_actions = QHBoxLayout()
        self.open_selected_short_btn = PushButton("Открыть выбранный")
        self.open_selected_short_btn.clicked.connect(self._open_selected_rendered)
        self.send_to_batch_btn = PrimaryPushButton("Отправить выбранные в пакетные субтитры")
        self.send_to_batch_btn.clicked.connect(self._send_selected_to_batch)
        rendered_actions.addWidget(self.open_selected_short_btn)
        rendered_actions.addStretch(1)
        rendered_actions.addWidget(self.send_to_batch_btn)
        rendered_layout.addLayout(rendered_actions)
        stage4_layout.addWidget(self.rendered_card)
        self.main_layout.addWidget(self.stage4_card)

        self.main_layout.addStretch(1)

        self.scroll_area.setWidget(self.scroll_content)
        root_layout.addWidget(self.scroll_area)

        self._set_stage(0)
        self._set_active_progress_stage(0)
        self._refresh_template_preset_combo(select_path=self.template_path)
        # Важный UX-фикс: откладываем загрузку пресета до следующего цикла UI,
        # чтобы старт окна не подвисал на тяжёлой инициализации предпросмотра.
        QTimer.singleShot(0, self._load_initial_template_deferred)

    def _refresh_template_preset_combo(self, select_path: Optional[Path] = None):
        if not hasattr(self, "template_preset_combo"):
            return
        try:
            self.templates_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        combo = self.template_preset_combo
        combo.blockSignals(True)
        combo.clear()
        # Пустой пункт = режим создания нового пресета (аналогично вкладке стилей субтитров)
        combo.addItem("")

        template_files = sorted(self.templates_dir.glob("*.json"), key=lambda p: p.stem.lower())
        for p in template_files:
            combo.addItem(p.stem)

        selected = Path(select_path) if select_path else Path(self.template_path)
        selected_stem = selected.stem if selected else ""
        last_state = self._read_template_state()
        last_preset_name = str(last_state.get("last_preset_name", "") or "").strip()

        target_stem = selected_stem or last_preset_name
        if target_stem:
            idx = combo.findText(target_stem)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
        else:
            combo.setCurrentIndex(0)

        combo.blockSignals(False)

    def _on_template_preset_selected(self, preset_name: str):
        preset_name = (preset_name or "").strip()
        if not preset_name:
            return
        target = self.templates_dir / f"{preset_name}.json"
        if not target.exists():
            return
        self._load_layout_template(target, persist_last=True)
        self._persist_last_template_path(target, preset_name=preset_name)

    def _save_template_as_named_preset(self):
        selected_name = (self.template_preset_combo.currentText() or "").strip()
        if selected_name and (self.templates_dir / f"{selected_name}.json").exists():
            preset_name = selected_name
            mode = "updated"
        else:
            dialog = TemplatePresetNameDialog(self)
            if not dialog.exec():
                return
            preset_name = dialog.nameLineEdit.text().strip()
            if not preset_name:
                return
            mode = "created"
        safe_name = (
            preset_name.replace("\\", "_")
            .replace("/", "_")
            .replace(":", "_")
            .replace("*", "_")
            .replace("?", "_")
            .replace('"', "_")
            .replace("<", "_")
            .replace(">", "_")
            .replace("|", "_")
            .strip()
        )
        if not safe_name:
            return
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.template_path = self.templates_dir / f"{safe_name}.json"
        self._save_layout_template()
        self._refresh_template_preset_combo(select_path=self.template_path)
        InfoBar.success(
            "Пресет сохранён",
            (
                f"Обновлён пресет: {safe_name}"
                if mode == "updated"
                else f"Создан пресет: {safe_name}"
            ),
            duration=2200,
            position=InfoBarPosition.TOP,
            parent=self,
        )

    def _load_initial_template_deferred(self):
        try:
            self._load_layout_template()
        except Exception:
            pass

    def _set_active_progress_stage(self, stage_idx: int):
        self._active_progress_stage = int(stage_idx or 0)
        self.stage1_progress_wrap.setVisible(self._active_progress_stage == 1)
        self.stage2_progress_wrap.setVisible(self._active_progress_stage == 2)
        self.stage4_progress_wrap.setVisible(self._active_progress_stage == 4)

    def _set_stage_progress(self, value: int, message: str = ""):
        value = max(0, min(100, int(value)))
        if self._active_progress_stage == 1:
            self.stage1_progress_bar.setValue(value)
            if message:
                self.stage1_progress_label.setText(message)
        elif self._active_progress_stage == 2:
            self.stage2_progress_bar.setValue(value)
            if message:
                self.stage2_progress_label.setText(message)
        elif self._active_progress_stage == 4:
            self.stage4_progress_bar.setValue(value)
            if message:
                self.stage4_progress_label.setText(message)

    def _on_keep_aspect_changed(self):
        enabled = self.keep_aspect_checkbox.isChecked()
        self.source_preview.set_keep_aspect(enabled)
        self.output_preview.set_keep_aspect(enabled)
        self.effects_preview.set_keep_aspect(enabled)

    def _on_layout_changed(self):
        self._refresh_output_composite_preview()

    def _on_webcam_area_toggled(self):
        enabled = bool(self.dual_layer_enabled.isChecked()) if hasattr(self, "dual_layer_enabled") else True
        if hasattr(self, "source_preview"):
            self.source_preview.set_webcam_enabled(enabled)
        if hasattr(self, "output_preview"):
            self.output_preview.set_webcam_enabled(enabled)
        self._refresh_output_composite_preview()

    def _apply_theme_style(self):
        p = get_theme_palette()
        dark = bool(p.get("is_dark"))
        self.source_preview.set_theme(dark)
        self.output_preview.set_theme(dark)
        self.effects_preview.set_theme(dark)

        row_alt_bg = "#252526" if dark else "#F7F7F7"
        selected_bg = p["accent"]
        selected_fg = "#FFFFFF" if dark else "#FFFFFF"

        self.setStyleSheet(
            f"""
            QWidget#AutoShortsInterface {{ background: {p['window_bg']}; }}
            QScrollArea {{ border: none; background: transparent; }}
            QScrollArea > QWidget > QWidget {{ background: {p['window_bg']}; }}
            QLabel, BodyLabel, StrongBodyLabel {{ color: {p['text']}; }}
            CardWidget {{ border-radius: 10px; }}
            CardWidget#shortsVideoCard, CardWidget#shortsStageCard,
            CardWidget#shortsTemplateCard, CardWidget#shortsStage1Card,
            CardWidget#shortsStage2Card, CardWidget#shortsStage3Card, CardWidget#shortsStage4Card {{
                background: {p['card_bg']};
                border: 1px solid {p['border']};
            }}
            QTableWidget {{ background: {p['card_bg']}; color: {p['text']}; gridline-color: {p['border']}; }}
            QTableWidget::item {{ background: {p['card_bg']}; color: {p['text']}; }}
            QTableWidget::item:alternate {{ background: {row_alt_bg}; color: {p['text']}; }}
            QHeaderView::section {{ background: {p['panel_bg']}; color: {p['text']}; border: none; padding: 4px; }}
            QTableWidget::item:selected {{ background: {selected_bg}; color: {selected_fg}; }}

            QScrollBar:vertical {{
                background: transparent;
                width: 12px;
                margin: 2px;
            }}
            QScrollBar::handle:vertical {{
                background: {p['border']};
                min-height: 40px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {p['accent']};
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {{
                background: transparent;
                height: 0px;
            }}

            QScrollBar:horizontal {{
                background: transparent;
                height: 12px;
                margin: 2px;
            }}
            QScrollBar::handle:horizontal {{
                background: {p['border']};
                min-width: 40px;
                border-radius: 6px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background: {p['accent']};
            }}
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal,
            QScrollBar::add-page:horizontal,
            QScrollBar::sub-page:horizontal {{
                background: transparent;
                width: 0px;
            }}
            """
        )

    def _set_stage(self, stage_idx: int):
        labels = [self.stage_1, self.stage_2, self.stage_3, self.stage_4]
        for i, label in enumerate(labels, start=1):
            if i < stage_idx:
                label.setText(f"✅ {label.text().split(' ', 1)[-1]}")
            elif i == stage_idx:
                label.setText(f"🔄 {label.text().split(' ', 1)[-1]}")
            else:
                label.setText(label.text().replace("✅ ", "").replace("🔄 ", ""))

    def _on_file_select(self):
        desktop_path = QStandardPaths.writableLocation(QStandardPaths.DesktopLocation)
        video_formats = " ".join(f"*.{fmt.value}" for fmt in SupportedVideoFormats)
        audio_formats = " ".join(f"*.{fmt.value}" for fmt in SupportedAudioFormats)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите видео/аудио",
            desktop_path,
            f"Медиафайлы ({video_formats} {audio_formats})",
        )
        if file_path:
            self.video_path = file_path
            self.video_label.setText(file_path)
            self.asr_payload = {}
            self.candidates = []
            self.table.setRowCount(0)
            self._update_llm_token_estimate()
            self._set_stage(1)
            self._sync_range_with_video_duration(file_path)
            self._load_source_preview_frame(file_path, self.preview_time_s.value())

    def _reload_preview_frame(self):
        if not self.video_path:
            return
        self._load_source_preview_frame(self.video_path, self.preview_time_s.value())

    def _sync_range_with_video_duration(self, video_path: str):
        duration_s = self._probe_video_duration_s(video_path)
        self.video_duration_s = max(1, duration_s)
        self.original_video_duration_s = self.video_duration_s
        self.selected_start_s = 0
        self.selected_end_s = self.video_duration_s
        self.range_slider.set_bounds(0, self.video_duration_s)
        self.range_slider.set_values(self.selected_start_s, self.selected_end_s, emit_signal=False)
        self._update_range_labels()
        # Не меняем min/max кандидатов автоматически при выборе ролика.
        # Рекомендации применяются только по кнопке "Рекомендовать по длине".

    def _recommend_candidate_limits(self, duration_s: int):
        d = max(1, int(duration_s or 1))
        minutes = d / 60.0
        target = int(round(minutes * 2.4 + 4))
        target = max(8, min(160, target))
        rec_min = max(4, int(round(target * 0.65)))
        rec_max = max(rec_min + 2, int(round(target * 1.65)))
        rec_max = min(260, rec_max)
        return rec_min, rec_max

    def _apply_recommended_candidate_limits(self):
        rec_min, rec_max = self._recommend_candidate_limits(self.original_video_duration_s)
        self.min_candidates_spin.setValue(rec_min)
        self.max_candidates_spin.setValue(rec_max)

    def _on_candidate_limits_changed(self, _value: int):
        if self.max_candidates_spin.value() < self.min_candidates_spin.value():
            self.max_candidates_spin.setValue(self.min_candidates_spin.value())
        self._update_llm_token_estimate()

    def _on_range_slider_changed(self, start_s: int, end_s: int):
        self.selected_start_s = int(start_s)
        self.selected_end_s = int(end_s)
        if self.selected_end_s <= self.selected_start_s:
            self.selected_end_s = min(self.video_duration_s, self.selected_start_s + 1)
        self._update_range_labels()

    def _update_range_labels(self):
        self.range_start_label.setText(f"С: {self._fmt_s(self.selected_start_s)}")
        self.range_end_label.setText(f"До: {self._fmt_s(self.selected_end_s)}")
        self.range_span_label.setText(
            f"Длительность: {self._fmt_s(max(1, self.selected_end_s - self.selected_start_s))}"
        )

    @staticmethod
    def _probe_video_duration_s(video_path: str) -> int:
        try:
            p = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    video_path,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=(
                    subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
                ),
            )
            if p.returncode == 0:
                raw = (p.stdout or "").strip().splitlines()
                if raw:
                    return max(1, int(float(raw[0])))
        except Exception:
            pass
        return 1

    def _start_full_pipeline(self):
        self._autonomous_run = True
        self.autonomous_checkbox.setChecked(True)
        self._start_transcribe(autonomous=True)

    def _start_analyze(self):
        # backward compatibility со старой кнопкой/сигналами
        self._start_transcribe(autonomous=False)

    def _start_transcribe(self, autonomous: bool = None):
        from app.thread.auto_shorts_thread import AutoShortsTranscribeThread

        if not self.video_path:
            InfoBar.warning(
                "Внимание",
                "Сначала выберите видео",
                duration=2000,
                position=InfoBarPosition.TOP,
                parent=self,
            )
            return

        if autonomous is None:
            autonomous = self.autonomous_checkbox.isChecked()
        self._autonomous_run = bool(autonomous)

        self._set_stage(1)
        self.transcribe_btn.setEnabled(False)
        self.select_candidates_btn.setEnabled(False)
        self.run_all_btn.setEnabled(False)
        self.render_btn.setEnabled(False)
        self._set_active_progress_stage(1)
        self._set_stage_progress(0, "Этап 1/4: Whisper...")

        range_start = int(self.selected_start_s)
        range_end = int(self.selected_end_s)
        if range_end <= range_start:
            InfoBar.warning(
                "Внимание",
                "Время конца диапазона должно быть больше времени старта",
                duration=2500,
                position=InfoBarPosition.TOP,
                parent=self,
            )
            self.transcribe_btn.setEnabled(True)
            self.select_candidates_btn.setEnabled(True)
            self.run_all_btn.setEnabled(True)
            self.render_btn.setEnabled(True)
            return

        self.transcribe_thread = AutoShortsTranscribeThread(
            self.video_path,
            range_enabled=True,
            range_start_s=range_start,
            range_end_s=range_end,
        )
        self.transcribe_thread.progress.connect(self._on_progress)
        self.transcribe_thread.finished.connect(self._on_transcribe_finished)
        self.transcribe_thread.error.connect(self._on_error)
        self.transcribe_thread.start()

    def _on_transcribe_finished(self, asr_payload: Dict):
        self.asr_payload = dict(asr_payload or {})
        self._update_llm_token_estimate()
        self.transcribe_btn.setEnabled(True)
        self.select_candidates_btn.setEnabled(True)
        self.run_all_btn.setEnabled(True)
        self.render_btn.setEnabled(True)
        self._set_stage(2)
        self._set_stage_progress(100, "Whisper завершён. Запустите этап 2: отбор кандидатов")
        self._set_active_progress_stage(0)
        InfoBar.success(
            "Этап 1 завершён",
            "Whisper завершён. Теперь можно запускать LLM-отбор кандидатов.",
            duration=3000,
            position=InfoBarPosition.TOP,
            parent=self,
        )
        if self._autonomous_run:
            self._start_candidate_selection(autonomous=True)

    def _start_candidate_selection(self, autonomous: bool = None):
        from app.thread.auto_shorts_thread import AutoShortsCandidateThread

        if not self.video_path:
            InfoBar.warning("Внимание", "Сначала выберите видео", duration=2000, position=InfoBarPosition.TOP, parent=self)
            return

        if not self.asr_payload:
            InfoBar.warning(
                "Внимание",
                "Сначала выполните этап 1 (Whisper)",
                duration=2500,
                position=InfoBarPosition.TOP,
                parent=self,
            )
            return

        if autonomous is None:
            autonomous = self.autonomous_checkbox.isChecked()
        self._autonomous_run = bool(autonomous)

        min_d = self.min_duration.value()
        max_d = self.max_duration.value()
        if min_d >= max_d:
            InfoBar.warning(
                "Внимание",
                "Минимальная длительность должна быть меньше максимальной",
                duration=2500,
                position=InfoBarPosition.TOP,
                parent=self,
            )
            return

        self._set_stage(2)
        self.transcribe_btn.setEnabled(False)
        self.select_candidates_btn.setEnabled(False)
        self.run_all_btn.setEnabled(False)
        self.render_btn.setEnabled(False)
        self._set_active_progress_stage(2)
        self._set_stage_progress(0, "Этап 2/4: LLM-отбор кандидатов...")

        self.candidate_thread = AutoShortsCandidateThread(
            self.asr_payload,
            min_duration_s=min_d,
            max_duration_s=max_d,
            repeat_similarity_percent=self.repeat_similarity_spin.value(),
            min_candidates=self.min_candidates_spin.value(),
            max_candidates=self.max_candidates_spin.value(),
            auto_filter_weak_candidates=bool(self.auto_filter_weak_checkbox.isChecked()),
            auto_filter_profile=self._auto_filter_profile_value(),
            interest_threshold_percent=self.interest_threshold_spin.value(),
            llm_search_intensity=self.llm_search_intensity_spin.value(),
        )
        self.candidate_thread.progress.connect(self._on_progress)
        self.candidate_thread.finished.connect(self._on_candidate_selection_finished)
        self.candidate_thread.error.connect(self._on_error)
        self.candidate_thread.start()

    def _on_candidate_selection_finished(self, candidates: List[Dict]):
        self.transcribe_btn.setEnabled(True)
        self.select_candidates_btn.setEnabled(True)
        self.run_all_btn.setEnabled(True)
        self.render_btn.setEnabled(True)
        self.candidates = candidates
        self._fill_table(candidates)
        self._set_stage(3)
        self._set_stage_progress(100, f"Этап 2 завершён. Найдено кандидатов: {len(candidates)}")
        self._set_active_progress_stage(0)
        InfoBar.success(
            "Этап 2 завершён",
            f"Найдено моментов: {len(candidates)}. Проверьте выбор перед рендером.",
            duration=3000,
            position=InfoBarPosition.TOP,
            parent=self,
        )
        if self._autonomous_run:
            self._start_render(autonomous=True)

    def _fill_table(self, candidates: List[Dict]):
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        p = get_theme_palette()
        dark = bool(p.get("is_dark"))
        fg = QColor(p["text"])
        bg_even = QColor(p["card_bg"])
        bg_odd = QColor("#252526" if dark else "#F7F7F7")
        for row, c in enumerate(candidates):
            self.table.insertRow(row)
            cb = CheckBox("", self)
            cb.setChecked(True)
            self.table.setCellWidget(row, 0, cb)

            start_s = int(c["start_ms"] // 1000)
            end_s = int(c["end_ms"] // 1000)
            timerange = f"{self._fmt_s(start_s)} - {self._fmt_s(end_s)}"
            duration = f"{(c['end_ms'] - c['start_ms']) / 1000:.1f}с"

            timerange_item = QTableWidgetItem(timerange)
            timerange_item.setData(Qt.UserRole, int(row))
            self.table.setItem(row, 1, timerange_item)
            self.table.setItem(row, 2, QTableWidgetItem(duration))
            self.table.setItem(row, 3, QTableWidgetItem(str(c.get("score", ""))))
            self.table.setItem(row, 4, QTableWidgetItem(str(c.get("quality_score", ""))))
            self.table.setItem(row, 5, QTableWidgetItem(str(c.get("quality_grade", ""))))
            self.table.setItem(row, 6, QTableWidgetItem(str(c.get("hook_score", ""))))
            self.table.setItem(row, 7, QTableWidgetItem(str(c.get("pause_ratio", ""))))
            self.table.setItem(row, 8, QTableWidgetItem(str(c.get("title", ""))))
            self.table.setItem(row, 9, QTableWidgetItem(str(c.get("viral_title", ""))))
            info_text = str(c.get("reason", ""))
            excerpt = str(c.get("excerpt", ""))
            tags = c.get("explainability_tags") or []
            if tags:
                info_text = f"{info_text}\nТеги: {', '.join(str(t) for t in tags)}"
            if excerpt:
                info_text = f"{info_text}\n{excerpt}"
            self.table.setItem(row, 10, QTableWidgetItem(info_text))

            row_bg = bg_even if row % 2 == 0 else bg_odd
            for col in range(1, self.table.columnCount()):
                item = self.table.item(row, col)
                if item:
                    item.setForeground(fg)
                    item.setBackground(row_bg)

        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(10, max(420, self.table.columnWidth(10)))
        self.table.setSortingEnabled(True)

    def _apply_candidates_table_tooltips(self):
        tips = {
            0: "Отметка, какие кандидаты отправлять в рендер.",
            1: "Таймкод начала и конца фрагмента в исходном видео.",
            2: "Длительность фрагмента.",
            3: "Score: базовая оценка кандидата (до финального quality-ранжирования).",
            4: "Q: итоговая оценка интереса/качества (0..100), основной ориентир для отбора.",
            5: "Grade: буквенная категория качества (A — лучший, D — слабый).",
            6: "Hook: сила захвата внимания в начале фрагмента (0..10).",
            7: "Pause: доля пауз/тишины внутри кандидата (меньше — обычно лучше).",
            8: "Развёрнутый описательный заголовок содержания фрагмента.",
            9: "Краткий вирусный заголовок для YouTube Shorts (используется и в именах файлов).",
            10: "Почему кандидат был выбран: причина + теги + фрагмент текста.",
        }
        for idx, tip in tips.items():
            item = self.table.horizontalHeaderItem(idx)
            if item:
                item.setToolTip(tip)

    def _collect_selected(self) -> List[Dict]:
        selected = []
        for row in range(self.table.rowCount()):
            cb = self.table.cellWidget(row, 0)
            if not (cb and cb.isChecked()):
                continue
            idx_item = self.table.item(row, 1)
            try:
                src_idx = int(idx_item.data(Qt.UserRole)) if idx_item else row
            except Exception:
                src_idx = row
            if 0 <= src_idx < len(self.candidates):
                selected.append(self.candidates[src_idx])
        return selected

    def _start_render(self, autonomous: bool = None):
        from app.thread.auto_shorts_thread import AutoShortsRenderThread

        if autonomous is None:
            autonomous = self.autonomous_checkbox.isChecked()
        self._autonomous_run = bool(autonomous)

        if not self.video_path:
            return
        selected = self._collect_selected()
        selected = self._sort_candidates_for_render(selected)
        if not selected:
            InfoBar.warning(
                "Внимание",
                "Выберите хотя бы один фрагмент",
                duration=2000,
                position=InfoBarPosition.TOP,
                parent=self,
            )
            return

        self._set_stage(4)
        self.transcribe_btn.setEnabled(False)
        self.select_candidates_btn.setEnabled(False)
        self.run_all_btn.setEnabled(False)
        self.render_btn.setEnabled(False)
        self.stop_render_btn.setEnabled(True)
        self._set_active_progress_stage(4)
        self._set_stage_progress(0, "Этап 4/4: Рендер шортсов...")

        output_dir = str(self._build_safe_output_dir(self.video_path))
        self.last_output_dir = output_dir
        self.output_hint_label.setText(f"Папка результата: {output_dir}")
        self.render_thread = AutoShortsRenderThread(
            self.video_path,
            selected,
            output_dir,
            layout_template=self._build_layout_template(),
            render_backend=self._get_render_backend(),
            render_options=self._build_render_options(),
        )
        self.render_thread.progress.connect(self._on_progress)
        self.render_thread.finished.connect(self._on_render_finished)
        self.render_thread.error.connect(self._on_error)
        self.render_thread.start()

    @staticmethod
    def _build_safe_output_dir(video_path: str) -> Path:
        """Генерирует безопасную папку вывода шортсов с контролем длины пути."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_stem = TaskFactory._safe_fs_name(Path(video_path).stem, max_len=48)
        dir_name = TaskFactory._safe_fs_name(
            f"шорты_{base_stem}_{stamp}", max_len=120
        ).replace(" ", "_")

        root_dir = WORK_PATH / "shorts"
        root_dir.mkdir(parents=True, exist_ok=True)

        max_path_len = int(getattr(TaskFactory, "MAX_PATH_LEN", 240) or 240)
        out_dir = root_dir / dir_name
        if len(str(out_dir)) <= max_path_len:
            return out_dir

        digest = hashlib.md5(dir_name.encode("utf-8", errors="ignore")).hexdigest()[:8]
        max_name_len = max(16, max_path_len - len(str(root_dir)) - 1)
        head_len = max(6, max_name_len - 9)
        short_name = dir_name[:head_len].rstrip(" ._")
        if not short_name:
            short_name = "шорты"
        return root_dir / f"{short_name}_{digest}"

    def _on_render_finished(self, files: List[str]):
        self.transcribe_btn.setEnabled(True)
        self.select_candidates_btn.setEnabled(True)
        self.run_all_btn.setEnabled(True)
        self.render_btn.setEnabled(True)
        self.stop_render_btn.setEnabled(False)
        self._autonomous_run = False
        self._set_stage_progress(100, f"Готово. Создано шортсов: {len(files)}")
        if self.last_output_dir:
            self.output_hint_label.setText(f"Папка результата: {self.last_output_dir}")
        self.rendered_files = list(files)
        self._populate_rendered_list(self.rendered_files)
        InfoBar.success(
            "Шортсы созданы",
            f"Сохранено файлов: {len(files)}\n{self.last_output_dir}",
            duration=3500,
            position=InfoBarPosition.TOP,
            parent=self,
        )

    def _on_progress(self, value: int, message: str):
        self._set_stage_progress(value, message)

    def _on_error(self, message: str):
        self.transcribe_btn.setEnabled(True)
        self.select_candidates_btn.setEnabled(True)
        self.run_all_btn.setEnabled(True)
        self.render_btn.setEnabled(True)
        self.stop_render_btn.setEnabled(False)
        self._autonomous_run = False
        InfoBar.error(
            "Ошибка",
            message,
            duration=3500,
            position=InfoBarPosition.TOP,
            parent=self,
        )

    def _open_output_folder(self):
        if self.last_output_dir and Path(self.last_output_dir).exists():
            target_dir = Path(self.last_output_dir)
        elif self.video_path:
            target_dir = Path(self.video_path).parent
        else:
            return
        if sys.platform == "win32":
            os.startfile(str(target_dir))

    def _build_layout_template(self) -> Dict:
        src = self.source_preview.get_layers()
        out = self.output_preview.get_layers()
        return {
            "enabled": self.dual_layer_enabled.isChecked(),
            "source_canvas": {
                "w": int(self.source_width),
                "h": int(self.source_height),
            },
            "webcam": {
                "crop_x": src["webcam"]["x"],
                "crop_y": src["webcam"]["y"],
                "crop_w": src["webcam"]["w"],
                "crop_h": src["webcam"]["h"],
                "out_x": out["webcam"]["x"],
                "out_y": out["webcam"]["y"],
                "out_w": out["webcam"]["w"],
                "out_h": out["webcam"]["h"],
            },
            "game": {
                "crop_x": src["game"]["x"],
                "crop_y": src["game"]["y"],
                "crop_w": src["game"]["w"],
                "crop_h": src["game"]["h"],
                "out_x": out["game"]["x"],
                "out_y": out["game"]["y"],
                "out_w": out["game"]["w"],
                "out_h": out["game"]["h"],
            },
            "webcam_fx": {
                "brightness": self.wc_brightness.value() / 100.0,
                "contrast": self.wc_contrast.value() / 100.0,
                "saturation": self.wc_saturation.value() / 100.0,
                "gamma": self.wc_gamma.value() / 100.0,
                "temperature": self.wc_temperature.value() / 100.0,
            },
            "game_fx": {
                "brightness": self.gm_brightness.value() / 100.0,
                "contrast": self.gm_contrast.value() / 100.0,
                "saturation": self.gm_saturation.value() / 100.0,
                "gamma": self.gm_gamma.value() / 100.0,
                "temperature": self.gm_temperature.value() / 100.0,
            },
            "render_settings": {
                "backend": self._get_render_backend(),
                "fps": (self.render_fps_combo.currentText() or "30").strip(),
                "resolution": (self.render_resolution_combo.currentText() or "1080x1920").strip(),
                "quality": (self.render_quality_combo.currentText() or "Сбалансированное").strip(),
                "sort_mode": self._get_render_sort_mode(),
                "filename_include_title": bool(self.filename_include_title_checkbox.isChecked()),
                "filename_include_time_range": bool(self.filename_include_time_checkbox.isChecked()),
                "filename_include_quality": bool(self.filename_include_quality_checkbox.isChecked()),
                "filename_include_score": bool(self.filename_include_score_checkbox.isChecked()),
                "filename_include_hook": bool(self.filename_include_hook_checkbox.isChecked()),
                "filename_include_grade": bool(self.filename_include_grade_checkbox.isChecked()),
                "repeat_similarity_percent": int(self.repeat_similarity_spin.value()),
                "speech_min_coverage_percent": int(self.speech_min_coverage_spin.value()),
                "speech_merge_gap_ms": int(self.speech_merge_gap_spin.value()),
                "speech_pre_pad_ms": int(self.speech_pre_pad_spin.value()),
                "speech_post_pad_ms": int(self.speech_post_pad_spin.value()),
                "clip_head_pad_ms": int(self.clip_head_pad_spin.value()),
                "clip_tail_pad_ms": int(self.clip_tail_pad_spin.value()),
            },
            "ui_settings": {
                "preview_time_s": int(self.preview_time_s.value()),
                "keep_aspect": bool(self.keep_aspect_checkbox.isChecked()),
                "autonomous": bool(self.autonomous_checkbox.isChecked()),
                "range_start_s": int(self.selected_start_s),
                "range_end_s": int(self.selected_end_s),
                "min_duration_s": int(self.min_duration.value()),
                "max_duration_s": int(self.max_duration.value()),
                "min_candidates": int(self.min_candidates_spin.value()),
                "max_candidates": int(self.max_candidates_spin.value()),
                "auto_filter_weak_enabled": bool(self.auto_filter_weak_checkbox.isChecked()),
                "auto_filter_profile": self._auto_filter_profile_value(),
                "interest_threshold_percent": int(self.interest_threshold_spin.value()),
                "llm_search_intensity": int(self.llm_search_intensity_spin.value()),
                "render_sort_mode": self._get_render_sort_mode(),
                "filename_include_title": bool(self.filename_include_title_checkbox.isChecked()),
                "filename_include_time_range": bool(self.filename_include_time_checkbox.isChecked()),
                "filename_include_quality": bool(self.filename_include_quality_checkbox.isChecked()),
                "filename_include_score": bool(self.filename_include_score_checkbox.isChecked()),
                "filename_include_hook": bool(self.filename_include_hook_checkbox.isChecked()),
                "filename_include_grade": bool(self.filename_include_grade_checkbox.isChecked()),
            },
        }

    def _reset_layout_template(self):
        self.source_preview.set_layers(
            QRectF(0, 0, self.source_width * 0.45, self.source_height * 0.35),
            QRectF(0, self.source_height * 0.25, self.source_width, self.source_height * 0.75),
        )
        self.output_preview.set_layers(
            QRectF(0, 0, 1080, 640),
            QRectF(0, 640, 1080, 1280),
        )
        self.wc_brightness.setValue(0)
        self.wc_contrast.setValue(100)
        self.wc_saturation.setValue(100)
        self.wc_gamma.setValue(100)
        self.wc_temperature.setValue(0)
        self.gm_brightness.setValue(0)
        self.gm_contrast.setValue(100)
        self.gm_saturation.setValue(100)
        self.gm_gamma.setValue(100)
        self.gm_temperature.setValue(0)
        self.wc_brightness_slider.setValue(0)
        self.wc_contrast_slider.setValue(100)
        self.wc_saturation_slider.setValue(100)
        self.wc_gamma_slider.setValue(100)
        self.wc_temperature_slider.setValue(0)
        self.gm_brightness_slider.setValue(0)
        self.gm_contrast_slider.setValue(100)
        self.gm_saturation_slider.setValue(100)
        self.gm_gamma_slider.setValue(100)
        self.gm_temperature_slider.setValue(0)
        self._refresh_output_composite_preview()

    def _reset_fx_controls(self):
        self.wc_brightness.setValue(0)
        self.wc_contrast.setValue(100)
        self.wc_saturation.setValue(100)
        self.wc_gamma.setValue(100)
        self.wc_temperature.setValue(0)

        self.gm_brightness.setValue(0)
        self.gm_contrast.setValue(100)
        self.gm_saturation.setValue(100)
        self.gm_gamma.setValue(100)
        self.gm_temperature.setValue(0)

        self._refresh_output_composite_preview()

    def _save_layout_template(self):
        try:
            self.template_path.parent.mkdir(parents=True, exist_ok=True)
            self.template_path.write_text(
                json.dumps(self._build_layout_template(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._persist_last_template_path(
                self.template_path,
                preset_name=self.template_path.stem,
            )
            self._refresh_template_preset_combo(select_path=self.template_path)
            InfoBar.success(
                "Пресет сохранён",
                str(self.template_path),
                duration=2000,
                position=InfoBarPosition.TOP,
                parent=self,
            )
        except Exception as e:
            InfoBar.error("Ошибка", f"Не удалось сохранить пресет: {e}", duration=3000, parent=self)

    def _save_layout_template_as(self):
        try:
            self.templates_dir.mkdir(parents=True, exist_ok=True)
            default_name = self.template_path.name if self.template_path else "template.json"
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Сохранить шаблон как",
                str(self.templates_dir / default_name),
                "JSON (*.json)",
            )
            if not file_path:
                return
            chosen = Path(file_path)
            if chosen.suffix.lower() != ".json":
                chosen = chosen.with_suffix(".json")
            self.template_path = chosen
            self._save_layout_template()
            self._refresh_template_preset_combo(select_path=self.template_path)
        except Exception as e:
            InfoBar.error("Ошибка", f"Не удалось сохранить шаблон: {e}", duration=3000, parent=self)

    def _choose_and_load_layout_template(self):
        try:
            self.templates_dir.mkdir(parents=True, exist_ok=True)
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Загрузить шаблон",
                str(self.templates_dir),
                "JSON (*.json)",
            )
            if not file_path:
                return
            self._load_layout_template(Path(file_path), persist_last=True)
            self._refresh_template_preset_combo(select_path=self.template_path)
        except Exception as e:
            InfoBar.error("Ошибка", f"Не удалось загрузить шаблон: {e}", duration=3000, parent=self)

    def _load_layout_template(
        self,
        template_path: Optional[Path] = None,
        persist_last: bool = True,
        apply_ui_settings: bool = True,
    ):
        try:
            path_to_load = Path(template_path) if template_path else self.template_path
            if not path_to_load.exists():
                return
            data = json.loads(path_to_load.read_text(encoding="utf-8"))
            self.template_path = path_to_load
            if persist_last:
                self._persist_last_template_path(
                    self.template_path,
                    preset_name=self.template_path.stem,
                )
            source_canvas = data.get("source_canvas", {}) if isinstance(data, dict) else {}
            try:
                saved_w = int(source_canvas.get("w", self.source_width))
                saved_h = int(source_canvas.get("h", self.source_height))
            except Exception:
                saved_w, saved_h = int(self.source_width), int(self.source_height)

            # Целевой canvas должен совпадать с текущим кадром видео, иначе
            # координаты crop начинают "съезжать" между предпросмотром и рендером.
            target_w = int(self.source_width)
            target_h = int(self.source_height)
            if target_w < 2 or target_h < 2:
                target_w = max(2, saved_w)
                target_h = max(2, saved_h)

            self.source_preview.set_canvas_size(target_w, target_h)
            self.source_width, self.source_height = target_w, target_h

            sx = target_w / max(1, saved_w)
            sy = target_h / max(1, saved_h)

            def _scaled(v, s):
                try:
                    return int(round(float(v) * s))
                except Exception:
                    return 0

            wc = data.get("webcam", {})
            gm = data.get("game", {})
            self.dual_layer_enabled.setChecked(bool(data.get("enabled", True)))

            self.source_preview.set_layers(
                QRectF(
                    _scaled(wc.get("crop_x", 0), sx),
                    _scaled(wc.get("crop_y", 0), sy),
                    max(2, _scaled(wc.get("crop_w", 320), sx)),
                    max(2, _scaled(wc.get("crop_h", 240), sy)),
                ),
                QRectF(
                    _scaled(gm.get("crop_x", 0), sx),
                    _scaled(gm.get("crop_y", 0), sy),
                    max(2, _scaled(gm.get("crop_w", 640), sx)),
                    max(2, _scaled(gm.get("crop_h", 360), sy)),
                ),
            )
            self.output_preview.set_layers(
                QRectF(int(wc.get("out_x", 0)), int(wc.get("out_y", 0)), int(wc.get("out_w", 1080)), int(wc.get("out_h", 640))),
                QRectF(int(gm.get("out_x", 0)), int(gm.get("out_y", 640)), int(gm.get("out_w", 1080)), int(gm.get("out_h", 1280))),
            )
            wfx = data.get("webcam_fx", {}) if isinstance(data, dict) else {}
            gfx = data.get("game_fx", {}) if isinstance(data, dict) else {}
            self.wc_brightness.setValue(int(round(float(wfx.get("brightness", 0.0) or 0.0) * 100)))
            self.wc_contrast.setValue(int(round(float(wfx.get("contrast", 1.0) or 1.0) * 100)))
            self.wc_saturation.setValue(int(round(float(wfx.get("saturation", 1.0) or 1.0) * 100)))
            self.wc_gamma.setValue(int(round(float(wfx.get("gamma", 1.0) or 1.0) * 100)))
            self.wc_temperature.setValue(int(round(float(wfx.get("temperature", 0.0) or 0.0) * 100)))
            self.gm_brightness.setValue(int(round(float(gfx.get("brightness", 0.0) or 0.0) * 100)))
            self.gm_contrast.setValue(int(round(float(gfx.get("contrast", 1.0) or 1.0) * 100)))
            self.gm_saturation.setValue(int(round(float(gfx.get("saturation", 1.0) or 1.0) * 100)))
            self.gm_gamma.setValue(int(round(float(gfx.get("gamma", 1.0) or 1.0) * 100)))
            self.gm_temperature.setValue(int(round(float(gfx.get("temperature", 0.0) or 0.0) * 100)))
            self.wc_brightness_slider.setValue(self.wc_brightness.value())
            self.wc_contrast_slider.setValue(self.wc_contrast.value())
            self.wc_saturation_slider.setValue(self.wc_saturation.value())
            self.wc_gamma_slider.setValue(self.wc_gamma.value())
            self.wc_temperature_slider.setValue(self.wc_temperature.value())
            self.gm_brightness_slider.setValue(self.gm_brightness.value())
            self.gm_contrast_slider.setValue(self.gm_contrast.value())
            self.gm_saturation_slider.setValue(self.gm_saturation.value())
            self.gm_gamma_slider.setValue(self.gm_gamma.value())
            self.gm_temperature_slider.setValue(self.gm_temperature.value())

            render_settings = data.get("render_settings", {}) if isinstance(data, dict) else {}
            if isinstance(render_settings, dict):
                backend = str(render_settings.get("backend", "") or "").strip().lower()
                if backend in {"auto", "cpu", "gpu", "cuda"}:
                    self.render_backend_combo.setCurrentIndex(self._backend_to_index(backend))

                fps_text = str(render_settings.get("fps", "") or "").strip()
                if fps_text in {"Исходный", "30", "60"}:
                    self.render_fps_combo.setCurrentText(fps_text)

                resolution_text = str(render_settings.get("resolution", "") or "").strip()
                if resolution_text in {"1080x1920", "720x1280", "1440x2560", "Исходное"}:
                    self.render_resolution_combo.setCurrentText(resolution_text)

                quality_text = str(render_settings.get("quality", "") or "").strip()
                if quality_text in {"Высокое", "Сбалансированное", "Быстрое"}:
                    self.render_quality_combo.setCurrentText(quality_text)

                self._set_render_sort_mode(
                    str(render_settings.get("sort_mode", self._get_render_sort_mode()) or self._get_render_sort_mode())
                )
                self.filename_include_title_checkbox.setChecked(
                    bool(render_settings.get("filename_include_title", self.filename_include_title_checkbox.isChecked()))
                )
                self.filename_include_time_checkbox.setChecked(
                    bool(render_settings.get("filename_include_time_range", self.filename_include_time_checkbox.isChecked()))
                )
                self.filename_include_quality_checkbox.setChecked(
                    bool(render_settings.get("filename_include_quality", self.filename_include_quality_checkbox.isChecked()))
                )
                self.filename_include_score_checkbox.setChecked(
                    bool(render_settings.get("filename_include_score", self.filename_include_score_checkbox.isChecked()))
                )
                self.filename_include_hook_checkbox.setChecked(
                    bool(render_settings.get("filename_include_hook", self.filename_include_hook_checkbox.isChecked()))
                )
                self.filename_include_grade_checkbox.setChecked(
                    bool(render_settings.get("filename_include_grade", self.filename_include_grade_checkbox.isChecked()))
                )

                try:
                    self.repeat_similarity_spin.setValue(
                        int(render_settings.get("repeat_similarity_percent", self.repeat_similarity_spin.value()))
                    )
                    self.speech_min_coverage_spin.setValue(
                        int(render_settings.get("speech_min_coverage_percent", self.speech_min_coverage_spin.value()))
                    )
                    self.speech_merge_gap_spin.setValue(
                        int(render_settings.get("speech_merge_gap_ms", self.speech_merge_gap_spin.value()))
                    )
                    self.speech_pre_pad_spin.setValue(
                        int(render_settings.get("speech_pre_pad_ms", self.speech_pre_pad_spin.value()))
                    )
                    self.speech_post_pad_spin.setValue(
                        int(render_settings.get("speech_post_pad_ms", self.speech_post_pad_spin.value()))
                    )
                    self.clip_head_pad_spin.setValue(
                        int(render_settings.get("clip_head_pad_ms", self.clip_head_pad_spin.value()))
                    )
                    self.clip_tail_pad_spin.setValue(
                        int(render_settings.get("clip_tail_pad_ms", self.clip_tail_pad_spin.value()))
                    )
                except Exception:
                    pass

            ui_settings = data.get("ui_settings", {}) if isinstance(data, dict) else {}
            if apply_ui_settings and isinstance(ui_settings, dict):
                try:
                    self.preview_time_s.setValue(
                        int(ui_settings.get("preview_time_s", self.preview_time_s.value()))
                    )
                    self.keep_aspect_checkbox.setChecked(
                        bool(ui_settings.get("keep_aspect", self.keep_aspect_checkbox.isChecked()))
                    )
                    self.autonomous_checkbox.setChecked(
                        bool(ui_settings.get("autonomous", self.autonomous_checkbox.isChecked()))
                    )
                    self.min_duration.setValue(
                        int(ui_settings.get("min_duration_s", self.min_duration.value()))
                    )
                    self.max_duration.setValue(
                        int(ui_settings.get("max_duration_s", self.max_duration.value()))
                    )
                    self.min_candidates_spin.setValue(
                        int(ui_settings.get("min_candidates", self.min_candidates_spin.value()))
                    )
                    self.max_candidates_spin.setValue(
                        int(ui_settings.get("max_candidates", self.max_candidates_spin.value()))
                    )
                    self.auto_filter_weak_checkbox.setChecked(
                        bool(ui_settings.get("auto_filter_weak_enabled", self.auto_filter_weak_checkbox.isChecked()))
                    )
                    self._set_auto_filter_profile_value(
                        str(ui_settings.get("auto_filter_profile", self._auto_filter_profile_value()) or "balanced")
                    )
                    self.interest_threshold_spin.setValue(
                        int(ui_settings.get("interest_threshold_percent", self.interest_threshold_spin.value()))
                    )
                    self.llm_search_intensity_spin.setValue(
                        int(ui_settings.get("llm_search_intensity", self.llm_search_intensity_spin.value()))
                    )

                    self._set_render_sort_mode(
                        str(ui_settings.get("render_sort_mode", self._get_render_sort_mode()) or self._get_render_sort_mode())
                    )
                    self.filename_include_title_checkbox.setChecked(
                        bool(ui_settings.get("filename_include_title", self.filename_include_title_checkbox.isChecked()))
                    )
                    self.filename_include_time_checkbox.setChecked(
                        bool(ui_settings.get("filename_include_time_range", self.filename_include_time_checkbox.isChecked()))
                    )
                    self.filename_include_quality_checkbox.setChecked(
                        bool(ui_settings.get("filename_include_quality", self.filename_include_quality_checkbox.isChecked()))
                    )
                    self.filename_include_score_checkbox.setChecked(
                        bool(ui_settings.get("filename_include_score", self.filename_include_score_checkbox.isChecked()))
                    )
                    self.filename_include_hook_checkbox.setChecked(
                        bool(ui_settings.get("filename_include_hook", self.filename_include_hook_checkbox.isChecked()))
                    )
                    self.filename_include_grade_checkbox.setChecked(
                        bool(ui_settings.get("filename_include_grade", self.filename_include_grade_checkbox.isChecked()))
                    )

                    loaded_start = int(ui_settings.get("range_start_s", self.selected_start_s))
                    loaded_end = int(ui_settings.get("range_end_s", self.selected_end_s))
                    self.selected_start_s = max(0, loaded_start)
                    self.selected_end_s = max(self.selected_start_s + 1, loaded_end)
                    self.range_slider.set_values(self.selected_start_s, self.selected_end_s, emit_signal=False)
                    self._update_range_labels()
                except Exception:
                    pass

            self._refresh_output_composite_preview()
            self._update_llm_token_estimate()
            self._refresh_template_preset_combo(select_path=self.template_path)
        except Exception:
            pass

    def _update_llm_token_estimate(self):
        if not hasattr(self, "llm_tokens_hint_label"):
            return

        asr_json = (self.asr_payload or {}).get("asr_json")
        if not asr_json:
            self.llm_tokens_hint_label.setText("Оценка токенов LLM: появится после этапа 1 (Whisper)")
            return

        segments = self._extract_asr_segments(asr_json)
        if not segments:
            self.llm_tokens_hint_label.setText("Оценка токенов LLM: не удалось извлечь сегменты из Whisper")
            return

        # В реальном пайплайне размер пакетов зависит от "Интенсивности LLM-поиска".
        intensity = int(self.llm_search_intensity_spin.value()) if hasattr(self, "llm_search_intensity_spin") else 3
        packet_size_map = {1: 180, 2: 160, 3: 140, 4: 120, 5: 100}
        overlap_map = {1: 42, 2: 38, 3: 35, 4: 30, 5: 26}
        packet_size = packet_size_map.get(intensity, 140)
        overlap = overlap_map.get(intensity, 35)
        packets = self._estimate_packet_count(len(segments), packet_size=packet_size, overlap=overlap)

        sample_rows = []
        for idx, seg in enumerate(segments[:packet_size]):
            sample_rows.append(
                {
                    "idx": int(idx),
                    "start_ms": int(seg["start_ms"]),
                    "end_ms": int(seg["end_ms"]),
                    "text": str(seg["text"]),
                }
            )

        # Ближе к реальности LM Studio: считаем токены по длине JSON-пакета + системного промпта.
        user_payload = f"Сегменты:\n{json.dumps(sample_rows, ensure_ascii=False)}"
        min_s = max(8, int(self.min_duration.value()))
        max_s = max(min_s + 5, int(self.max_duration.value()))
        system_payload = (
            "Ты enterprise-редактор YouTube Shorts. Найди лучшие моменты удержания. "
            "Критерии: hook в первые 2-5 секунд, эмоция, конфликт/неожиданность, панчлайн, кульминация, потенциал для шеринга. "
            "Верни СТРОГО JSON: "
            "{\"items\":[{\"start_idx\":int,\"end_idx\":int,\"score\":0-100,\"title\":str,\"reason\":str,"
            "\"hook\":0-10,\"emotion\":0-10,\"novelty\":0-10,\"shareability\":0-10}]}. "
            f"Длительность каждого фрагмента {min_s}-{max_s} секунд. "
            "Не придумывай таймкоды, используй только переданные idx."
        )

        # Калибровка под реальное поведение LM Studio:
        # для JSON с таймкодами/индексами и RU/EN текста фактическая токенизация
        # обычно заметно «плотнее», чем грубое 1 токен ~= 3 символа.
        # Используем более реалистичный коэффициент 1.6 символа/токен.
        per_packet_input_tokens = max(128, int((len(system_payload) + len(user_payload)) / 1.6))

        # На один пакет число кандидатов зависит от интенсивности LLM.
        target_candidates = max(1, min(28, 10 + intensity * 4))
        per_packet_output_tokens = 180 + target_candidates * 75
        per_packet_total = per_packet_input_tokens + per_packet_output_tokens

        total_input_tokens = per_packet_input_tokens * packets
        total_output_tokens = per_packet_output_tokens * packets
        total_tokens = total_input_tokens + total_output_tokens

        recommended_ctx = int(per_packet_total * 1.2)

        self.llm_tokens_hint_label.setText(
            f"Оценка LLM (приближенно к LM Studio): на 1 пакет вход ~{per_packet_input_tokens}, "
            f"выход ~{per_packet_output_tokens}, всего ~{per_packet_total}; пакетов: {packets}. "
            f"Суммарно за этап: вход ~{total_input_tokens}, выход ~{total_output_tokens}, всего ~{total_tokens}. "
            f"Рекомендуемый context в LM Studio: от {recommended_ctx} (на один запрос)."
        )

    def _auto_filter_profile_value(self) -> str:
        text = (self.auto_filter_profile_combo.currentText() or "Сбалансированный").strip().lower()
        if "мяг" in text:
            return "soft"
        if "строг" in text:
            return "strict"
        return "balanced"

    def _set_auto_filter_profile_value(self, profile: str):
        p = str(profile or "balanced").strip().lower()
        if p == "soft":
            self.auto_filter_profile_combo.setCurrentText("Мягкий")
        elif p == "strict":
            self.auto_filter_profile_combo.setCurrentText("Строгий")
        else:
            self.auto_filter_profile_combo.setCurrentText("Сбалансированный")

    @staticmethod
    def _collect_asr_text(asr_json: Dict) -> str:
        segments = AutoShortsInterface._extract_asr_segments(asr_json)
        if not segments:
            return ""
        return "\n".join(str(s.get("text", "")).strip() for s in segments if str(s.get("text", "")).strip())

    @staticmethod
    def _extract_asr_segments(asr_json: Dict) -> List[Dict[str, int | str]]:
        if not isinstance(asr_json, dict):
            return []

        # Формат проекта: {"1": {...}, "2": {...}} c полями original_subtitle/start_time/end_time
        numeric_keys = [k for k in asr_json.keys() if str(k).isdigit()]
        segments: List[Dict[str, int | str]] = []
        if numeric_keys:
            for k in sorted(numeric_keys, key=lambda x: int(x)):
                item = asr_json.get(k)
                if not isinstance(item, dict):
                    continue
                text = str(item.get("original_subtitle") or "").strip()
                if not text:
                    continue
                try:
                    start_ms = int(item.get("start_time", 0))
                    end_ms = int(item.get("end_time", start_ms))
                except Exception:
                    continue
                segments.append({"text": text, "start_ms": start_ms, "end_ms": end_ms})
            return segments

        # Fallback на альтернативные структуры
        raw_segments = asr_json.get("segments")
        if isinstance(raw_segments, list):
            for seg in raw_segments:
                if not isinstance(seg, dict):
                    continue
                text = str(seg.get("text") or seg.get("original_subtitle") or "").strip()
                if not text:
                    continue
                try:
                    start_ms = int(seg.get("start_ms", seg.get("start_time", 0)))
                    end_ms = int(seg.get("end_ms", seg.get("end_time", start_ms)))
                except Exception:
                    continue
                segments.append({"text": text, "start_ms": start_ms, "end_ms": end_ms})
        return segments

    @staticmethod
    def _estimate_packet_count(total_segments: int, packet_size: int, overlap: int) -> int:
        total = max(0, int(total_segments or 0))
        if total <= 0:
            return 1
        step = max(1, int(packet_size) - int(overlap))
        count = 1
        start = 0
        while start + packet_size < total:
            start += step
            count += 1
        return max(1, count)

    def _resolve_startup_template_path(self) -> Path:
        legacy_default = APPDATA_PATH / "shorts_layout_template.json"
        try:
            raw = self._read_template_state()
            p = str(raw.get("last_template", "") or "").strip()
            if p:
                candidate = Path(p)
                if candidate.exists():
                    return candidate
        except Exception:
            pass
        if legacy_default.exists():
            return legacy_default
        return self.templates_dir / "default.json"

    def _read_template_state(self) -> Dict:
        try:
            if not self.template_state_path.exists():
                return {}
            raw = json.loads(self.template_state_path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except Exception:
            return {}

    def _persist_last_template_path(self, path: Path, preset_name: str = ""):
        try:
            state = self._read_template_state()
            state["last_template"] = str(path)
            if preset_name:
                state["last_preset_name"] = str(preset_name)
            self.template_state_path.parent.mkdir(parents=True, exist_ok=True)
            self.template_state_path.write_text(
                json.dumps(state, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _link_fx_control_pair(self, spin: SpinBox, slider: QSlider):
        spin.valueChanged.connect(slider.setValue)
        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(lambda _v: self._schedule_fx_preview_refresh())

    @staticmethod
    def _clamp_u8(v: float) -> int:
        return max(0, min(255, int(round(v))))

    def _apply_fx_preview(self, pixmap: QPixmap, layer: str, fast_mode: bool = True) -> QPixmap:
        if pixmap.isNull():
            return pixmap

        if layer == "webcam":
            brightness = self.wc_brightness.value() / 100.0
            contrast = self.wc_contrast.value() / 100.0
            saturation = self.wc_saturation.value() / 100.0
            gamma = self.wc_gamma.value() / 100.0
            temperature = self.wc_temperature.value() / 100.0
        else:
            brightness = self.gm_brightness.value() / 100.0
            contrast = self.gm_contrast.value() / 100.0
            saturation = self.gm_saturation.value() / 100.0
            gamma = self.gm_gamma.value() / 100.0
            temperature = self.gm_temperature.value() / 100.0

        cache_key = (
            str(layer),
            bool(fast_mode),
            int(pixmap.cacheKey()),
            int(pixmap.width()),
            int(pixmap.height()),
            round(float(brightness), 4),
            round(float(contrast), 4),
            round(float(saturation), 4),
            round(float(gamma), 4),
            round(float(temperature), 4),
        )
        cached = self._fx_layer_cache.get(cache_key)
        if cached is not None and not cached.isNull():
            return QPixmap(cached)

        if (
            abs(brightness) < 1e-6
            and abs(contrast - 1.0) < 1e-6
            and abs(saturation - 1.0) < 1e-6
            and abs(gamma - 1.0) < 1e-6
            and abs(temperature) < 1e-6
        ):
            return pixmap

        work_pix = pixmap
        if fast_mode:
            max_side = 220
            side = max(work_pix.width(), work_pix.height())
            if side > max_side:
                scale = max_side / float(side)
                work_pix = work_pix.scaled(
                    max(1, int(work_pix.width() * scale)),
                    max(1, int(work_pix.height() * scale)),
                    Qt.IgnoreAspectRatio,
                    Qt.SmoothTransformation,
                )

        img = work_pix.toImage().convertToFormat(4)  # QImage.Format_ARGB32
        w, h = img.width(), img.height()
        bright_add = brightness * 255.0
        gamma = max(0.3, min(3.0, float(gamma)))
        warm = max(-0.4, min(0.4, float(temperature)))
        red_gain = max(0.55, min(1.45, 1.0 + warm * 0.55))
        blue_gain = max(0.55, min(1.45, 1.0 - warm * 0.55))
        # Важно: финальный Auto Shorts рендер сейчас не применяет unsharp в filter graph
        # (из-за проблем стабильности ffmpeg на части систем).
        # Поэтому в превью НЕ добавляем имитацию резкости, чтобы кадр был ближе к результату рендера.
        contrast_total = contrast

        for y in range(h):
            for x in range(w):
                c = img.pixelColor(x, y)
                r, g, b, a = c.red(), c.green(), c.blue(), c.alpha()

                r = (r - 127.5) * contrast_total + 127.5 + bright_add
                g = (g - 127.5) * contrast_total + 127.5 + bright_add
                b = (b - 127.5) * contrast_total + 127.5 + bright_add

                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                r = lum + (r - lum) * saturation
                g = lum + (g - lum) * saturation
                b = lum + (b - lum) * saturation

                # gamma и temperature применяем после базовой цветокоррекции
                r = 255.0 * pow(max(0.0, min(255.0, r)) / 255.0, 1.0 / gamma)
                g = 255.0 * pow(max(0.0, min(255.0, g)) / 255.0, 1.0 / gamma)
                b = 255.0 * pow(max(0.0, min(255.0, b)) / 255.0, 1.0 / gamma)
                r *= red_gain
                b *= blue_gain

                img.setPixelColor(
                    x,
                    y,
                    QColor(self._clamp_u8(r), self._clamp_u8(g), self._clamp_u8(b), a),
                )

        result = QPixmap.fromImage(img)
        if fast_mode and result.size() != pixmap.size():
            result = result.scaled(
                pixmap.size(),
                Qt.IgnoreAspectRatio,
                Qt.SmoothTransformation,
            )
        if len(self._fx_layer_cache) > 96:
            self._fx_layer_cache.clear()
        self._fx_layer_cache[cache_key] = QPixmap(result)
        return result

    @staticmethod
    def _eq_tail_for_preview(layer_cfg: Dict) -> str:
        fx = layer_cfg if isinstance(layer_cfg, dict) else {}
        brightness = float(fx.get("brightness", 0.0) or 0.0)
        contrast = float(fx.get("contrast", 1.0) or 1.0)
        saturation = float(fx.get("saturation", 1.0) or 1.0)
        gamma = float(fx.get("gamma", 1.0) or 1.0)
        temperature = float(fx.get("temperature", 0.0) or 0.0)
        parts = []
        if (
            abs(brightness) > 1e-6
            or abs(contrast - 1.0) > 1e-6
            or abs(saturation - 1.0) > 1e-6
            or abs(gamma - 1.0) > 1e-6
        ):
            gamma = max(0.3, min(3.0, gamma))
            parts.append(
                f"eq=brightness={brightness:.3f}:contrast={contrast:.3f}:saturation={saturation:.3f}:gamma={gamma:.3f}"
            )
        if abs(temperature) > 1e-6:
            warm = max(-0.4, min(0.4, temperature))
            rr = max(0.55, min(1.45, 1.0 + warm * 0.55))
            bb = max(0.55, min(1.45, 1.0 - warm * 0.55))
            parts.append(f"colorchannelmixer=rr={rr:.3f}:bb={bb:.3f}")
        return ("," + ",".join(parts)) if parts else ""

    def _render_exact_fx_preview_frame(self):
        if not self.video_path or not Path(self.video_path).exists():
            InfoBar.warning(
                "Внимание",
                "Сначала выберите видео",
                duration=2500,
                position=InfoBarPosition.TOP,
                parent=self,
            )
            return

        fd, temp_img = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        try:
            tpl = self._build_layout_template()
            out_w, out_h = 1080, 1920
            raw_seek_s = max(0, int(self.preview_time_s.value()))
            duration_s = max(1, int(self._probe_video_duration_s(self.video_path)))
            seek_s = min(raw_seek_s, max(0, duration_s - 1))

            def _looks_black_frame(pix: QPixmap) -> bool:
                if pix.isNull():
                    return True
                img = pix.toImage()
                if img.isNull():
                    return True
                w, h = img.width(), img.height()
                step_x = max(1, w // 24)
                step_y = max(1, h // 24)
                for y in range(0, h, step_y):
                    for x in range(0, w, step_x):
                        c = img.pixelColor(x, y)
                        if c.red() > 3 or c.green() > 3 or c.blue() > 3:
                            return False
                return True

            def _run_ffmpeg(cmd: List[str]) -> subprocess.CompletedProcess:
                return subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=(
                        subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
                    ),
                )

            if bool(tpl.get("enabled", True)):
                wc = tpl.get("webcam", {})
                gm = tpl.get("game", {})
                wc_fx = self._eq_tail_for_preview(tpl.get("webcam_fx", {}))
                gm_fx = self._eq_tail_for_preview(tpl.get("game_fx", {}))

                def _iv(d: Dict, k: str, default: int) -> int:
                    try:
                        return int(d.get(k, default))
                    except Exception:
                        return int(default)

                wc_crop_x = max(0, _iv(wc, "crop_x", 0))
                wc_crop_y = max(0, _iv(wc, "crop_y", 0))
                wc_crop_w = max(2, _iv(wc, "crop_w", max(2, self.source_width // 2)))
                wc_crop_h = max(2, _iv(wc, "crop_h", max(2, self.source_height // 3)))
                wc_out_x = max(0, _iv(wc, "out_x", 0))
                wc_out_y = max(0, _iv(wc, "out_y", 0))
                wc_out_w = max(2, _iv(wc, "out_w", out_w))
                wc_out_h = max(2, _iv(wc, "out_h", out_h // 3))

                gm_crop_x = max(0, _iv(gm, "crop_x", 0))
                gm_crop_y = max(0, _iv(gm, "crop_y", 0))
                gm_crop_w = max(2, _iv(gm, "crop_w", max(2, self.source_width)))
                gm_crop_h = max(2, _iv(gm, "crop_h", max(2, self.source_height // 2)))
                gm_out_x = max(0, _iv(gm, "out_x", 0))
                gm_out_y = max(0, _iv(gm, "out_y", out_h // 3))
                gm_out_w = max(2, _iv(gm, "out_w", out_w))
                gm_out_h = max(2, _iv(gm, "out_h", (out_h * 2) // 3))

                filter_complex = (
                    f"color=size={out_w}x{out_h}:color=black[base];"
                    f"[0:v]split=2[src_cam][src_game];"
                    f"[src_cam]crop={wc_crop_w}:{wc_crop_h}:{wc_crop_x}:{wc_crop_y}{wc_fx},"
                    f"scale={wc_out_w}:{wc_out_h}:flags=lanczos[cam];"
                    f"[src_game]crop={gm_crop_w}:{gm_crop_h}:{gm_crop_x}:{gm_crop_y}{gm_fx},"
                    f"scale={gm_out_w}:{gm_out_h}:flags=lanczos[game];"
                    f"[base][cam]overlay={wc_out_x}:{wc_out_y}[tmp];"
                    f"[tmp][game]overlay={gm_out_x}:{gm_out_y}[vout]"
                )
                cmd_primary = [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    str(seek_s),
                    "-i",
                    self.video_path,
                    "-frames:v",
                    "1",
                    "-filter_complex",
                    filter_complex,
                    "-map",
                    "[vout]",
                    "-y",
                    temp_img,
                ]

                cmd_fallback = [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    self.video_path,
                    "-ss",
                    str(seek_s),
                    "-frames:v",
                    "1",
                    "-filter_complex",
                    filter_complex,
                    "-map",
                    "[vout]",
                    "-y",
                    temp_img,
                ]
            else:
                cmd_primary = [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    str(seek_s),
                    "-i",
                    self.video_path,
                    "-frames:v",
                    "1",
                    "-vf",
                    f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase:flags=lanczos,crop={out_w}:{out_h}",
                    "-y",
                    temp_img,
                ]
                cmd_fallback = [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    self.video_path,
                    "-ss",
                    str(seek_s),
                    "-frames:v",
                    "1",
                    "-vf",
                    f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase:flags=lanczos,crop={out_w}:{out_h}",
                    "-y",
                    temp_img,
                ]

            proc = _run_ffmpeg(cmd_primary)
            pix = QPixmap(temp_img) if Path(temp_img).exists() else QPixmap()

            # На некоторых источниках ffmpeg с -ss до -i отдаёт «чёрный» кадр в filter_complex.
            # Если кадр пустой/чёрный, повторяем рендер с -ss после -i (более надёжно).
            if proc.returncode != 0 or pix.isNull() or _looks_black_frame(pix):
                proc2 = _run_ffmpeg(cmd_fallback)
                pix2 = QPixmap(temp_img) if Path(temp_img).exists() else QPixmap()
                if proc2.returncode == 0 and not pix2.isNull() and not _looks_black_frame(pix2):
                    pix = pix2
                elif proc.returncode != 0 and proc2.returncode != 0:
                    raise RuntimeError(((proc2.stderr or proc.stderr) or "Ошибка FFmpeg")[-800:])
                elif pix.isNull() and pix2.isNull():
                    raise RuntimeError("FFmpeg вернул пустой кадр")

            self.effects_preview.set_background(pix)
            InfoBar.success(
                "Точный предпросмотр готов",
                "Кадр справа обновлён через FFmpeg и ближе к итоговому рендеру",
                duration=2200,
                position=InfoBarPosition.TOP,
                parent=self,
            )
        except Exception as e:
            InfoBar.error(
                "Ошибка предпросмотра",
                f"Не удалось построить точный FX-кадр: {e}",
                duration=3200,
                position=InfoBarPosition.TOP,
                parent=self,
            )
        finally:
            try:
                Path(temp_img).unlink(missing_ok=True)
            except Exception:
                pass

    def _populate_rendered_list(self, files: List[str]):
        self.rendered_list.clear()
        for f in files or []:
            item = QListWidgetItem(Path(f).name)
            item.setToolTip(f)
            item.setData(Qt.UserRole, f)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.rendered_list.addItem(item)

    def _selected_rendered_files(self) -> List[str]:
        selected = []
        for i in range(self.rendered_list.count()):
            item = self.rendered_list.item(i)
            if item and item.checkState() == Qt.Checked:
                path = item.data(Qt.UserRole)
                if path and Path(path).exists():
                    selected.append(str(path))
        return selected

    def _open_selected_rendered(self):
        files = self._selected_rendered_files()
        if not files:
            return
        if sys.platform == "win32":
            for f in files[:3]:
                os.startfile(f)

    def _send_selected_to_batch(self):
        files = self._selected_rendered_files()
        if not files:
            InfoBar.warning(
                "Внимание",
                "Выберите хотя бы один готовый шортс",
                duration=2500,
                position=InfoBarPosition.TOP,
                parent=self,
            )
            return

        main_win = self.window()
        batch = getattr(main_win, "batchProcessInterface", None)
        if batch is None and hasattr(main_win, "_ensure_batch_interface"):
            try:
                main_win._ensure_batch_interface()
                batch = getattr(main_win, "batchProcessInterface", None)
            except Exception:
                batch = None
        if not batch:
            InfoBar.error(
                "Ошибка",
                "Не удалось найти вкладку пакетной обработки",
                duration=3000,
                position=InfoBarPosition.TOP,
                parent=self,
            )
            return

        try:
            batch.task_type_combo.setCurrentText(str(BatchTaskType.FULL_PROCESS))
            batch.add_files(files)
            if hasattr(main_win, "switchTo"):
                main_win.switchTo(batch)
            InfoBar.success(
                "Готово",
                f"Отправлено в пакетную обработку: {len(files)}",
                duration=3000,
                position=InfoBarPosition.TOP,
                parent=self,
            )
        except Exception as e:
            InfoBar.error(
                "Ошибка",
                f"Не удалось отправить в пакетную обработку: {e}",
                duration=3500,
                position=InfoBarPosition.TOP,
                parent=self,
            )

    def _load_source_preview_frame(self, video_path: str, seek_s: int = 2):
        try:
            self._fx_layer_cache.clear()
            fd, temp_img = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            cmd = [
                "ffmpeg",
                "-ss",
                str(max(0, int(seek_s))),
                "-i",
                video_path,
                "-frames:v",
                "1",
                "-y",
                temp_img,
            ]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=(
                    subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
                ),
            )
            if proc.returncode == 0 and Path(temp_img).exists():
                pix = QPixmap(temp_img)
                if not pix.isNull():
                    self.source_frame_pixmap = QPixmap(pix)
                    self.source_width = pix.width()
                    self.source_height = pix.height()
                    self.source_preview.set_canvas_size(self.source_width, self.source_height)
                    self.source_preview.set_background(pix)
                    # Если пресет сохранён — применяем его, иначе дефолт
                    if self.template_path.exists():
                        # Важно: при выборе нового ролика не перетираем только что
                        # рассчитанный диапазон анализа (0..длина ролика) значениями
                        # из пресета. Иначе ползунок диапазона не растягивается на весь ролик.
                        self._load_layout_template(persist_last=False, apply_ui_settings=False)
                    else:
                        self._reset_layout_template()
                    self._refresh_output_composite_preview()
            try:
                Path(temp_img).unlink(missing_ok=True)
            except Exception:
                pass
        except Exception:
            pass

    def _compose_output_canvas(self, apply_fx: bool = False, fx_fast_mode: bool = True) -> QPixmap:
        if self.source_frame_pixmap.isNull():
            return QPixmap()

        out_w, out_h = 1080, 1920
        canvas = QPixmap(out_w, out_h)
        canvas.fill(QColor(0, 0, 0))

        src_layers = self.source_preview.get_layers()
        out_layers = self.output_preview.get_layers()
        use_webcam = bool(self.dual_layer_enabled.isChecked()) if hasattr(self, "dual_layer_enabled") else True
        layer_keys = ["webcam", "game"] if use_webcam else ["game"]
        painter = QPainter(canvas)
        try:
            for key in layer_keys:
                sr = src_layers.get(key, {})
                dr = out_layers.get(key, {})

                sx = max(0, int(sr.get("x", 0)))
                sy = max(0, int(sr.get("y", 0)))
                sw = max(1, int(sr.get("w", 1)))
                sh = max(1, int(sr.get("h", 1)))

                if sx + sw > self.source_frame_pixmap.width():
                    sw = max(1, self.source_frame_pixmap.width() - sx)
                if sy + sh > self.source_frame_pixmap.height():
                    sh = max(1, self.source_frame_pixmap.height() - sy)

                dx = max(0, int(dr.get("x", 0)))
                dy = max(0, int(dr.get("y", 0)))
                dw = max(1, int(dr.get("w", 1)))
                dh = max(1, int(dr.get("h", 1)))
                if dx + dw > out_w:
                    dw = max(1, out_w - dx)
                if dy + dh > out_h:
                    dh = max(1, out_h - dy)

                crop = self.source_frame_pixmap.copy(sx, sy, sw, sh)
                if crop.isNull():
                    continue
                if apply_fx:
                    crop = self._apply_fx_preview(crop, key, fast_mode=fx_fast_mode)
                painter.drawPixmap(QRectF(dx, dy, dw, dh), crop, QRectF(0, 0, crop.width(), crop.height()))
        finally:
            painter.end()

        return canvas

    def _schedule_fx_preview_refresh(self):
        if not hasattr(self, "effects_preview"):
            return
        self._fx_preview_timer.start()

    def _render_fx_preview_now(self):
        if not hasattr(self, "effects_preview"):
            return
        fx_canvas = self._compose_output_canvas(apply_fx=True, fx_fast_mode=True)
        self.effects_preview.set_background(fx_canvas)

    def _refresh_output_composite_preview(self):
        canvas = self._compose_output_canvas(apply_fx=False)
        self.output_preview.set_background(canvas)
        self._schedule_fx_preview_refresh()

    def _select_all(self):
        for row in range(self.table.rowCount()):
            cb = self.table.cellWidget(row, 0)
            if cb:
                cb.setChecked(True)

    def _clear_select(self):
        for row in range(self.table.rowCount()):
            cb = self.table.cellWidget(row, 0)
            if cb:
                cb.setChecked(False)

    def _stop_render(self):
        if hasattr(self, "render_thread") and self.render_thread and self.render_thread.isRunning():
            self.render_thread.request_cancel()
            self.stage4_progress_label.setText("Остановка рендера...")
            self.stop_render_btn.setEnabled(False)

    def _get_render_backend(self) -> str:
        idx = self.render_backend_combo.currentIndex()
        return {0: "auto", 1: "cpu", 2: "gpu", 3: "cuda"}.get(idx, "auto")

    @staticmethod
    def _backend_to_index(backend: str) -> int:
        b = (backend or "auto").strip().lower()
        return {"auto": 0, "cpu": 1, "gpu": 2, "cuda": 3}.get(b, 0)

    def _on_render_backend_changed(self, _index: int):
        try:
            cfg.set(cfg.auto_shorts_render_backend, self._get_render_backend())
        except Exception:
            pass

    def _build_render_options(self) -> Dict:
        fps_text = (self.render_fps_combo.currentText() or "30").strip()
        if fps_text == "Исходный":
            fps_mode = "source"
        elif fps_text == "60":
            fps_mode = "60"
        else:
            fps_mode = "30"

        resolution_text = (self.render_resolution_combo.currentText() or "1080x1920").strip()
        if resolution_text == "Исходное":
            resolution_mode = "source"
            resolution_value = "source"
        else:
            resolution_mode = "fixed"
            resolution_value = resolution_text

        quality_text = (self.render_quality_combo.currentText() or "Сбалансированное").strip()
        quality_profile = {
            "Высокое": "high",
            "Быстрое": "fast",
        }.get(quality_text, "balanced")

        return {
            "fps_mode": fps_mode,
            "resolution_mode": resolution_mode,
            "resolution": resolution_value,
            "quality_profile": quality_profile,
            "sort_mode": self._get_render_sort_mode(),
            "clip_head_pad_ms": int(self.clip_head_pad_spin.value()),
            "clip_tail_pad_ms": int(self.clip_tail_pad_spin.value()),
            "speech_pre_pad_ms": int(self.speech_pre_pad_spin.value()),
            "speech_post_pad_ms": int(self.speech_post_pad_spin.value()),
            "speech_merge_gap_ms": int(self.speech_merge_gap_spin.value()),
            "speech_min_coverage_percent": int(self.speech_min_coverage_spin.value()),
            "filename_options": {
                "include_title": bool(self.filename_include_title_checkbox.isChecked()),
                "include_time_range": bool(self.filename_include_time_checkbox.isChecked()),
                "include_quality": bool(self.filename_include_quality_checkbox.isChecked()),
                "include_score": bool(self.filename_include_score_checkbox.isChecked()),
                "include_hook": bool(self.filename_include_hook_checkbox.isChecked()),
                "include_grade": bool(self.filename_include_grade_checkbox.isChecked()),
            },
        }

    def _get_render_sort_mode(self) -> str:
        idx = int(self.render_sort_combo.currentIndex()) if hasattr(self, "render_sort_combo") else 0
        return {
            0: "time_asc",
            1: "time_desc",
            2: "quality_desc",
            3: "score_desc",
            4: "hook_desc",
            5: "duration_desc",
            6: "duration_asc",
        }.get(idx, "time_asc")

    def _set_render_sort_mode(self, sort_mode: str):
        mode = str(sort_mode or "time_asc").strip().lower()
        idx = {
            "time_asc": 0,
            "time_desc": 1,
            "quality_desc": 2,
            "score_desc": 3,
            "hook_desc": 4,
            "duration_desc": 5,
            "duration_asc": 6,
        }.get(mode, 0)
        if hasattr(self, "render_sort_combo"):
            self.render_sort_combo.setCurrentIndex(idx)

    def _sort_candidates_for_render(self, candidates: List[Dict]) -> List[Dict]:
        if not candidates:
            return []
        mode = self._get_render_sort_mode()
        items = list(candidates)

        def _duration_ms(item: Dict) -> int:
            try:
                return max(0, int(item.get("end_ms", 0)) - int(item.get("start_ms", 0)))
            except Exception:
                return 0

        if mode == "time_desc":
            items.sort(key=lambda x: int(x.get("start_ms", 0) or 0), reverse=True)
        elif mode == "quality_desc":
            items.sort(key=lambda x: float(x.get("quality_score", 0.0) or 0.0), reverse=True)
        elif mode == "score_desc":
            items.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
        elif mode == "hook_desc":
            items.sort(key=lambda x: float(x.get("hook_score", 0.0) or 0.0), reverse=True)
        elif mode == "duration_desc":
            items.sort(key=_duration_ms, reverse=True)
        elif mode == "duration_asc":
            items.sort(key=_duration_ms)
        else:
            items.sort(key=lambda x: int(x.get("start_ms", 0) or 0))
        return items

    @staticmethod
    def _fmt_s(total: int) -> str:
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        return f"{h:02}:{m:02}:{s:02}"

    def closeEvent(self, event):
        if hasattr(self, "transcribe_thread") and self.transcribe_thread.isRunning():
            self.transcribe_thread.terminate()
        if hasattr(self, "candidate_thread") and self.candidate_thread.isRunning():
            self.candidate_thread.terminate()
        if hasattr(self, "render_thread") and self.render_thread.isRunning():
            self.render_thread.terminate()
        super().closeEvent(event)


class TemplatePresetNameDialog(MessageBoxBase):
    """Диалог сохранения пользовательского пресета шортсов"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.titleLabel = BodyLabel("Сохранить пресет шортсов", self)
        self.nameLineEdit = LineEdit(self)

        self.nameLineEdit.setPlaceholderText("Введите название пресета")
        self.nameLineEdit.setClearButtonEnabled(True)

        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.nameLineEdit)

        self.yesButton.setText("Сохранить")
        self.cancelButton.setText("Отмена")
        self.widget.setMinimumWidth(380)
        self.yesButton.setDisabled(True)
        self.nameLineEdit.textChanged.connect(self._validateInput)

    def _validateInput(self, text: str):
        self.yesButton.setEnabled(bool((text or "").strip()))
