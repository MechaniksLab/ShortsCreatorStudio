import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from PyQt5.QtCore import QPointF, QRect, QRectF, Qt, QStandardPaths, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QScrollArea, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
from qfluentwidgets import Action, BodyLabel, CardWidget, CheckBox, ComboBox, CommandBar, FluentIcon, InfoBar, InfoBarPosition, PrimaryPushButton, ProgressBar, PushButton, StrongBodyLabel, SpinBox, isDarkTheme

from app.common.config import cfg
from app.config import APPDATA_PATH, WORK_PATH
from app.core.entities import SupportedAudioFormats, SupportedVideoFormats
from app.thread.auto_shorts_thread import AutoShortsAnalyzeThread, AutoShortsRenderThread


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

        for name in ["game", "webcam"]:
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
        self.last_output_dir: str = ""
        self.template_path = APPDATA_PATH / "shorts_layout_template.json"
        self.source_width = 1920
        self.source_height = 1080
        self.source_frame_pixmap = QPixmap()
        self.video_duration_s = 0
        self.selected_start_s = 0
        self.selected_end_s = 1

        self._init_ui()
        self._apply_theme_style()

    def _init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_area = QScrollArea(self)
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
        self.stage_1 = QLabel("1) Анализ видео")
        self.stage_2 = QLabel("2) Поиск интересных моментов")
        self.stage_3 = QLabel("3) Выбор кандидатов")
        self.stage_4 = QLabel("4) Рендер шортсов")
        for w in [self.stage_1, self.stage_2, self.stage_3, self.stage_4]:
            stage_layout.addWidget(w)
        self.main_layout.addWidget(self.stage_card)

        self.control_card = CardWidget(self)
        self.control_card.setObjectName("shortsControlCard")
        control_layout = QVBoxLayout(self.control_card)

        title_row = QHBoxLayout()
        title_row.addWidget(StrongBodyLabel("Параметры поиска"))
        title_row.addStretch(1)
        control_layout.addLayout(title_row)

        params_row = QHBoxLayout()
        params_row.addWidget(BodyLabel("Мин. длительность (сек):"))
        self.min_duration = SpinBox(self)
        self.min_duration.setRange(8, 90)
        self.min_duration.setValue(15)
        params_row.addWidget(self.min_duration)

        params_row.addWidget(BodyLabel("Макс. длительность (сек):"))
        self.max_duration = SpinBox(self)
        self.max_duration.setRange(12, 180)
        self.max_duration.setValue(60)
        params_row.addWidget(self.max_duration)

        self.analyze_btn = PrimaryPushButton("Найти интересные моменты")
        self.analyze_btn.clicked.connect(self._start_analyze)
        params_row.addStretch(1)
        params_row.addWidget(self.analyze_btn)
        control_layout.addLayout(params_row)

        control_layout.addWidget(BodyLabel("Диапазон анализа (всегда активен):"))
        self.range_slider = TimeRangeSlider(self)
        self.range_slider.set_bounds(0, 1)
        self.range_slider.set_values(0, 1, emit_signal=False)
        self.range_slider.rangeChanged.connect(self._on_range_slider_changed)
        control_layout.addWidget(self.range_slider)

        range_info_row = QHBoxLayout()
        self.range_start_label = BodyLabel("С: 00:00")
        self.range_end_label = BodyLabel("До: 00:01")
        self.range_span_label = BodyLabel("Длительность: 00:01")
        range_info_row.addWidget(self.range_start_label)
        range_info_row.addStretch(1)
        range_info_row.addWidget(self.range_end_label)
        range_info_row.addStretch(1)
        range_info_row.addWidget(self.range_span_label)
        control_layout.addLayout(range_info_row)

        hint = BodyLabel(
            "По умолчанию выбран весь ролик. Перемещайте левый и правый маркеры, чтобы ограничить промежуток анализа."
        )
        hint.setWordWrap(True)
        control_layout.addWidget(hint)
        self.main_layout.addWidget(self.control_card)

        self.template_card = CardWidget(self)
        self.template_card.setObjectName("shortsTemplateCard")
        template_layout = QVBoxLayout(self.template_card)
        template_layout.addWidget(StrongBodyLabel("Наглядный шаблон монтажа"))

        frame_row = QHBoxLayout()
        frame_row.addWidget(BodyLabel("Кадр предпросмотра (сек):"))
        self.preview_time_s = SpinBox(self)
        self.preview_time_s.setRange(0, 24 * 3600)
        self.preview_time_s.setValue(2)
        frame_row.addWidget(self.preview_time_s)
        self.refresh_preview_btn = PushButton("Обновить кадр")
        self.refresh_preview_btn.clicked.connect(self._reload_preview_frame)
        frame_row.addWidget(self.refresh_preview_btn)
        frame_row.addStretch(1)
        self.keep_aspect_checkbox = CheckBox("Сохранять пропорции")
        self.keep_aspect_checkbox.setChecked(True)
        self.keep_aspect_checkbox.stateChanged.connect(self._on_keep_aspect_changed)
        frame_row.addWidget(self.keep_aspect_checkbox)
        template_layout.addLayout(frame_row)

        template_layout.addWidget(BodyLabel("1) На исходном кадре перетяните и растяните области WEBCAM и GAME (кроп)."))
        self.source_preview = LayerPreviewWidget(1920, 1080, self)
        self.source_preview.set_keep_aspect(True)
        template_layout.addWidget(self.source_preview)

        template_layout.addWidget(BodyLabel("2) На вертикальном кадре 1080x1920 перетяните и растяните размещение слоёв."))
        self.output_preview = LayerPreviewWidget(1080, 1920, self)
        self.output_preview.set_keep_aspect(True)
        self.output_preview.set_background(QPixmap())
        self.source_preview.changed.connect(lambda _: self._refresh_output_composite_preview())
        self.output_preview.changed.connect(lambda _: self._refresh_output_composite_preview())
        template_layout.addWidget(self.output_preview)

        row_tpl_actions = QHBoxLayout()
        self.dual_layer_enabled = CheckBox("Включить двухслойный шаблон")
        self.dual_layer_enabled.setChecked(True)
        self.reset_template_btn = PushButton("Сбросить шаблон")
        self.reset_template_btn.clicked.connect(self._reset_layout_template)
        self.load_template_btn = PushButton("Загрузить шаблон")
        self.load_template_btn.clicked.connect(self._load_layout_template)
        self.save_template_btn = PushButton("Сохранить шаблон")
        self.save_template_btn.clicked.connect(self._save_layout_template)
        row_tpl_actions.addWidget(self.dual_layer_enabled)
        row_tpl_actions.addStretch(1)
        row_tpl_actions.addWidget(self.reset_template_btn)
        row_tpl_actions.addWidget(self.load_template_btn)
        row_tpl_actions.addWidget(self.save_template_btn)
        template_layout.addLayout(row_tpl_actions)
        self.main_layout.addWidget(self.template_card)

        self.table = QTableWidget(self)
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ["Выбрать", "Таймкод", "Длит.", "Score", "Заголовок", "Причина / Фрагмент"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(False)
        self.table.verticalHeader().setDefaultSectionSize(64)
        self.main_layout.addWidget(self.table)

        self.bottom_card = CardWidget(self)
        self.bottom_card.setObjectName("shortsBottomCard")
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
        self.render_btn = PrimaryPushButton("Сделать шортсы из выбранных")
        self.render_btn.clicked.connect(self._start_render)
        self.stop_render_btn = PushButton("Стоп")
        self.stop_render_btn.setEnabled(False)
        self.stop_render_btn.clicked.connect(self._stop_render)
        bottom_layout.addWidget(self.select_all_btn)
        bottom_layout.addWidget(self.clear_select_btn)
        bottom_layout.addWidget(self.render_backend_label)
        bottom_layout.addWidget(self.render_backend_combo)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(self.stop_render_btn)
        bottom_layout.addWidget(self.render_btn)
        self.main_layout.addWidget(self.bottom_card)

        self.progress_bar = ProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label = BodyLabel("Ожидание")
        self.main_layout.addWidget(self.progress_bar)
        self.main_layout.addWidget(self.progress_label)

        self.output_hint_label = BodyLabel("Папка результата: пока нет")
        self.output_hint_label.setWordWrap(True)
        self.main_layout.addWidget(self.output_hint_label)

        self.main_layout.addStretch(1)

        self.scroll_area.setWidget(self.scroll_content)
        root_layout.addWidget(self.scroll_area)

        self._set_stage(0)
        self._load_layout_template()

    def _on_keep_aspect_changed(self):
        enabled = self.keep_aspect_checkbox.isChecked()
        self.source_preview.set_keep_aspect(enabled)
        self.output_preview.set_keep_aspect(enabled)

    def _apply_theme_style(self):
        dark = isDarkTheme()
        self.source_preview.set_theme(dark)
        self.output_preview.set_theme(dark)

        if dark:
            self.setStyleSheet(
                """
                QWidget#AutoShortsInterface { background: #202124; }
                QScrollArea { border: none; background: transparent; }
                QScrollArea > QWidget > QWidget { background: #202124; }
                QLabel, BodyLabel, StrongBodyLabel { color: #EDEDED; }
                CardWidget { border-radius: 10px; }
                CardWidget#shortsVideoCard, CardWidget#shortsStageCard, CardWidget#shortsControlCard,
                CardWidget#shortsTemplateCard, CardWidget#shortsBottomCard {
                    background: #2A2B2E;
                    border: 1px solid #3A3D42;
                }
                QTableWidget { background: #2A2B2E; color: #EDEDED; gridline-color: #3A3D42; }
                QTableWidget::item { background: #2A2B2E; color: #EDEDED; }
                QTableWidget::item:alternate { background: #242529; color: #EDEDED; }
                QHeaderView::section { background: #31343A; color: #EDEDED; border: none; padding: 4px; }
                QTableWidget::item:selected { background: #3B82F6; color: white; }
                """
            )
        else:
            self.setStyleSheet(
                """
                QWidget#AutoShortsInterface { background: #f5f6f8; }
                QScrollArea { border: none; background: transparent; }
                QScrollArea > QWidget > QWidget { background: #f5f6f8; }
                CardWidget { border-radius: 10px; }
                CardWidget#shortsVideoCard, CardWidget#shortsStageCard, CardWidget#shortsControlCard,
                CardWidget#shortsTemplateCard, CardWidget#shortsBottomCard {
                    background: #FFFFFF;
                    border: 1px solid #E5E7EB;
                }
                QTableWidget { background: #FFFFFF; color: #202124; gridline-color: #E5E7EB; }
                QTableWidget::item { background: #FFFFFF; color: #202124; }
                QTableWidget::item:alternate { background: #F9FAFB; color: #202124; }
                QHeaderView::section { background: #F3F4F6; color: #202124; border: none; padding: 4px; }
                QTableWidget::item:selected { background: #DBEAFE; color: #111827; }
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
            self.candidates = []
            self.table.setRowCount(0)
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
        self.selected_start_s = 0
        self.selected_end_s = self.video_duration_s
        self.range_slider.set_bounds(0, self.video_duration_s)
        self.range_slider.set_values(self.selected_start_s, self.selected_end_s, emit_signal=False)
        self._update_range_labels()

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

    def _start_analyze(self):
        if not self.video_path:
            InfoBar.warning(
                "Внимание",
                "Сначала выберите видео",
                duration=2000,
                position=InfoBarPosition.TOP,
                parent=self,
            )
            return

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

        self._set_stage(1)
        self.analyze_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Анализ...")

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
            self.analyze_btn.setEnabled(True)
            return

        self.analyze_thread = AutoShortsAnalyzeThread(
            self.video_path,
            min_d,
            max_d,
            range_enabled=True,
            range_start_s=range_start,
            range_end_s=range_end,
        )
        self.analyze_thread.progress.connect(self._on_progress)
        self.analyze_thread.finished.connect(self._on_analyze_finished)
        self.analyze_thread.error.connect(self._on_error)
        self.analyze_thread.start()

    def _on_analyze_finished(self, candidates: List[Dict]):
        self.analyze_btn.setEnabled(True)
        self.candidates = candidates
        self._fill_table(candidates)
        self._set_stage(3)
        self.progress_label.setText(f"Готово. Найдено кандидатов: {len(candidates)}")
        InfoBar.success(
            "Анализ завершен",
            f"Найдено моментов: {len(candidates)}. Выберите нужные.",
            duration=3000,
            position=InfoBarPosition.TOP,
            parent=self,
        )

    def _fill_table(self, candidates: List[Dict]):
        self.table.setRowCount(0)
        dark = isDarkTheme()
        fg = QColor("#EDEDED") if dark else QColor("#202124")
        bg_even = QColor("#2A2B2E") if dark else QColor("#FFFFFF")
        bg_odd = QColor("#242529") if dark else QColor("#F9FAFB")
        for row, c in enumerate(candidates):
            self.table.insertRow(row)
            cb = CheckBox("", self)
            cb.setChecked(True)
            self.table.setCellWidget(row, 0, cb)

            start_s = int(c["start_ms"] // 1000)
            end_s = int(c["end_ms"] // 1000)
            timerange = f"{self._fmt_s(start_s)} - {self._fmt_s(end_s)}"
            duration = f"{(c['end_ms'] - c['start_ms']) / 1000:.1f}с"

            self.table.setItem(row, 1, QTableWidgetItem(timerange))
            self.table.setItem(row, 2, QTableWidgetItem(duration))
            self.table.setItem(row, 3, QTableWidgetItem(str(c.get("score", ""))))
            self.table.setItem(row, 4, QTableWidgetItem(str(c.get("title", ""))))
            info_text = str(c.get("reason", ""))
            excerpt = str(c.get("excerpt", ""))
            if excerpt:
                info_text = f"{info_text}\n{excerpt}"
            self.table.setItem(row, 5, QTableWidgetItem(info_text))

            row_bg = bg_even if row % 2 == 0 else bg_odd
            for col in range(1, self.table.columnCount()):
                item = self.table.item(row, col)
                if item:
                    item.setForeground(fg)
                    item.setBackground(row_bg)

        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(5, max(360, self.table.columnWidth(5)))

    def _collect_selected(self) -> List[Dict]:
        selected = []
        for row in range(self.table.rowCount()):
            cb = self.table.cellWidget(row, 0)
            if cb and cb.isChecked() and row < len(self.candidates):
                selected.append(self.candidates[row])
        return selected

    def _start_render(self):
        if not self.video_path:
            return
        selected = self._collect_selected()
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
        self.render_btn.setEnabled(False)
        self.stop_render_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Рендер шортсов...")

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = str(WORK_PATH / "shorts" / f"shorts_{Path(self.video_path).stem}_{stamp}")
        self.last_output_dir = output_dir
        self.output_hint_label.setText(f"Папка результата: {output_dir}")
        self.render_thread = AutoShortsRenderThread(
            self.video_path,
            selected,
            output_dir,
            layout_template=self._build_layout_template(),
            render_backend=self._get_render_backend(),
        )
        self.render_thread.progress.connect(self._on_progress)
        self.render_thread.finished.connect(self._on_render_finished)
        self.render_thread.error.connect(self._on_error)
        self.render_thread.start()

    def _on_render_finished(self, files: List[str]):
        self.render_btn.setEnabled(True)
        self.stop_render_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        self.progress_label.setText(f"Готово. Создано шортсов: {len(files)}")
        if self.last_output_dir:
            self.output_hint_label.setText(f"Папка результата: {self.last_output_dir}")
        InfoBar.success(
            "Шортсы созданы",
            f"Сохранено файлов: {len(files)}\n{self.last_output_dir}",
            duration=3500,
            position=InfoBarPosition.TOP,
            parent=self,
        )

    def _on_progress(self, value: int, message: str):
        self.progress_bar.setValue(max(0, min(100, int(value))))
        self.progress_label.setText(message)
        if value >= 70:
            self._set_stage(2)

    def _on_error(self, message: str):
        self.analyze_btn.setEnabled(True)
        self.render_btn.setEnabled(True)
        self.stop_render_btn.setEnabled(False)
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
        self._refresh_output_composite_preview()

    def _save_layout_template(self):
        try:
            self.template_path.parent.mkdir(parents=True, exist_ok=True)
            self.template_path.write_text(
                json.dumps(self._build_layout_template(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            InfoBar.success(
                "Шаблон сохранён",
                str(self.template_path),
                duration=2000,
                position=InfoBarPosition.TOP,
                parent=self,
            )
        except Exception as e:
            InfoBar.error("Ошибка", f"Не удалось сохранить шаблон: {e}", duration=3000, parent=self)

    def _load_layout_template(self):
        try:
            if not self.template_path.exists():
                return
            data = json.loads(self.template_path.read_text(encoding="utf-8"))
            wc = data.get("webcam", {})
            gm = data.get("game", {})
            self.dual_layer_enabled.setChecked(bool(data.get("enabled", True)))

            self.source_preview.set_layers(
                QRectF(int(wc.get("crop_x", 0)), int(wc.get("crop_y", 0)), int(wc.get("crop_w", 320)), int(wc.get("crop_h", 240))),
                QRectF(int(gm.get("crop_x", 0)), int(gm.get("crop_y", 0)), int(gm.get("crop_w", 640)), int(gm.get("crop_h", 360))),
            )
            self.output_preview.set_layers(
                QRectF(int(wc.get("out_x", 0)), int(wc.get("out_y", 0)), int(wc.get("out_w", 1080)), int(wc.get("out_h", 640))),
                QRectF(int(gm.get("out_x", 0)), int(gm.get("out_y", 640)), int(gm.get("out_w", 1080)), int(gm.get("out_h", 1280))),
            )
            self._refresh_output_composite_preview()
        except Exception:
            pass

    def _load_source_preview_frame(self, video_path: str, seek_s: int = 2):
        try:
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
                    # Если шаблон сохранён — применяем его, иначе дефолт
                    if self.template_path.exists():
                        self._load_layout_template()
                    else:
                        self._reset_layout_template()
                    self._refresh_output_composite_preview()
            try:
                Path(temp_img).unlink(missing_ok=True)
            except Exception:
                pass
        except Exception:
            pass

    def _refresh_output_composite_preview(self):
        if self.source_frame_pixmap.isNull():
            self.output_preview.set_background(QPixmap())
            return

        out_w, out_h = 1080, 1920
        canvas = QPixmap(out_w, out_h)
        canvas.fill(QColor(0, 0, 0))

        src_layers = self.source_preview.get_layers()
        out_layers = self.output_preview.get_layers()
        painter = QPainter(canvas)
        try:
            for key in ["webcam", "game"]:
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
                painter.drawPixmap(QRectF(dx, dy, dw, dh), crop, QRectF(0, 0, crop.width(), crop.height()))
        finally:
            painter.end()

        self.output_preview.set_background(canvas)

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
            self.progress_label.setText("Остановка рендера...")
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

    @staticmethod
    def _fmt_s(total: int) -> str:
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        return f"{h:02}:{m:02}:{s:02}"

    def closeEvent(self, event):
        # Тихо сохраняем последний шаблон при закрытии, чтобы он сохранялся между перезапусками
        try:
            self.template_path.parent.mkdir(parents=True, exist_ok=True)
            self.template_path.write_text(
                json.dumps(self._build_layout_template(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
        if hasattr(self, "analyze_thread") and self.analyze_thread.isRunning():
            self.analyze_thread.terminate()
        if hasattr(self, "render_thread") and self.render_thread.isRunning():
            self.render_thread.terminate()
        super().closeEvent(event)
