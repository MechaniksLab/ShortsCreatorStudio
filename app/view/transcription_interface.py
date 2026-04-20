# -*- coding: utf-8 -*-

import datetime
import os
import subprocess
import sys
import time
from pathlib import Path

from PyQt5.QtCore import *
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    Action,
    BodyLabel,
    CardWidget,
    CommandBar,
    FluentIcon,
    InfoBar,
    InfoBarPosition,
    PillPushButton,
    PrimaryPushButton,
    ProgressBar,
    ProgressRing,
    PushButton,
    RoundMenu,
    TransparentDropDownPushButton,
    setFont,
)

from app.common.config import cfg
from app.common.signal_bus import signalBus
from app.components.LanguageSettingDialog import LanguageSettingDialog
from app.components.transcription_setting_card import TranscriptionSettingCard
from app.config import RESOURCE_PATH
from app.core.entities import (
    SupportedAudioFormats,
    SupportedVideoFormats,
    TranscribeModelEnum,
    TranscribeTask,
    VideoInfo,
)
from app.core.task_factory import TaskFactory
from app.thread.transcript_thread import TranscriptThread
from app.thread.video_info_thread import VideoInfoThread

DEFAULT_THUMBNAIL_PATH = RESOURCE_PATH / "assets" / "default_thumbnail.jpg"


class VideoInfoCard(CardWidget):
    finished = pyqtSignal(TranscribeTask)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_signals()
        self.task = None
        self.video_info = None
        self.transcription_interface = parent
        self._transcribe_started_at = None

    def setup_ui(self):
        self.setFixedHeight(150)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(20, 15, 20, 15)
        self.layout.setSpacing(20)

        self.setup_thumbnail()
        self.setup_info_layout()
        self.setup_button_layout()

    def setup_thumbnail(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        default_thumbnail_path = os.path.join(DEFAULT_THUMBNAIL_PATH)

        self.video_thumbnail = QLabel(self)
        self.video_thumbnail.setFixedSize(208, 117)
        self.video_thumbnail.setStyleSheet("background-color: #1E1F22;")
        self.video_thumbnail.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap(default_thumbnail_path).scaled(
            self.video_thumbnail.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_thumbnail.setPixmap(pixmap)
        self.layout.addWidget(self.video_thumbnail, 0, Qt.AlignLeft)

    def setup_info_layout(self):
        self.info_layout = QVBoxLayout()
        self.info_layout.setContentsMargins(3, 8, 3, 8)
        self.info_layout.setSpacing(8)

        self.video_title = BodyLabel("Перетащите аудио или видео файл", self)
        self.video_title.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        self.video_title.setWordWrap(True)
        self.info_layout.addWidget(self.video_title, alignment=Qt.AlignTop)

        self.details_layout = QHBoxLayout()
        self.details_layout.setSpacing(15)

        self.resolution_info = self.create_pill_button("Качество", 110)
        self.file_size_info = self.create_pill_button("Размер", 110)
        self.duration_info = self.create_pill_button("Длительность", 100)

        self.progress_ring = ProgressRing(self)
        self.progress_ring.setFixedSize(20, 20)
        self.progress_ring.setStrokeWidth(4)
        self.progress_ring.hide()

        self.details_layout.addWidget(self.resolution_info)
        self.details_layout.addWidget(self.file_size_info)
        self.details_layout.addWidget(self.duration_info)
        self.details_layout.addWidget(self.progress_ring)
        self.details_layout.addStretch(1)
        self.info_layout.addLayout(self.details_layout)

        # Линейный прогресс + оценка оставшегося времени
        self.transcribe_progress_bar = ProgressBar(self)
        self.transcribe_progress_bar.setRange(0, 100)
        self.transcribe_progress_bar.setValue(0)
        self.transcribe_progress_bar.hide()

        self.transcribe_progress_label = BodyLabel("", self)
        self.transcribe_progress_label.hide()

        self.info_layout.addWidget(self.transcribe_progress_bar)
        self.info_layout.addWidget(self.transcribe_progress_label)
        self.layout.addLayout(self.info_layout)

    @staticmethod
    def _format_eta(seconds: float) -> str:
        sec = max(0, int(seconds))
        h, rem = divmod(sec, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h:d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def create_pill_button(self, text, width):
        button = PillPushButton(text, self)
        button.setCheckable(False)
        setFont(button, 11)
        # button.setFixedWidth(width)
        button.setMinimumWidth(50)
        return button

    def setup_button_layout(self):
        self.button_layout = QVBoxLayout()
        self.open_folder_button = PushButton("Открыть папку", self)
        self.start_button = PrimaryPushButton("Начать распознавание", self)
        self.button_layout.addWidget(self.open_folder_button)
        self.button_layout.addWidget(self.start_button)

        self.start_button.setDisabled(True)

        button_widget = QWidget()
        button_widget.setLayout(self.button_layout)
        button_widget.setFixedWidth(130)
        self.layout.addWidget(button_widget)

    def update_info(self, video_info: VideoInfo):
        """更新视频信息显示"""
        # self.reset_ui()
        self.video_info = video_info

        self.video_title.setText(video_info.file_name.rsplit(".", 1)[0])
        self.resolution_info.setText("Качество: " + f"{video_info.width}x{video_info.height}")
        file_size_mb = os.path.getsize(video_info.file_path) / 1024 / 1024
        self.file_size_info.setText("Размер: " + f"{file_size_mb:.1f} MB")
        duration = datetime.timedelta(seconds=int(video_info.duration_seconds))
        self.duration_info.setText("Длительность: " + f"{duration}")
        if self.transcription_interface and self.transcription_interface.is_processing:
            self.start_button.setEnabled(False)
        else:
            self.start_button.setEnabled(True)
        self.update_thumbnail(video_info.thumbnail_path)

    def update_thumbnail(self, thumbnail_path):
        """更新视频缩略图"""
        if not Path(thumbnail_path).exists():
            thumbnail_path = RESOURCE_PATH / "assets" / "audio-thumbnail.png"

        pixmap = QPixmap(str(thumbnail_path)).scaled(
            self.video_thumbnail.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_thumbnail.setPixmap(pixmap)

    def setup_signals(self):
        self.start_button.clicked.connect(self.on_start_button_clicked)
        self.open_folder_button.clicked.connect(self.on_open_folder_clicked)

    def show_language_settings(self):
        """显示语言设置对话框"""
        dialog = LanguageSettingDialog(self.window())
        if dialog.exec_():
            return True
        return False

    def on_start_button_clicked(self):
        """开始转录按钮点击事件"""
        if self.task and not self.task.need_next_task:
            need_language_settings = cfg.transcribe_model.value in [
                TranscribeModelEnum.WHISPER_CPP,
                TranscribeModelEnum.WHISPER_API,
                TranscribeModelEnum.FASTER_WHISPER,
            ]
            if need_language_settings and not self.show_language_settings():
                return
        self.progress_ring.show()
        self.progress_ring.setValue(0)
        self.transcribe_progress_bar.show()
        self.transcribe_progress_bar.setValue(0)
        self.transcribe_progress_label.setText("Подготовка распознавания...")
        self.transcribe_progress_label.show()
        self._transcribe_started_at = time.perf_counter()
        self.start_button.setDisabled(True)
        self.start_transcription()

    def on_open_folder_clicked(self):
        """打开文件夹按钮点击事件"""
        if self.task and self.task.output_path:
            original_subtitle_save_path = Path(self.task.output_path)
            target_dir = str(
                original_subtitle_save_path.parent
                if original_subtitle_save_path.exists()
                else Path(self.task.file_path).parent
            )
            if sys.platform == "win32":
                os.startfile(target_dir)
            elif sys.platform == "darwin":  # macOS
                subprocess.run(["open", target_dir])
            else:  # Linux
                subprocess.run(["xdg-open", target_dir])
        else:
            InfoBar.warning(
                "Внимание",
                "Нет доступной папки с субтитрами",
                duration=2000,
                parent=self,
            )

    def start_transcription(self, need_create_task=True):
        """开始转录过程"""
        self.transcription_interface.is_processing = True
        self.start_button.setEnabled(False)

        if need_create_task:
            self.task = TaskFactory.create_transcribe_task(self.video_info.file_path)

        self.transcript_thread = TranscriptThread(self.task)
        self.transcript_thread.finished.connect(self.on_transcript_finished)
        self.transcript_thread.progress.connect(self.on_transcript_progress)
        self.transcript_thread.error.connect(self.on_transcript_error)
        self.transcript_thread.start()

    def on_transcript_progress(self, value, message):
        """更新转录进度"""
        value = max(0, min(100, int(value)))
        self.start_button.setText(message)
        self.progress_ring.setValue(value)
        self.transcribe_progress_bar.setValue(value)

        eta_text = ""
        if self._transcribe_started_at and value > 0 and value < 100:
            elapsed = max(0.0, time.perf_counter() - self._transcribe_started_at)
            total_est = elapsed / (value / 100.0)
            remain = max(0.0, total_est - elapsed)
            eta_text = f" • осталось ~ {self._format_eta(remain)}"

        self.transcribe_progress_label.setText(f"{value}%{eta_text} — {message}")

    def on_transcript_error(self, error):
        """处理转录错误"""
        self.start_button.setEnabled(True)
        self.start_button.setText("Повторить распознавание")
        self.start_button.setEnabled(True)
        self._transcribe_started_at = None
        self.transcribe_progress_bar.setValue(0)
        self.transcribe_progress_label.setText("Распознавание завершилось с ошибкой")
        InfoBar.error(
            "Ошибка распознавания",
            self.tr(error),
            duration=3000,
            parent=self.parent().parent(),
        )

    def on_transcript_finished(self, task):
        """转录完成处理"""
        self.start_button.setEnabled(True)
        self.start_button.setText("Распознавание завершено")
        self._transcribe_started_at = None
        self.progress_ring.setValue(100)
        self.transcribe_progress_bar.setValue(100)
        self.transcribe_progress_label.setText("100% — распознавание завершено")
        self.finished.emit(task)

    def reset_ui(self):
        """重置UI状态"""
        self.start_button.setDisabled(False)
        self.start_button.setText("Начать распознавание")
        self.progress_ring.setValue(0)
        self._transcribe_started_at = None
        self.transcribe_progress_bar.hide()
        self.transcribe_progress_bar.setValue(0)
        self.transcribe_progress_label.hide()
        self.transcribe_progress_label.setText("")

    def set_task(self, task):
        """设置任务并更新UI"""
        self.task = task
        self.reset_ui()

    def stop(self):
        if hasattr(self, "transcript_thread"):
            self.transcript_thread.terminate()


class TranscriptionInterface(QWidget):
    """转录界面类,用于显示视频信息和转录进度"""

    finished = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAcceptDrops(True)
        self.task = None
        self.is_processing = False

        self._init_ui()
        self._setup_signals()
        self._set_value()

    def _init_ui(self):
        """初始化UI"""
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setObjectName("main_layout")
        self.main_layout.setSpacing(20)

        # 添加命令栏
        self._setup_command_bar()

        self.video_info_card = VideoInfoCard(self)
        self.main_layout.addWidget(self.video_info_card)

        # 添加转录设置卡片
        self.transcription_setting_card = TranscriptionSettingCard(self)
        self.main_layout.addWidget(self.transcription_setting_card)

    def _setup_command_bar(self):
        """设置命令栏"""
        self.command_bar = CommandBar(self)
        self.command_bar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.command_bar.setFixedHeight(40)

        # 添加打开文件按钮
        self.open_file_action = Action(FluentIcon.FOLDER, "Открыть файл")
        self.open_file_action.triggered.connect(self._on_file_select)
        self.command_bar.addAction(self.open_file_action)

        self.command_bar.addSeparator()

        # 添加转录模型选择按钮
        self.model_button = TransparentDropDownPushButton(
            "Модель распознавания", self, FluentIcon.SETTING
        )
        self.model_button.setFixedHeight(34)
        self.model_button.setMinimumWidth(180)

        self.model_menu = RoundMenu(parent=self)
        for model in TranscribeModelEnum:
            if "接口" in model.value or "api" in model.value.lower():
                self.model_menu.addActions(
                    [
                        Action(FluentIcon.GLOBE, model.value),
                    ]
                )
            else:
                self.model_menu.addActions(
                    [
                        Action(FluentIcon.ROBOT, model.value),
                    ]
                )
        self.model_button.setMenu(self.model_menu)
        self.command_bar.addWidget(self.model_button)

        self.main_layout.addWidget(self.command_bar)

    def _setup_signals(self):
        """设置信号连接"""
        self.video_info_card.finished.connect(self._on_transcript_finished)

        # 设置模型选择菜单的信号连接
        for action in self.model_menu.actions():
            action.triggered.connect(
                lambda checked, text=action.text(): self.on_transcription_model_changed(
                    text
                )
            )

        # 全局信号连接
        signalBus.transcription_model_changed.connect(
            self.on_transcription_model_changed
        )

    def _set_value(self):
        """设置转录模型"""
        model_name = cfg.get(cfg.transcribe_model).value
        # self.model_button.setText(self.tr(model_name))
        self.on_transcription_model_changed(model_name)

    def on_transcription_model_changed(self, model_name: str):
        """处理转录模型改变"""
        self.model_button.setText(self.tr(model_name))
        self.transcription_setting_card.on_model_changed(model_name)
        for model in TranscribeModelEnum:
            if model.value == model_name:
                cfg.set(cfg.transcribe_model, model)
                break

    def _on_transcript_finished(self, task: TranscribeTask):
        """转录完成处理"""
        self.is_processing = False
        if task.need_next_task:
            self.finished.emit(task.output_path, task.file_path)

            InfoBar.success(
                "Распознавание завершено",
                "Запускаю обработку субтитров...",
                duration=3000,
                position=InfoBarPosition.BOTTOM,
                parent=self.parent(),
            )

    def _on_file_select(self):
        """文件选择处理"""
        desktop_path = QStandardPaths.writableLocation(QStandardPaths.DesktopLocation)
        file_dialog = QFileDialog()

        video_formats = " ".join(f"*.{fmt.value}" for fmt in SupportedVideoFormats)
        audio_formats = " ".join(f"*.{fmt.value}" for fmt in SupportedAudioFormats)
        filter_str = f"Медиафайлы ({video_formats} {audio_formats});;Видеофайлы ({video_formats});;Аудиофайлы ({audio_formats})"

        file_path, _ = file_dialog.getOpenFileName(
            self, "Выберите медиафайл", desktop_path, filter_str
        )
        if file_path:
            self.update_info(file_path)

    def update_info(self, file_path):
        """设置UI"""
        self.video_info_thread = VideoInfoThread(file_path)
        self.video_info_thread.finished.connect(self.video_info_card.update_info)
        self.video_info_thread.error.connect(self._on_video_info_error)
        self.video_info_thread.start()

    def _on_video_info_error(self, error_msg):
        """处理视频信息提取错误"""
        self.is_processing = False
        InfoBar.error("Ошибка", self.tr(error_msg), duration=3000, parent=self)

    def set_task(self, task: TranscribeTask):
        """设置任务并更新UI"""
        self.task = task
        self.video_info_card.set_task(self.task)
        self.update_info(self.task.file_path)

    def process(self):
        """主处理函数"""
        self.is_processing = True
        self.video_info_card.start_transcription(need_create_task=False)

    def dragEnterEvent(self, event):
        """拖拽进入事件处理"""
        event.accept() if event.mimeData().hasUrls() else event.ignore()

    def dropEvent(self, event):
        """拖拽放下事件处理"""
        if self.is_processing:

            InfoBar.warning(
                "Внимание",
                "Идёт обработка. Дождитесь завершения текущей задачи",
                duration=3000,
                parent=self,
            )
            return

        files = [u.toLocalFile() for u in event.mimeData().urls()]
        for file_path in files:
            if not os.path.isfile(file_path):
                continue

            file_ext = os.path.splitext(file_path)[1][1:].lower()

            # 检查文件格式是否支持
            supported_formats = {fmt.value for fmt in SupportedVideoFormats} | {
                fmt.value for fmt in SupportedAudioFormats
            }
            is_supported = file_ext in supported_formats

            if is_supported:
                self.update_info(file_path)
                InfoBar.success(
                    "Импорт выполнен",
                    "Запускаю распознавание речи",
                    duration=3000,
                    parent=self,
                )
                break
            else:
                InfoBar.error(
                    "Неподдерживаемый формат: " + file_ext,
                    "Перетащите аудио или видео файл",
                    duration=3000,
                    parent=self,
                )

    def closeEvent(self, event):
        self.video_info_card.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    window = TranscriptionInterface()
    window.show()
    sys.exit(app.exec_())
