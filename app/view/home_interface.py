from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QSizePolicy, QStackedWidget, QVBoxLayout, QWidget
from qfluentwidgets import SegmentedWidget

from app.common.theme_manager import get_theme_palette
from app.core.task_factory import TaskFactory
from app.view.auto_shorts_interface import AutoShortsInterface
from app.view.anti_mate_interface import AntiMateInterface
from app.view.subtitle_interface import SubtitleInterface
from app.view.task_creation_interface import TaskCreationInterface
from app.view.transcription_interface import TranscriptionInterface
from app.view.video_translate_interface import VideoTranslateInterface
from app.view.video_synthesis_interface import VideoSynthesisInterface


class HomeInterface(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        # 设置对象名称和样式
        self.setObjectName("HomeInterface")
        self.refresh_theme()

        # 创建分段控件和堆叠控件
        self.pivot = SegmentedWidget(self)
        self.pivot.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        self.stackedWidget = QStackedWidget(self)
        self.vBoxLayout = QVBoxLayout(self)

        # 添加子界面
        self.task_creation_interface = TaskCreationInterface(self)
        self.transcription_interface = TranscriptionInterface(self)
        self.subtitle_optimization_interface = SubtitleInterface(self)
        self.video_synthesis_interface = VideoSynthesisInterface(self)
        self.auto_shorts_interface = AutoShortsInterface(self)
        self.anti_mate_interface = AntiMateInterface(self)
        self.video_translate_interface = VideoTranslateInterface(self)

        self.addSubInterface(
            self.auto_shorts_interface,
            "AutoShortsInterface",
            "Создание шортсов",
        )
        self.addSubInterface(
            self.task_creation_interface,
            "TaskCreationInterface",
            "Создание субтитров",
        )
        self.addSubInterface(
            self.transcription_interface, "TranscriptionInterface", "Распознавание речи"
        )
        self.addSubInterface(
            self.subtitle_optimization_interface,
            "SubtitleInterface",
            "Оптимизация и перевод субтитров",
        )
        self.addSubInterface(
            self.video_synthesis_interface,
            "VideoSynthesisInterface",
            "Синтез видео с субтитрами",
        )
        self.addSubInterface(
            self.anti_mate_interface,
            "AntiMateInterface",
            "Система Антимат",
        )
        self.addSubInterface(
            self.video_translate_interface,
            "VideoTranslateInterface",
            "Переозвучка/Перевод видео",
        )
        self.vBoxLayout.addWidget(self.pivot)
        self.vBoxLayout.addWidget(self.stackedWidget)
        self.vBoxLayout.setContentsMargins(30, 10, 30, 30)

        self.stackedWidget.currentChanged.connect(self.onCurrentIndexChanged)
        self.stackedWidget.setCurrentWidget(self.task_creation_interface)
        self.pivot.setCurrentItem("TaskCreationInterface")

        self.task_creation_interface.finished.connect(self.switch_to_transcription)
        self.transcription_interface.finished.connect(
            self.switch_to_subtitle_optimization
        )
        self.subtitle_optimization_interface.finished.connect(
            self.switch_to_video_synthesis
        )

    def switch_to_transcription(self, file_path):
        # 切换到转录界面
        transcribe_task = TaskFactory.create_transcribe_task(
            file_path, need_next_task=True
        )
        self.transcription_interface.set_task(transcribe_task)
        self.transcription_interface.process()
        self.stackedWidget.setCurrentWidget(self.transcription_interface)
        self.pivot.setCurrentItem("TranscriptionInterface")

    def switch_to_subtitle_optimization(self, file_path, video_path):
        # 切换到字幕处理界面
        subtitle_task = TaskFactory.create_subtitle_task(
            file_path, video_path, need_next_task=True
        )
        self.subtitle_optimization_interface.set_task(subtitle_task)
        self.subtitle_optimization_interface.process()
        self.stackedWidget.setCurrentWidget(self.subtitle_optimization_interface)
        self.pivot.setCurrentItem("SubtitleInterface")

    def switch_to_video_synthesis(self, video_path, subtitle_path):
        # 切换到视频合成界面
        synthesis_task = TaskFactory.create_synthesis_task(
            video_path, subtitle_path, need_next_task=True
        )
        self.video_synthesis_interface.set_task(synthesis_task)
        self.video_synthesis_interface.process()
        self.stackedWidget.setCurrentWidget(self.video_synthesis_interface)
        self.pivot.setCurrentItem("VideoSynthesisInterface")

    def addSubInterface(self, widget, objectName, text):
        # 添加子界面到堆叠控件和分段控件
        widget.setObjectName(objectName)
        self.stackedWidget.addWidget(widget)
        self.pivot.addItem(
            routeKey=objectName,
            text=text,
            onClick=lambda: self.stackedWidget.setCurrentWidget(widget),
        )

    def onCurrentIndexChanged(self, index):
        # 当堆叠控件的当前索引改变时，更新分段控件的当前项
        widget = self.stackedWidget.widget(index)
        if widget:
            self.pivot.setCurrentItem(widget.objectName())

    def closeEvent(self, event):
        # 关闭事件，关闭所有子界面
        self.task_creation_interface.close()
        self.transcription_interface.close()
        self.subtitle_optimization_interface.close()
        self.video_synthesis_interface.close()
        self.auto_shorts_interface.close()
        self.anti_mate_interface.close()
        self.video_translate_interface.close()
        super().closeEvent(event)

    def refresh_theme(self):
        p = get_theme_palette()
        self.setStyleSheet(f"HomeInterface{{background: {p['window_bg']};}}")
