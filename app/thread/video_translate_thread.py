import datetime

from PyQt5.QtCore import QThread, pyqtSignal

from app.core.entities import VideoTranslateTask
from app.core.utils.logger import setup_logger
from app.core.video_translate import VideoTranslationProcessor

logger = setup_logger("video_translate_thread")


class VideoTranslateThread(QThread):
    finished = pyqtSignal(VideoTranslateTask)
    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)

    def __init__(self, task: VideoTranslateTask):
        super().__init__()
        self.task = task
        self.processor = VideoTranslationProcessor()

    def run(self):
        try:
            logger.info("\n=========== видео-перевод: старт ===========")
            logger.info("时间：%s", datetime.datetime.now())

            result = self.processor.process(
                self.task,
                progress_cb=self._on_progress,
            )
            self.finished.emit(result)
        except Exception as e:
            logger.exception("Ошибка в video translate pipeline: %s", e)
            self.error.emit(str(e))

    def _on_progress(self, value: int, message: str):
        self.progress.emit(int(value), str(message))
