from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QProgressBar,
    QFileDialog,
    QSizePolicy,
)
from PyQt5.QtGui import QDesktopServices, QColor, QFont
from PyQt5.QtCore import QUrl
from qfluentwidgets import (
    ComboBox,
    PushButton,
    TableWidget,
    ProgressBar,
    InfoBar,
    InfoBarPosition,
    RoundMenu,
    Action,
    FluentIcon as FIF,
)
import os

from app.thread.batch_process_thread import (
    BatchProcessThread,
    BatchTask,
    BatchTaskStatus,
)
from app.core.entities import (
    SupportedAudioFormats,
    SupportedVideoFormats,
    SupportedSubtitleFormats,
)
from app.core.entities import BatchTaskType, BatchTaskStatus


class BatchProcessInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("batchProcessInterface")
        self.setWindowTitle("Пакетная обработка")
        self.setAcceptDrops(True)
        self.batch_thread = BatchProcessThread()

        self.init_ui()
        self.setup_connections()

    def init_ui(self):
        # Создаём основной макет
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(8)

        # Верхняя панель управления
        top_layout = QHBoxLayout()
        top_layout.setSpacing(8)

        # Выбор типа задачи
        self.task_type_combo = ComboBox()
        self.task_type_combo.addItems([str(task_type) for task_type in BatchTaskType])
        self.task_type_combo.setCurrentText(str(BatchTaskType.FULL_PROCESS))

        # Кнопки управления
        self.add_file_btn = PushButton("Добавить файлы", icon=FIF.ADD)
        self.start_all_btn = PushButton("Запустить обработку", icon=FIF.PLAY)
        self.clear_btn = PushButton("Очистить список", icon=FIF.DELETE)

        # Добавляем элементы на верхнюю панель
        top_layout.addWidget(self.task_type_combo)
        top_layout.addWidget(self.add_file_btn)
        top_layout.addWidget(self.clear_btn)

        top_layout.addStretch()
        top_layout.addWidget(self.start_all_btn)

        # Создаём таблицу задач
        self.task_table = TableWidget()
        self.task_table.setColumnCount(3)
        self.task_table.setHorizontalHeaderLabels(["Имя файла", "Прогресс", "Статус"])

        # Настройки внешнего вида таблицы
        self.task_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.task_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self.task_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self.task_table.setColumnWidth(1, 250)  # ширина колонки прогресса
        self.task_table.setColumnWidth(2, 160)  # ширина колонки статуса

        # Высота строк
        self.task_table.verticalHeader().setDefaultSectionSize(40)  # высота строки по умолчанию

        # Границы таблицы
        self.task_table.setBorderVisible(True)
        self.task_table.setBorderRadius(12)

        # Запрещаем редактирование таблицы
        self.task_table.setEditTriggers(QTableWidget.NoEditTriggers)

        # Политика изменения размера таблицы
        self.task_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.task_table.setMinimumHeight(300)  # минимальная высота

        # Подключаем сигнал двойного клика
        self.task_table.doubleClicked.connect(self.on_table_double_clicked)

        # Добавляем в основной макет
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.task_table)

        # Подключаем сигналы
        self.add_file_btn.clicked.connect(self.on_add_file_clicked)
        self.start_all_btn.clicked.connect(self.start_all_tasks)
        self.clear_btn.clicked.connect(self.clear_tasks)
        self.task_type_combo.currentTextChanged.connect(self.on_task_type_changed)

    def setup_connections(self):
        # Сигналы потока пакетной обработки
        self.batch_thread.task_progress.connect(self.update_task_progress)
        self.batch_thread.task_error.connect(self.on_task_error)
        self.batch_thread.task_completed.connect(self.on_task_completed)

        # Контекстное меню таблицы
        self.task_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.task_table.customContextMenuRequested.connect(self.show_context_menu)

    def on_add_file_clicked(self):
        task_type = self.task_type_combo.currentText()
        file_filter = ""
        if task_type in [
            BatchTaskType.TRANSCRIBE,
            BatchTaskType.TRANS_SUB,
            BatchTaskType.FULL_PROCESS,
        ]:
            # Получаем все поддерживаемые аудио/видео форматы
            audio_formats = [f"*.{fmt.value}" for fmt in SupportedAudioFormats]
            video_formats = [f"*.{fmt.value}" for fmt in SupportedVideoFormats]
            formats = audio_formats + video_formats
            file_filter = f"Аудио/видео файлы ({' '.join(formats)})"
        elif task_type == BatchTaskType.SUBTITLE:
            # Получаем все поддерживаемые форматы субтитров
            subtitle_formats = [f"*.{fmt.value}" for fmt in SupportedSubtitleFormats]
            file_filter = f"Файлы субтитров ({' '.join(subtitle_formats)})"

        files, _ = QFileDialog.getOpenFileNames(self, "Выберите файлы", "", file_filter)
        if files:
            self.add_files(files)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [url.toLocalFile() for url in event.mimeData().urls()]
        self.add_files(files)

    def add_files(self, file_paths):
        task_type = BatchTaskType(self.task_type_combo.currentText())

        # Проверяем существование файлов и собираем отсутствующие
        non_existent_files = []
        valid_files = []
        for file_path in file_paths:
            if not os.path.exists(file_path):
                non_existent_files.append(os.path.basename(file_path))
            else:
                valid_files.append(file_path)

        # Если есть отсутствующие файлы — показываем предупреждение
        if non_existent_files:
            InfoBar.warning(
                title="Файлы не найдены",
                content=f"Следующие файлы не существуют:\n{', '.join(non_existent_files)}",
                duration=3000,
                position=InfoBarPosition.TOP,
                parent=self,
            )

        # Если нет валидных файлов — выходим
        if not valid_files:
            return

        # Сортируем валидные файлы по имени
        valid_files.sort(key=lambda x: os.path.basename(x).lower())

        # Если таблица пустая, автодетектим тип файла и переключаем тип задачи
        if self.task_table.rowCount() == 0 and self.task_type_combo.currentIndex() == 0:
            first_file = valid_files[0].lower()
            is_subtitle = any(
                first_file.endswith(f".{fmt.value}") for fmt in SupportedSubtitleFormats
            )
            is_media = any(
                first_file.endswith(f".{fmt.value}") for fmt in SupportedAudioFormats
            ) or any(
                first_file.endswith(f".{fmt.value}") for fmt in SupportedVideoFormats
            )
            if is_subtitle:
                self.task_type_combo.setCurrentText(str(BatchTaskType.SUBTITLE))
                task_type = BatchTaskType.SUBTITLE
            # elif is_media:
            #     self.task_type_combo.setCurrentText(str(BatchTaskType.FULL_PROCESS))
            #     task_type = BatchTaskType.FULL_PROCESS

        # Фильтруем файлы по типу
        valid_files = self.filter_files(valid_files, task_type)

        if not valid_files:
            InfoBar.warning(
                title="Некорректные файлы",
                content="Выберите файлы подходящего типа",
                duration=3000,
                position=InfoBarPosition.TOP,
                parent=self,
            )
            return

        for file_path in valid_files:
            # Проверяем, не добавлена ли уже такая задача
            exists = False
            for row in range(self.task_table.rowCount()):
                if self.task_table.item(row, 0).toolTip() == file_path:
                    exists = True
                    InfoBar.warning(
                        title="Задача уже существует",
                        content="Этот файл уже добавлен в очередь",
                        duration=2000,
                        position=InfoBarPosition.TOP_RIGHT,
                        parent=self,
                    )
                    break

            if not exists:
                self.add_task_to_table(file_path)

    def filter_files(self, file_paths, task_type: BatchTaskType):
        valid_extensions = {}

        # Поддерживаемые расширения зависят от типа задачи
        if task_type in [
            BatchTaskType.TRANSCRIBE,
            BatchTaskType.TRANS_SUB,
            BatchTaskType.FULL_PROCESS,
        ]:
            valid_extensions = {f".{fmt.value}" for fmt in SupportedAudioFormats} | {
                f".{fmt.value}" for fmt in SupportedVideoFormats
            }
        elif task_type == BatchTaskType.SUBTITLE:
            valid_extensions = {f".{fmt.value}" for fmt in SupportedSubtitleFormats}

        return [
            f
            for f in file_paths
            if any(f.lower().endswith(ext) for ext in valid_extensions)
        ]

    def add_task_to_table(self, file_path):
        row = self.task_table.rowCount()
        self.task_table.insertRow(row)

        # Имя файла
        file_name = QTableWidgetItem(os.path.basename(file_path))
        file_name.setToolTip(file_path)
        self.task_table.setItem(row, 0, file_name)

        # Прогресс-бар
        progress_bar = ProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        progress_bar.setFixedHeight(18)
        self.task_table.setCellWidget(row, 1, progress_bar)

        # Статус
        status = QTableWidgetItem(str(BatchTaskStatus.WAITING))
        status.setTextAlignment(Qt.AlignCenter)
        status.setForeground(Qt.gray)  # цвет текста статуса по умолчанию
        font = QFont()
        font.setBold(True)
        status.setFont(font)
        self.task_table.setItem(row, 2, status)

    def show_context_menu(self, pos):
        row = self.task_table.rowAt(pos.y())
        if row < 0:
            return

        menu = RoundMenu(parent=self)
        file_path = self.task_table.item(row, 0).toolTip()
        status = self.task_table.item(row, 2).text()

        start_action = Action(FIF.PLAY, "Запустить")
        start_action.triggered.connect(lambda: self.start_task(file_path))
        menu.addAction(start_action)

        cancel_action = Action(FIF.CLOSE, "Отменить")
        cancel_action.triggered.connect(lambda: self.cancel_task(file_path))
        menu.addAction(cancel_action)

        menu.addSeparator()
        open_folder_action = Action(FIF.FOLDER, "Открыть папку вывода")
        open_folder_action.triggered.connect(lambda: self.open_output_folder(file_path))
        menu.addAction(open_folder_action)

        if status != str(BatchTaskStatus.WAITING):
            start_action.setEnabled(False)

        menu.exec_(self.task_table.viewport().mapToGlobal(pos))

    def open_output_folder(self, file_path: str):
        # Определяем папку вывода по типу задачи и пути файла
        task_type = BatchTaskType(self.task_type_combo.currentText())
        file_dir = os.path.dirname(file_path)

        if task_type == BatchTaskType.FULL_PROCESS:
            # Для полного процесса вывод рядом с исходным видео
            output_dir = file_dir
        else:
            # Для остальных задач вывод рядом с исходным файлом
            output_dir = file_dir

        # Открываем папку
        QDesktopServices.openUrl(QUrl.fromLocalFile(output_dir))

    def update_task_progress(self, file_path: str, progress: int, status: str):
        for row in range(self.task_table.rowCount()):
            if self.task_table.item(row, 0).toolTip() == file_path:
                # Обновляем прогресс-бар
                progress_bar = self.task_table.cellWidget(row, 1)
                progress_bar.setValue(progress)
                # Обновляем статус
                self.task_table.item(row, 2).setText(status)
                break

    def on_task_error(self, file_path: str, error: str):
        for row in range(self.task_table.rowCount()):
            if self.task_table.item(row, 0).toolTip() == file_path:
                status_item = self.task_table.item(row, 2)
                status_item.setText(str(BatchTaskStatus.FAILED))
                status_item.setToolTip(error)
                break

    def on_task_completed(self, file_path: str):
        for row in range(self.task_table.rowCount()):
            if self.task_table.item(row, 0).toolTip() == file_path:
                self.task_table.item(row, 2).setText(str(BatchTaskStatus.COMPLETED))
                self.task_table.item(row, 2).setForeground(QColor("#13A10E"))
                break

    def start_all_tasks(self):
        # Проверяем, есть ли задачи
        if self.task_table.rowCount() == 0:
            InfoBar.warning(
                title="Нет задач",
                content="Сначала добавьте файлы для обработки",
                duration=2000,
                position=InfoBarPosition.TOP,
                parent=self,
            )
            return

        # Проверяем, есть ли задачи в ожидании
        waiting_tasks = 0
        for row in range(self.task_table.rowCount()):
            if self.task_table.item(row, 2).text() == str(BatchTaskStatus.WAITING):
                waiting_tasks += 1

        if waiting_tasks == 0:
            InfoBar.warning(
                title="Нет ожидающих задач",
                content="Все задачи уже выполняются или завершены",
                duration=2000,
                position=InfoBarPosition.TOP,
                parent=self,
            )
            return

        # Показываем уведомление о старте обработки
        InfoBar.success(
            title="Запуск обработки",
            content=f"Запущена обработка задач: {waiting_tasks}",
            duration=2000,
            position=InfoBarPosition.TOP,
            parent=self,
        )
        # Запускаем обработку задач
        for row in range(self.task_table.rowCount()):
            file_path = self.task_table.item(row, 0).toolTip()
            status = self.task_table.item(row, 2).text()
            if status == str(BatchTaskStatus.WAITING):
                task_type = BatchTaskType(self.task_type_combo.currentText())
                batch_task = BatchTask(file_path, task_type)
                self.batch_thread.add_task(batch_task)

    def start_task(self, file_path: str):
        # Показываем уведомление о старте обработки
        file_name = os.path.basename(file_path)
        InfoBar.success(
            title="Запуск обработки",
            content=f"Запущена обработка файла: {file_name}",
            duration=2000,
            position=InfoBarPosition.TOP,
            parent=self,
        )

        # Создаём и добавляем одну задачу
        task_type = BatchTaskType(self.task_type_combo.currentText())
        batch_task = BatchTask(file_path, task_type)
        self.batch_thread.add_task(batch_task)

    def cancel_task(self, file_path: str):
        self.batch_thread.stop_task(file_path)
        # Удаляем задачу из таблицы
        for row in range(self.task_table.rowCount()):
            if self.task_table.item(row, 0).toolTip() == file_path:
                self.task_table.removeRow(row)
                break

    def clear_tasks(self):
        self.batch_thread.stop_all()
        self.task_table.setRowCount(0)

    def on_task_type_changed(self, task_type):
        # Очищаем текущий список задач
        self.clear_tasks()

    def closeEvent(self, event):
        self.batch_thread.stop_all()
        super().closeEvent(event)

    def on_table_double_clicked(self, index):
        """Обрабатывает двойной клик по строке таблицы."""
        row = index.row()
        file_path = self.task_table.item(row, 0).toolTip()
        self.open_output_folder(file_path)
