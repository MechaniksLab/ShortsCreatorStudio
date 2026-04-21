import os

import psutil
from PyQt5.QtCore import QSize, QThread, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QIcon
from PyQt5.QtWidgets import QApplication, QProgressDialog
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import (
    FluentWindow,
    MessageBox,
    NavigationAvatarWidget,
    NavigationItemPosition,
    SplashScreen,
)

from app.common.config import cfg
from app.common.theme_manager import apply_vscode_theme, get_theme_palette
from app.components.DonateDialog import DonateDialog
from app.config import APP_ICON_PATH, APP_NAME, APP_SPLASH_LOGO_PATH, GITHUB_REPO_URL
from app.core.github_update_manager import GitHubUpdateManager
from app.core.video_translate import VideoTranslateServiceManager
from app.thread.version_manager_thread import VersionManager
from app.view.batch_process_interface import BatchProcessInterface
from app.view.home_interface import HomeInterface
from app.view.setting_interface import SettingInterface
from app.view.subtitle_style_interface import SubtitleStyleInterface

LOGO_PATH = APP_ICON_PATH
APP_WINDOW_TITLE_RU = "Лаборатория Механика - Студия создания шортсов"


class MainWindow(FluentWindow):

    def __init__(self):
        super().__init__()
        self.initWindow()
        self.updateManager = GitHubUpdateManager()

        # 创建子界面
        self.homeInterface = HomeInterface(self)
        self.settingInterface = SettingInterface(self)
        self.subtitleStyleInterface = SubtitleStyleInterface(self)
        self.batchProcessInterface = BatchProcessInterface(self)

        # 初始化版本管理器
        self.versionManager = VersionManager()
        self.versionManager.newVersionAvailable.connect(self.onNewVersion)
        self.versionManager.announcementAvailable.connect(self.onAnnouncement)

        # 创建版本检查线程
        self.versionThread = QThread()
        self.versionManager.moveToThread(self.versionThread)
        self.versionThread.started.connect(self.versionManager.performCheck)
        self.versionThread.start()

        # 初始化导航界面
        self.initNavigation()
        apply_vscode_theme(refresh_widgets=True)
        self.splashScreen.finish()

        # Ненавязчивая проверка обновлений из репозитория
        try:
            if bool(cfg.checkUpdateAtStartUp.value):
                self._start_repo_update_check_async()
        except Exception:
            pass

        # 注册退出处理， 清理进程
        import atexit

        atexit.register(self.stop)

    def initNavigation(self):
        """初始化导航栏"""
        # 添加导航项
        self.addSubInterface(self.homeInterface, FIF.HOME, "Главная")
        self.addSubInterface(self.batchProcessInterface, FIF.VIDEO, "Пакетная обработка")
        self.addSubInterface(self.subtitleStyleInterface, FIF.FONT, "Стиль субтитров")

        self.navigationInterface.addSeparator()

        # 在底部添加自定义小部件
        self.navigationInterface.addItem(
            routeKey="avatar",
            text="GitHub",
            icon=FIF.GITHUB,
            onClick=self.onGithubDialog,
            position=NavigationItemPosition.BOTTOM,
        )
        self.addSubInterface(
            self.settingInterface,
            FIF.SETTING,
            self.tr("Settings"),
            NavigationItemPosition.BOTTOM,
        )

        # 设置默认界面
        self.switchTo(self.homeInterface)

    def switchTo(self, interface):
        self.setWindowTitle(APP_WINDOW_TITLE_RU)
        self.stackedWidget.setCurrentWidget(interface, popOut=False)

    def initWindow(self):
        """初始化窗口"""
        self.resize(1050, 800)
        self.setMinimumWidth(700)
        self.setWindowIcon(QIcon(str(LOGO_PATH)))
        self.setWindowTitle(APP_WINDOW_TITLE_RU)

        self.setMicaEffectEnabled(cfg.get(cfg.micaEnabled))

        # 创建启动画面
        self.splashScreen = SplashScreen(QIcon(str(APP_SPLASH_LOGO_PATH)), self)
        self.splashScreen.setIconSize(QSize(170, 170))
        self.splashScreen.raise_()

        # 设置窗口位置, 居中
        desktop = QApplication.desktop().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w // 2 - self.width() // 2, h // 2 - self.height() // 2)

        self.show()
        QApplication.processEvents()

    def onGithubDialog(self):
        """打开GitHub"""
        w = MessageBox(
            "Информация о GitHub",
            "MechaniksLab Creator Studio разработан автором как независимый проект и размещён на GitHub."
            " Буду рад вашим Star и Fork. Если столкнётесь с проблемами или багами —"
            " пожалуйста, создайте Issue.\n\n https://github.com/MechaniksLab/ShortsCreatorStudio",
            self,
        )
        w.yesButton.setText("Открыть GitHub")
        w.cancelButton.setText("Поддержать автора")
        if w.exec():
            QDesktopServices.openUrl(QUrl(GITHUB_REPO_URL))
        else:
            # 点击"支持作者"按钮时打开捐赠对话框
            donate_dialog = DonateDialog(self)
            donate_dialog.exec_()

    def onNewVersion(self, version, force_update, update_info, download_url):
        """新版本提示"""
        title = "Доступна новая версия" if not force_update else "Текущая версия отключена"
        content = f"Найдена новая версия {version}\n\n{update_info}"
        w = MessageBox(title, content, self)
        w.yesButton.setText("Обновить сейчас")
        w.cancelButton.setText("Позже" if not force_update else "Выйти из программы")
        if w.exec():
            QDesktopServices.openUrl(QUrl(download_url))
        if force_update:
            QApplication.quit()

    def onAnnouncement(self, content):
        """显示公告"""
        w = MessageBox("Объявление", content, self)
        w.yesButton.setText("Понятно")
        w.cancelButton.hide()
        w.exec()

    def _check_repo_update_silently(self):
        info = self.updateManager.check_update()
        if not info.get("has_update"):
            return

        latest = info.get("latest") or {}
        message = str(latest.get("message") or "").strip()
        short_sha = str(latest.get("sha") or "")[:8]

        text = f"Доступно обновление из GitHub ({short_sha})."
        if message:
            text += f"\n\nПоследний коммит:\n{message[:400]}"

        box = MessageBox("Доступно обновление", text, self)
        box.yesButton.setText("Обновить и перезапустить")
        box.cancelButton.setText("Позже")
        if box.exec():
            result = self.updateManager.apply_update_and_restart()
            if result.get("ok"):
                QApplication.quit()
            else:
                err = str(result.get("error") or "Не удалось применить обновление")
                fail_box = MessageBox("Ошибка обновления", err, self)
                fail_box.yesButton.setText("ОК")
                fail_box.cancelButton.hide()
                fail_box.exec()

    def _start_repo_update_check_async(self):
        self.repoUpdateThread = RepoUpdateCheckThread()
        self.repoUpdateThread.finishedCheck.connect(self._on_repo_update_check_finished)
        self.repoUpdateThread.failed.connect(self._on_repo_update_check_failed)
        self.repoUpdateThread.start()

    def _on_repo_update_check_finished(self, info):
        if not isinstance(info, dict):
            return
        if not info.get("has_update"):
            return

        latest = info.get("latest") or {}
        message = str(latest.get("message") or "").strip()
        short_sha = str(latest.get("sha") or "")[:8]

        text = f"Доступно обновление из GitHub ({short_sha})."
        if message:
            text += f"\n\nПоследний коммит:\n{message[:400]}"

        box = MessageBox("Доступно обновление", text, self)
        box.yesButton.setText("Обновить и перезапустить")
        box.cancelButton.setText("Позже")
        if box.exec():
            self._start_apply_update_with_progress()

    def _on_repo_update_check_failed(self, _error: str):
        # Проверка на старте должна быть максимально ненавязчивой.
        pass

    def _start_apply_update_with_progress(self):
        self.updateProgressDialog = QProgressDialog("Подготовка обновления...", None, 0, 100, self)
        self.updateProgressDialog.setWindowTitle("Обновление")
        self.updateProgressDialog.setCancelButton(None)
        self.updateProgressDialog.setAutoClose(False)
        self.updateProgressDialog.setAutoReset(False)
        self.updateProgressDialog.setMinimumDuration(0)
        self.updateProgressDialog.setValue(0)
        self._style_update_progress_dialog()
        self.updateProgressDialog.show()

        self.applyUpdateThread = ApplyUpdateThread()
        self.applyUpdateThread.progressChanged.connect(self._on_apply_update_progress)
        self.applyUpdateThread.finishedResult.connect(self._on_apply_update_finished)
        self.applyUpdateThread.start()

    def _on_apply_update_progress(self, percent: int, text: str):
        if hasattr(self, "updateProgressDialog") and self.updateProgressDialog is not None:
            self.updateProgressDialog.setLabelText(text)
            self.updateProgressDialog.setValue(max(0, min(100, int(percent))))

    def _on_apply_update_finished(self, result: object):
        if hasattr(self, "updateProgressDialog") and self.updateProgressDialog is not None:
            self.updateProgressDialog.close()

        result = result if isinstance(result, dict) else {}
        if result.get("ok"):
            QApplication.quit()
            return

        err = str(result.get("error") or "Не удалось применить обновление")
        fail_box = MessageBox("Ошибка обновления", err, self)
        fail_box.yesButton.setText("ОК")
        fail_box.cancelButton.hide()
        fail_box.exec()

    def _style_update_progress_dialog(self):
        if not hasattr(self, "updateProgressDialog") or self.updateProgressDialog is None:
            return
        p = get_theme_palette()
        self.updateProgressDialog.setStyleSheet(
            f"""
            QProgressDialog {{
                background-color: {p['card_bg']};
                color: {p['text']};
                border: 1px solid {p['border']};
                border-radius: 8px;
            }}
            QProgressBar {{
                border: 1px solid {p['border']};
                border-radius: 6px;
                text-align: center;
                background: {p['panel_bg']};
                color: {p['text']};
            }}
            QProgressBar::chunk {{
                background-color: {p['accent']};
                border-radius: 5px;
            }}
            QLabel {{
                color: {p['text']};
            }}
            """
        )

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if hasattr(self, "splashScreen"):
            self.splashScreen.resize(self.size())

    def closeEvent(self, event):
        # 关闭所有子界面
        # self.homeInterface.close()
        # self.batchProcessInterface.close()
        # self.subtitleStyleInterface.close()
        # self.settingInterface.close()
        super().closeEvent(event)

        # 强制退出应用程序
        QApplication.quit()

        # 确保所有线程和进程都被终止 要是一些错误退出就不会处理了。
        # import os
        # os._exit(0)

    def stop(self):
        # Остановка встроенных локальных сервисов (video translate)
        try:
            VideoTranslateServiceManager.instance().shutdown_all()
        except Exception:
            pass

        # Закрываем только медиа-процессы, не трогаем updater/restart скрипты
        process = psutil.Process(os.getpid())
        kill_names = {
            "ffmpeg.exe",
            "ffprobe.exe",
            "faster-whisper-xxl.exe",
            "faster-whisper-xxl-cpu.exe",
        }
        for child in process.children(recursive=True):
            try:
                name = str(child.name() or "").lower()
                if name in kill_names:
                    child.kill()
            except Exception:
                pass


class RepoUpdateCheckThread(QThread):
    finishedCheck = pyqtSignal(object)
    failed = pyqtSignal(str)

    def run(self):
        try:
            info = GitHubUpdateManager().check_update()
            self.finishedCheck.emit(info)
        except Exception as e:
            self.failed.emit(str(e))


class ApplyUpdateThread(QThread):
    progressChanged = pyqtSignal(int, str)
    finishedResult = pyqtSignal(object)

    def run(self):
        manager = GitHubUpdateManager()
        try:
            result = manager.apply_update_and_restart(
                progress_cb=lambda p, t: self.progressChanged.emit(int(p), str(t))
            )
        except Exception as e:
            result = {"ok": False, "error": str(e)}
        self.finishedResult.emit(result)
