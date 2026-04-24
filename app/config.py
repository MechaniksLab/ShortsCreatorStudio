import logging
import os
from pathlib import Path

VERSION = "v1.5.0"
YEAR = 2026
APP_NAME = "MechaniksLab Creator Studio"
AUTHOR = "MechaniksLab"

HELP_URL = "https://github.com/MechaniksLab/ShortsCreatorStudio"
GITHUB_REPO_URL = "https://github.com/MechaniksLab/ShortsCreatorStudio"
RELEASE_URL = "https://github.com/MechaniksLab/ShortsCreatorStudio/releases/latest"
FEEDBACK_URL = "https://github.com/MechaniksLab/ShortsCreatorStudio/issues"
UPDATE_REPO_OWNER = "MechaniksLab"
UPDATE_REPO_NAME = "ShortsCreatorStudio"
UPDATE_REPO_BRANCH = "master"

# 路径
ROOT_PATH = Path(__file__).parent
PROJECT_ROOT = ROOT_PATH.parent

RESOURCE_PATH = ROOT_PATH.parent / "resource"
ICONS_PATH = ROOT_PATH.parent / "icons"
APPDATA_PATH = ROOT_PATH.parent / "AppData"
WORK_PATH = ROOT_PATH.parent / "work-dir"
RUNTIME_SCRIPTS_PATH = PROJECT_ROOT / "runtime" / "Scripts"


BIN_PATH = RESOURCE_PATH / "bin"
ASSETS_PATH = RESOURCE_PATH / "assets"
SUBTITLE_STYLE_PATH = RESOURCE_PATH / "subtitle_style"
APP_ICON_PATH = ICONS_PATH / "favicon.ico"
APP_LOGO_PATH = ICONS_PATH / "android-chrome-192x192.png"
APP_SPLASH_LOGO_PATH = ICONS_PATH / "android-chrome-512x512.png"

LOG_PATH = APPDATA_PATH / "logs"
SETTINGS_PATH = APPDATA_PATH / "settings.json"
CACHE_PATH = APPDATA_PATH / "cache"
MODEL_PATH = APPDATA_PATH / "models"

FASER_WHISPER_PATH = BIN_PATH / "Faster-Whisper-XXL"

# 日志配置
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# 环境变量添加 bin 路径，添加到PATH开头以优先使用
os.environ["PATH"] = str(BIN_PATH) + os.pathsep + os.environ["PATH"]
os.environ["PATH"] = str(FASER_WHISPER_PATH) + os.pathsep + os.environ["PATH"]
if RUNTIME_SCRIPTS_PATH.exists():
    os.environ["PATH"] = str(RUNTIME_SCRIPTS_PATH) + os.pathsep + os.environ["PATH"]

# 添加 VLC 路径
os.environ["PYTHON_VLC_MODULE_PATH"] = str(BIN_PATH / "vlc")

# 创建路径
for p in [CACHE_PATH, LOG_PATH, WORK_PATH, MODEL_PATH]:
    p.mkdir(parents=True, exist_ok=True)
