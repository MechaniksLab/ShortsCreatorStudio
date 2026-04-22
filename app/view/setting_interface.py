import os
import subprocess
import webbrowser
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QColor
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import ComboBoxSettingCard, CustomColorSettingCard, ExpandLayout
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import (
    HyperlinkCard,
    InfoBar,
    MessageBox,
    OptionsSettingCard,
    PrimaryPushSettingCard,
    PushSettingCard,
    RangeSettingCard,
    ScrollArea,
    SettingCardGroup,
    SwitchSettingCard,
)

from app.common.config import cfg
from app.common.theme_manager import apply_vscode_theme, get_theme_palette
from app.common.signal_bus import signalBus
from app.components.EditComboBoxSettingCard import EditComboBoxSettingCard
from app.components.LineEditSettingCard import LineEditSettingCard
from app.config import APP_NAME, AUTHOR, FEEDBACK_URL, HELP_URL, LOG_PATH, RELEASE_URL, VERSION, YEAR
from app.config import MODEL_PATH, PROJECT_ROOT
from app.core.entities import LLMServiceEnum, TranscribeModelEnum, TranslatorServiceEnum, language_value_to_ru, TargetLanguageEnum
from app.core.utils.test_opanai import get_openai_models, test_openai
from app.core.github_update_manager import GitHubUpdateManager
from app.thread.version_manager_thread import VersionManager
from app.components.MySettingCard import ComboBoxSettingCard as MyComboBoxSettingCard
from app.components.MySettingCard import ColorSettingCard


class SettingInterface(ScrollArea):
    """设置界面"""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle("Настройки")
        self.githubUpdateManager = GitHubUpdateManager()
        self.scrollWidget = QWidget()
        self.expandLayout = ExpandLayout(self.scrollWidget)
        self._vt_install_thread = None
        self._vt_install_dialog = None
        self.settingLabel = QLabel("Настройки", self)

        # 初始化所有设置组
        self.__initGroups()
        # 初始化所有配置卡片
        self.__initCards()
        # 初始化界面
        self.__initWidget()
        # 初始化布局
        self.__initLayout()
        # 连接信号和槽
        self.__connectSignalToSlot()

    def __initGroups(self):
        """初始化所有设置组"""
        # 转录配置组
        self.transcribeGroup = SettingCardGroup("Параметры распознавания", self.scrollWidget)
        # LLM配置组
        self.llmGroup = SettingCardGroup("Параметры LLM", self.scrollWidget)
        # 翻译服务组
        self.translate_serviceGroup = SettingCardGroup(
            "Сервис перевода", self.scrollWidget
        )
        # 翻译与优化组
        self.translateGroup = SettingCardGroup("Перевод и оптимизация", self.scrollWidget)
        # 视频-перевод с клонированием голоса
        self.videoTranslateGroup = SettingCardGroup(
            "Перевод видео", self.scrollWidget
        )
        # 字幕合成配置组
        self.subtitleGroup = SettingCardGroup(
            "Параметры синтеза субтитров", self.scrollWidget
        )
        # 保存配置组
        self.saveGroup = SettingCardGroup("Сохранение", self.scrollWidget)
        # 个性化组
        self.personalGroup = SettingCardGroup("Персонализация", self.scrollWidget)
        # 关于组
        self.aboutGroup = SettingCardGroup("О программе", self.scrollWidget)
        self.updateGroup = SettingCardGroup("Обновления из GitHub", self.scrollWidget)

    def __initCards(self):
        """初始化所有配置卡片"""
        # 转录配置卡片
        self.transcribeModelCard = ComboBoxSettingCard(
            cfg.transcribe_model,
            FIF.MICROPHONE,
            "Модель распознавания",
            "Модель ASR для преобразования речи в текст",
            texts=[model.value for model in cfg.transcribe_model.validator.options],
            parent=self.transcribeGroup,
        )

        # LLM配置卡片
        self.__createLLMServiceCards()

        # 翻译配置卡片
        self.__createTranslateServiceCards()

        # 翻译与优化配置卡片
        self.subtitleCorrectCard = SwitchSettingCard(
            FIF.EDIT,
            "Коррекция субтитров",
            "Выполнять коррекцию сгенерированных субтитров",
            cfg.need_optimize,
            self.translateGroup,
        )
        self.subtitleTranslateCard = SwitchSettingCard(
            FIF.LANGUAGE,
            "Перевод субтитров",
            "Выполнять перевод субтитров в процессе обработки",
            cfg.need_translate,
            self.translateGroup,
        )
        self.targetLanguageCard = ComboBoxSettingCard(
            cfg.target_language,
            FIF.LANGUAGE,
            "Целевой язык",
            "Выберите язык перевода субтитров",
            texts=[language_value_to_ru(lang.value) for lang in cfg.target_language.validator.options],
            parent=self.translateGroup,
        )

        # Видео-перевод
        self.videoTranslateTargetLanguageCard = ComboBoxSettingCard(
            cfg.video_translate_target_language,
            FIF.LANGUAGE,
            "Язык дубляжа",
            "Целевой язык для перевода речи в видео",
            texts=[language_value_to_ru(lang.value) for lang in cfg.video_translate_target_language.validator.options],
            parent=self.videoTranslateGroup,
        )
        self.videoTranslateVoiceProviderCard = ComboBoxSettingCard(
            cfg.video_translate_voice_provider,
            FIF.MICROPHONE,
            "Провайдер клонирования голоса",
            "Движок TTS/Voice Clone для озвучки",
            texts=[
                "auto",
                "xtts",
                "openvoice",
                "fish_speech",
                "elevenlabs",
                "azure",
                "cartesia",
            ],
            parent=self.videoTranslateGroup,
        )
        self.videoTranslateVoiceQualityCard = ComboBoxSettingCard(
            cfg.video_translate_voice_quality,
            FIF.SPEED_HIGH,
            "Качество озвучки",
            "Профиль качества/скорости голосового клонирования",
            texts=["fast", "balanced", "high", "studio"],
            parent=self.videoTranslateGroup,
        )
        self.videoTranslateVoiceRefModeCard = ComboBoxSettingCard(
            cfg.video_translate_voice_reference_mode,
            FIF.PEOPLE,
            "Режим voice reference",
            "Auto — автоматически искать эталон голоса, Manual — использовать JSON mapping",
            texts=["auto", "manual"],
            parent=self.videoTranslateGroup,
        )
        self.videoTranslateEnableDiarizationCard = SwitchSettingCard(
            FIF.PEOPLE,
            "Определение спикеров (Diarization)",
            "Определять разных говорящих и поддерживать их раздельные голоса",
            cfg.video_translate_enable_diarization,
            self.videoTranslateGroup,
        )
        self.videoTranslateExpectedSpeakersCard = RangeSettingCard(
            cfg.video_translate_expected_speaker_count,
            FIF.PEOPLE,
            "Ожидаемое число спикеров",
            "0 = автоопределение. Полезно для повышения стабильности diarization",
            parent=self.videoTranslateGroup,
        )
        self.videoTranslateSourceSeparationCard = SwitchSettingCard(
            FIF.MUSIC,
            "Разделение голоса и фона",
            "Отделять речь от музыки/шума перед переводом",
            cfg.video_translate_enable_source_separation,
            self.videoTranslateGroup,
        )
        self.videoTranslateKeepMusicCard = SwitchSettingCard(
            FIF.MUSIC,
            "Сохранять фоновую музыку",
            "Подмешивать оригинальный фон к переведённой озвучке",
            cfg.video_translate_keep_background_music,
            self.videoTranslateGroup,
        )
        self.videoTranslateEnableLipsyncCard = SwitchSettingCard(
            FIF.VIDEO,
            "Липсинк (экспериментально)",
            "Дополнительная подгонка артикуляции под новый язык",
            cfg.video_translate_enable_lipsync,
            self.videoTranslateGroup,
        )
        self.videoTranslateManualVoiceMapCard = LineEditSettingCard(
            cfg.video_translate_manual_voice_map_json,
            FIF.DICTIONARY,
            "Manual Voice Map (JSON)",
            "Пример: {\"default\":\"VOICE_ID\",\"spk_1\":\"VOICE_ID_2\"}",
            "{}",
            self.videoTranslateGroup,
        )
        self.videoTranslateElevenLabsApiCard = LineEditSettingCard(
            cfg.video_translate_elevenlabs_api_key,
            FIF.FINGERPRINT,
            "ElevenLabs API Key",
            "Ключ API ElevenLabs для облачного voice cloning",
            "",
            self.videoTranslateGroup,
        )
        self.videoTranslateAzureKeyCard = LineEditSettingCard(
            cfg.video_translate_azure_speech_key,
            FIF.FINGERPRINT,
            "Azure Speech Key",
            "Ключ Azure Speech (если используете provider=azure)",
            "",
            self.videoTranslateGroup,
        )
        self.videoTranslateAzureRegionCard = LineEditSettingCard(
            cfg.video_translate_azure_speech_region,
            FIF.GLOBE,
            "Azure Region",
            "Регион Azure Speech, например: westeurope",
            "",
            self.videoTranslateGroup,
        )
        self.videoTranslateCartesiaApiCard = LineEditSettingCard(
            cfg.video_translate_cartesia_api_key,
            FIF.FINGERPRINT,
            "Cartesia API Key",
            "Ключ Cartesia (если используете provider=cartesia)",
            "",
            self.videoTranslateGroup,
        )
        self.videoTranslateXttsPathCard = LineEditSettingCard(
            cfg.video_translate_xtts_model_path,
            FIF.FOLDER,
            "Путь к XTTS модели (опционально)",
            "Можно оставить пустым: XTTS загрузится/используется автоматически из runtime",
            "",
            self.videoTranslateGroup,
        )
        self.videoTranslateOpenVoicePathCard = LineEditSettingCard(
            cfg.video_translate_openvoice_model_path,
            FIF.FOLDER,
            "Путь к OpenVoice модели (опционально)",
            "Нужен только если вы реально выбрали provider=openvoice",
            "",
            self.videoTranslateGroup,
        )
        self.videoTranslateFishSpeechPathCard = LineEditSettingCard(
            cfg.video_translate_fish_speech_model_path,
            FIF.FOLDER,
            "Путь к Fish-Speech модели (опционально)",
            "Нужен только если вы реально выбрали provider=fish_speech",
            "",
            self.videoTranslateGroup,
        )
        self.videoTranslateLocalEndpointCard = LineEditSettingCard(
            cfg.video_translate_local_tts_endpoint,
            FIF.LINK,
            "Local TTS Endpoint",
            "Локальный HTTP endpoint XTTS/OpenVoice/Fish-Speech сервера",
            "http://127.0.0.1:8020",
            self.videoTranslateGroup,
        )
        self.videoTranslateAutonomousModeCard = SwitchSettingCard(
            FIF.ROBOT,
            "Автономный режим",
            "Программа сама поднимает локальные сервисы и использует локальный пайплайн",
            cfg.video_translate_autonomous_mode,
            self.videoTranslateGroup,
        )
        self.videoTranslateAutoDownloadCard = SwitchSettingCard(
            FIF.DOWNLOAD,
            "Автозагрузка моделей",
            "Автоматически догружать отсутствующие локальные модели при первом запуске",
            cfg.video_translate_auto_download_models,
            self.videoTranslateGroup,
        )
        self.videoTranslateCheckEnvCard = PushSettingCard(
            "Проверить",
            FIF.INFO,
            "Проверить окружение Video Translate",
            "Показывает готовность XTTS/ASR/диаризации и точную причину ошибок",
            self.videoTranslateGroup,
        )
        self.videoTranslateInstallCpuCard = PushSettingCard(
            "Установить CPU",
            FIF.DOWNLOAD,
            "Установить зависимости Video Translate (CPU)",
            "Установит/обновит XTTS, faster-whisper, pyannote и зависимости в runtime",
            self.videoTranslateGroup,
        )
        self.videoTranslateInstallGpuCard = PushSettingCard(
            "Установить GPU (CUDA 12.4)",
            FIF.DOWNLOAD,
            "Установить зависимости Video Translate (GPU)",
            "Установит CUDA-пакеты PyTorch и зависимости для лучшего качества/скорости",
            self.videoTranslateGroup,
        )
        self.videoTranslateDownloadXttsCard = PushSettingCard(
            "Скачать XTTS",
            FIF.DOWNLOAD,
            "Скачать/подготовить модель XTTS v2",
            "Предзагрузка XTTS-модели в runtime (чтобы не ждать первого запуска)",
            self.videoTranslateGroup,
        )
        self.videoTranslateOpenModuleCard = PushSettingCard(
            "Открыть",
            FIF.VIDEO,
            "Основные параметры перевода видео",
            "Базовые настройки (язык, качество, кэш, провайдер, параллелизм) находятся во вкладке «Перевод видео»",
            self.videoTranslateGroup,
        )

        # Все пользовательские параметры перенесены на отдельную вкладку
        # "Перевод видео". В окне настроек скрываем их, чтобы они не
        # перекрывали заголовок группы и не дублировали UI.
        self.videoTranslateTargetLanguageCard.setVisible(False)
        self.videoTranslateVoiceProviderCard.setVisible(False)
        self.videoTranslateVoiceQualityCard.setVisible(False)
        self.videoTranslateVoiceRefModeCard.setVisible(False)
        self.videoTranslateEnableDiarizationCard.setVisible(False)
        self.videoTranslateExpectedSpeakersCard.setVisible(False)
        self.videoTranslateSourceSeparationCard.setVisible(False)
        self.videoTranslateKeepMusicCard.setVisible(False)
        self.videoTranslateEnableLipsyncCard.setVisible(False)
        self.videoTranslateLocalEndpointCard.setVisible(False)

        # Также скрываем продвинутые/необязательные поля.
        self.videoTranslateManualVoiceMapCard.setVisible(False)
        self.videoTranslateElevenLabsApiCard.setVisible(False)
        self.videoTranslateAzureKeyCard.setVisible(False)
        self.videoTranslateAzureRegionCard.setVisible(False)
        self.videoTranslateCartesiaApiCard.setVisible(False)
        self.videoTranslateXttsPathCard.setVisible(False)
        self.videoTranslateOpenVoicePathCard.setVisible(False)
        self.videoTranslateFishSpeechPathCard.setVisible(False)

        # 字幕合成配置卡片
        self.subtitleStyleCard = HyperlinkCard(
            "",
            "Изменить",
            FIF.FONT,
            "Стиль субтитров",
            "Выбор стиля субтитров (цвет, размер, шрифт и т.д.)",
            self.subtitleGroup,
        )
        self.subtitleLayoutCard = HyperlinkCard(
            "",
            "Изменить",
            FIF.FONT,
            "Расположение субтитров",
            "Выбор макета субтитров (одноязычные/двуязычные)",
            self.subtitleGroup,
        )
        self.needVideoCard = SwitchSettingCard(
            FIF.VIDEO,
            "Синтезировать видео",
            "Если включено — создавать видео, если выключено — пропускать",
            cfg.need_video,
            self.subtitleGroup,
        )
        self.softSubtitleCard = SwitchSettingCard(
            FIF.FONT,
            "Мягкие субтитры",
            "Включено: субтитры можно отключать в плеере. Выключено: субтитры вшиваются в кадр",
            cfg.soft_subtitle,
            self.subtitleGroup,
        )

        # 保存配置卡片
        self.savePathCard = PushSettingCard(
            "Рабочая папка",
            FIF.SAVE,
            "Путь к рабочей директории",
            cfg.get(cfg.work_dir),
            self.saveGroup,
        )

        # 个性化配置卡片
        self.themeCard = OptionsSettingCard(
            cfg.themeMode,
            FIF.BRUSH,
            "Тема приложения",
            "Изменение внешнего вида приложения",
            texts=["Светлая", "Тёмная", "Как в системе"],
            parent=self.personalGroup,
        )
        self.themeColorCard = CustomColorSettingCard(
            cfg.themeColor,
            FIF.PALETTE,
            "Акцентный цвет",
            "Изменение акцентного цвета приложения",
            self.personalGroup,
        )
        self.uiWindowBgCard = ColorSettingCard(
            self._cfg_color_or_default(cfg.ui_window_bg, "#1E1E1E"),
            FIF.BRUSH,
            "Фон окна",
            "Основной фон страницы/рабочей области",
            self.personalGroup,
        )
        self.uiPanelBgCard = ColorSettingCard(
            self._cfg_color_or_default(cfg.ui_panel_bg, "#252526"),
            FIF.PALETTE,
            "Фон панелей",
            "Цвет фона для панелей и областей ввода",
            self.personalGroup,
        )
        self.uiCardBgCard = ColorSettingCard(
            self._cfg_color_or_default(cfg.ui_card_bg, "#2D2D30"),
            FIF.PALETTE,
            "Фон карточек",
            "Цвет карточек и блоков настроек",
            self.personalGroup,
        )
        self.uiBorderColorCard = ColorSettingCard(
            self._cfg_color_or_default(cfg.ui_border_color, "#3C3C3C"),
            FIF.BRUSH,
            "Цвет границ",
            "Границы таблиц, карточек и полей",
            self.personalGroup,
        )
        self.uiTextColorCard = ColorSettingCard(
            self._cfg_color_or_default(cfg.ui_text_color, "#D4D4D4"),
            FIF.FONT,
            "Основной цвет текста",
            "Базовый цвет текста интерфейса",
            self.personalGroup,
        )
        self.applyThemeCard = PushSettingCard(
            "Применить",
            FIF.BRUSH,
            "Применить тему и цвета",
            "Применяет выбранные цвета и тему без перезапуска",
            self.personalGroup,
        )
        self.zoomCard = OptionsSettingCard(
            cfg.dpiScale,
            FIF.ZOOM,
            "Масштаб интерфейса",
            "Изменение размера виджетов и шрифтов",
            texts=["100%", "125%", "150%", "175%", "200%", "Как в системе"],
            parent=self.personalGroup,
        )
        self.languageCard = ComboBoxSettingCard(
            cfg.language,
            FIF.LANGUAGE,
            "Язык",
            "Выберите язык интерфейса",
            texts=[
                "Китайский (упрощённый)",
                "Китайский (традиционный)",
                "English",
                "Как в системе",
            ],
            parent=self.personalGroup,
        )

        # 关于卡片
        self.helpCard = HyperlinkCard(
            HELP_URL,
            "Открыть страницу помощи",
            FIF.HELP,
            "Помощь",
            f"Новые функции и советы по использованию {APP_NAME}",
            self.aboutGroup,
        )
        self.feedbackCard = PrimaryPushSettingCard(
            "Оставить отзыв",
            FIF.FEEDBACK,
            "Оставить отзыв",
            f"Ваш отзыв помогает улучшать {APP_NAME}",
            self.aboutGroup,
        )
        self.aboutCard = PrimaryPushSettingCard(
            "Проверить обновления",
            FIF.INFO,
            "О программе",
            "© "
            + "Все права защищены"
            + f" {YEAR}, {AUTHOR}. "
            + "Версия"
            + " "
            + VERSION,
            self.aboutGroup,
        )

        # 更新（GitHub）
        self.checkUpdateAtStartupCard = SwitchSettingCard(
            FIF.UPDATE,
            "Проверять обновления при старте",
            "Ненавязчиво проверять новый коммит в официальном репозитории",
            cfg.checkUpdateAtStartUp,
            self.updateGroup,
        )
        self.checkRepoUpdateCard = PushSettingCard(
            "Проверить",
            FIF.SYNC,
            "Проверить обновление сейчас",
            "Проверяет последний коммит в GitHub",
            self.updateGroup,
        )
        self.applyRepoUpdateCard = PrimaryPushSettingCard(
            "Обновить и перезапустить",
            FIF.DOWNLOAD,
            "Применить обновление из GitHub",
            "Скачает актуальный код и перезапустит программу",
            self.updateGroup,
        )

        # 添加卡片到对应的组
        self.translateGroup.addSettingCard(self.subtitleCorrectCard)
        self.translateGroup.addSettingCard(self.subtitleTranslateCard)
        self.translateGroup.addSettingCard(self.targetLanguageCard)

        # Основные пользовательские настройки перенесены во вкладку "Перевод видео".
        # Здесь оставляем только сервисные/инфраструктурные параметры.
        self.videoTranslateGroup.addSettingCard(self.videoTranslateOpenModuleCard)
        self.videoTranslateGroup.addSettingCard(self.videoTranslateAutonomousModeCard)
        self.videoTranslateGroup.addSettingCard(self.videoTranslateAutoDownloadCard)
        self.videoTranslateGroup.addSettingCard(self.videoTranslateCheckEnvCard)
        self.videoTranslateGroup.addSettingCard(self.videoTranslateInstallCpuCard)
        self.videoTranslateGroup.addSettingCard(self.videoTranslateInstallGpuCard)
        self.videoTranslateGroup.addSettingCard(self.videoTranslateDownloadXttsCard)

        self.subtitleGroup.addSettingCard(self.subtitleStyleCard)
        self.subtitleGroup.addSettingCard(self.subtitleLayoutCard)
        self.subtitleGroup.addSettingCard(self.needVideoCard)
        self.subtitleGroup.addSettingCard(self.softSubtitleCard)

        self.saveGroup.addSettingCard(self.savePathCard)

        self.personalGroup.addSettingCard(self.themeCard)
        self.personalGroup.addSettingCard(self.themeColorCard)
        self.personalGroup.addSettingCard(self.uiWindowBgCard)
        self.personalGroup.addSettingCard(self.uiPanelBgCard)
        self.personalGroup.addSettingCard(self.uiCardBgCard)
        self.personalGroup.addSettingCard(self.uiBorderColorCard)
        self.personalGroup.addSettingCard(self.uiTextColorCard)
        self.personalGroup.addSettingCard(self.applyThemeCard)
        self.personalGroup.addSettingCard(self.zoomCard)
        self.personalGroup.addSettingCard(self.languageCard)

        self.aboutGroup.addSettingCard(self.helpCard)
        self.aboutGroup.addSettingCard(self.feedbackCard)
        self.aboutGroup.addSettingCard(self.aboutCard)

        self.updateGroup.addSettingCard(self.checkUpdateAtStartupCard)
        self.updateGroup.addSettingCard(self.checkRepoUpdateCard)
        self.updateGroup.addSettingCard(self.applyRepoUpdateCard)

    def __createLLMServiceCards(self):
        """创建LLM服务相关的配置卡片"""
        # 服务选择卡片
        self.llmServiceCard = ComboBoxSettingCard(
            cfg.llm_service,
            FIF.ROBOT,
            "Сервис LLM",
            "Выберите сервис LLM для сегментации, оптимизации и перевода субтитров",
            texts=[service.value for service in cfg.llm_service.validator.options],
            parent=self.llmGroup,
        )

        # 创建OPENAI官方API链接卡片
        self.openaiOfficialApiCard = HyperlinkCard(
            "https://api.videocaptioner.cn/register?aff=UrLB",
            "Перейти",
            FIF.DEVELOPER_TOOLS,
            f"Официальный API {APP_NAME}",
            "Интеграция нескольких LLM, поддержка быстрой оптимизации и перевода",
            self.llmGroup,
        )
        # 默认隐藏
        self.openaiOfficialApiCard.setVisible(False)

        # 定义每个服务的配置
        service_configs = {
            LLMServiceEnum.OPENAI: {
                "prefix": "openai",
                "api_key_cfg": cfg.openai_api_key,
                "api_base_cfg": cfg.openai_api_base,
                "model_cfg": cfg.openai_model,
                "default_base": "https://api.openai.com/v1",
                "default_models": [
                    "gpt-4o-mini",
                    "gpt-4o",
                    "claude-3-5-sonnet-20241022",
                ],
            },
            LLMServiceEnum.SILICON_CLOUD: {
                "prefix": "silicon_cloud",
                "api_key_cfg": cfg.silicon_cloud_api_key,
                "api_base_cfg": cfg.silicon_cloud_api_base,
                "model_cfg": cfg.silicon_cloud_model,
                "default_base": "https://api.siliconflow.cn/v1",
                "default_models": ["deepseek-ai/DeepSeek-V3"],
            },
            LLMServiceEnum.DEEPSEEK: {
                "prefix": "deepseek",
                "api_key_cfg": cfg.deepseek_api_key,
                "api_base_cfg": cfg.deepseek_api_base,
                "model_cfg": cfg.deepseek_model,
                "default_base": "https://api.deepseek.com/v1",
                "default_models": ["deepseek-chat"],
            },
            LLMServiceEnum.OLLAMA: {
                "prefix": "ollama",
                "api_key_cfg": cfg.ollama_api_key,
                "api_base_cfg": cfg.ollama_api_base,
                "model_cfg": cfg.ollama_model,
                "default_base": "http://localhost:11434/v1",
                "default_models": ["qwen2.5:7b"],
            },
            LLMServiceEnum.LM_STUDIO: {
                "prefix": "LM Studio",
                "api_key_cfg": cfg.lm_studio_api_key,
                "api_base_cfg": cfg.lm_studio_api_base,
                "model_cfg": cfg.lm_studio_model,
                "default_base": "http://localhost:1234/v1",
                "default_models": ["qwen2.5:7b"],
            },
            LLMServiceEnum.GEMINI: {
                "prefix": "gemini",
                "api_key_cfg": cfg.gemini_api_key,
                "api_base_cfg": cfg.gemini_api_base,
                "model_cfg": cfg.gemini_model,
                "default_base": "https://generativelanguage.googleapis.com/v1beta/openai/",
                "default_models": ["gemini-2.0-flash-exp"],
            },
            LLMServiceEnum.CHATGLM: {
                "prefix": "chatglm",
                "api_key_cfg": cfg.chatglm_api_key,
                "api_base_cfg": cfg.chatglm_api_base,
                "model_cfg": cfg.chatglm_model,
                "default_base": "https://open.bigmodel.cn/api/paas/v4",
                "default_models": ["glm-4-flash"],
            },
            LLMServiceEnum.PUBLIC: {
                "prefix": "public",
                "api_key_cfg": cfg.public_api_key,
                "api_base_cfg": cfg.public_api_base,
                "model_cfg": cfg.public_model,
                "default_base": "https://api.public-model.com/v1",
                "default_models": ["public-model"],
            },
        }

        # 创建服务配置映射
        self.llm_service_configs = {}

        # 为每个服务创建配置卡片
        for service, config in service_configs.items():
            prefix = config["prefix"]

            # 如果是公益模型，只添加配置不创建卡片
            if service == LLMServiceEnum.PUBLIC:
                self.llm_service_configs[service] = {
                    "cards": [],
                    "api_base": None,
                    "api_key": None,
                    "model": None,
                }
                continue

            # 创建API Key卡片
            api_key_card = LineEditSettingCard(
                config["api_key_cfg"],
                FIF.FINGERPRINT,
                "API Key",
                f"Введите API Key для {service.value}",
                "sk-" if service != LLMServiceEnum.OLLAMA else "",
                self.llmGroup,
            )
            setattr(self, f"{prefix}_api_key_card", api_key_card)

            # 创建Base URL卡片
            api_base_card = LineEditSettingCard(
                config["api_base_cfg"],
                FIF.LINK,
                "Base URL",
                f"Введите Base URL для {service.value} (должен содержать /v1)",
                config["default_base"],
                self.llmGroup,
            )
            setattr(self, f"{prefix}_api_base_card", api_base_card)

            # 创建模型选择卡片
            model_card = EditComboBoxSettingCard(
                config["model_cfg"],
                FIF.ROBOT,
                "Модель",
                f"Выберите модель {service.value}",
                config["default_models"],
                self.llmGroup,
            )
            setattr(self, f"{prefix}_model_card", model_card)

            # 存储服务配置
            cards = [api_key_card, api_base_card, model_card]

            self.llm_service_configs[service] = {
                "cards": cards,
                "api_base": api_base_card,
                "api_key": api_key_card,
                "model": model_card,
            }

        # 创建检查连接卡片
        self.checkLLMConnectionCard = PushSettingCard(
            "Проверить соединение",
            FIF.LINK,
            "Проверка соединения LLM",
            "Проверить доступность API и получить список моделей",
            self.llmGroup,
        )

        # 初始化显示状态
        self.__onLLMServiceChanged(self.llmServiceCard.comboBox.currentText())

    def __createTranslateServiceCards(self):
        """创建翻译服务相关的配置卡片"""
        # 翻译服务选择卡片
        self.translatorServiceCard = ComboBoxSettingCard(
            cfg.translator_service,
            FIF.ROBOT,
            "Сервис перевода",
            "Выберите сервис перевода",
            texts=[
                service.value for service in cfg.translator_service.validator.options
            ],
            parent=self.translate_serviceGroup,
        )

        # 反思翻译开关
        self.needReflectTranslateCard = SwitchSettingCard(
            FIF.EDIT,
            "Рефлексивный перевод",
            "Улучшает качество перевода, но требует больше времени и токенов",
            cfg.need_reflect_translate,
            self.translate_serviceGroup,
        )

        # DeepLx端点配置
        self.deeplxEndpointCard = LineEditSettingCard(
            cfg.deeplx_endpoint,
            FIF.LINK,
            "Бэкенд DeepLx",
            "Введите адрес DeepLx (обязательно при использовании deeplx)",
            "https://api.deeplx.org/translate",
            self.translate_serviceGroup,
        )

        # 批处理大小配置
        self.batchSizeCard = RangeSettingCard(
            cfg.batch_size,
            FIF.ALIGNMENT,
            "Размер пакета",
            "Количество субтитров в одном пакете (рекомендуется кратно 10)",
            parent=self.translate_serviceGroup,
        )

        # 线程数配置
        self.threadNumCard = RangeSettingCard(
            cfg.thread_num,
            FIF.SPEED_HIGH,
            "Количество потоков",
            "Число параллельных запросов: чем больше (в рамках лимитов), тем выше скорость",
            parent=self.translate_serviceGroup,
        )

        # 添加卡片到翻译服务组
        self.translate_serviceGroup.addSettingCard(self.translatorServiceCard)
        self.translate_serviceGroup.addSettingCard(self.needReflectTranslateCard)
        self.translate_serviceGroup.addSettingCard(self.deeplxEndpointCard)
        self.translate_serviceGroup.addSettingCard(self.batchSizeCard)
        self.translate_serviceGroup.addSettingCard(self.threadNumCard)

        # 初始化显示状态
        self.__onTranslatorServiceChanged(
            self.translatorServiceCard.comboBox.currentText()
        )

    def __initWidget(self):
        self.resize(1000, 800)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setViewportMargins(0, 80, 0, 20)
        self.setWidget(self.scrollWidget)
        self.setWidgetResizable(True)
        self.setObjectName("settingInterface")

        # 初始化样式表
        self.scrollWidget.setObjectName("scrollWidget")
        self.settingLabel.setObjectName("settingLabel")

        # 初始化翻译服务配置卡片的显示状态
        self.__onTranslatorServiceChanged(
            self.translatorServiceCard.comboBox.currentText()
        )

        self.setStyleSheet(
            """        
            SettingInterface, #scrollWidget {
                background-color: transparent;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QLabel#settingLabel {
                font: 33px 'Microsoft YaHei';
                background-color: transparent;
                color: #D4D4D4;
            }
        """
        )
        self.refresh_theme()

    def __initLayout(self):
        """初始化布局"""
        self.settingLabel.move(36, 30)

        # 添加转录配置卡片
        self.transcribeGroup.addSettingCard(self.transcribeModelCard)

        # 添加LLM配置卡片
        self.llmGroup.addSettingCard(self.llmServiceCard)
        # 添加OPENAI官方API链接卡片
        self.llmGroup.addSettingCard(self.openaiOfficialApiCard)
        for config in self.llm_service_configs.values():
            for card in config["cards"]:
                self.llmGroup.addSettingCard(card)
        self.llmGroup.addSettingCard(self.checkLLMConnectionCard)

        # 将所有组添加到布局
        self.expandLayout.setSpacing(28)
        self.expandLayout.setContentsMargins(36, 10, 36, 0)
        self.expandLayout.addWidget(self.updateGroup)
        self.expandLayout.addWidget(self.transcribeGroup)
        self.expandLayout.addWidget(self.llmGroup)
        self.expandLayout.addWidget(self.translate_serviceGroup)
        self.expandLayout.addWidget(self.translateGroup)
        self.expandLayout.addWidget(self.videoTranslateGroup)
        self.expandLayout.addWidget(self.subtitleGroup)
        self.expandLayout.addWidget(self.saveGroup)
        self.expandLayout.addWidget(self.personalGroup)
        self.expandLayout.addWidget(self.aboutGroup)

    def __connectSignalToSlot(self):
        """连接信号与槽"""
        cfg.appRestartSig.connect(self.__showRestartTooltip)

        # LLM服务切换
        self.llmServiceCard.comboBox.currentTextChanged.connect(
            self.__onLLMServiceChanged
        )

        # 翻译服务切换
        self.translatorServiceCard.comboBox.currentTextChanged.connect(
            self.__onTranslatorServiceChanged
        )

        # 检查 LLM 连接
        self.checkLLMConnectionCard.clicked.connect(self.checkLLMConnection)

        # 保存路径
        self.savePathCard.clicked.connect(self.__onsavePathCardClicked)

        # 字幕样式修改跳转
        self.subtitleStyleCard.linkButton.clicked.connect(
            lambda: self.window().switchTo(self.window().subtitleStyleInterface)
        )
        self.subtitleLayoutCard.linkButton.clicked.connect(
            lambda: self.window().switchTo(self.window().subtitleStyleInterface)
        )

        # 个性化
        self.themeCard.optionChanged.connect(lambda _ci: self._on_theme_mode_changed())
        self.applyThemeCard.clicked.connect(self._apply_theme_from_controls)

        # 反馈
        self.feedbackCard.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(FEEDBACK_URL))
        )

        # 关于
        self.aboutCard.clicked.connect(self.checkUpdate)

        # GitHub update
        self.checkRepoUpdateCard.clicked.connect(self._check_repo_update_now)
        self.applyRepoUpdateCard.clicked.connect(self._apply_repo_update_now)

        # 全局 signalBus
        self.transcribeModelCard.comboBox.currentTextChanged.connect(
            signalBus.transcription_model_changed
        )
        self.subtitleCorrectCard.checkedChanged.connect(
            signalBus.subtitle_optimization_changed
        )
        self.subtitleTranslateCard.checkedChanged.connect(
            signalBus.subtitle_translation_changed
        )
        self.targetLanguageCard.comboBox.currentTextChanged.connect(
            signalBus.target_language_changed
        )
        self.softSubtitleCard.checkedChanged.connect(signalBus.soft_subtitle_changed)
        self.needVideoCard.checkedChanged.connect(signalBus.need_video_changed)

        self.videoTranslateCheckEnvCard.clicked.connect(self._check_video_translate_env)
        self.videoTranslateInstallCpuCard.clicked.connect(
            lambda: self._install_video_translate_runtime("cpu")
        )
        self.videoTranslateInstallGpuCard.clicked.connect(
            lambda: self._install_video_translate_runtime("gpu-cu124")
        )
        self.videoTranslateDownloadXttsCard.clicked.connect(self._download_xtts_model)
        self.videoTranslateOpenModuleCard.clicked.connect(
            lambda: self.window().switchTo(self.window().video_translate_interface)
        )
        self.videoTranslateVoiceProviderCard.comboBox.currentTextChanged.connect(
            self.__onVideoTranslateProviderChanged
        )

    def _cfg_color_or_default(self, item, fallback: str) -> QColor:
        value = cfg.get(item)
        q = QColor(str(value) if value is not None else "")
        if not q.isValid():
            q = QColor(fallback)
        return q

    def _on_theme_mode_changed(self):
        palette = get_theme_palette()
        if cfg.get(cfg.themeMode).name == "LIGHT":
            self.uiWindowBgCard.setColor(QColor("#F3F3F3"))
            self.uiPanelBgCard.setColor(QColor("#FFFFFF"))
            self.uiCardBgCard.setColor(QColor("#FFFFFF"))
            self.uiBorderColorCard.setColor(QColor("#E1E1E1"))
            self.uiTextColorCard.setColor(QColor("#1F1F1F"))
        elif cfg.get(cfg.themeMode).name == "DARK":
            self.uiWindowBgCard.setColor(QColor("#1E1E1E"))
            self.uiPanelBgCard.setColor(QColor("#252526"))
            self.uiCardBgCard.setColor(QColor("#2D2D30"))
            self.uiBorderColorCard.setColor(QColor("#3C3C3C"))
            self.uiTextColorCard.setColor(QColor("#D4D4D4"))
        else:
            self.uiWindowBgCard.setColor(QColor(palette["window_bg"]))
            self.uiPanelBgCard.setColor(QColor(palette["panel_bg"]))
            self.uiCardBgCard.setColor(QColor(palette["card_bg"]))
            self.uiBorderColorCard.setColor(QColor(palette["border"]))
            self.uiTextColorCard.setColor(QColor(palette["text"]))

    def _apply_theme_from_controls(self):
        cfg.set(cfg.ui_window_bg, self.uiWindowBgCard.colorPicker.color.name(QColor.HexRgb))
        cfg.set(cfg.ui_panel_bg, self.uiPanelBgCard.colorPicker.color.name(QColor.HexRgb))
        cfg.set(cfg.ui_card_bg, self.uiCardBgCard.colorPicker.color.name(QColor.HexRgb))
        cfg.set(cfg.ui_border_color, self.uiBorderColorCard.colorPicker.color.name(QColor.HexRgb))
        cfg.set(cfg.ui_text_color, self.uiTextColorCard.colorPicker.color.name(QColor.HexRgb))

        apply_vscode_theme(refresh_widgets=True)
        self.refresh_theme()
        InfoBar.success(
            "Тема применена",
            "Новые цвета интерфейса применены в реальном времени",
            duration=2200,
            parent=self,
        )

    def refresh_theme(self):
        p = get_theme_palette()
        self.settingLabel.setStyleSheet(
            f"font: 33px 'Microsoft YaHei'; background: transparent; color: {p['text']};"
        )

    def __showRestartTooltip(self):
        """显示重启提示"""
        InfoBar.success(
            "Успешно",
            "Настройки вступят в силу после перезапуска",
            duration=1500,
            parent=self,
        )

    def _check_video_translate_env(self):
        runtime_python = PROJECT_ROOT / "runtime" / "python.exe"
        script = PROJECT_ROOT / "scripts" / "check_video_translate_env.py"
        if not runtime_python.exists() or not script.exists():
            InfoBar.error(
                "Ошибка",
                "Не найден runtime python или скрипт проверки окружения",
                duration=3500,
                parent=self,
            )
            return
        try:
            p = subprocess.run(
                [
                    str(runtime_python),
                    str(script),
                    "--runtime-python",
                    str(runtime_python),
                    "--model",
                    str(cfg.faster_whisper_model.value.value),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=(subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0),
            )
            out = ((p.stdout or "") + "\n" + (p.stderr or "")).strip()
            dlg = _TextReportDialog("Отчёт Video Translate", self)
            dlg.set_text(out if out else "Нет вывода")
            dlg.exec()
        except Exception as e:
            InfoBar.error("Ошибка", f"Проверка не выполнена: {e}", duration=4000, parent=self)

    def _install_video_translate_runtime(self, profile: str):
        runtime_python = PROJECT_ROOT / "runtime" / "python.exe"
        script = PROJECT_ROOT / "scripts" / "setup_video_translate_runtime.py"
        if not runtime_python.exists() or not script.exists():
            InfoBar.error(
                "Ошибка",
                "Не найден runtime python или скрипт установки",
                duration=3500,
                parent=self,
            )
            return
        try:
            if self._vt_install_thread and self._vt_install_thread.isRunning():
                InfoBar.warning(
                    "Установка уже выполняется",
                    "Дождитесь завершения текущей установки зависимостей.",
                    duration=3000,
                    parent=self,
                )
                return

            self._vt_install_dialog = _TextReportDialog(
                f"Установка Video Translate ({profile})",
                self,
                with_progress=True,
            )
            self._vt_install_dialog.set_text("Запуск установки...\n")
            self._vt_install_dialog.show()

            self._vt_install_thread = _VideoTranslateInstallThread(
                runtime_python=str(runtime_python),
                setup_script=str(script),
                profile=str(profile),
                log_file=str(LOG_PATH / "video_translate_install.log"),
            )
            self._vt_install_thread.log_line.connect(self._vt_install_dialog.append_text)
            self._vt_install_thread.progress.connect(self._vt_install_dialog.set_progress)
            self._vt_install_thread.finished_status.connect(self._on_vt_install_finished)
            self._vt_install_thread.start()

            InfoBar.success(
                "Установка запущена",
                "Открылось окно прогресса. После завершения перезапустите программу.",
                duration=3500,
                parent=self,
            )
        except Exception as e:
            InfoBar.error("Ошибка", f"Не удалось запустить установку: {e}", duration=4500, parent=self)

    def _download_xtts_model(self):
        runtime_python = PROJECT_ROOT / "runtime" / "python.exe"
        script = PROJECT_ROOT / "scripts" / "download_xtts_model.py"
        if not runtime_python.exists() or not script.exists():
            InfoBar.error(
                "Ошибка",
                "Не найден runtime python или скрипт загрузки XTTS",
                duration=3500,
                parent=self,
            )
            return
        try:
            if self._vt_install_thread and self._vt_install_thread.isRunning():
                InfoBar.warning(
                    "Задача уже выполняется",
                    "Дождитесь завершения текущей установки/загрузки.",
                    duration=3000,
                    parent=self,
                )
                return

            self._vt_install_dialog = _TextReportDialog(
                "Загрузка XTTS v2",
                self,
                with_progress=True,
            )
            self._vt_install_dialog.set_text("Запуск загрузки XTTS...\n")
            self._vt_install_dialog.show()

            self._vt_install_thread = _VideoTranslateInstallThread(
                runtime_python=str(runtime_python),
                setup_script="",
                profile="",
                log_file=str(LOG_PATH / "video_translate_install.log"),
                custom_cmd=[str(runtime_python), str(script)],
            )
            self._vt_install_thread.log_line.connect(self._vt_install_dialog.append_text)
            self._vt_install_thread.progress.connect(self._vt_install_dialog.set_progress)
            self._vt_install_thread.finished_status.connect(self._on_vt_install_finished)
            self._vt_install_thread.start()
        except Exception as e:
            InfoBar.error("Ошибка", f"Не удалось запустить загрузку XTTS: {e}", duration=4500, parent=self)

    def _on_vt_install_finished(self, ok: bool, message: str):
        if self._vt_install_dialog:
            self._vt_install_dialog.append_text(f"\n{message}\n")
            self._vt_install_dialog.set_progress(100 if ok else 0)
        if ok:
            InfoBar.success("Установка завершена", message, duration=4500, parent=self)
        else:
            InfoBar.error("Установка завершилась с ошибкой", message, duration=6500, parent=self)

    def __onsavePathCardClicked(self):
        """处理保存路径卡片点击事件"""
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку", "./")
        if not folder or cfg.get(cfg.work_dir) == folder:
            return
        cfg.set(cfg.work_dir, folder)
        self.savePathCard.setContent(folder)

    def checkLLMConnection(self):
        """检查 LLM 连接"""
        # 获取当前选中的服务
        current_service = LLMServiceEnum(self.llmServiceCard.comboBox.currentText())

        # 获取服务配置
        service_config = self.llm_service_configs.get(current_service)
        if not service_config:
            return

        # 如果是公益模型，使用配置文件中的值
        if current_service == LLMServiceEnum.PUBLIC:
            api_base = cfg.public_api_base.value
            api_key = cfg.public_api_key.value
            model = cfg.public_model.value
        else:
            api_base = (
                service_config["api_base"].lineEdit.text()
                if service_config["api_base"]
                else ""
            )
            api_key = (
                service_config["api_key"].lineEdit.text()
                if service_config["api_key"]
                else ""
            )
            model = (
                service_config["model"].comboBox.currentText()
                if service_config["model"]
                else ""
            )

        # 检查 API Base 是否属于网址
        if not api_base.startswith("http"):
            InfoBar.error(
                "Ошибка",
                "Введите корректный API Base (должен содержать /v1)",
                duration=3000,
                parent=self,
            )
            return

        # 禁用检查按钮，显示加载状态
        self.checkLLMConnectionCard.button.setEnabled(False)
        self.checkLLMConnectionCard.button.setText("Проверка...")

        # 创建并启动线程
        self.connection_thread = LLMConnectionThread(api_base, api_key, model)
        self.connection_thread.finished.connect(self.onConnectionCheckFinished)
        self.connection_thread.error.connect(self.onConnectionCheckError)
        self.connection_thread.start()

    def onConnectionCheckError(self, message):
        """处理连接检查错误事件"""
        self.checkLLMConnectionCard.button.setEnabled(True)
        self.checkLLMConnectionCard.button.setText("Проверить соединение")
        InfoBar.error("Ошибка проверки LLM", message, duration=3000, parent=self)

    def onConnectionCheckFinished(self, is_success, message, models):
        """处理连接检查完成事件"""
        self.checkLLMConnectionCard.button.setEnabled(True)
        self.checkLLMConnectionCard.button.setText("Проверить соединение")

        # 获取当前服务
        current_service = LLMServiceEnum(self.llmServiceCard.comboBox.currentText())

        if models:
            # 更新当前服务的模型列表
            service_config = self.llm_service_configs.get(current_service)
            if service_config and service_config["model"]:
                temp = service_config["model"].comboBox.currentText()
                service_config["model"].setItems(models)
                service_config["model"].comboBox.setCurrentText(temp)

            InfoBar.success(
                "Список моделей получен",
                "Всего моделей: " + str(len(models)),
                duration=3000,
                parent=self,
            )
        if not is_success:
            InfoBar.error(
                "Ошибка проверки LLM", message, duration=3000, parent=self
            )
        else:
            InfoBar.success(
                "Проверка LLM успешна", message, duration=3000, parent=self
            )

    def checkUpdate(self):
        webbrowser.open(RELEASE_URL)

    def _check_repo_update_now(self):
        try:
            info = self.githubUpdateManager.check_update()
            if info.get("baseline_initialized"):
                text = "База обновлений инициализирована. Следующий новый коммит будет предложен как обновление."
            elif info.get("has_update"):
                latest = info.get("latest") or {}
                text = (
                    f"Найден новый коммит: {str(latest.get('sha') or '')[:8]}\n\n"
                    f"{str(latest.get('message') or '').strip()[:500]}"
                )
            else:
                text = "У вас актуальная версия относительно последнего коммита выбранной ветки."
            box = MessageBox("Проверка обновлений", text, self)
            box.yesButton.setText("ОК")
            box.cancelButton.hide()
            box.exec()
        except Exception as e:
            InfoBar.error("Ошибка", f"Проверка обновления не удалась: {e}", duration=3500, parent=self)

    def _apply_repo_update_now(self):
        try:
            result = self.githubUpdateManager.apply_update_and_restart()
            if result.get("ok"):
                box = MessageBox(
                    "Обновление запущено",
                    "Файлы обновления скачаны. Приложение будет закрыто для применения обновления и перезапуска.",
                    self,
                )
                box.yesButton.setText("ОК")
                box.cancelButton.hide()
                box.exec()
                from PyQt5.QtWidgets import QApplication

                QApplication.quit()
            else:
                InfoBar.error(
                    "Ошибка обновления",
                    str(result.get("error") or "Не удалось применить обновление"),
                    duration=4500,
                    parent=self,
                )
        except Exception as e:
            InfoBar.error("Ошибка", f"Обновление не удалось: {e}", duration=4500, parent=self)

    def __onLLMServiceChanged(self, service):
        """处理LLM服务切换事件"""
        current_service = LLMServiceEnum(service)

        # 隐藏所有卡片
        for config in self.llm_service_configs.values():
            for card in config["cards"]:
                card.setVisible(False)

        # 隐藏OPENAI官方API链接卡片
        self.openaiOfficialApiCard.setVisible(False)

        # 显示选中服务的卡片
        if current_service in self.llm_service_configs:
            for card in self.llm_service_configs[current_service]["cards"]:
                card.setVisible(True)

            # 为OLLAMA和LM_STUDIO设置默认API Key
            service_config = self.llm_service_configs[current_service]
            if current_service == LLMServiceEnum.OLLAMA and service_config["api_key"]:
                # 如果API Key为空，设置默认值"ollama"
                if not service_config["api_key"].lineEdit.text():
                    service_config["api_key"].lineEdit.setText("ollama")
            if (
                current_service == LLMServiceEnum.LM_STUDIO
                and service_config["api_key"]
            ):
                # 如果API Key为空，设置默认值 "lm-studio"
                if not service_config["api_key"].lineEdit.text():
                    service_config["api_key"].lineEdit.setText("lm-studio")

            # 如果是OPENAI服务，显示官方API链接卡片
            if current_service == LLMServiceEnum.OPENAI:
                self.openaiOfficialApiCard.setVisible(True)

        # 更新布局
        self.llmGroup.adjustSize()
        self.expandLayout.update()

    def __onTranslatorServiceChanged(self, service):
        openai_cards = [
            self.needReflectTranslateCard,
            self.batchSizeCard,
        ]
        deeplx_cards = [self.deeplxEndpointCard]

        all_cards = openai_cards + deeplx_cards
        for card in all_cards:
            card.setVisible(False)

        # 根据选择的服务显示相应的配置卡片
        if service in [TranslatorServiceEnum.DEEPLX.value]:
            for card in deeplx_cards:
                card.setVisible(True)
        elif service in [TranslatorServiceEnum.OPENAI.value]:
            for card in openai_cards:
                card.setVisible(True)

        # 更新布局
        self.translate_serviceGroup.adjustSize()
        self.expandLayout.update()

    def __onVideoTranslateProviderChanged(self, provider: str):
        provider = (provider or "").strip().lower()

        # Сначала прячем всё продвинутое
        self.videoTranslateManualVoiceMapCard.setVisible(False)
        self.videoTranslateElevenLabsApiCard.setVisible(False)
        self.videoTranslateAzureKeyCard.setVisible(False)
        self.videoTranslateAzureRegionCard.setVisible(False)
        self.videoTranslateCartesiaApiCard.setVisible(False)
        self.videoTranslateXttsPathCard.setVisible(False)
        self.videoTranslateOpenVoicePathCard.setVisible(False)
        self.videoTranslateFishSpeechPathCard.setVisible(False)

        # Показываем только то, что реально нужно выбранному провайдеру
        if provider in {"auto", "xtts"}:
            # XTTS path опционален, оставляем скрытым для простоты
            pass
        elif provider == "openvoice":
            self.videoTranslateOpenVoicePathCard.setVisible(True)
        elif provider == "fish_speech":
            self.videoTranslateFishSpeechPathCard.setVisible(True)
        elif provider == "elevenlabs":
            self.videoTranslateElevenLabsApiCard.setVisible(True)
        elif provider == "azure":
            self.videoTranslateAzureKeyCard.setVisible(True)
            self.videoTranslateAzureRegionCard.setVisible(True)
        elif provider == "cartesia":
            self.videoTranslateCartesiaApiCard.setVisible(True)

        # Manual voice map нужен только в manual mode
        if str(cfg.video_translate_voice_reference_mode.value).lower() == "manual":
            self.videoTranslateManualVoiceMapCard.setVisible(True)

        self.videoTranslateGroup.adjustSize()
        self.expandLayout.update()


class LLMConnectionThread(QThread):
    finished = pyqtSignal(bool, str, list)
    error = pyqtSignal(str)

    def __init__(self, api_base, api_key, model):
        super().__init__()
        self.api_base = api_base
        self.api_key = api_key
        self.model = model

    def run(self):
        """检查 LLM 连接并获取模型列表"""
        try:
            is_success, message = test_openai(self.api_base, self.api_key, self.model)
            models = get_openai_models(self.api_base, self.api_key)
            self.finished.emit(is_success, message, models)
        except Exception as e:
            self.error.emit(str(e))


class _TextReportDialog(QDialog):
    def __init__(self, title: str, parent=None, with_progress: bool = False):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 620)
        self._with_progress = with_progress

        layout = QVBoxLayout(self)
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit, 1)

        if with_progress:
            from qfluentwidgets import ProgressBar

            self.progress_bar = ProgressBar(self)
            self.progress_bar.setValue(0)
            layout.addWidget(self.progress_bar)
        else:
            self.progress_bar = None

        btn_row = QHBoxLayout()
        self.copy_btn = QPushButton("Копировать", self)
        self.clear_btn = QPushButton("Очистить", self)
        self.close_btn = QPushButton("Закрыть", self)
        btn_row.addWidget(self.copy_btn)
        btn_row.addWidget(self.clear_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.close_btn)
        layout.addLayout(btn_row)

        self.copy_btn.clicked.connect(self.copy_text)
        self.clear_btn.clicked.connect(self.text_edit.clear)
        self.close_btn.clicked.connect(self.accept)

    def set_text(self, text: str):
        self.text_edit.setPlainText(text or "")

    def append_text(self, text: str):
        if not text:
            return
        self.text_edit.moveCursor(self.text_edit.textCursor().End)
        self.text_edit.insertPlainText(text)
        self.text_edit.moveCursor(self.text_edit.textCursor().End)

    def copy_text(self):
        QApplication.clipboard().setText(self.text_edit.toPlainText())
        InfoBar.success("Скопировано", "Текст отчёта скопирован в буфер", duration=1200, parent=self)

    def set_progress(self, value: int):
        if self.progress_bar is not None:
            self.progress_bar.setValue(max(0, min(100, int(value))))


class _VideoTranslateInstallThread(QThread):
    log_line = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished_status = pyqtSignal(bool, str)

    def __init__(
        self,
        runtime_python: str,
        setup_script: str,
        profile: str,
        log_file: str,
        custom_cmd: list[str] | None = None,
    ):
        super().__init__()
        self.runtime_python = runtime_python
        self.setup_script = setup_script
        self.profile = profile
        self.log_file = log_file
        self.custom_cmd = custom_cmd

    def run(self):
        cmd = self.custom_cmd or [
            self.runtime_python,
            self.setup_script,
            "--runtime-python",
            self.runtime_python,
            "--profile",
            self.profile,
            "--upgrade-pip",
        ]
        run_steps = 0
        total_steps = 4
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "a", encoding="utf-8", errors="replace") as lf:
            lf.write("\n" + "=" * 40 + "\n")
            lf.write("CMD: " + subprocess.list2cmdline(cmd) + "\n")
            try:
                p = subprocess.Popen(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=(
                        subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
                    ),
                )
                self.progress.emit(2)
                while True:
                    line = p.stdout.readline() if p.stdout else ""
                    if not line and p.poll() is not None:
                        break
                    if not line:
                        continue
                    lf.write(line)
                    self.log_line.emit(line)
                    if line.strip().startswith("[RUN]"):
                        run_steps += 1
                        self.progress.emit(min(95, int((run_steps / total_steps) * 100)))

                rc = p.wait()
                if rc == 0:
                    self.progress.emit(100)
                    self.finished_status.emit(True, "Установка зависимостей выполнена успешно")
                else:
                    self.finished_status.emit(False, f"Установка завершилась с кодом {rc}. Лог: {self.log_file}")
            except Exception as e:
                lf.write(f"\nERROR: {e}\n")
                self.finished_status.emit(False, f"Ошибка установки: {e}. Лог: {self.log_file}")
