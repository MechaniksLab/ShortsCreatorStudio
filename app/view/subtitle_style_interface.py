import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QColor, QFontDatabase
from PyQt5.QtWidgets import QFileDialog, QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import BodyLabel, CardWidget
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import (
    ImageLabel,
    InfoBar,
    InfoBarPosition,
    LineEdit,
    MessageBoxBase,
    PushSettingCard,
    ScrollArea,
    SettingCardGroup,
    Slider,
    PushButton,
)

from app.common.config import cfg
from app.common.signal_bus import signalBus
from app.components.MySettingCard import (
    ColorSettingCard,
    ComboBoxSettingCard,
    DoubleSpinBoxSettingCard,
    SpinBoxSettingCard,
)
from app.config import SUBTITLE_STYLE_PATH, ASSETS_PATH, APPDATA_PATH
from app.core.entities import SplitTypeEnum
from app.core.subtitle_processor.effect_manager import EffectManager
from app.core.utils.subtitle_preview import generate_preview, generate_preview_video

LAYOUT_LABEL_TO_VALUE = {
    "Перевод сверху": "译文在上",
    "Оригинал сверху": "原文在上",
    "Только перевод": "仅译文",
    "Только оригинал": "仅原文",
}
LAYOUT_VALUE_TO_LABEL = {v: k for k, v in LAYOUT_LABEL_TO_VALUE.items()}

ORIENTATION_LABEL_TO_VALUE = {
    "Горизонтальный": "横屏",
    "Вертикальный": "竖屏",
}
ORIENTATION_VALUE_TO_LABEL = {v: k for k, v in ORIENTATION_LABEL_TO_VALUE.items()}

STYLE_PRESET_LABEL_TO_VALUE = {
    "Свой (Custom)": "custom",
    "TikTok Dynamic": "tiktok_dynamic",
    "YouTube Shorts Clean": "shorts_clean",
    "Minimal Classic": "minimal_classic",
    "Karaoke Pro": "karaoke_pro",
    "Cinema Gradient": "cinema_gradient",
    "Neon Pulse": "neon_pulse",
    "Gaming Glitch": "gaming_glitch",
    "Podcast Focus": "podcast_focus",
    "Travel Vlog": "travel_vlog",
    "Dramatic Trailer": "dramatic_trailer",
}
STYLE_PRESET_VALUE_TO_LABEL = {v: k for k, v in STYLE_PRESET_LABEL_TO_VALUE.items()}

TOGGLE_LABEL_TO_VALUE = {
    "Выкл": False,
    "Вкл": True,
}
TOGGLE_VALUE_TO_LABEL = {v: k for k, v in TOGGLE_LABEL_TO_VALUE.items()}

GRADIENT_MODE_LABEL_TO_VALUE = {
    "Без градиента": "off",
    "2 цвета": "two_color",
    "Радужный": "rainbow",
}
GRADIENT_MODE_VALUE_TO_LABEL = {v: k for k, v in GRADIENT_MODE_LABEL_TO_VALUE.items()}

SPEAKER_COLOR_MODE_LABEL_TO_VALUE = {
    "Выкл": "off",
    "Чередовать по репликам": "alternate",
}
SPEAKER_COLOR_MODE_VALUE_TO_LABEL = {
    v: k for k, v in SPEAKER_COLOR_MODE_LABEL_TO_VALUE.items()
}

SPLIT_TYPE_LABEL_TO_VALUE = {
    "По предложениям": SplitTypeEnum.SENTENCE,
    "По словам": SplitTypeEnum.SEMANTIC,
}
SPLIT_TYPE_VALUE_TO_LABEL = {
    SplitTypeEnum.SENTENCE: "По предложениям",
    SplitTypeEnum.SEMANTIC: "По словам",
    SplitTypeEnum.SENTENCE.value: "По предложениям",
    SplitTypeEnum.SEMANTIC.value: "По словам",
}

MOTION_DIRECTION_LABEL_TO_VALUE = {
    "Снизу вверх": "up",
    "Сверху вниз": "down",
    "Слева направо": "left",
    "Справа налево": "right",
}
MOTION_DIRECTION_VALUE_TO_LABEL = {
    v: k for k, v in MOTION_DIRECTION_LABEL_TO_VALUE.items()
}

MOTION_EASING_LABEL_TO_VALUE = {
    "Ease Out (мягкий финиш)": "ease_out",
    "Ease In (мягкий старт)": "ease_in",
    "Ease In-Out": "ease_in_out",
    "Linear": "linear",
}
MOTION_EASING_VALUE_TO_LABEL = {v: k for k, v in MOTION_EASING_LABEL_TO_VALUE.items()}

SENTENCE_MODE_EFFECTS = {
    "none",
    "fade_in",
    "fade_out",
    "fade_in_out",
    "typewriter",
    "word_highlight",
    "shine",
}

WORD_MODE_EFFECTS = {
    "bounce",
    "pulse",
    "wave",
    "spin",
    "zoom_in",
    "swing",
    "slide_up",
    "slide_left",
    "pop_rotate",
    "shake",
    "neon_flicker",
    "glitch",
    "twinkle",
    "rainbow",
}

PREVIEW_TEXTS_SENTENCE = {
    "Длинный текст": (
        "This is a long text used for testing subtitle preview and style settings.",
        "这是一段用于测试字幕预览和样式设置的长文本内容",
    ),
    "Средний текст": (
        "Welcome to apply for the prestigious South China Normal University!",
        "欢迎报考百年名校华南师范大学",
    ),
    "Короткий текст": ("Elementary school students know this", "小学二年级的都知道"),
}

PREVIEW_TEXTS_WORD = {
    "По словам — коротко": (
        "я иду домой после работы",
        "я иду домой после работы",
    ),
    "По словам — ритм": (
        "вчера неожиданно случилось невероятное приключение",
        "вчера неожиданно случилось невероятное приключение",
    ),
    "По словам — акцент": (
        "старт пауза ускорение финиш аплодисменты",
        "старт пауза ускорение финиш аплодисменты",
    ),
}

DEFAULT_BG_LANDSCAPE = {
    "path": ASSETS_PATH / "default_bg_landscape.png",
    "width": 1280,
    "height": 720,
}
DEFAULT_BG_PORTRAIT = {
    "path": ASSETS_PATH / "default_bg_portrait.png",
    "width": 480,
    "height": 852,
}

CUSTOM_SUBTITLE_PRESETS_PATH = APPDATA_PATH / "subtitle_custom_presets.json"


class PreviewThread(QThread):
    previewReady = pyqtSignal(str, int)

    def __init__(
        self,
        style_str: str,
        preview_text: Tuple[str, Optional[str]],
        bg_path: str,
        width: int,
        height: int,
        effect_type: str,
        effect_duration_ms: int,
        effect_intensity: float,
        rainbow_end_color: str,
        motion_direction: str,
        motion_amplitude: float,
        motion_easing: str,
        motion_jitter: float,
        motion_blur_strength: float,
        karaoke_mode: bool,
        karaoke_window_ms: int,
        auto_contrast: bool,
        anti_flicker: bool,
        gradient_mode: str,
        gradient_color_1: str,
        gradient_color_2: str,
        safe_area_enabled: bool,
        safe_margin_x: int,
        safe_margin_y: int,
        speaker_color_mode: str,
        preview_time_sec: float,
        request_id: int,
    ):
        """
        Args:
            style_str: ASS 样式字符串
            preview_text: 预览文本元组 (主字幕, 副字幕), 副字幕可选
        """
        super().__init__()
        self.style_str = style_str
        self.preview_text = preview_text
        self.bg_path = bg_path
        self.width = width
        self.height = height
        self.effect_type = effect_type
        self.effect_duration_ms = effect_duration_ms
        self.effect_intensity = effect_intensity
        self.rainbow_end_color = rainbow_end_color
        self.motion_direction = motion_direction
        self.motion_amplitude = motion_amplitude
        self.motion_easing = motion_easing
        self.motion_jitter = motion_jitter
        self.motion_blur_strength = motion_blur_strength
        self.karaoke_mode = karaoke_mode
        self.karaoke_window_ms = karaoke_window_ms
        self.auto_contrast = auto_contrast
        self.anti_flicker = anti_flicker
        self.gradient_mode = gradient_mode
        self.gradient_color_1 = gradient_color_1
        self.gradient_color_2 = gradient_color_2
        self.safe_area_enabled = safe_area_enabled
        self.safe_margin_x = safe_margin_x
        self.safe_margin_y = safe_margin_y
        self.speaker_color_mode = speaker_color_mode
        self.preview_time_sec = preview_time_sec
        self.request_id = request_id

    def run(self):
        frame_token = f"{self.request_id}_{int(self.preview_time_sec * 1000)}"
        preview_path = generate_preview(
            style_str=self.style_str,
            preview_text=self.preview_text,
            bg_path=self.bg_path,
            width=self.width,
            height=self.height,
            effect_type=self.effect_type,
            effect_duration_ms=self.effect_duration_ms,
            effect_intensity=self.effect_intensity,
            rainbow_end_color=self.rainbow_end_color,
            motion_direction=self.motion_direction,
            motion_amplitude=self.motion_amplitude,
            motion_easing=self.motion_easing,
            motion_jitter=self.motion_jitter,
            motion_blur_strength=self.motion_blur_strength,
            karaoke_mode=self.karaoke_mode,
            karaoke_window_ms=self.karaoke_window_ms,
            auto_contrast=self.auto_contrast,
            anti_flicker=self.anti_flicker,
            gradient_mode=self.gradient_mode,
            gradient_color_1=self.gradient_color_1,
            gradient_color_2=self.gradient_color_2,
            safe_area_enabled=self.safe_area_enabled,
            safe_margin_x=self.safe_margin_x,
            safe_margin_y=self.safe_margin_y,
            speaker_color_mode=self.speaker_color_mode,
            preview_time_sec=self.preview_time_sec,
            frame_token=frame_token,
        )
        self.previewReady.emit(preview_path, self.request_id)


class SubtitleStyleInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("SubtitleStyleInterface")
        self.setWindowTitle("Настройка стиля субтитров")

        # 创建主布局
        self.hBoxLayout = QHBoxLayout(self)

        # 初始化界面组件
        self._initSettingsArea()
        self._initPreviewArea()
        self._initSettingCards()
        self._initLayout()
        self._initStyle()

        # 添加一个标志位来控制是否触发onSettingChanged
        self._loading_style = False

        # Live preview state (должен быть инициализирован до __setValues,
        # т.к. loadStyle() внутри __setValues вызывает updatePreview())
        self._live_timer = QTimer(self)
        self._live_timer.setInterval(40)  # ~25 FPS
        self._live_timer.timeout.connect(self._onLiveTimerTick)
        self._live_playing = False

        self._preview_debounce_timer = QTimer(self)
        self._preview_debounce_timer.setSingleShot(True)
        self._preview_debounce_timer.setInterval(80)
        self._preview_debounce_timer.timeout.connect(self._renderPreviewNow)

        self._preview_thread = None
        self._preview_pending = False
        self._preview_request_id = 0
        self._latest_applied_request_id = -1

        # 设置初始值,加载样式
        self.__setValues()

        # 连接信号
        self.connectSignals()

        self._update_motion_controls_state()

    def _initSettingsArea(self):
        """初始化左侧设置区域"""
        self.settingsScrollArea = ScrollArea()
        self.settingsScrollArea.setFixedWidth(350)
        self.settingsWidget = QWidget()
        self.settingsLayout = QVBoxLayout(self.settingsWidget)
        self.settingsScrollArea.setWidget(self.settingsWidget)
        self.settingsScrollArea.setWidgetResizable(True)

        # Группа 1: быстрый старт и ключевые решения
        self.quickStartGroup = SettingCardGroup("1) Быстрый старт", self.settingsWidget)

        # Группа 2: разметка/сегментация текста
        self.segmentationGroup = SettingCardGroup("2) Разбиение и структура", self.settingsWidget)

        # Группа 3: анимация и динамика
        self.effectsGroup = SettingCardGroup("3) Эффекты и движение", self.settingsWidget)

        # Группа 4: читаемость и безопасность кадра
        self.readabilityGroup = SettingCardGroup(
            "4) Читаемость и безопасная зона", self.settingsWidget
        )

        # Группа 5: цветовая логика
        self.colorGroup = SettingCardGroup("5) Цвет и акценты", self.settingsWidget)

        # Группа 6-7: базовые стили
        self.mainGroup = SettingCardGroup("Стиль основного субтитра", self.settingsWidget)
        self.subGroup = SettingCardGroup("Стиль дополнительного субтитра", self.settingsWidget)

        # Группа 8: предпросмотр
        self.previewGroup = SettingCardGroup("Предпросмотр", self.settingsWidget)

    def _initPreviewArea(self):
        """初始化右侧预览区域"""
        self.previewScrollArea = ScrollArea()
        self.previewScrollArea.setWidgetResizable(True)
        self.previewScrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.previewCard = CardWidget()
        self.previewLayout = QVBoxLayout(self.previewCard)
        self.previewLayout.setSpacing(16)

        # 顶部预览区域
        self.previewTopWidget = QWidget()
        self.previewTopWidget.setFixedHeight(430)
        self.previewTopLayout = QVBoxLayout(self.previewTopWidget)

        self.previewLabel = BodyLabel("Предпросмотр")
        self.previewImage = ImageLabel()
        self.previewImage.setAlignment(Qt.AlignCenter)
        self.previewTopLayout.addWidget(self.previewImage, 0, Qt.AlignCenter)
        self.previewTopLayout.setAlignment(Qt.AlignVCenter)

        # 底部控件区域
        self.previewBottomWidget = QWidget()
        self.previewBottomLayout = QVBoxLayout(self.previewBottomWidget)

        self.timelineWidget = QWidget()
        self.timelineLayout = QHBoxLayout(self.timelineWidget)
        self.timelineLayout.setContentsMargins(0, 0, 0, 0)
        self.timelineLayout.setSpacing(8)

        self.playPreviewButton = PushButton("▶ Live", self.timelineWidget)
        self.playPreviewButton.setFixedWidth(78)
        self.timelineSlider = Slider(Qt.Horizontal, self.timelineWidget)
        self.timelineSlider.setRange(0, 1000)
        self.timelineSlider.setValue(0)
        self.timelineSlider.setSingleStep(40)
        self.timelineLabel = BodyLabel("0.00 c", self.timelineWidget)
        self.timelineLabel.setMinimumWidth(55)

        self.timelineLayout.addWidget(self.playPreviewButton)
        self.timelineLayout.addWidget(self.timelineSlider, 1)
        self.timelineLayout.addWidget(self.timelineLabel)

        self.styleNameComboBox = ComboBoxSettingCard(
            FIF.VIEW, "Выбор стиля", "Выберите сохранённый пресет стиля", texts=[]
        )

        self.newStyleButton = PushSettingCard(
            "Новый стиль",
            FIF.ADD,
            "Новый стиль",
            "Создать новый пресет на основе текущих настроек",
        )

        self.openStyleFolderButton = PushSettingCard(
            "Открыть папку стилей",
            FIF.FOLDER,
            "Открыть папку стилей",
            "Открыть папку стилей в проводнике",
        )

        self.customPresetComboBox = ComboBoxSettingCard(
            FIF.BOOK_SHELF,
            "Кастомный пресет",
            "Сохранённые наборы ваших настроек стиля и эффектов",
            texts=[],
        )

        self.saveCustomPresetButton = PushSettingCard(
            "Сохранить текущий пресет",
            FIF.SAVE,
            "Сохранить текущий пресет",
            "Сохранить все текущие настройки во внутренний пользовательский пресет",
        )

        self.previewEffectButton = PushSettingCard(
            "Показать анимацию",
            FIF.PLAY,
            "Показать анимацию эффекта",
            "Сгенерировать короткое видео предпросмотра и открыть его",
        )

        self.customPreviewTextLabel = BodyLabel("Свой текст предпросмотра")
        self.customPreviewTextInput = LineEdit(self.previewBottomWidget)
        self.customPreviewTextInput.setPlaceholderText(
            "Введите свой текст (можно: оригинал || перевод)"
        )
        self.customPreviewTextInput.setClearButtonEnabled(True)

        self.wordTimestampHintLabel = BodyLabel(
            "⚠ Для режима «по словам» и точного karaoke нужны word timestamps из ASR.")
        self.wordTimestampHintLabel.setVisible(False)
        self.wordTimestampHintLabel.setStyleSheet(
            "color: #F5A524; font-size: 12px; padding: 2px 0;"
        )

        # Live-блок размещаем выше — сразу под окном предпросмотра
        self.previewBottomLayout.addWidget(self.timelineWidget)
        self.previewBottomLayout.addWidget(self.styleNameComboBox)
        self.previewBottomLayout.addWidget(self.newStyleButton)
        self.previewBottomLayout.addWidget(self.openStyleFolderButton)
        self.previewBottomLayout.addWidget(self.customPresetComboBox)
        self.previewBottomLayout.addWidget(self.saveCustomPresetButton)
        self.previewBottomLayout.addWidget(self.previewEffectButton)
        self.previewBottomLayout.addWidget(self.customPreviewTextLabel)
        self.previewBottomLayout.addWidget(self.customPreviewTextInput)
        self.previewBottomLayout.addWidget(self.wordTimestampHintLabel)

        self.previewLayout.addWidget(self.previewTopWidget)
        self.previewLayout.addWidget(self.previewBottomWidget)
        self.previewLayout.addStretch(1)

        self.previewScrollArea.setWidget(self.previewCard)

    def _initSettingCards(self):
        """初始化所有设置卡片"""
        self.effect_options = EffectManager.get_effect_options()

        # 字幕排布设置
        self.layoutCard = ComboBoxSettingCard(
            FIF.ALIGNMENT,
            "Расположение субтитров",
            "Порядок отображения основного и дополнительного текста",
            texts=list(LAYOUT_LABEL_TO_VALUE.keys()),
        )

        self.effectCard = ComboBoxSettingCard(
            FIF.BRUSH,
            "Эффект субтитров",
            "Выберите анимацию появления субтитров",
            texts=list(self.effect_options.keys()),
        )

        self.effectDurationCard = SpinBoxSettingCard(
            FIF.STOP_WATCH,
            "Длительность эффекта (мс)",
            "Сколько длится анимация появления",
            minimum=20,
            maximum=10000,
        )

        self.effectIntensityCard = SpinBoxSettingCard(
            FIF.ZOOM,
            "Интенсивность эффекта (%)",
            "Сила выраженности анимации",
            minimum=1,
            maximum=500,
        )

        self.rainbowEndColorCard = ColorSettingCard(
            QColor(0, 0, 255),
            FIF.PALETTE,
            "Конечный цвет радуги",
            "Цвет, в который переходит текст при эффекте Радуга",
        )

        self.presetCard = ComboBoxSettingCard(
            FIF.BOOK_SHELF,
            "Пресет стиля",
            "Готовые наборы для шортсов",
            texts=list(STYLE_PRESET_LABEL_TO_VALUE.keys()),
        )

        self.motionDirectionCard = ComboBoxSettingCard(
            FIF.MOVE,
            "Направление анимации",
            "Работает для: Прыжок, Волна, Скольжение, Дрожание, Увеличение",
            texts=list(MOTION_DIRECTION_LABEL_TO_VALUE.keys()),
        )

        self.motionAmplitudeCard = SpinBoxSettingCard(
            FIF.ZOOM,
            "Амплитуда движения (%)",
            "Работает для motion-эффектов (см. выше)",
            minimum=1,
            maximum=500,
        )

        self.motionEasingCard = ComboBoxSettingCard(
            FIF.SPEED_HIGH,
            "Плавность (easing)",
            "Профиль скорости для motion-эффектов",
            texts=list(MOTION_EASING_LABEL_TO_VALUE.keys()),
        )

        self.motionJitterCard = SpinBoxSettingCard(
            FIF.SYNC,
            "Дрожание (jitter, %) ",
            "Микро-смещение позиции (только motion-эффекты)",
            minimum=0,
            maximum=200,
        )

        self.motionBlurCard = SpinBoxSettingCard(
            FIF.HIGHTLIGHT,
            "Motion Blur",
            "Дополнительное размытие во время движения",
            minimum=0,
            maximum=20,
        )

        self.karaokeModeCard = ComboBoxSettingCard(
            FIF.MUSIC,
            "Караоке-режим",
            "Подсветка слов по времени",
            texts=list(TOGGLE_LABEL_TO_VALUE.keys()),
        )

        self.karaokeWindowCard = SpinBoxSettingCard(
            FIF.STOP_WATCH,
            "Окно караоке (мс)",
            "Длительность подсветки слов",
            minimum=50,
            maximum=8000,
        )

        # Единая точка настроек сегментации/пунктуации (перенесено из окна шестерёнки)
        self.needSplitCard = ComboBoxSettingCard(
            FIF.TILES,
            "Умное разбиение (LLM)",
            "Вкл: разбиение на фразы/предложения. Выкл: исходные сегменты",
            texts=list(TOGGLE_LABEL_TO_VALUE.keys()),
        )

        self.splitTypeCard = ComboBoxSettingCard(
            FIF.ALIGNMENT,
            "Тип разбиения",
            "Как делить субтитры при включённом умном разбиении",
            texts=list(SPLIT_TYPE_LABEL_TO_VALUE.keys()),
        )

        self.maxWordCountCjkCard = SpinBoxSettingCard(
            FIF.FONT_SIZE,
            "Макс. символов CJK",
            "Лимит длины строки для китайского/японского/корейского",
            minimum=8,
            maximum=100,
        )

        self.maxWordCountEnglishCard = SpinBoxSettingCard(
            FIF.FONT_SIZE,
            "Макс. слов (англ.)",
            "Лимит длины строки для английского",
            minimum=8,
            maximum=100,
        )

        self.removePunctuationCard = ComboBoxSettingCard(
            FIF.EDIT,
            "Убирать конечную пунктуацию",
            "Удалять хвостовые знаки препинания в конце строк",
            texts=list(TOGGLE_LABEL_TO_VALUE.keys()),
        )

        self.processedSubtitleCacheCard = ComboBoxSettingCard(
            FIF.SAVE,
            "Кэшировать обработанные субтитры",
            "При смене только стиля повторно использовать результат без прогона LLM",
            texts=list(TOGGLE_LABEL_TO_VALUE.keys()),
        )

        self.autoContrastCard = ComboBoxSettingCard(
            FIF.CONSTRACT,
            "Авто-контраст",
            "Добавляет адаптивную читаемость (обводка/тень)",
            texts=list(TOGGLE_LABEL_TO_VALUE.keys()),
        )

        self.antiFlickerCard = ComboBoxSettingCard(
            FIF.SYNC,
            "Анти-мигание",
            "Сглаживает резкие анимационные скачки",
            texts=list(TOGGLE_LABEL_TO_VALUE.keys()),
        )

        self.gradientModeCard = ComboBoxSettingCard(
            FIF.PALETTE,
            "Градиент текста",
            "Режим многоцветной раскраски",
            texts=list(GRADIENT_MODE_LABEL_TO_VALUE.keys()),
        )

        self.gradientColor1Card = ColorSettingCard(
            QColor(255, 255, 255),
            FIF.PALETTE,
            "Градиент: цвет 1",
            "Начальный цвет 2-цветного градиента",
        )

        self.gradientColor2Card = ColorSettingCard(
            QColor(102, 204, 255),
            FIF.PALETTE,
            "Градиент: цвет 2",
            "Конечный цвет 2-цветного градиента",
        )

        self.speakerColorModeCard = ComboBoxSettingCard(
            FIF.PEOPLE,
            "Раскраска по спикерам",
            "Базовый режим: чередование цветов по репликам",
            texts=list(SPEAKER_COLOR_MODE_LABEL_TO_VALUE.keys()),
        )

        self.safeAreaCard = ComboBoxSettingCard(
            FIF.FIT_PAGE,
            "Safe Area",
            "Удерживать субтитры внутри безопасной зоны",
            texts=list(TOGGLE_LABEL_TO_VALUE.keys()),
        )

        self.safeMarginXCard = SpinBoxSettingCard(
            FIF.ALIGNMENT,
            "Safe Area: отступ X (%)",
            "Отступ от левого/правого края",
            minimum=0,
            maximum=40,
        )

        self.safeMarginYCard = SpinBoxSettingCard(
            FIF.ALIGNMENT,
            "Safe Area: отступ Y (%)",
            "Отступ от верхнего/нижнего края",
            minimum=0,
            maximum=40,
        )

        # 垂直间距
        self.verticalSpacingCard = SpinBoxSettingCard(
            FIF.ALIGNMENT,
            "Вертикальный отступ",
            "Расстояние от нижнего края экрана",
            minimum=8,
            maximum=10000,
        )

        # 主字幕样式设置
        self.mainFontCard = ComboBoxSettingCard(
            FIF.FONT,
            "Шрифт основного субтитра",
            "Выбор шрифта основного текста",
            texts=["Arial"],
        )

        self.mainSizeCard = SpinBoxSettingCard(
            FIF.FONT_SIZE,
            "Размер основного субтитра",
            "Размер шрифта основного текста",
            minimum=8,
            maximum=1000,
        )

        self.mainSpacingCard = DoubleSpinBoxSettingCard(
            FIF.ALIGNMENT,
            "Интервал символов (основной)",
            "Межбуквенное расстояние основного текста",
            minimum=0.0,
            maximum=10.0,
            decimals=1,
        )

        self.mainColorCard = ColorSettingCard(
            QColor(255, 255, 255),
            FIF.PALETTE,
            "Цвет основного субтитра",
            "Цвет текста основного субтитра",
        )

        self.mainOutlineColorCard = ColorSettingCard(
            QColor(0, 0, 0),
            FIF.PALETTE,
            "Цвет обводки (основной)",
            "Цвет контура основного текста",
        )

        self.mainOutlineSizeCard = DoubleSpinBoxSettingCard(
            FIF.ZOOM,
            "Толщина обводки (основной)",
            "Толщина контура основного текста",
            minimum=0.0,
            maximum=10.0,
            decimals=1,
        )

        self.mainShadowCard = DoubleSpinBoxSettingCard(
            FIF.BRUSH,
            "Тень (основной)",
            "Смещение тени основного текста",
            minimum=0.0,
            maximum=20.0,
            decimals=1,
        )

        self.mainShadowColorCard = ColorSettingCard(
            QColor(0, 0, 0, 160),
            FIF.PALETTE,
            "Цвет тени (основной)",
            "Цвет тени основного текста",
        )

        self.mainBlurCard = DoubleSpinBoxSettingCard(
            FIF.HIGHTLIGHT,
            "Свечение/размытие (основной)",
            "Сила мягкого свечения основного текста",
            minimum=0.0,
            maximum=10.0,
            decimals=1,
        )

        # 副字幕样式设置
        self.subFontCard = ComboBoxSettingCard(
            FIF.FONT,
            "Шрифт дополнительного субтитра",
            "Выбор шрифта дополнительного текста",
            texts=["Arial"],
        )

        self.subSizeCard = SpinBoxSettingCard(
            FIF.FONT_SIZE,
            "Размер дополнительного субтитра",
            "Размер шрифта дополнительного текста",
            minimum=8,
            maximum=1000,
        )

        self.subSpacingCard = DoubleSpinBoxSettingCard(
            FIF.ALIGNMENT,
            "Интервал символов (доп.)",
            "Межбуквенное расстояние дополнительного текста",
            minimum=0.0,
            maximum=50.0,
            decimals=1,
        )

        self.subColorCard = ColorSettingCard(
            QColor(255, 255, 255),
            FIF.PALETTE,
            "Цвет дополнительного субтитра",
            "Цвет текста дополнительного субтитра",
        )

        self.subOutlineColorCard = ColorSettingCard(
            QColor(0, 0, 0),
            FIF.PALETTE,
            "Цвет обводки (доп.)",
            "Цвет контура дополнительного текста",
        )

        self.subOutlineSizeCard = DoubleSpinBoxSettingCard(
            FIF.ZOOM,
            "Толщина обводки (доп.)",
            "Толщина контура дополнительного текста",
            minimum=0.0,
            maximum=50.0,
            decimals=1,
        )

        self.subShadowCard = DoubleSpinBoxSettingCard(
            FIF.BRUSH,
            "Тень (доп.)",
            "Смещение тени дополнительного текста",
            minimum=0.0,
            maximum=20.0,
            decimals=1,
        )

        self.subShadowColorCard = ColorSettingCard(
            QColor(0, 0, 0, 160),
            FIF.PALETTE,
            "Цвет тени (доп.)",
            "Цвет тени дополнительного текста",
        )

        self.subBlurCard = DoubleSpinBoxSettingCard(
            FIF.HIGHTLIGHT,
            "Свечение/размытие (доп.)",
            "Сила мягкого свечения дополнительного текста",
            minimum=0.0,
            maximum=10.0,
            decimals=1,
        )

        # 预览设置
        self.previewTextCard = ComboBoxSettingCard(
            FIF.MESSAGE,
            "Текст для предпросмотра",
            "Выберите набор текста для проверки стиля",
            texts=list(PREVIEW_TEXTS_SENTENCE.keys()),
            parent=self.previewGroup,
        )

        self.orientationCard = ComboBoxSettingCard(
            FIF.LAYOUT,
            "Ориентация предпросмотра",
            "Горизонтальный или вертикальный формат",
            texts=list(ORIENTATION_LABEL_TO_VALUE.keys()),
            parent=self.previewGroup,
        )

        self.previewImageCard = PushSettingCard(
            "Выбрать изображение",
            FIF.PHOTO,
            "Фон предпросмотра",
            "Выберите фоновую картинку для предпросмотра",
            parent=self.previewGroup,
        )

        self.liveDurationCard = SpinBoxSettingCard(
            FIF.STOP_WATCH,
            "Длительность Live-предпросмотра (мс)",
            "Диапазон времени для ползунка и режима Live",
            minimum=500,
            maximum=15000,
        )

    def _initLayout(self):
        """初始化布局"""
        # 添加卡片到组
        # 1) Быстрый старт
        self.quickStartGroup.addSettingCard(self.presetCard)
        self.quickStartGroup.addSettingCard(self.layoutCard)
        self.quickStartGroup.addSettingCard(self.effectCard)

        # 2) Разбиение и структура
        self.segmentationGroup.addSettingCard(self.needSplitCard)
        self.segmentationGroup.addSettingCard(self.splitTypeCard)
        self.segmentationGroup.addSettingCard(self.maxWordCountCjkCard)
        self.segmentationGroup.addSettingCard(self.maxWordCountEnglishCard)
        self.segmentationGroup.addSettingCard(self.removePunctuationCard)
        self.segmentationGroup.addSettingCard(self.processedSubtitleCacheCard)
        self.segmentationGroup.addSettingCard(self.karaokeModeCard)
        self.segmentationGroup.addSettingCard(self.karaokeWindowCard)

        # 3) Эффекты и движение
        self.effectsGroup.addSettingCard(self.effectDurationCard)
        self.effectsGroup.addSettingCard(self.effectIntensityCard)
        self.effectsGroup.addSettingCard(self.motionDirectionCard)
        self.effectsGroup.addSettingCard(self.motionAmplitudeCard)
        self.effectsGroup.addSettingCard(self.motionEasingCard)
        self.effectsGroup.addSettingCard(self.motionJitterCard)
        self.effectsGroup.addSettingCard(self.motionBlurCard)

        # 4) Читаемость и безопасная зона
        self.readabilityGroup.addSettingCard(self.autoContrastCard)
        self.readabilityGroup.addSettingCard(self.antiFlickerCard)
        self.readabilityGroup.addSettingCard(self.safeAreaCard)
        self.readabilityGroup.addSettingCard(self.safeMarginXCard)
        self.readabilityGroup.addSettingCard(self.safeMarginYCard)
        self.readabilityGroup.addSettingCard(self.verticalSpacingCard)

        # 5) Цвет и акценты
        self.colorGroup.addSettingCard(self.gradientModeCard)
        self.colorGroup.addSettingCard(self.gradientColor1Card)
        self.colorGroup.addSettingCard(self.gradientColor2Card)
        self.colorGroup.addSettingCard(self.rainbowEndColorCard)
        self.colorGroup.addSettingCard(self.speakerColorModeCard)

        self.mainGroup.addSettingCard(self.mainFontCard)
        self.mainGroup.addSettingCard(self.mainSizeCard)
        self.mainGroup.addSettingCard(self.mainSpacingCard)
        self.mainGroup.addSettingCard(self.mainColorCard)
        self.mainGroup.addSettingCard(self.mainOutlineColorCard)
        self.mainGroup.addSettingCard(self.mainOutlineSizeCard)
        self.mainGroup.addSettingCard(self.mainShadowCard)
        self.mainGroup.addSettingCard(self.mainShadowColorCard)
        self.mainGroup.addSettingCard(self.mainBlurCard)

        self.subGroup.addSettingCard(self.subFontCard)
        self.subGroup.addSettingCard(self.subSizeCard)
        self.subGroup.addSettingCard(self.subSpacingCard)
        self.subGroup.addSettingCard(self.subColorCard)
        self.subGroup.addSettingCard(self.subOutlineColorCard)
        self.subGroup.addSettingCard(self.subOutlineSizeCard)
        self.subGroup.addSettingCard(self.subShadowCard)
        self.subGroup.addSettingCard(self.subShadowColorCard)
        self.subGroup.addSettingCard(self.subBlurCard)

        self.previewGroup.addSettingCard(self.previewTextCard)
        self.previewGroup.addSettingCard(self.orientationCard)
        self.previewGroup.addSettingCard(self.liveDurationCard)
        self.previewGroup.addSettingCard(self.previewImageCard)

        # 添加组到布局
        self.settingsLayout.addWidget(self.quickStartGroup)
        self.settingsLayout.addWidget(self.segmentationGroup)
        self.settingsLayout.addWidget(self.effectsGroup)
        self.settingsLayout.addWidget(self.readabilityGroup)
        self.settingsLayout.addWidget(self.colorGroup)
        self.settingsLayout.addWidget(self.mainGroup)
        self.settingsLayout.addWidget(self.subGroup)
        self.settingsLayout.addWidget(self.previewGroup)
        self.settingsLayout.addStretch(1)

        # 添加左右两侧到主布局
        self.hBoxLayout.addWidget(self.settingsScrollArea)
        self.hBoxLayout.addWidget(self.previewScrollArea, 1)

    def _initStyle(self):
        """初始化样式"""
        self.settingsWidget.setObjectName("settingsWidget")
        self.setStyleSheet(
            """        
            SubtitleStyleInterface, #settingsWidget {
                background-color: transparent;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """
        )

    def __setValues(self):
        """设置初始值"""
        # 设置字幕排布
        self.layoutCard.comboBox.setCurrentText(
            LAYOUT_VALUE_TO_LABEL.get(cfg.get(cfg.subtitle_layout), "Перевод сверху")
        )
        # 设置字幕效果
        current_effect = cfg.get(cfg.subtitle_effect)
        current_effect_label = next(
            (
                label
                for label, value in self.effect_options.items()
                if value == current_effect
            ),
            "Без эффекта",
        )
        self.effectCard.comboBox.setCurrentText(current_effect_label)
        self.effectDurationCard.spinBox.setValue(cfg.get(cfg.subtitle_effect_duration))
        self.effectIntensityCard.spinBox.setValue(cfg.get(cfg.subtitle_effect_intensity))
        rainbow_hex = cfg.get(cfg.subtitle_rainbow_end_color) or "#0000FF"
        self.rainbowEndColorCard.setColor(QColor(rainbow_hex))
        self.presetCard.comboBox.setCurrentText(
            STYLE_PRESET_VALUE_TO_LABEL.get(
                cfg.get(cfg.subtitle_style_preset), "Свой (Custom)"
            )
        )
        self.motionDirectionCard.comboBox.setCurrentText(
            MOTION_DIRECTION_VALUE_TO_LABEL.get(
                cfg.get(cfg.subtitle_motion_direction), "Снизу вверх"
            )
        )
        self.motionAmplitudeCard.spinBox.setValue(
            int(cfg.get(cfg.subtitle_motion_amplitude))
        )
        self.motionEasingCard.comboBox.setCurrentText(
            MOTION_EASING_VALUE_TO_LABEL.get(
                cfg.get(cfg.subtitle_motion_easing), "Ease Out (мягкий финиш)"
            )
        )
        self.motionJitterCard.spinBox.setValue(int(cfg.get(cfg.subtitle_motion_jitter)))
        self.motionBlurCard.spinBox.setValue(int(cfg.get(cfg.subtitle_motion_blur_strength)))
        self.karaokeModeCard.comboBox.setCurrentText(
            TOGGLE_VALUE_TO_LABEL.get(bool(cfg.get(cfg.subtitle_karaoke_mode)), "Выкл")
        )
        self.karaokeWindowCard.spinBox.setValue(
            int(cfg.get(cfg.subtitle_karaoke_window_ms))
        )
        self.needSplitCard.comboBox.setCurrentText(
            TOGGLE_VALUE_TO_LABEL.get(bool(cfg.get(cfg.need_split)), "Выкл")
        )
        self.splitTypeCard.comboBox.setCurrentText(
            SPLIT_TYPE_VALUE_TO_LABEL.get(
                cfg.get(cfg.split_type),
                "По предложениям",
            )
        )
        self._refresh_effect_options_by_split_mode()
        self._refresh_preview_text_options()
        self.liveDurationCard.spinBox.setValue(
            int(cfg.get(cfg.subtitle_preview_live_duration_ms))
        )
        self._apply_live_duration_ms(int(cfg.get(cfg.subtitle_preview_live_duration_ms)))
        self.customPreviewTextInput.setText(
            str(cfg.get(cfg.subtitle_preview_custom_text) or "")
        )
        self.maxWordCountCjkCard.spinBox.setValue(int(cfg.get(cfg.max_word_count_cjk)))
        self.maxWordCountEnglishCard.spinBox.setValue(
            int(cfg.get(cfg.max_word_count_english))
        )
        self.removePunctuationCard.comboBox.setCurrentText(
            TOGGLE_VALUE_TO_LABEL.get(
                bool(cfg.get(cfg.needs_remove_punctuation)),
                "Вкл",
            )
        )
        self.processedSubtitleCacheCard.comboBox.setCurrentText(
            TOGGLE_VALUE_TO_LABEL.get(
                bool(cfg.get(cfg.use_processed_subtitle_cache)),
                "Вкл",
            )
        )
        self.autoContrastCard.comboBox.setCurrentText(
            TOGGLE_VALUE_TO_LABEL.get(bool(cfg.get(cfg.subtitle_auto_contrast)), "Выкл")
        )
        self.antiFlickerCard.comboBox.setCurrentText(
            TOGGLE_VALUE_TO_LABEL.get(bool(cfg.get(cfg.subtitle_anti_flicker)), "Вкл")
        )
        self.gradientModeCard.comboBox.setCurrentText(
            GRADIENT_MODE_VALUE_TO_LABEL.get(
                cfg.get(cfg.subtitle_gradient_mode), "Без градиента"
            )
        )
        self.gradientColor1Card.setColor(
            QColor(cfg.get(cfg.subtitle_gradient_color_1) or "#FFFFFF")
        )
        self.gradientColor2Card.setColor(
            QColor(cfg.get(cfg.subtitle_gradient_color_2) or "#66CCFF")
        )
        self.speakerColorModeCard.comboBox.setCurrentText(
            SPEAKER_COLOR_MODE_VALUE_TO_LABEL.get(
                cfg.get(cfg.subtitle_speaker_color_mode),
                "Выкл",
            )
        )
        self.safeAreaCard.comboBox.setCurrentText(
            TOGGLE_VALUE_TO_LABEL.get(bool(cfg.get(cfg.subtitle_safe_area_enabled)), "Вкл")
        )
        self.safeMarginXCard.spinBox.setValue(int(cfg.get(cfg.subtitle_safe_margin_x)))
        self.safeMarginYCard.spinBox.setValue(int(cfg.get(cfg.subtitle_safe_margin_y)))
        # 设置字幕样式
        self.styleNameComboBox.comboBox.setCurrentText(cfg.get(cfg.subtitle_style_name))

        # 获取系统字体,设置comboBox的选项
        fontDatabase = QFontDatabase()
        fontFamilies = fontDatabase.families()
        self.mainFontCard.addItems(fontFamilies)
        self.subFontCard.addItems(fontFamilies)

        # 设置字体选项框最大显示数量
        self.mainFontCard.comboBox.setMaxVisibleItems(12)
        self.subFontCard.comboBox.setMaxVisibleItems(12)

        # 获取样式目录下的所有txt文件名
        style_files = [f.stem for f in SUBTITLE_STYLE_PATH.glob("*.txt")]
        if "default" in style_files:
            style_files.insert(0, style_files.pop(style_files.index("default")))
        else:
            style_files.insert(0, "default")
            self.saveStyle("default")
        self.styleNameComboBox.comboBox.addItems(style_files)

        # 加载默认样式
        subtitle_style_name = cfg.get(cfg.subtitle_style_name)
        if subtitle_style_name in style_files:
            self.loadStyle(subtitle_style_name)
            self.styleNameComboBox.comboBox.setCurrentText(subtitle_style_name)
        else:
            self.loadStyle(style_files[0])
            self.styleNameComboBox.comboBox.setCurrentText(style_files[0])

        self._refresh_custom_preset_items()

    def connectSignals(self):
        """连接所有设置变更的信号到预览更新函数"""
        # 字幕排布
        self.layoutCard.currentTextChanged.connect(self.onSettingChanged)
        self.layoutCard.currentTextChanged.connect(
            lambda text: cfg.set(
                cfg.subtitle_layout,
                LAYOUT_LABEL_TO_VALUE.get(text, "译文在上"),
            )
        )
        self.effectCard.currentTextChanged.connect(self.onSettingChanged)
        self.effectCard.currentTextChanged.connect(
            lambda text: cfg.set(
                cfg.subtitle_effect,
                self.effect_options.get(text, "none"),
            )
        )
        self.effectCard.currentTextChanged.connect(
            lambda _: self._update_motion_controls_state()
        )
        self.effectDurationCard.spinBox.valueChanged.connect(self.onSettingChanged)
        self.effectDurationCard.spinBox.valueChanged.connect(
            lambda value: cfg.set(cfg.subtitle_effect_duration, int(value))
        )
        self.effectIntensityCard.spinBox.valueChanged.connect(self.onSettingChanged)
        self.effectIntensityCard.spinBox.valueChanged.connect(
            lambda value: cfg.set(cfg.subtitle_effect_intensity, int(value))
        )
        self.rainbowEndColorCard.colorChanged.connect(self.onSettingChanged)
        self.rainbowEndColorCard.colorChanged.connect(
            lambda color: cfg.set(cfg.subtitle_rainbow_end_color, color.name())
        )
        self.presetCard.currentTextChanged.connect(self.onPresetChanged)
        self.presetCard.currentTextChanged.connect(
            lambda text: cfg.set(
                cfg.subtitle_style_preset,
                STYLE_PRESET_LABEL_TO_VALUE.get(text, "custom"),
            )
        )
        self.motionDirectionCard.currentTextChanged.connect(self.onSettingChanged)
        self.motionDirectionCard.currentTextChanged.connect(
            lambda text: cfg.set(
                cfg.subtitle_motion_direction,
                MOTION_DIRECTION_LABEL_TO_VALUE.get(text, "up"),
            )
        )
        self.motionAmplitudeCard.spinBox.valueChanged.connect(self.onSettingChanged)
        self.motionAmplitudeCard.spinBox.valueChanged.connect(
            lambda value: cfg.set(cfg.subtitle_motion_amplitude, int(value))
        )
        self.motionEasingCard.currentTextChanged.connect(self.onSettingChanged)
        self.motionEasingCard.currentTextChanged.connect(
            lambda text: cfg.set(
                cfg.subtitle_motion_easing,
                MOTION_EASING_LABEL_TO_VALUE.get(text, "ease_out"),
            )
        )
        self.motionJitterCard.spinBox.valueChanged.connect(self.onSettingChanged)
        self.motionJitterCard.spinBox.valueChanged.connect(
            lambda value: cfg.set(cfg.subtitle_motion_jitter, int(value))
        )
        self.motionBlurCard.spinBox.valueChanged.connect(self.onSettingChanged)
        self.motionBlurCard.spinBox.valueChanged.connect(
            lambda value: cfg.set(cfg.subtitle_motion_blur_strength, int(value))
        )
        self.karaokeModeCard.currentTextChanged.connect(self.onSettingChanged)
        self.karaokeModeCard.currentTextChanged.connect(
            lambda text: cfg.set(
                cfg.subtitle_karaoke_mode,
                TOGGLE_LABEL_TO_VALUE.get(text, False),
            )
        )
        self.karaokeWindowCard.spinBox.valueChanged.connect(self.onSettingChanged)
        self.karaokeWindowCard.spinBox.valueChanged.connect(
            lambda value: cfg.set(cfg.subtitle_karaoke_window_ms, int(value))
        )
        self.needSplitCard.currentTextChanged.connect(self.onSettingChanged)
        self.needSplitCard.currentTextChanged.connect(
            lambda text: cfg.set(
                cfg.need_split,
                TOGGLE_LABEL_TO_VALUE.get(text, False),
            )
        )
        self.splitTypeCard.currentTextChanged.connect(self._on_split_type_changed)
        self.splitTypeCard.currentTextChanged.connect(
            lambda text: cfg.set(
                cfg.split_type,
                SPLIT_TYPE_LABEL_TO_VALUE.get(text, SplitTypeEnum.SENTENCE.value),
            )
        )
        self.maxWordCountCjkCard.spinBox.valueChanged.connect(self.onSettingChanged)
        self.maxWordCountCjkCard.spinBox.valueChanged.connect(
            lambda value: cfg.set(cfg.max_word_count_cjk, int(value))
        )
        self.maxWordCountEnglishCard.spinBox.valueChanged.connect(self.onSettingChanged)
        self.maxWordCountEnglishCard.spinBox.valueChanged.connect(
            lambda value: cfg.set(cfg.max_word_count_english, int(value))
        )
        self.removePunctuationCard.currentTextChanged.connect(self.onSettingChanged)
        self.removePunctuationCard.currentTextChanged.connect(
            lambda text: cfg.set(
                cfg.needs_remove_punctuation,
                TOGGLE_LABEL_TO_VALUE.get(text, True),
            )
        )
        self.processedSubtitleCacheCard.currentTextChanged.connect(self.onSettingChanged)
        self.processedSubtitleCacheCard.currentTextChanged.connect(
            lambda text: cfg.set(
                cfg.use_processed_subtitle_cache,
                TOGGLE_LABEL_TO_VALUE.get(text, True),
            )
        )
        self.autoContrastCard.currentTextChanged.connect(self.onSettingChanged)
        self.autoContrastCard.currentTextChanged.connect(
            lambda text: cfg.set(
                cfg.subtitle_auto_contrast,
                TOGGLE_LABEL_TO_VALUE.get(text, False),
            )
        )
        self.antiFlickerCard.currentTextChanged.connect(self.onSettingChanged)
        self.antiFlickerCard.currentTextChanged.connect(
            lambda text: cfg.set(
                cfg.subtitle_anti_flicker,
                TOGGLE_LABEL_TO_VALUE.get(text, True),
            )
        )
        self.gradientModeCard.currentTextChanged.connect(self.onSettingChanged)
        self.gradientModeCard.currentTextChanged.connect(
            lambda text: cfg.set(
                cfg.subtitle_gradient_mode,
                GRADIENT_MODE_LABEL_TO_VALUE.get(text, "off"),
            )
        )
        self.gradientColor1Card.colorChanged.connect(self.onSettingChanged)
        self.gradientColor1Card.colorChanged.connect(
            lambda color: cfg.set(cfg.subtitle_gradient_color_1, color.name())
        )
        self.gradientColor2Card.colorChanged.connect(self.onSettingChanged)
        self.gradientColor2Card.colorChanged.connect(
            lambda color: cfg.set(cfg.subtitle_gradient_color_2, color.name())
        )
        self.speakerColorModeCard.currentTextChanged.connect(self.onSettingChanged)
        self.speakerColorModeCard.currentTextChanged.connect(
            lambda text: cfg.set(
                cfg.subtitle_speaker_color_mode,
                SPEAKER_COLOR_MODE_LABEL_TO_VALUE.get(text, "off"),
            )
        )
        self.safeAreaCard.currentTextChanged.connect(self.onSettingChanged)
        self.safeAreaCard.currentTextChanged.connect(
            lambda text: cfg.set(
                cfg.subtitle_safe_area_enabled,
                TOGGLE_LABEL_TO_VALUE.get(text, True),
            )
        )
        self.safeMarginXCard.spinBox.valueChanged.connect(self.onSettingChanged)
        self.safeMarginXCard.spinBox.valueChanged.connect(
            lambda value: cfg.set(cfg.subtitle_safe_margin_x, int(value))
        )
        self.safeMarginYCard.spinBox.valueChanged.connect(self.onSettingChanged)
        self.safeMarginYCard.spinBox.valueChanged.connect(
            lambda value: cfg.set(cfg.subtitle_safe_margin_y, int(value))
        )
        # 垂直间距
        self.verticalSpacingCard.spinBox.valueChanged.connect(self.onSettingChanged)

        # 主字幕样式
        self.mainFontCard.currentTextChanged.connect(self.onSettingChanged)
        self.mainSizeCard.spinBox.valueChanged.connect(self.onSettingChanged)
        self.mainSpacingCard.spinBox.valueChanged.connect(self.onSettingChanged)
        self.mainColorCard.colorChanged.connect(self.onSettingChanged)
        self.mainOutlineColorCard.colorChanged.connect(self.onSettingChanged)
        self.mainOutlineSizeCard.spinBox.valueChanged.connect(self.onSettingChanged)
        self.mainShadowCard.spinBox.valueChanged.connect(self.onSettingChanged)
        self.mainShadowColorCard.colorChanged.connect(self.onSettingChanged)
        self.mainBlurCard.spinBox.valueChanged.connect(self.onSettingChanged)

        # 副字幕样式
        self.subFontCard.currentTextChanged.connect(self.onSettingChanged)
        self.subSizeCard.spinBox.valueChanged.connect(self.onSettingChanged)
        self.subSpacingCard.spinBox.valueChanged.connect(self.onSettingChanged)
        self.subColorCard.colorChanged.connect(self.onSettingChanged)
        self.subOutlineColorCard.colorChanged.connect(self.onSettingChanged)
        self.subOutlineSizeCard.spinBox.valueChanged.connect(self.onSettingChanged)
        self.subShadowCard.spinBox.valueChanged.connect(self.onSettingChanged)
        self.subShadowColorCard.colorChanged.connect(self.onSettingChanged)
        self.subBlurCard.spinBox.valueChanged.connect(self.onSettingChanged)

        # 预览设置
        self.previewTextCard.currentTextChanged.connect(self.onSettingChanged)
        self.orientationCard.currentTextChanged.connect(self.onOrientationChanged)
        self.liveDurationCard.spinBox.valueChanged.connect(self.onLiveDurationChanged)
        self.previewImageCard.clicked.connect(self.selectPreviewImage)
        self.customPreviewTextInput.textChanged.connect(self.onCustomPreviewTextChanged)
        self.previewEffectButton.clicked.connect(self.showEffectPreview)
        self.playPreviewButton.clicked.connect(self.toggleLivePreview)
        self.timelineSlider.valueChanged.connect(self.onTimelineChanged)

        # 连接样式切换信号
        self.styleNameComboBox.currentTextChanged.connect(self.loadStyle)
        self.newStyleButton.clicked.connect(self.createNewStyle)
        self.openStyleFolderButton.clicked.connect(self.on_open_style_folder_clicked)
        self.customPresetComboBox.currentTextChanged.connect(self.onCustomPresetChanged)
        self.saveCustomPresetButton.clicked.connect(self.saveCurrentCustomPreset)

        # 连接字幕排布信号
        self.layoutCard.comboBox.currentTextChanged.connect(
            lambda text: signalBus.subtitle_layout_changed.emit(
                LAYOUT_LABEL_TO_VALUE.get(text, "译文在上")
            )
        )
        signalBus.subtitle_layout_changed.connect(self.on_subtitle_layout_changed)

    def on_open_style_folder_clicked(self):
        """打开样式文件夹"""
        if sys.platform == "win32":
            os.startfile(SUBTITLE_STYLE_PATH)
        elif sys.platform == "darwin":  # macOS
            subprocess.run(["open", SUBTITLE_STYLE_PATH])
        else:  # Linux
            subprocess.run(["xdg-open", SUBTITLE_STYLE_PATH])

    def _read_custom_presets(self) -> dict:
        try:
            if not CUSTOM_SUBTITLE_PRESETS_PATH.exists():
                return {}
            raw = CUSTOM_SUBTITLE_PRESETS_PATH.read_text(encoding="utf-8")
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _write_custom_presets(self, data: dict):
        try:
            CUSTOM_SUBTITLE_PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
            CUSTOM_SUBTITLE_PRESETS_PATH.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _collect_custom_preset_state(self) -> dict:
        def _color_hex(c: QColor) -> str:
            try:
                return c.name(QColor.HexArgb)
            except Exception:
                return c.name()

        return {
            "style_name": self.styleNameComboBox.comboBox.currentText(),
            "vertical_spacing": int(self.verticalSpacingCard.spinBox.value()),

            "layout_label": self.layoutCard.comboBox.currentText(),
            "effect_label": self.effectCard.comboBox.currentText(),
            "effect_duration": int(self.effectDurationCard.spinBox.value()),
            "effect_intensity": int(self.effectIntensityCard.spinBox.value()),
            "rainbow_end_color": _color_hex(self.rainbowEndColorCard.colorPicker.color),
            "preset_label": self.presetCard.comboBox.currentText(),
            "motion_direction_label": self.motionDirectionCard.comboBox.currentText(),
            "motion_amplitude": int(self.motionAmplitudeCard.spinBox.value()),
            "motion_easing_label": self.motionEasingCard.comboBox.currentText(),
            "motion_jitter": int(self.motionJitterCard.spinBox.value()),
            "motion_blur": int(self.motionBlurCard.spinBox.value()),
            "karaoke_mode_label": self.karaokeModeCard.comboBox.currentText(),
            "karaoke_window_ms": int(self.karaokeWindowCard.spinBox.value()),
            "need_split_label": self.needSplitCard.comboBox.currentText(),
            "split_type_label": self.splitTypeCard.comboBox.currentText(),
            "max_word_count_cjk": int(self.maxWordCountCjkCard.spinBox.value()),
            "max_word_count_english": int(self.maxWordCountEnglishCard.spinBox.value()),
            "remove_punctuation_label": self.removePunctuationCard.comboBox.currentText(),
            "processed_cache_label": self.processedSubtitleCacheCard.comboBox.currentText(),
            "auto_contrast_label": self.autoContrastCard.comboBox.currentText(),
            "anti_flicker_label": self.antiFlickerCard.comboBox.currentText(),
            "gradient_mode_label": self.gradientModeCard.comboBox.currentText(),
            "gradient_color_1": _color_hex(self.gradientColor1Card.colorPicker.color),
            "gradient_color_2": _color_hex(self.gradientColor2Card.colorPicker.color),
            "speaker_color_mode_label": self.speakerColorModeCard.comboBox.currentText(),
            "safe_area_label": self.safeAreaCard.comboBox.currentText(),
            "safe_margin_x": int(self.safeMarginXCard.spinBox.value()),
            "safe_margin_y": int(self.safeMarginYCard.spinBox.value()),

            "main_font": self.mainFontCard.comboBox.currentText(),
            "main_size": int(self.mainSizeCard.spinBox.value()),
            "main_font_size": int(self.mainSizeCard.spinBox.value()),
            "main_spacing": float(self.mainSpacingCard.spinBox.value()),
            "main_color": _color_hex(self.mainColorCard.colorPicker.color),
            "main_outline_color": _color_hex(self.mainOutlineColorCard.colorPicker.color),
            "main_outline_size": float(self.mainOutlineSizeCard.spinBox.value()),
            "main_shadow": float(self.mainShadowCard.spinBox.value()),
            "main_shadow_color": _color_hex(self.mainShadowColorCard.colorPicker.color),
            "main_blur": float(self.mainBlurCard.spinBox.value()),

            "sub_font": self.subFontCard.comboBox.currentText(),
            "sub_size": int(self.subSizeCard.spinBox.value()),
            "sub_font_size": int(self.subSizeCard.spinBox.value()),
            "sub_spacing": float(self.subSpacingCard.spinBox.value()),
            "sub_color": _color_hex(self.subColorCard.colorPicker.color),
            "sub_outline_color": _color_hex(self.subOutlineColorCard.colorPicker.color),
            "sub_outline_size": float(self.subOutlineSizeCard.spinBox.value()),
            "sub_shadow": float(self.subShadowCard.spinBox.value()),
            "sub_shadow_color": _color_hex(self.subShadowColorCard.colorPicker.color),
            "sub_blur": float(self.subBlurCard.spinBox.value()),

            "preview_text_label": self.previewTextCard.comboBox.currentText(),
            "preview_custom_text": self.customPreviewTextInput.text(),
            "orientation_label": self.orientationCard.comboBox.currentText(),
        }

    def _apply_custom_preset_state(self, state: dict):
        if not isinstance(state, dict):
            return

        def _qcolor(value: str, fallback: QColor) -> QColor:
            try:
                c = QColor(str(value or ""))
                return c if c.isValid() else fallback
            except Exception:
                return fallback

        def _as_int(value, fallback: int) -> int:
            try:
                return int(value)
            except Exception:
                return int(fallback)

        def _as_float(value, fallback: float) -> float:
            try:
                return float(value)
            except Exception:
                return float(fallback)

        self._loading_style = True
        try:
            style_name = str(state.get("style_name", "") or "").strip()
            if style_name:
                self.styleNameComboBox.comboBox.setCurrentText(style_name)

            self.verticalSpacingCard.spinBox.setValue(
                _as_int(state.get("vertical_spacing", self.verticalSpacingCard.spinBox.value()), self.verticalSpacingCard.spinBox.value())
            )

            self.layoutCard.comboBox.setCurrentText(
                str(state.get("layout_label", self.layoutCard.comboBox.currentText()))
            )
            self.effectCard.comboBox.setCurrentText(
                str(state.get("effect_label", self.effectCard.comboBox.currentText()))
            )
            self.effectDurationCard.spinBox.setValue(
                _as_int(state.get("effect_duration", self.effectDurationCard.spinBox.value()), self.effectDurationCard.spinBox.value())
            )
            self.effectIntensityCard.spinBox.setValue(
                _as_int(state.get("effect_intensity", self.effectIntensityCard.spinBox.value()), self.effectIntensityCard.spinBox.value())
            )
            self.rainbowEndColorCard.setColor(
                _qcolor(
                    str(state.get("rainbow_end_color", "")),
                    self.rainbowEndColorCard.colorPicker.color,
                )
            )
            self.presetCard.comboBox.setCurrentText(
                str(state.get("preset_label", self.presetCard.comboBox.currentText()))
            )
            self.motionDirectionCard.comboBox.setCurrentText(
                str(state.get("motion_direction_label", self.motionDirectionCard.comboBox.currentText()))
            )
            self.motionAmplitudeCard.spinBox.setValue(
                _as_int(state.get("motion_amplitude", self.motionAmplitudeCard.spinBox.value()), self.motionAmplitudeCard.spinBox.value())
            )
            self.motionEasingCard.comboBox.setCurrentText(
                str(state.get("motion_easing_label", self.motionEasingCard.comboBox.currentText()))
            )
            self.motionJitterCard.spinBox.setValue(
                _as_int(state.get("motion_jitter", self.motionJitterCard.spinBox.value()), self.motionJitterCard.spinBox.value())
            )
            self.motionBlurCard.spinBox.setValue(
                _as_int(state.get("motion_blur", self.motionBlurCard.spinBox.value()), self.motionBlurCard.spinBox.value())
            )
            self.karaokeModeCard.comboBox.setCurrentText(
                str(state.get("karaoke_mode_label", self.karaokeModeCard.comboBox.currentText()))
            )
            self.karaokeWindowCard.spinBox.setValue(
                _as_int(state.get("karaoke_window_ms", self.karaokeWindowCard.spinBox.value()), self.karaokeWindowCard.spinBox.value())
            )
            self.needSplitCard.comboBox.setCurrentText(
                str(state.get("need_split_label", self.needSplitCard.comboBox.currentText()))
            )
            self.splitTypeCard.comboBox.setCurrentText(
                str(state.get("split_type_label", self.splitTypeCard.comboBox.currentText()))
            )
            self.maxWordCountCjkCard.spinBox.setValue(
                _as_int(state.get("max_word_count_cjk", self.maxWordCountCjkCard.spinBox.value()), self.maxWordCountCjkCard.spinBox.value())
            )
            self.maxWordCountEnglishCard.spinBox.setValue(
                _as_int(state.get("max_word_count_english", self.maxWordCountEnglishCard.spinBox.value()), self.maxWordCountEnglishCard.spinBox.value())
            )
            self.removePunctuationCard.comboBox.setCurrentText(
                str(state.get("remove_punctuation_label", self.removePunctuationCard.comboBox.currentText()))
            )
            self.processedSubtitleCacheCard.comboBox.setCurrentText(
                str(state.get("processed_cache_label", self.processedSubtitleCacheCard.comboBox.currentText()))
            )
            self.autoContrastCard.comboBox.setCurrentText(
                str(state.get("auto_contrast_label", self.autoContrastCard.comboBox.currentText()))
            )
            self.antiFlickerCard.comboBox.setCurrentText(
                str(state.get("anti_flicker_label", self.antiFlickerCard.comboBox.currentText()))
            )
            self.gradientModeCard.comboBox.setCurrentText(
                str(state.get("gradient_mode_label", self.gradientModeCard.comboBox.currentText()))
            )
            self.gradientColor1Card.setColor(
                _qcolor(
                    str(state.get("gradient_color_1", "")),
                    self.gradientColor1Card.colorPicker.color,
                )
            )
            self.gradientColor2Card.setColor(
                _qcolor(
                    str(state.get("gradient_color_2", "")),
                    self.gradientColor2Card.colorPicker.color,
                )
            )
            self.speakerColorModeCard.comboBox.setCurrentText(
                str(state.get("speaker_color_mode_label", self.speakerColorModeCard.comboBox.currentText()))
            )
            self.safeAreaCard.comboBox.setCurrentText(
                str(state.get("safe_area_label", self.safeAreaCard.comboBox.currentText()))
            )
            self.safeMarginXCard.spinBox.setValue(
                _as_int(state.get("safe_margin_x", self.safeMarginXCard.spinBox.value()), self.safeMarginXCard.spinBox.value())
            )
            self.safeMarginYCard.spinBox.setValue(
                _as_int(state.get("safe_margin_y", self.safeMarginYCard.spinBox.value()), self.safeMarginYCard.spinBox.value())
            )

            self.mainFontCard.comboBox.setCurrentText(
                str(state.get("main_font", self.mainFontCard.comboBox.currentText()))
            )
            self.mainSizeCard.spinBox.setValue(
                _as_int(
                    state.get(
                        "main_size",
                        state.get("main_font_size", self.mainSizeCard.spinBox.value()),
                    ),
                    self.mainSizeCard.spinBox.value(),
                )
            )
            self.mainSpacingCard.spinBox.setValue(
                _as_float(state.get("main_spacing", self.mainSpacingCard.spinBox.value()), self.mainSpacingCard.spinBox.value())
            )
            self.mainColorCard.setColor(
                _qcolor(str(state.get("main_color", "")), self.mainColorCard.colorPicker.color)
            )
            self.mainOutlineColorCard.setColor(
                _qcolor(
                    str(state.get("main_outline_color", "")),
                    self.mainOutlineColorCard.colorPicker.color,
                )
            )
            self.mainOutlineSizeCard.spinBox.setValue(
                _as_float(state.get("main_outline_size", self.mainOutlineSizeCard.spinBox.value()), self.mainOutlineSizeCard.spinBox.value())
            )
            self.mainShadowCard.spinBox.setValue(
                _as_float(state.get("main_shadow", self.mainShadowCard.spinBox.value()), self.mainShadowCard.spinBox.value())
            )
            self.mainShadowColorCard.setColor(
                _qcolor(
                    str(state.get("main_shadow_color", "")),
                    self.mainShadowColorCard.colorPicker.color,
                )
            )
            self.mainBlurCard.spinBox.setValue(
                _as_float(state.get("main_blur", self.mainBlurCard.spinBox.value()), self.mainBlurCard.spinBox.value())
            )

            self.subFontCard.comboBox.setCurrentText(
                str(state.get("sub_font", self.subFontCard.comboBox.currentText()))
            )
            self.subSizeCard.spinBox.setValue(
                _as_int(
                    state.get(
                        "sub_size",
                        state.get("sub_font_size", self.subSizeCard.spinBox.value()),
                    ),
                    self.subSizeCard.spinBox.value(),
                )
            )
            self.subSpacingCard.spinBox.setValue(
                _as_float(state.get("sub_spacing", self.subSpacingCard.spinBox.value()), self.subSpacingCard.spinBox.value())
            )
            self.subColorCard.setColor(
                _qcolor(str(state.get("sub_color", "")), self.subColorCard.colorPicker.color)
            )
            self.subOutlineColorCard.setColor(
                _qcolor(
                    str(state.get("sub_outline_color", "")),
                    self.subOutlineColorCard.colorPicker.color,
                )
            )
            self.subOutlineSizeCard.spinBox.setValue(
                _as_float(state.get("sub_outline_size", self.subOutlineSizeCard.spinBox.value()), self.subOutlineSizeCard.spinBox.value())
            )
            self.subShadowCard.spinBox.setValue(
                _as_float(state.get("sub_shadow", self.subShadowCard.spinBox.value()), self.subShadowCard.spinBox.value())
            )
            self.subShadowColorCard.setColor(
                _qcolor(
                    str(state.get("sub_shadow_color", "")),
                    self.subShadowColorCard.colorPicker.color,
                )
            )
            self.subBlurCard.spinBox.setValue(
                _as_float(state.get("sub_blur", self.subBlurCard.spinBox.value()), self.subBlurCard.spinBox.value())
            )

            self.previewTextCard.comboBox.setCurrentText(
                str(state.get("preview_text_label", self.previewTextCard.comboBox.currentText()))
            )
            self.customPreviewTextInput.setText(
                str(state.get("preview_custom_text", self.customPreviewTextInput.text()))
            )
            self.orientationCard.comboBox.setCurrentText(
                str(state.get("orientation_label", self.orientationCard.comboBox.currentText()))
            )
        finally:
            self._loading_style = False

        self._update_motion_controls_state()
        self.onSettingChanged()

    def _refresh_custom_preset_items(self, select_name: str = ""):
        data = self._read_custom_presets()
        items = sorted((data.get("presets") or {}).keys(), key=lambda x: x.lower())
        last_selected = str(data.get("last_selected_preset", "") or "").strip()
        self.customPresetComboBox.comboBox.blockSignals(True)
        self.customPresetComboBox.comboBox.clear()
        self.customPresetComboBox.comboBox.addItem("—")
        if items:
            self.customPresetComboBox.comboBox.addItems(items)

        if select_name in items:
            target = select_name
        elif last_selected in items:
            target = last_selected
        else:
            target = "—"
        self.customPresetComboBox.comboBox.setCurrentText(target)
        self.customPresetComboBox.comboBox.blockSignals(False)

        # Автоматически применяем последний использованный пресет при открытии вкладки
        if target != "—":
            self.onCustomPresetChanged(target)

    def saveCurrentCustomPreset(self):
        data = self._read_custom_presets()
        presets = data.get("presets") if isinstance(data.get("presets"), dict) else {}

        selected_name = (self.customPresetComboBox.comboBox.currentText() or "").strip()
        if selected_name and selected_name != "—" and selected_name in presets:
            preset_name = selected_name
            mode = "updated"
        else:
            dialog = PresetNameDialog(self)
            if not dialog.exec():
                return
            preset_name = dialog.nameLineEdit.text().strip()
            if not preset_name:
                return
            mode = "created"

        presets[preset_name] = self._collect_custom_preset_state()
        data["presets"] = presets
        data["last_selected_preset"] = preset_name
        self._write_custom_presets(data)
        self._refresh_custom_preset_items(select_name=preset_name)

        InfoBar.success(
            title="Успешно",
            content=(
                f"Кастомный пресет обновлён: {preset_name}"
                if mode == "updated"
                else f"Кастомный пресет сохранён: {preset_name}"
            ),
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=1800,
            parent=self,
        )

    def onCustomPresetChanged(self, preset_name: str):
        preset_name = (preset_name or "").strip()
        if not preset_name or preset_name == "—":
            return

        data = self._read_custom_presets()
        presets = data.get("presets") if isinstance(data.get("presets"), dict) else {}
        state = presets.get(preset_name)
        if not isinstance(state, dict):
            return

        data["last_selected_preset"] = preset_name
        self._write_custom_presets(data)
        self._apply_custom_preset_state(state)

    def on_subtitle_layout_changed(self, layout: str):
        cfg.subtitle_layout.value = layout
        self.layoutCard.setCurrentText(LAYOUT_VALUE_TO_LABEL.get(layout, "Перевод сверху"))

    def onOrientationChanged(self):
        """当预览方向改变时调用"""
        orientation_label = self.orientationCard.comboBox.currentText()
        orientation = ORIENTATION_LABEL_TO_VALUE.get(orientation_label, "横屏")
        preview_image = (
            DEFAULT_BG_LANDSCAPE if orientation == "横屏" else DEFAULT_BG_PORTRAIT
        )
        cfg.set(cfg.subtitle_preview_image, str(Path(preview_image["path"])))
        self._schedulePreviewUpdate()

    def onSettingChanged(self):
        """当任何设置改变时调用"""
        # 如果正在加载样式，不触发更新
        if self._loading_style:
            return

        self._update_word_timestamp_hint()

        self._schedulePreviewUpdate()
        # 获取当前选择的样式名称
        current_style = self.styleNameComboBox.comboBox.currentText()
        if current_style:
            self.saveStyle(current_style)  # 自动保存为当前选择的样式
        else:
            self.saveStyle("default")  # 如果没有选择样式,保存为默认样式

    def _on_split_type_changed(self):
        self._refresh_effect_options_by_split_mode()
        self._refresh_preview_text_options()
        self._update_word_timestamp_hint()
        self.onSettingChanged()

    def _is_word_mode(self) -> bool:
        return self.splitTypeCard.comboBox.currentText() == "По словам"

    def _get_allowed_effect_values(self) -> set[str]:
        return WORD_MODE_EFFECTS if self._is_word_mode() else SENTENCE_MODE_EFFECTS

    def _refresh_effect_options_by_split_mode(self):
        allowed = self._get_allowed_effect_values()
        current_label = self.effectCard.comboBox.currentText()

        filtered = [
            label
            for label, value in self.effect_options.items()
            if value in allowed
        ]
        if not filtered:
            filtered = ["Без эффекта"]

        self.effectCard.comboBox.blockSignals(True)
        self.effectCard.comboBox.clear()
        self.effectCard.comboBox.addItems(filtered)
        if current_label in filtered:
            self.effectCard.comboBox.setCurrentText(current_label)
        else:
            self.effectCard.comboBox.setCurrentIndex(0)
        self.effectCard.comboBox.blockSignals(False)

        # Применяем значение в cfg после фильтрации
        selected_label = self.effectCard.comboBox.currentText()
        cfg.set(cfg.subtitle_effect, self.effect_options.get(selected_label, "none"))
        self._update_motion_controls_state()

    def _current_preview_texts(self):
        split_label = self.splitTypeCard.comboBox.currentText()
        if split_label == "По словам":
            return PREVIEW_TEXTS_WORD
        return PREVIEW_TEXTS_SENTENCE

    def _resolve_preview_text_pair(self, base_pair: Tuple[str, Optional[str]]) -> Tuple[str, Optional[str]]:
        custom = (self.customPreviewTextInput.text() or "").strip()
        if not custom:
            return base_pair

        if "||" in custom:
            left, right = custom.split("||", 1)
            main = (left or "").strip()
            sub = (right or "").strip() or None
            return main, sub

        return custom, custom

    def _refresh_preview_text_options(self):
        texts_map = self._current_preview_texts()
        current = self.previewTextCard.comboBox.currentText()
        self.previewTextCard.comboBox.blockSignals(True)
        self.previewTextCard.comboBox.clear()
        self.previewTextCard.comboBox.addItems(list(texts_map.keys()))
        if current in texts_map:
            self.previewTextCard.comboBox.setCurrentText(current)
        else:
            self.previewTextCard.comboBox.setCurrentIndex(0)
        self.previewTextCard.comboBox.blockSignals(False)

    @staticmethod
    def _select_word_for_preview(text: Optional[str], timeline_ratio: float) -> Optional[str]:
        raw = (text or "").strip()
        if not raw:
            return text

        words = [w for w in raw.split() if w]
        if len(words) <= 1:
            return raw

        ratio = max(0.0, min(0.9999, float(timeline_ratio or 0.0)))
        idx = int(ratio * len(words))
        idx = max(0, min(len(words) - 1, idx))
        return words[idx]

    def _update_motion_controls_state(self):
        effect_label = self.effectCard.comboBox.currentText()
        effect_value = self.effect_options.get(effect_label, "none")
        is_word_mode = self._is_word_mode()
        enabled = is_word_mode and EffectManager.is_motion_customizable(effect_value)

        if is_word_mode and self.karaokeModeCard.comboBox.currentText() != "Выкл":
            self.karaokeModeCard.comboBox.setCurrentText("Выкл")

        self.karaokeModeCard.setEnabled(not is_word_mode)
        self.karaokeWindowCard.setEnabled(not is_word_mode)

        self.motionDirectionCard.setEnabled(enabled)
        self.motionAmplitudeCard.setEnabled(enabled)
        self.motionEasingCard.setEnabled(enabled)
        self.motionJitterCard.setEnabled(enabled)
        self.motionBlurCard.setEnabled(enabled)
        self._update_word_timestamp_hint()

    def _update_word_timestamp_hint(self):
        split_is_word = self.splitTypeCard.comboBox.currentText() == "По словам"
        effect_value = self.effect_options.get(self.effectCard.comboBox.currentText(), "none")
        effect_is_word = effect_value == "word_highlight"
        karaoke_on = TOGGLE_LABEL_TO_VALUE.get(
            self.karaokeModeCard.comboBox.currentText(), False
        )
        need_hint = split_is_word or effect_is_word or karaoke_on
        self.wordTimestampHintLabel.setVisible(need_hint)

    def toggleLivePreview(self):
        self._live_playing = not self._live_playing
        if self._live_playing:
            self.playPreviewButton.setText("⏸ Live")
            self._live_timer.start()
        else:
            self.playPreviewButton.setText("▶ Live")
            self._live_timer.stop()

    def _onLiveTimerTick(self):
        next_value = self.timelineSlider.value() + self.timelineSlider.singleStep()
        if next_value > self.timelineSlider.maximum():
            next_value = self.timelineSlider.minimum()
        self.timelineSlider.setValue(next_value)

    def onTimelineChanged(self, value: int):
        self.timelineLabel.setText(f"{value / 1000:.2f} c")
        self._schedulePreviewUpdate(immediate=self._live_playing)

    def _apply_live_duration_ms(self, duration_ms: int):
        duration = max(500, min(15000, int(duration_ms or 1000)))
        prev_max = max(1, self.timelineSlider.maximum())
        prev_ratio = self.timelineSlider.value() / prev_max

        self.timelineSlider.setRange(0, duration)
        self.timelineSlider.setValue(min(duration, int(prev_ratio * duration)))
        self.timelineLabel.setText(f"{self.timelineSlider.value() / 1000:.2f} c")

    def onLiveDurationChanged(self, value: int):
        cfg.set(cfg.subtitle_preview_live_duration_ms, int(value))
        self._apply_live_duration_ms(int(value))
        self.onSettingChanged()

    def onCustomPreviewTextChanged(self, text: str):
        cfg.set(cfg.subtitle_preview_custom_text, text or "")
        self.onSettingChanged()

    def onPresetChanged(self, preset_label: str):
        if self._loading_style:
            return

        preset = STYLE_PRESET_LABEL_TO_VALUE.get(preset_label, "custom")
        if preset == "custom":
            self.onSettingChanged()
            return

        self._loading_style = True
        try:
            # Сбрасываем «скрытые» цветовые модификаторы по умолчанию,
            # чтобы ручной цвет текста оставался управляемым после смены пресета.
            self.speakerColorModeCard.comboBox.setCurrentText("Выкл")

            if preset == "tiktok_dynamic":
                self.effectCard.comboBox.setCurrentText("Подсветка по словам")
                self.effectDurationCard.spinBox.setValue(450)
                self.effectIntensityCard.spinBox.setValue(140)
                self.motionDirectionCard.comboBox.setCurrentText("Снизу вверх")
                self.motionAmplitudeCard.spinBox.setValue(140)
                self.motionEasingCard.comboBox.setCurrentText("Ease Out (мягкий финиш)")
                self.motionJitterCard.spinBox.setValue(12)
                self.mainOutlineSizeCard.spinBox.setValue(2.8)
                self.mainShadowCard.spinBox.setValue(2.2)
            elif preset == "shorts_clean":
                self.effectCard.comboBox.setCurrentText("Плавное появление")
                self.effectDurationCard.spinBox.setValue(260)
                self.effectIntensityCard.spinBox.setValue(90)
                self.motionDirectionCard.comboBox.setCurrentText("Снизу вверх")
                self.motionAmplitudeCard.spinBox.setValue(80)
                self.motionEasingCard.comboBox.setCurrentText("Linear")
                self.motionJitterCard.spinBox.setValue(0)
                self.mainOutlineSizeCard.spinBox.setValue(2.0)
                self.mainShadowCard.spinBox.setValue(1.2)
            elif preset == "minimal_classic":
                self.effectCard.comboBox.setCurrentText("Без эффекта")
                self.effectDurationCard.spinBox.setValue(200)
                self.effectIntensityCard.spinBox.setValue(100)
                self.motionDirectionCard.comboBox.setCurrentText("Снизу вверх")
                self.motionAmplitudeCard.spinBox.setValue(60)
                self.motionEasingCard.comboBox.setCurrentText("Ease In-Out")
                self.motionJitterCard.spinBox.setValue(0)
                self.mainOutlineSizeCard.spinBox.setValue(1.6)
                self.mainShadowCard.spinBox.setValue(0.8)
                self.karaokeModeCard.comboBox.setCurrentText("Выкл")
                self.autoContrastCard.comboBox.setCurrentText("Выкл")
                self.antiFlickerCard.comboBox.setCurrentText("Вкл")
                self.gradientModeCard.comboBox.setCurrentText("Без градиента")
            elif preset == "karaoke_pro":
                self.effectCard.comboBox.setCurrentText("Подсветка по словам")
                self.effectDurationCard.spinBox.setValue(600)
                self.effectIntensityCard.spinBox.setValue(130)
                self.motionDirectionCard.comboBox.setCurrentText("Снизу вверх")
                self.motionAmplitudeCard.spinBox.setValue(110)
                self.motionEasingCard.comboBox.setCurrentText("Ease Out (мягкий финиш)")
                self.motionJitterCard.spinBox.setValue(6)
                self.karaokeModeCard.comboBox.setCurrentText("Вкл")
                self.karaokeWindowCard.spinBox.setValue(1400)
                self.autoContrastCard.comboBox.setCurrentText("Вкл")
                self.antiFlickerCard.comboBox.setCurrentText("Вкл")
                self.gradientModeCard.comboBox.setCurrentText("2 цвета")
                self.gradientColor1Card.setColor(QColor("#FFF4B2"))
                self.gradientColor2Card.setColor(QColor("#FF6FAE"))
            elif preset == "cinema_gradient":
                self.effectCard.comboBox.setCurrentText("Плавное появление")
                self.effectDurationCard.spinBox.setValue(380)
                self.effectIntensityCard.spinBox.setValue(105)
                self.motionDirectionCard.comboBox.setCurrentText("Снизу вверх")
                self.motionAmplitudeCard.spinBox.setValue(85)
                self.motionEasingCard.comboBox.setCurrentText("Ease In-Out")
                self.motionJitterCard.spinBox.setValue(0)
                self.karaokeModeCard.comboBox.setCurrentText("Выкл")
                self.autoContrastCard.comboBox.setCurrentText("Вкл")
                self.antiFlickerCard.comboBox.setCurrentText("Вкл")
                self.gradientModeCard.comboBox.setCurrentText("2 цвета")
                self.gradientColor1Card.setColor(QColor("#E8F3FF"))
                self.gradientColor2Card.setColor(QColor("#6AB8FF"))
            elif preset == "neon_pulse":
                self.effectCard.comboBox.setCurrentText("Неоновое мерцание")
                self.effectDurationCard.spinBox.setValue(520)
                self.effectIntensityCard.spinBox.setValue(150)
                self.motionDirectionCard.comboBox.setCurrentText("Снизу вверх")
                self.motionAmplitudeCard.spinBox.setValue(130)
                self.motionEasingCard.comboBox.setCurrentText("Ease Out (мягкий финиш)")
                self.motionJitterCard.spinBox.setValue(10)
                self.karaokeModeCard.comboBox.setCurrentText("Выкл")
                self.autoContrastCard.comboBox.setCurrentText("Вкл")
                self.antiFlickerCard.comboBox.setCurrentText("Вкл")
                self.gradientModeCard.comboBox.setCurrentText("Радужный")
                self.motionBlurCard.spinBox.setValue(4)
            elif preset == "gaming_glitch":
                self.effectCard.comboBox.setCurrentText("Глитч")
                self.effectDurationCard.spinBox.setValue(420)
                self.effectIntensityCard.spinBox.setValue(180)
                self.motionDirectionCard.comboBox.setCurrentText("Снизу вверх")
                self.motionAmplitudeCard.spinBox.setValue(120)
                self.motionEasingCard.comboBox.setCurrentText("Ease In-Out")
                self.motionJitterCard.spinBox.setValue(18)
                self.motionBlurCard.spinBox.setValue(6)
                self.autoContrastCard.comboBox.setCurrentText("Вкл")
                self.gradientModeCard.comboBox.setCurrentText("2 цвета")
                self.gradientColor1Card.setColor(QColor("#86F7FF"))
                self.gradientColor2Card.setColor(QColor("#FF6FD8"))
            elif preset == "podcast_focus":
                self.effectCard.comboBox.setCurrentText("Плавное появление")
                self.effectDurationCard.spinBox.setValue(240)
                self.effectIntensityCard.spinBox.setValue(90)
                self.motionJitterCard.spinBox.setValue(0)
                self.motionBlurCard.spinBox.setValue(0)
                self.karaokeModeCard.comboBox.setCurrentText("Выкл")
                self.autoContrastCard.comboBox.setCurrentText("Вкл")
                self.gradientModeCard.comboBox.setCurrentText("Без градиента")
                self.speakerColorModeCard.comboBox.setCurrentText("Чередовать по репликам")
            elif preset == "travel_vlog":
                self.effectCard.comboBox.setCurrentText("Скольжение снизу")
                self.effectDurationCard.spinBox.setValue(360)
                self.effectIntensityCard.spinBox.setValue(120)
                self.motionDirectionCard.comboBox.setCurrentText("Снизу вверх")
                self.motionAmplitudeCard.spinBox.setValue(125)
                self.motionEasingCard.comboBox.setCurrentText("Ease Out (мягкий финиш)")
                self.motionJitterCard.spinBox.setValue(5)
                self.motionBlurCard.spinBox.setValue(3)
                self.gradientModeCard.comboBox.setCurrentText("2 цвета")
                self.gradientColor1Card.setColor(QColor("#FFF0B3"))
                self.gradientColor2Card.setColor(QColor("#6CD4FF"))
            elif preset == "dramatic_trailer":
                self.effectCard.comboBox.setCurrentText("Поп + поворот")
                self.effectDurationCard.spinBox.setValue(500)
                self.effectIntensityCard.spinBox.setValue(160)
                self.motionDirectionCard.comboBox.setCurrentText("Снизу вверх")
                self.motionAmplitudeCard.spinBox.setValue(140)
                self.motionEasingCard.comboBox.setCurrentText("Ease In-Out")
                self.motionJitterCard.spinBox.setValue(8)
                self.motionBlurCard.spinBox.setValue(5)
                self.autoContrastCard.comboBox.setCurrentText("Вкл")
                self.gradientModeCard.comboBox.setCurrentText("2 цвета")
                self.gradientColor1Card.setColor(QColor("#FFFFFF"))
                self.gradientColor2Card.setColor(QColor("#FF9A7A"))
        finally:
            self._loading_style = False

        self._schedulePreviewUpdate()
        current_style = self.styleNameComboBox.comboBox.currentText() or "default"
        self.saveStyle(current_style)

    def selectPreviewImage(self):
        """选择预览背景图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите фоновое изображение",
            "",
            "Файлы изображений (*.png *.jpg *.jpeg)",
        )
        if file_path:
            cfg.set(cfg.subtitle_preview_image, file_path)
            self._schedulePreviewUpdate()

    def _schedulePreviewUpdate(self, immediate: bool = False):
        self._preview_pending = True
        if immediate or self._live_playing:
            if self._preview_thread and self._preview_thread.isRunning():
                return
            self._renderPreviewNow()
            return
        self._preview_debounce_timer.start()

    def _renderPreviewNow(self):
        if not self._preview_pending:
            return

        if self._preview_thread and self._preview_thread.isRunning():
            return

        self._preview_pending = False
        self._preview_request_id += 1
        request_id = self._preview_request_id

        # 生成 ASS 样式字符串
        style_str = self.generateAssStyles()

        # 获取预览文本
        preview_texts = self._current_preview_texts()
        main_text, sub_text = preview_texts[self.previewTextCard.comboBox.currentText()]
        main_text, sub_text = self._resolve_preview_text_pair((main_text, sub_text))

        # 字幕布局
        layout = self.layoutCard.comboBox.currentText()
        layout_value = LAYOUT_LABEL_TO_VALUE.get(layout, "译文在上")
        if layout_value == "译文在上":
            main_text, sub_text = sub_text, main_text
        elif layout_value == "原文在上":
            main_text, sub_text = main_text, sub_text
        elif layout_value == "仅译文":
            main_text, sub_text = sub_text, None
        elif layout_value == "仅原文":
            main_text, sub_text = main_text, None

        timeline_ms = self.timelineSlider.value()
        timeline_ratio = timeline_ms / max(1, self.timelineSlider.maximum())
        if self._is_word_mode():
            main_text = self._select_word_for_preview(main_text, timeline_ratio)
            if sub_text:
                sub_text = self._select_word_for_preview(sub_text, timeline_ratio)

        # 获取预览方向
        orientation_label = self.orientationCard.comboBox.currentText()
        orientation = ORIENTATION_LABEL_TO_VALUE.get(orientation_label, "横屏")
        default_preview = (
            DEFAULT_BG_LANDSCAPE if orientation == "横屏" else DEFAULT_BG_PORTRAIT
        )

        # 检查是否存在用户自定义背景图片
        user_bg_path = cfg.get(cfg.subtitle_preview_image)
        if user_bg_path and Path(user_bg_path).exists():
            path = user_bg_path
            width = default_preview["width"]
            height = default_preview["height"]
        else:
            path = default_preview["path"]
            width = default_preview["width"]
            height = default_preview["height"]

        effect_type = cfg.get(cfg.subtitle_effect)
        if effect_type != "none":
            preview_window_ms = self.timelineSlider.maximum()
            effect_ms = min(
                max(50, int(cfg.get(cfg.subtitle_effect_duration))),
                preview_window_ms,
            )
            preview_time_sec = (
                (timeline_ms / max(1, self.timelineSlider.maximum()))
                * (effect_ms / 1000.0)
            )
        else:
            preview_time_sec = timeline_ms / 1000.0

        self._preview_thread = PreviewThread(
            style_str=style_str,
            preview_text=(main_text, sub_text),
            bg_path=path,
            width=width,
            height=height,
            effect_type=effect_type,
            effect_duration_ms=cfg.get(cfg.subtitle_effect_duration),
            effect_intensity=cfg.get(cfg.subtitle_effect_intensity) / 100,
            rainbow_end_color=cfg.get(cfg.subtitle_rainbow_end_color),
            motion_direction=cfg.get(cfg.subtitle_motion_direction),
            motion_amplitude=cfg.get(cfg.subtitle_motion_amplitude) / 100,
            motion_easing=cfg.get(cfg.subtitle_motion_easing),
            motion_jitter=cfg.get(cfg.subtitle_motion_jitter) / 100,
            motion_blur_strength=cfg.get(cfg.subtitle_motion_blur_strength),
            karaoke_mode=(not self._is_word_mode()) and bool(cfg.get(cfg.subtitle_karaoke_mode)),
            karaoke_window_ms=cfg.get(cfg.subtitle_karaoke_window_ms),
            auto_contrast=cfg.get(cfg.subtitle_auto_contrast),
            anti_flicker=cfg.get(cfg.subtitle_anti_flicker),
            gradient_mode=cfg.get(cfg.subtitle_gradient_mode),
            gradient_color_1=cfg.get(cfg.subtitle_gradient_color_1),
            gradient_color_2=cfg.get(cfg.subtitle_gradient_color_2),
            safe_area_enabled=cfg.get(cfg.subtitle_safe_area_enabled),
            safe_margin_x=cfg.get(cfg.subtitle_safe_margin_x),
            safe_margin_y=cfg.get(cfg.subtitle_safe_margin_y),
            speaker_color_mode=cfg.get(cfg.subtitle_speaker_color_mode),
            preview_time_sec=preview_time_sec,
            request_id=request_id,
        )
        self._preview_thread.previewReady.connect(self.onPreviewReady)
        self._preview_thread.finished.connect(self._onPreviewThreadFinished)
        self._preview_thread.start()

    def _onPreviewThreadFinished(self):
        if self._preview_pending:
            self._renderPreviewNow()

    def generateAssStyles(self) -> str:
        """生成 ASS 样式字符串"""
        style_format = "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding"

        def _qcolor_to_ass(color: QColor, with_alpha: bool = True) -> str:
            r, g, b, a = color.getRgb()
            # Qt хранит alpha как непрозрачность (0=прозрачный, 255=непрозрачный),
            # а ASS хранит alpha как прозрачность (00=непрозрачный, FF=прозрачный)
            ass_alpha = 255 - a if with_alpha else 0
            aa = f"{ass_alpha:02X}"
            return f"&H{aa}{b:02X}{g:02X}{r:02X}"

        # 从控件获取当前设置
        # 获取垂直间距
        vertical_spacing = int(
            self.verticalSpacingCard.spinBox.value()
        )  # 转换为ASS单位

        # 提取主字幕样式元素
        main_font = self.mainFontCard.comboBox.currentText()
        main_size = self.mainSizeCard.spinBox.value()

        # 获取颜色值并转换为 ASS 格式 (AABBGGRR)
        main_color_hex = self.mainColorCard.colorPicker.color.name()
        main_outline_hex = self.mainOutlineColorCard.colorPicker.color.name()
        main_color = (
            f"&H00{main_color_hex[5:7]}{main_color_hex[3:5]}{main_color_hex[1:3]}"
        )
        main_outline_color = (
            f"&H00{main_outline_hex[5:7]}{main_outline_hex[3:5]}{main_outline_hex[1:3]}"
        )
        main_spacing = self.mainSpacingCard.spinBox.value()
        main_outline_size = self.mainOutlineSizeCard.spinBox.value()
        main_shadow = self.mainShadowCard.spinBox.value()
        main_shadow_color = _qcolor_to_ass(self.mainShadowColorCard.colorPicker.color)
        main_blur = self.mainBlurCard.spinBox.value()

        # 提取副字幕样式元素
        sub_font = self.subFontCard.comboBox.currentText()
        sub_size = self.subSizeCard.spinBox.value()

        # 获取颜色值并转换为 ASS 格式 (AABBGGRR)
        sub_color_hex = self.subColorCard.colorPicker.color.name()
        sub_outline_hex = self.subOutlineColorCard.colorPicker.color.name()
        sub_color = f"&H00{sub_color_hex[5:7]}{sub_color_hex[3:5]}{sub_color_hex[1:3]}"
        sub_outline_color = (
            f"&H00{sub_outline_hex[5:7]}{sub_outline_hex[3:5]}{sub_outline_hex[1:3]}"
        )
        sub_spacing = self.subSpacingCard.spinBox.value()
        sub_outline_size = self.subOutlineSizeCard.spinBox.value()
        sub_shadow = self.subShadowCard.spinBox.value()
        sub_shadow_color = _qcolor_to_ass(self.subShadowColorCard.colorPicker.color)
        sub_blur = self.subBlurCard.spinBox.value()

        # 生成样式字符串
        # В ASS нет отдельного поля blur в строке Style, поэтому храним его
        # как комментарии вида ;VC_BLUR:StyleName=value и применяем в Dialogue.
        main_style = f"Style: Default,{main_font},{main_size},{main_color},&H000000FF,{main_outline_color},{main_shadow_color},-1,0,0,0,100,100,{main_spacing},0,1,{main_outline_size},{main_shadow},2,10,10,{vertical_spacing},1"
        sub_style = f"Style: Secondary,{sub_font},{sub_size},{sub_color},&H000000FF,{sub_outline_color},{sub_shadow_color},-1,0,0,0,100,100,{sub_spacing},0,1,{sub_outline_size},{sub_shadow},2,10,10,{vertical_spacing},1"

        extras = [
            f";VC_BLUR:Default={main_blur}",
            f";VC_BLUR:Secondary={sub_blur}",
        ]

        return (
            f"[V4+ Styles]\n{style_format}\n{main_style}\n{sub_style}\n"
            + "\n".join(extras)
        )

    def updatePreview(self):
        """请求更新预览（带防抖和串行渲染）"""
        self._schedulePreviewUpdate()

    def showEffectPreview(self):
        """Сгенерировать короткое видео предпросмотра эффекта и открыть его."""
        style_str = self.generateAssStyles()
        preview_texts = self._current_preview_texts()
        main_text, sub_text = preview_texts[self.previewTextCard.comboBox.currentText()]
        main_text, sub_text = self._resolve_preview_text_pair((main_text, sub_text))

        layout_value = LAYOUT_LABEL_TO_VALUE.get(
            self.layoutCard.comboBox.currentText(), "译文在上"
        )
        if layout_value == "译文在上":
            main_text, sub_text = sub_text, main_text
        elif layout_value == "原文在上":
            pass
        elif layout_value == "仅译文":
            main_text, sub_text = sub_text, None
        elif layout_value == "仅原文":
            sub_text = None

        orientation = ORIENTATION_LABEL_TO_VALUE.get(
            self.orientationCard.comboBox.currentText(), "横屏"
        )
        default_preview = DEFAULT_BG_LANDSCAPE if orientation == "横屏" else DEFAULT_BG_PORTRAIT
        user_bg_path = cfg.get(cfg.subtitle_preview_image)
        path = user_bg_path if user_bg_path and Path(user_bg_path).exists() else default_preview["path"]

        video_path = generate_preview_video(
            style_str=style_str,
            preview_text=(main_text, sub_text),
            bg_path=str(path),
            width=default_preview["width"],
            height=default_preview["height"],
            effect_type=cfg.get(cfg.subtitle_effect),
            effect_duration_ms=cfg.get(cfg.subtitle_effect_duration),
            effect_intensity=cfg.get(cfg.subtitle_effect_intensity) / 100,
            rainbow_end_color=cfg.get(cfg.subtitle_rainbow_end_color),
            motion_direction=cfg.get(cfg.subtitle_motion_direction),
            motion_amplitude=cfg.get(cfg.subtitle_motion_amplitude) / 100,
            motion_easing=cfg.get(cfg.subtitle_motion_easing),
            motion_jitter=cfg.get(cfg.subtitle_motion_jitter) / 100,
            motion_blur_strength=cfg.get(cfg.subtitle_motion_blur_strength),
            karaoke_mode=(not self._is_word_mode()) and bool(cfg.get(cfg.subtitle_karaoke_mode)),
            karaoke_window_ms=cfg.get(cfg.subtitle_karaoke_window_ms),
            auto_contrast=cfg.get(cfg.subtitle_auto_contrast),
            anti_flicker=cfg.get(cfg.subtitle_anti_flicker),
            gradient_mode=cfg.get(cfg.subtitle_gradient_mode),
            gradient_color_1=cfg.get(cfg.subtitle_gradient_color_1),
            gradient_color_2=cfg.get(cfg.subtitle_gradient_color_2),
            safe_area_enabled=cfg.get(cfg.subtitle_safe_area_enabled),
            safe_margin_x=cfg.get(cfg.subtitle_safe_margin_x),
            safe_margin_y=cfg.get(cfg.subtitle_safe_margin_y),
            speaker_color_mode=cfg.get(cfg.subtitle_speaker_color_mode),
            duration_sec=max(1.0, self.timelineSlider.maximum() / 1000.0),
            word_by_word_preview=self._is_word_mode(),
        )

        if sys.platform == "win32":
            os.startfile(video_path)
        elif sys.platform == "darwin":
            subprocess.run(["open", video_path])
        else:
            subprocess.run(["xdg-open", video_path])

    def onPreviewReady(self, preview_path, request_id: int):
        """预览图片生成完成的回调"""
        if request_id < self._latest_applied_request_id:
            return
        self._latest_applied_request_id = request_id
        self.previewImage.setImage(preview_path)
        self.updatePreviewImage()

    def updatePreviewImage(self):
        """更新预览图片"""
        height = int(self.previewTopWidget.height() * 0.98)
        width = int(self.previewTopWidget.width() * 0.98)
        self.previewImage.scaledToWidth(width)
        if self.previewImage.height() > height:
            self.previewImage.scaledToHeight(height)
        self.previewImage.setBorderRadius(8, 8, 8, 8)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.updatePreviewImage()

    def showEvent(self, event):
        """窗口显示事件"""
        super().showEvent(event)
        self.updatePreviewImage()

    def closeEvent(self, event):
        self._live_timer.stop()
        self._preview_debounce_timer.stop()
        if self._preview_thread and self._preview_thread.isRunning():
            self._preview_thread.quit()
            if not self._preview_thread.wait(500):
                self._preview_thread.terminate()
                self._preview_thread.wait(200)
        super().closeEvent(event)

    def loadStyle(self, style_name):
        """加载指定样式"""
        style_path = SUBTITLE_STYLE_PATH / f"{style_name}.txt"

        if not style_path.exists():
            return

        # 设置标志位，防止触发onSettingChanged
        self._loading_style = True

        with open(style_path, "r", encoding="utf-8") as f:
            style_content = f.read()

        # 解析样式内容
        for line in style_content.split("\n"):
            if line.startswith("Style: Default"):
                # 解析主字幕样式
                parts = line.split(",")
                self.mainFontCard.setCurrentText(parts[1])
                self.mainSizeCard.spinBox.setValue(int(parts[2]))

                vertical_spacing = int(parts[21])
                self.verticalSpacingCard.spinBox.setValue(vertical_spacing)

                # 将 &HAARRGGBB 格式转换为 QColor
                primary_color = parts[3].strip()
                if primary_color.startswith("&H"):
                    # 移除 &H 前缀,转换为 RGB
                    color_hex = primary_color[2:]
                    alpha = 255 - int(color_hex[0:2], 16)
                    blue = int(color_hex[2:4], 16)
                    green = int(color_hex[4:6], 16)
                    red = int(color_hex[6:8], 16)
                    self.mainColorCard.setColor(QColor(red, green, blue, alpha))

                outline_color = parts[5].strip()
                if outline_color.startswith("&H"):
                    color_hex = outline_color[2:]
                    alpha = 255 - int(color_hex[0:2], 16)
                    blue = int(color_hex[2:4], 16)
                    green = int(color_hex[4:6], 16)
                    red = int(color_hex[6:8], 16)
                    self.mainOutlineColorCard.setColor(QColor(red, green, blue, alpha))

                shadow_color = parts[6].strip()
                if shadow_color.startswith("&H") and len(shadow_color) >= 10:
                    color_hex = shadow_color[2:]
                    alpha = 255 - int(color_hex[0:2], 16)
                    blue = int(color_hex[2:4], 16)
                    green = int(color_hex[4:6], 16)
                    red = int(color_hex[6:8], 16)
                    self.mainShadowColorCard.setColor(QColor(red, green, blue, alpha))

                self.mainSpacingCard.spinBox.setValue(float(parts[13]))
                self.mainOutlineSizeCard.spinBox.setValue(float(parts[16]))
                self.mainShadowCard.spinBox.setValue(float(parts[17]))
                blur_match = re.search(r";VC_BLUR:Default=([0-9.]+)", style_content)
                if not blur_match:
                    blur_match = re.search(r"#EXTRAS:blur=([0-9.]+)", style_content)
                self.mainBlurCard.spinBox.setValue(
                    float(blur_match.group(1)) if blur_match else 0.0
                )
            elif line.startswith("Style: Secondary"):
                # 解析副字幕样式
                parts = line.split(",")
                self.subFontCard.setCurrentText(parts[1])
                self.subSizeCard.spinBox.setValue(int(parts[2]))
                # 将 &HAARRGGBB 格式转换为 QColor
                primary_color = parts[3].strip()
                if primary_color.startswith("&H"):
                    color_hex = primary_color[2:]
                    alpha = 255 - int(color_hex[0:2], 16)
                    blue = int(color_hex[2:4], 16)
                    green = int(color_hex[4:6], 16)
                    red = int(color_hex[6:8], 16)
                    self.subColorCard.setColor(QColor(red, green, blue, alpha))

                outline_color = parts[5].strip()
                if outline_color.startswith("&H"):
                    color_hex = outline_color[2:]
                    alpha = 255 - int(color_hex[0:2], 16)
                    blue = int(color_hex[2:4], 16)
                    green = int(color_hex[4:6], 16)
                    red = int(color_hex[6:8], 16)
                    self.subOutlineColorCard.setColor(QColor(red, green, blue, alpha))

                shadow_color = parts[6].strip()
                if shadow_color.startswith("&H") and len(shadow_color) >= 10:
                    color_hex = shadow_color[2:]
                    alpha = 255 - int(color_hex[0:2], 16)
                    blue = int(color_hex[2:4], 16)
                    green = int(color_hex[4:6], 16)
                    red = int(color_hex[6:8], 16)
                    self.subShadowColorCard.setColor(QColor(red, green, blue, alpha))

                self.subSpacingCard.spinBox.setValue(float(parts[13]))
                self.subOutlineSizeCard.spinBox.setValue(float(parts[16]))
                self.subShadowCard.spinBox.setValue(float(parts[17]))
                blur_match = re.search(r";VC_BLUR:Secondary=([0-9.]+)", style_content)
                if not blur_match:
                    blur_match = re.search(r"#EXTRAS:blur=([0-9.]+)", style_content)
                self.subBlurCard.spinBox.setValue(
                    float(blur_match.group(1)) if blur_match else 0.0
                )

        cfg.set(cfg.subtitle_style_name, style_name)

        # 重置标志位
        self._loading_style = False

        # 手动更新一次预览
        self.updatePreview()

        # 显示加载成功提示
        InfoBar.success(
            title="Успешно",
            content="Загружен стиль: " + style_name,
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=1500,
            parent=self,
        )

    def createNewStyle(self):
        """创建新样式"""
        dialog = StyleNameDialog(self)
        if dialog.exec():
            style_name = dialog.nameLineEdit.text().strip()
            if not style_name:
                return

            # 检查是否已存在同名样式
            if (SUBTITLE_STYLE_PATH / f"{style_name}.txt").exists():
                InfoBar.warning(
                    title="Внимание",
                    content="Стиль уже существует: " + style_name,
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.TOP,
                    duration=2000,
                    parent=self,
                )
                return

            # 保存新样式
            self.saveStyle(style_name)

            # 更新样式列表并选中新样式
            self.styleNameComboBox.addItem(style_name)
            self.styleNameComboBox.comboBox.setCurrentText(style_name)

            # 显示创建成功提示
            InfoBar.success(
                title="Успешно",
                content="Создан новый стиль: " + style_name,
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self,
            )

    def saveStyle(self, style_name):
        """保存样式
        Args:
            style_name (str): 样式名称
        """
        # 确保样式目录存在
        SUBTITLE_STYLE_PATH.mkdir(parents=True, exist_ok=True)

        # 生成样式内容并保存
        style_content = self.generateAssStyles()
        style_path = SUBTITLE_STYLE_PATH / f"{style_name}.txt"

        with open(style_path, "w", encoding="utf-8") as f:
            f.write(style_content)


class StyleNameDialog(MessageBoxBase):
    """样式名称输入对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.titleLabel = BodyLabel("Новый стиль", self)
        self.nameLineEdit = LineEdit(self)

        self.nameLineEdit.setPlaceholderText("Введите название стиля")
        self.nameLineEdit.setClearButtonEnabled(True)

        # 添加控件到布局
        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.nameLineEdit)

        # 设置按钮文本
        self.yesButton.setText("ОК")
        self.cancelButton.setText("Отмена")

        self.widget.setMinimumWidth(350)
        self.yesButton.setDisabled(True)
        self.nameLineEdit.textChanged.connect(self._validateInput)

    def _validateInput(self, text):
        self.yesButton.setEnabled(bool(text.strip()))


class PresetNameDialog(MessageBoxBase):
    """Диалог имени пользовательского пресета"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.titleLabel = BodyLabel("Новый кастомный пресет", self)
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

    def _validateInput(self, text):
        self.yesButton.setEnabled(bool(text.strip()))
