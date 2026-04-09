from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget
from qfluentwidgets import BodyLabel
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import MessageBoxBase, SwitchSettingCard, ComboBoxSettingCard

from app.common.config import cfg
from app.components.SpinBoxSettingCard import SpinBoxSettingCard
from app.core.entities import SplitTypeEnum


class SubtitleSettingDialog(MessageBoxBase):
    """字幕设置对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.titleLabel = BodyLabel("Параметры субтитров", self)

        # 创建设置卡片
        self.split_card = SwitchSettingCard(
            FIF.ALIGNMENT,
            "Разделение субтитров",
            "Использовать LLM для умного разбиения субтитров",
            cfg.need_split,
            self,
        )

        self.split_type_card = ComboBoxSettingCard(
            cfg.split_type,
            FIF.TILES,
            "Тип разбиения",
            "Разбивать по смыслу или по предложениям",
            texts=["По смыслу", "По предложениям"],
            parent=self,
        )

        self.word_count_cjk_card = SpinBoxSettingCard(
            cfg.max_word_count_cjk,
            FIF.TILES,
            "Макс. символов CJK",
            "Максимум символов в одной строке (китайский/японский/корейский)",
            minimum=8,
            maximum=50,
            parent=self,
        )

        self.word_count_english_card = SpinBoxSettingCard(
            cfg.max_word_count_english,
            FIF.TILES,
            "Макс. слов (английский)",
            "Максимум слов в одной строке для английского",
            minimum=8,
            maximum=50,
            parent=self,
        )

        self.remove_punctuation_card = SwitchSettingCard(
            FIF.ALIGNMENT,
            "Убирать конечную пунктуацию",
            "Удалять знаки препинания в конце строк",
            cfg.needs_remove_punctuation,
            self,
        )

        # 添加到布局
        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.split_card)
        self.viewLayout.addWidget(self.split_type_card)
        self.viewLayout.addWidget(self.word_count_cjk_card)
        self.viewLayout.addWidget(self.word_count_english_card)
        self.viewLayout.addWidget(self.remove_punctuation_card)
        # 设置间距

        self.viewLayout.setSpacing(10)

        # 设置窗口标题
        self.setWindowTitle("Параметры субтитров")

        # 只显示取消按钮
        self.yesButton.hide()
        self.cancelButton.setText("Закрыть")
