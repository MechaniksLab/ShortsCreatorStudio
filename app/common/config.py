# coding:utf-8
from enum import Enum

from PyQt5.QtCore import QLocale
from PyQt5.QtGui import QColor
import openai
from qfluentwidgets import (
    qconfig,
    QConfig,
    ConfigItem,
    OptionsConfigItem,
    BoolValidator,
    OptionsValidator,
    RangeConfigItem,
    RangeValidator,
    Theme,
    FolderValidator,
    ConfigSerializer,
    EnumSerializer,
)

from app.config import WORK_PATH, SETTINGS_PATH
from ..core.entities import (
    LLMServiceEnum,
    SplitTypeEnum,
    TargetLanguageEnum,
    TranscribeModelEnum,
    TranscribeLanguageEnum,
    TranslatorServiceEnum,
    WhisperModelEnum,
    FasterWhisperModelEnum,
    VadMethodEnum,
)


class Language(Enum):
    """软件语言"""

    CHINESE_SIMPLIFIED = QLocale(QLocale.Chinese, QLocale.China)
    CHINESE_TRADITIONAL = QLocale(QLocale.Chinese, QLocale.HongKong)
    ENGLISH = QLocale(QLocale.English)
    AUTO = QLocale()


class SubtitleLayoutEnum(Enum):
    """字幕布局"""

    TRANSLATE_ON_TOP = "译文在上"
    ORIGINAL_ON_TOP = "原文在上"
    ONLY_ORIGINAL = "仅原文"
    ONLY_TRANSLATE = "仅译文"


class LanguageSerializer(ConfigSerializer):
    """Language serializer"""

    def serialize(self, language):
        return language.value.name() if language != Language.AUTO else "Auto"

    def deserialize(self, value: str):
        return Language(QLocale(value)) if value != "Auto" else Language.AUTO


class Config(QConfig):
    """应用配置"""

    # LLM配置
    llm_service = OptionsConfigItem(
        "LLM",
        "LLMService",
        LLMServiceEnum.PUBLIC,
        OptionsValidator(LLMServiceEnum),
        EnumSerializer(LLMServiceEnum),
    )

    openai_model = ConfigItem("LLM", "OpenAI_Model", "gpt-4o-mini")
    openai_api_key = ConfigItem("LLM", "OpenAI_API_Key", "")
    openai_api_base = ConfigItem("LLM", "OpenAI_API_Base", "https://api.openai.com/v1")

    silicon_cloud_model = ConfigItem("LLM", "SiliconCloud_Model", "gpt-4o-mini")
    silicon_cloud_api_key = ConfigItem("LLM", "SiliconCloud_API_Key", "")
    silicon_cloud_api_base = ConfigItem(
        "LLM", "SiliconCloud_API_Base", "https://api.siliconflow.cn/v1"
    )

    deepseek_model = ConfigItem("LLM", "DeepSeek_Model", "deepseek-chat")
    deepseek_api_key = ConfigItem("LLM", "DeepSeek_API_Key", "")
    deepseek_api_base = ConfigItem(
        "LLM", "DeepSeek_API_Base", "https://api.deepseek.com/v1"
    )

    ollama_model = ConfigItem("LLM", "Ollama_Model", "llama2")
    ollama_api_key = ConfigItem("LLM", "Ollama_API_Key", "ollama")
    ollama_api_base = ConfigItem("LLM", "Ollama_API_Base", "http://localhost:11434/v1")

    lm_studio_model = ConfigItem("LLM", "LmStudio_Model", "qwen2.5:7b")
    lm_studio_api_key = ConfigItem("LLM", "LmStudio_API_Key", "lmstudio")
    lm_studio_api_base = ConfigItem(
        "LLM", "LmStudio_API_Base", "http://localhost:1234/v1"
    )

    gemini_model = ConfigItem("LLM", "Gemini_Model", "gemini-pro")
    gemini_api_key = ConfigItem("LLM", "Gemini_API_Key", "")
    gemini_api_base = ConfigItem(
        "LLM",
        "Gemini_API_Base",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    chatglm_model = ConfigItem("LLM", "ChatGLM_Model", "glm-4")
    chatglm_api_key = ConfigItem("LLM", "ChatGLM_API_Key", "")
    chatglm_api_base = ConfigItem(
        "LLM", "ChatGLM_API_Base", "https://open.bigmodel.cn/api/paas/v4"
    )

    # 公益模型
    public_model = ConfigItem("LLM", "Public_Model", "gpt-4o-mini")
    public_api_key = ConfigItem(
        "LLM", "Public_API_Key", "please-do-not-use-for-personal-purposes"
    )
    public_api_base = ConfigItem("LLM", "Public_API_Base", "https://ddg.bkfeng.top/v1")

    # ------------------- 翻译配置 -------------------
    translator_service = OptionsConfigItem(
        "Translate",
        "TranslatorServiceEnum",
        TranslatorServiceEnum.BING,
        OptionsValidator(TranslatorServiceEnum),
        EnumSerializer(TranslatorServiceEnum),
    )
    need_reflect_translate = ConfigItem(
        "Translate", "NeedReflectTranslate", False, BoolValidator()
    )
    deeplx_endpoint = ConfigItem("Translate", "DeeplxEndpoint", "")
    batch_size = RangeConfigItem("Translate", "BatchSize", 10, RangeValidator(5, 30))
    thread_num = RangeConfigItem("Translate", "ThreadNum", 10, RangeValidator(1, 100))

    # ------------------- 转录配置 -------------------
    transcribe_model = OptionsConfigItem(
        "Transcribe",
        "TranscribeModel",
        TranscribeModelEnum.BIJIAN,
        OptionsValidator(TranscribeModelEnum),
        EnumSerializer(TranscribeModelEnum),
    )
    use_asr_cache = ConfigItem("Transcribe", "UseASRCache", True, BoolValidator())
    transcribe_language = OptionsConfigItem(
        "Transcribe",
        "TranscribeLanguage",
        TranscribeLanguageEnum.ENGLISH,
        OptionsValidator(TranscribeLanguageEnum),
        EnumSerializer(TranscribeLanguageEnum),
    )

    # ------------------- Whisper Cpp 配置 -------------------
    whisper_model = OptionsConfigItem(
        "Whisper",
        "WhisperModel",
        WhisperModelEnum.TINY,
        OptionsValidator(WhisperModelEnum),
        EnumSerializer(WhisperModelEnum),
    )

    # ------------------- Faster Whisper 配置 -------------------
    faster_whisper_program = ConfigItem(
        "FasterWhisper",
        "Program",
        "faster-whisper-xxl.exe",
    )
    faster_whisper_model = OptionsConfigItem(
        "FasterWhisper",
        "Model",
        FasterWhisperModelEnum.TINY,
        OptionsValidator(FasterWhisperModelEnum),
        EnumSerializer(FasterWhisperModelEnum),
    )
    faster_whisper_model_dir = ConfigItem("FasterWhisper", "ModelDir", "")
    faster_whisper_device = OptionsConfigItem(
        "FasterWhisper", "Device", "cuda", OptionsValidator(["cuda", "cpu"])
    )
    # VAD 参数
    faster_whisper_vad_filter = ConfigItem(
        "FasterWhisper", "VadFilter", True, BoolValidator()
    )
    faster_whisper_vad_threshold = RangeConfigItem(
        "FasterWhisper", "VadThreshold", 0.4, RangeValidator(0, 1)
    )
    faster_whisper_vad_method = OptionsConfigItem(
        "FasterWhisper",
        "VadMethod",
        VadMethodEnum.SILERO_V4,
        OptionsValidator(VadMethodEnum),
        EnumSerializer(VadMethodEnum),
    )
    # 人声提取
    faster_whisper_ff_mdx_kim2 = ConfigItem(
        "FasterWhisper", "FfMdxKim2", False, BoolValidator()
    )
    # 文本处理参数
    faster_whisper_one_word = ConfigItem(
        "FasterWhisper", "OneWord", True, BoolValidator()
    )
    # 提示词
    faster_whisper_prompt = ConfigItem("FasterWhisper", "Prompt", "")

    # ------------------- Whisper API 配置 -------------------
    whisper_api_base = ConfigItem("WhisperAPI", "WhisperApiBase", "")
    whisper_api_key = ConfigItem("WhisperAPI", "WhisperApiKey", "")
    whisper_api_model = OptionsConfigItem("WhisperAPI", "WhisperApiModel", "")
    whisper_api_prompt = ConfigItem("WhisperAPI", "WhisperApiPrompt", "")

    # ------------------- 字幕配置 -------------------
    need_optimize = ConfigItem("Subtitle", "NeedOptimize", False, BoolValidator())
    need_translate = ConfigItem("Subtitle", "NeedTranslate", False, BoolValidator())
    use_subtitle_cache = ConfigItem("Subtitle", "UseSubtitleCache", True, BoolValidator())
    use_processed_subtitle_cache = ConfigItem(
        "Subtitle", "UseProcessedSubtitleCache", True, BoolValidator()
    )
    need_split = ConfigItem("Subtitle", "NeedSplit", False, BoolValidator())
    split_type = OptionsConfigItem(
        "Subtitle",
        "SplitType",
        SplitTypeEnum.SENTENCE,
        OptionsValidator(SplitTypeEnum),
        EnumSerializer(SplitTypeEnum),
    )
    target_language = OptionsConfigItem(
        "Subtitle",
        "TargetLanguage",
        TargetLanguageEnum.CHINESE_SIMPLIFIED,
        OptionsValidator(TargetLanguageEnum),
        EnumSerializer(TargetLanguageEnum),
    )
    max_word_count_cjk = ConfigItem(
        "Subtitle", "MaxWordCountCJK", 25, RangeValidator(8, 100)
    )
    max_word_count_english = ConfigItem(
        "Subtitle", "MaxWordCountEnglish", 20, RangeValidator(8, 100)
    )
    needs_remove_punctuation = ConfigItem(
        "Subtitle", "NeedsRemovePunctuation", True, BoolValidator()
    )
    custom_prompt_text = ConfigItem("Subtitle", "CustomPromptText", "")

    # ------------------- 字幕合成配置 -------------------
    soft_subtitle = ConfigItem("Video", "SoftSubtitle", False, BoolValidator())
    need_video = ConfigItem("Video", "NeedVideo", True, BoolValidator())

    # ------------------- 字幕样式配置 -------------------
    subtitle_style_name = ConfigItem("SubtitleStyle", "StyleName", "default")
    subtitle_layout = ConfigItem("SubtitleStyle", "Layout", "译文在上")
    subtitle_preview_image = ConfigItem("SubtitleStyle", "PreviewImage", "")
    subtitle_preview_custom_text = ConfigItem("SubtitleStyle", "PreviewCustomText", "")
    subtitle_preview_live_duration_ms = RangeConfigItem(
        "SubtitleStyle", "PreviewLiveDurationMs", 1000, RangeValidator(500, 15000)
    )
    subtitle_effect = ConfigItem("SubtitleStyle", "Effect", "none")
    subtitle_effect_duration = RangeConfigItem(
        "SubtitleStyle", "EffectDuration", 300, RangeValidator(20, 10000)
    )
    subtitle_effect_intensity = RangeConfigItem(
        "SubtitleStyle", "EffectIntensity", 100, RangeValidator(1, 500)
    )
    subtitle_rainbow_end_color = ConfigItem(
        "SubtitleStyle", "RainbowEndColor", "#0000FF"
    )
    subtitle_style_preset = OptionsConfigItem(
        "SubtitleStyle",
        "StylePreset",
        "custom",
        OptionsValidator([
            "custom",
            "tiktok_dynamic",
            "shorts_clean",
            "minimal_classic",
            "karaoke_pro",
            "cinema_gradient",
            "neon_pulse",
            "gaming_glitch",
            "podcast_focus",
            "travel_vlog",
            "dramatic_trailer",
        ]),
    )
    subtitle_motion_direction = OptionsConfigItem(
        "SubtitleStyle",
        "MotionDirection",
        "up",
        OptionsValidator(["up", "down", "left", "right"]),
    )
    subtitle_motion_amplitude = RangeConfigItem(
        "SubtitleStyle", "MotionAmplitude", 100, RangeValidator(1, 500)
    )
    subtitle_motion_easing = OptionsConfigItem(
        "SubtitleStyle",
        "MotionEasing",
        "ease_out",
        OptionsValidator(["ease_out", "ease_in", "ease_in_out", "linear"]),
    )
    subtitle_motion_jitter = RangeConfigItem(
        "SubtitleStyle", "MotionJitter", 0, RangeValidator(0, 200)
    )
    subtitle_motion_blur_strength = RangeConfigItem(
        "SubtitleStyle", "MotionBlurStrength", 0, RangeValidator(0, 20)
    )
    subtitle_karaoke_mode = ConfigItem(
        "SubtitleStyle", "KaraokeMode", False, BoolValidator()
    )
    subtitle_karaoke_window_ms = RangeConfigItem(
        "SubtitleStyle", "KaraokeWindowMs", 1200, RangeValidator(50, 8000)
    )
    subtitle_auto_contrast = ConfigItem(
        "SubtitleStyle", "AutoContrast", False, BoolValidator()
    )
    subtitle_anti_flicker = ConfigItem(
        "SubtitleStyle", "AntiFlicker", True, BoolValidator()
    )
    subtitle_gradient_mode = OptionsConfigItem(
        "SubtitleStyle",
        "GradientMode",
        "off",
        OptionsValidator(["off", "two_color", "rainbow"]),
    )
    subtitle_speaker_color_mode = OptionsConfigItem(
        "SubtitleStyle",
        "SpeakerColorMode",
        "off",
        OptionsValidator(["off", "alternate"],),
    )
    subtitle_gradient_color_1 = ConfigItem(
        "SubtitleStyle", "GradientColor1", "#FFFFFF"
    )
    subtitle_gradient_color_2 = ConfigItem(
        "SubtitleStyle", "GradientColor2", "#66CCFF"
    )
    subtitle_safe_area_enabled = ConfigItem(
        "SubtitleStyle", "SafeAreaEnabled", True, BoolValidator()
    )
    subtitle_safe_margin_x = RangeConfigItem(
        "SubtitleStyle", "SafeMarginX", 8, RangeValidator(0, 40)
    )
    subtitle_safe_margin_y = RangeConfigItem(
        "SubtitleStyle", "SafeMarginY", 10, RangeValidator(0, 40)
    )

    # ------------------- 保存配置 -------------------
    work_dir = ConfigItem("Save", "Work_Dir", WORK_PATH, FolderValidator())

    # ------------------- 软件页面配置 -------------------
    micaEnabled = ConfigItem("MainWindow", "MicaEnabled", False, BoolValidator())
    dpiScale = OptionsConfigItem(
        "MainWindow",
        "DpiScale",
        "Auto",
        OptionsValidator([1, 1.25, 1.5, 1.75, 2, "Auto"]),
        restart=True,
    )
    language = OptionsConfigItem(
        "MainWindow",
        "Language",
        Language.AUTO,
        OptionsValidator(Language),
        LanguageSerializer(),
        restart=True,
    )
    ui_window_bg = ConfigItem("MainWindow", "UiWindowBg", "")
    ui_panel_bg = ConfigItem("MainWindow", "UiPanelBg", "")
    ui_card_bg = ConfigItem("MainWindow", "UiCardBg", "")
    ui_border_color = ConfigItem("MainWindow", "UiBorderColor", "")
    ui_text_color = ConfigItem("MainWindow", "UiTextColor", "")
    auto_shorts_render_backend = OptionsConfigItem(
        "MainWindow",
        "AutoShortsRenderBackend",
        "auto",
        OptionsValidator(["auto", "cpu", "gpu", "cuda"]),
    )
    auto_shorts_clip_head_pad_ms = RangeConfigItem(
        "MainWindow", "AutoShortsClipHeadPadMs", 90, RangeValidator(0, 2000)
    )
    auto_shorts_clip_tail_pad_ms = RangeConfigItem(
        "MainWindow", "AutoShortsClipTailPadMs", 220, RangeValidator(0, 3000)
    )
    auto_shorts_speech_pre_pad_ms = RangeConfigItem(
        "MainWindow", "AutoShortsSpeechPrePadMs", 140, RangeValidator(0, 1500)
    )
    auto_shorts_speech_post_pad_ms = RangeConfigItem(
        "MainWindow", "AutoShortsSpeechPostPadMs", 160, RangeValidator(0, 2000)
    )
    auto_shorts_speech_merge_gap_ms = RangeConfigItem(
        "MainWindow", "AutoShortsSpeechMergeGapMs", 180, RangeValidator(60, 2000)
    )
    auto_shorts_speech_min_coverage_percent = RangeConfigItem(
        "MainWindow", "AutoShortsSpeechMinCoveragePercent", 74, RangeValidator(30, 100)
    )
    auto_shorts_repeat_similarity_percent = RangeConfigItem(
        "MainWindow", "AutoShortsRepeatSimilarityPercent", 82, RangeValidator(40, 100)
    )
    batch_synthesis_fps_mode = OptionsConfigItem(
        "MainWindow",
        "BatchSynthesisFpsMode",
        "source",
        OptionsValidator(["source", "30", "60"]),
    )
    batch_synthesis_resolution_mode = OptionsConfigItem(
        "MainWindow",
        "BatchSynthesisResolutionMode",
        "source",
        OptionsValidator(["source", "fixed"]),
    )
    batch_synthesis_resolution = OptionsConfigItem(
        "MainWindow",
        "BatchSynthesisResolution",
        "1080x1920",
        OptionsValidator(["1080x1920", "720x1280", "1440x2560"]),
    )
    batch_synthesis_quality_profile = OptionsConfigItem(
        "MainWindow",
        "BatchSynthesisQualityProfile",
        "high",
        OptionsValidator(["high", "balanced", "fast"]),
    )
    batch_synthesis_render_backend = OptionsConfigItem(
        "MainWindow",
        "BatchSynthesisRenderBackend",
        "gpu",
        OptionsValidator(["cpu", "gpu"]),
    )

    # ------------------- Video Translate -------------------
    video_translate_source_language = OptionsConfigItem(
        "VideoTranslate",
        "SourceLanguage",
        TranscribeLanguageEnum.ENGLISH,
        OptionsValidator(TranscribeLanguageEnum),
        EnumSerializer(TranscribeLanguageEnum),
    )
    video_translate_target_language = OptionsConfigItem(
        "VideoTranslate",
        "TargetLanguage",
        TargetLanguageEnum.ENGLISH,
        OptionsValidator(TargetLanguageEnum),
        EnumSerializer(TargetLanguageEnum),
    )
    video_translate_enable_diarization = ConfigItem(
        "VideoTranslate", "EnableDiarization", True, BoolValidator()
    )
    video_translate_expected_speaker_count = RangeConfigItem(
        "VideoTranslate", "ExpectedSpeakerCount", 0, RangeValidator(0, 12)
    )
    video_translate_enable_source_separation = ConfigItem(
        "VideoTranslate", "EnableSourceSeparation", True, BoolValidator()
    )
    video_translate_source_separation_mode = OptionsConfigItem(
        "VideoTranslate",
        "SourceSeparationMode",
        "demucs_plus_uvr",
        OptionsValidator(["auto", "demucs", "uvr_mdx_kim", "demucs_plus_uvr"]),
    )
    video_translate_keep_background_music = ConfigItem(
        "VideoTranslate", "KeepBackgroundMusic", True, BoolValidator()
    )
    video_translate_enable_lipsync = ConfigItem(
        "VideoTranslate", "EnableLipsync", False, BoolValidator()
    )
    video_translate_voice_provider = OptionsConfigItem(
        "VideoTranslate",
        "VoiceProvider",
        "auto",
        OptionsValidator(["auto", "xtts", "rvc", "openvoice", "fish_speech", "elevenlabs", "azure", "cartesia"]),
    )
    video_translate_voice_quality = OptionsConfigItem(
        "VideoTranslate",
        "VoiceQuality",
        "high",
        OptionsValidator(["fast", "balanced", "high", "studio"]),
    )
    video_translate_voice_reference_mode = OptionsConfigItem(
        "VideoTranslate",
        "VoiceReferenceMode",
        "auto",
        OptionsValidator(["auto", "manual"]),
    )
    video_translate_manual_voice_map_json = ConfigItem(
        "VideoTranslate", "ManualVoiceMapJson", ""
    )
    video_translate_xtts_model_path = ConfigItem(
        "VideoTranslate", "XTTSModelPath", ""
    )
    video_translate_openvoice_model_path = ConfigItem(
        "VideoTranslate", "OpenVoiceModelPath", ""
    )
    video_translate_fish_speech_model_path = ConfigItem(
        "VideoTranslate", "FishSpeechModelPath", ""
    )
    video_translate_rvc_runtime_python = ConfigItem(
        "VideoTranslate", "RVCRuntimePython", ""
    )
    video_translate_rvc_model_dir = ConfigItem(
        "VideoTranslate", "RVCModelDir", ""
    )
    video_translate_rvc_default_model = ConfigItem(
        "VideoTranslate", "RVCDefaultModel", ""
    )
    video_translate_rvc_auto_male_models = ConfigItem(
        "VideoTranslate", "RVCAutoMaleModels", ""
    )
    video_translate_rvc_auto_female_models = ConfigItem(
        "VideoTranslate", "RVCAutoFemaleModels", ""
    )
    video_translate_rvc_index_rate = RangeConfigItem(
        "VideoTranslate", "RVCIndexRate", 0.75, RangeValidator(0.0, 1.0)
    )
    video_translate_rvc_protect = RangeConfigItem(
        "VideoTranslate", "RVCProtect", 0.33, RangeValidator(0.0, 0.5)
    )
    video_translate_rvc_filter_radius = RangeConfigItem(
        "VideoTranslate", "RVCFilterRadius", 3, RangeValidator(0, 7)
    )
    video_translate_rvc_male_f0_up_key = RangeConfigItem(
        "VideoTranslate", "RVCMaleF0UpKey", 0, RangeValidator(-24, 24)
    )
    video_translate_rvc_female_f0_up_key = RangeConfigItem(
        "VideoTranslate", "RVCFemaleF0UpKey", 0, RangeValidator(-24, 24)
    )
    video_translate_local_tts_endpoint = ConfigItem(
        "VideoTranslate", "LocalTTSEndpoint", "http://127.0.0.1:8020"
    )
    video_translate_autonomous_mode = ConfigItem(
        "VideoTranslate", "AutonomousMode", True, BoolValidator()
    )
    video_translate_auto_download_models = ConfigItem(
        "VideoTranslate", "AutoDownloadModels", True, BoolValidator()
    )
    video_translate_tts_parallel_workers = RangeConfigItem(
        "VideoTranslate", "TTSParallelWorkers", 3, RangeValidator(1, 8)
    )
    video_translate_use_asr_cache = ConfigItem(
        "VideoTranslate", "UseASRCache", True, BoolValidator()
    )
    video_translate_use_translation_cache = ConfigItem(
        "VideoTranslate", "UseTranslationCache", True, BoolValidator()
    )
    video_translate_allow_speaker_overlap = ConfigItem(
        "VideoTranslate", "AllowSpeakerOverlap", True, BoolValidator()
    )
    video_translate_overlap_aware_mix = ConfigItem(
        "VideoTranslate", "OverlapAwareMix", True, BoolValidator()
    )
    video_translate_segment_qa_enabled = ConfigItem(
        "VideoTranslate", "SegmentQAEnabled", True, BoolValidator()
    )
    video_translate_segment_qa_retry_count = RangeConfigItem(
        "VideoTranslate", "SegmentQARetryCount", 1, RangeValidator(0, 4)
    )
    video_translate_segment_min_duration_ms = RangeConfigItem(
        "VideoTranslate", "SegmentMinDurationMs", 180, RangeValidator(80, 1000)
    )
    video_translate_segment_min_size_bytes = RangeConfigItem(
        "VideoTranslate", "SegmentMinSizeBytes", 2500, RangeValidator(200, 20000)
    )
    video_translate_segment_min_mean_db = RangeConfigItem(
        "VideoTranslate", "SegmentMinMeanDb", -43.0, RangeValidator(-70.0, -10.0)
    )
    video_translate_segment_max_peak_db = RangeConfigItem(
        "VideoTranslate", "SegmentMaxPeakDb", -0.2, RangeValidator(-3.0, 0.0)
    )
    video_translate_enable_background_ducking = ConfigItem(
        "VideoTranslate", "EnableBackgroundDucking", True, BoolValidator()
    )
    video_translate_preserve_background_loudness = ConfigItem(
        "VideoTranslate", "PreserveBackgroundLoudness", False, BoolValidator()
    )
    video_translate_aggressive_vocal_suppression = ConfigItem(
        "VideoTranslate", "AggressiveVocalSuppression", False, BoolValidator()
    )
    video_translate_reference_enhancement_enabled = ConfigItem(
        "VideoTranslate", "ReferenceEnhancementEnabled", True, BoolValidator()
    )
    video_translate_reference_min_mean_db = RangeConfigItem(
        "VideoTranslate", "ReferenceMinMeanDb", -48.0, RangeValidator(-70.0, -10.0)
    )
    video_translate_reference_target_total_sec = RangeConfigItem(
        "VideoTranslate", "ReferenceTargetTotalSec", 18.0, RangeValidator(4.0, 60.0)
    )
    video_translate_uvr_model_dir = ConfigItem(
        "VideoTranslate", "UVRModelDir", ""
    )
    video_translate_uvr_inst_hq3_model_name = ConfigItem(
        "VideoTranslate", "UVRInstHQ3ModelName", "UVR-MDX-NET-Inst_HQ_3.onnx"
    )
    video_translate_uvr_kim_vocal_model_name = ConfigItem(
        "VideoTranslate", "UVRKimVocalModelName", "Kim_Vocal_2.onnx"
    )
    video_translate_elevenlabs_api_key = ConfigItem(
        "VideoTranslate", "ElevenLabsApiKey", ""
    )
    video_translate_azure_speech_key = ConfigItem(
        "VideoTranslate", "AzureSpeechKey", ""
    )
    video_translate_azure_speech_region = ConfigItem(
        "VideoTranslate", "AzureSpeechRegion", ""
    )
    video_translate_cartesia_api_key = ConfigItem(
        "VideoTranslate", "CartesiaApiKey", ""
    )
    video_translate_video_decode_backend = OptionsConfigItem(
        "VideoTranslate",
        "VideoDecodeBackend",
        "auto",
        OptionsValidator(["auto", "cpu", "cuda"]),
    )
    video_translate_video_encode_backend = OptionsConfigItem(
        "VideoTranslate",
        "VideoEncodeBackend",
        "copy",
        OptionsValidator(["copy", "cpu", "nvenc"]),
    )

    # ------------------- 更新配置 -------------------
    checkUpdateAtStartUp = ConfigItem(
        "Update", "CheckUpdateAtStartUp", True, BoolValidator()
    )
    update_last_known_commit = ConfigItem("Update", "LastKnownCommit", "")
    update_repo_owner = ConfigItem("Update", "RepoOwner", "MechaniksLab")
    update_repo_name = ConfigItem("Update", "RepoName", "ShortsCreatorStudio")
    update_repo_branch = ConfigItem("Update", "RepoBranch", "master")

cfg = Config()
cfg.themeMode.value = Theme.DARK
cfg.themeColor.value = QColor("#ff28f08b")
qconfig.load(SETTINGS_PATH, cfg)
