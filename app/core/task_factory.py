import datetime
import hashlib
import re
from pathlib import Path
from typing import Optional

from app.common.config import cfg
from app.config import MODEL_PATH, SUBTITLE_STYLE_PATH
from app.core.entities import (
    LANGUAGES,
    FullProcessTask,
    LLMServiceEnum,
    SplitTypeEnum,
    SubtitleConfig,
    SubtitleTask,
    SynthesisConfig,
    SynthesisTask,
    TranscribeConfig,
    TranscribeModelEnum,
    TranscribeTask,
    TranscriptAndSubtitleTask,
)


class TaskFactory:
    """任务工厂类，用于创建各种类型的任务"""

    @staticmethod
    def _safe_fs_name(name: str, max_len: int = 72) -> str:
        """Безопасное имя для Windows-путей: чистим спецсимволы и ограничиваем длину."""
        raw = (name or "").strip()
        if not raw:
            return "untitled"

        raw = re.sub(r'[<>:"/\\|?*]+', " ", raw)
        raw = re.sub(r"[\x00-\x1F]", "", raw)
        raw = re.sub(r"\s+", " ", raw).strip(" .")
        if not raw:
            raw = "untitled"

        if len(raw) <= max_len:
            return raw

        digest = hashlib.md5(raw.encode("utf-8", errors="ignore")).hexdigest()[:8]
        head = raw[: max(16, max_len - 10)].rstrip(" .")
        return f"{head}_{digest}"

    @staticmethod
    def get_subtitle_style(style_name: str) -> str:
        """获取字幕样式内容

        Args:
            style_name: 样式名称

        Returns:
            str: 样式内容字符串，如果样式文件不存在则返回None
        """
        style_path = SUBTITLE_STYLE_PATH / f"{style_name}.txt"
        if style_path.exists():
            return style_path.read_text(encoding="utf-8")
        return None

    @staticmethod
    def create_transcribe_task(
        file_path: str, need_next_task: bool = False
    ) -> TranscribeTask:
        """创建转录任务"""

        split_type_cfg = cfg.split_type.value
        if isinstance(split_type_cfg, SplitTypeEnum):
            split_type_enum = split_type_cfg
        elif split_type_cfg == SplitTypeEnum.SENTENCE.value:
            split_type_enum = SplitTypeEnum.SENTENCE
        else:
            split_type_enum = SplitTypeEnum.SEMANTIC
        is_word_mode = split_type_enum == SplitTypeEnum.SEMANTIC

        # 词级时间戳需求：
        # - 全流程中由“是否需要断句”决定，避免意外始终变成按词字幕
        # - 单独转录页面可继续尊重 FasterWhisper OneWord 开关
        need_word_time_stamp = bool(cfg.need_split.value or is_word_mode)

        # 获取文件名
        file_name = Path(file_path).stem
        safe_file_name = TaskFactory._safe_fs_name(file_name, max_len=72)

        # 构建输出路径
        if need_next_task:
            output_path = str(
                Path(cfg.work_dir.value)
                / safe_file_name
                / "subtitle"
                / f"【原始字幕】{safe_file_name}-{cfg.transcribe_model.value.value}-{cfg.transcribe_language.value.value}.srt"
            )
        else:
            if cfg.transcribe_model.value == TranscribeModelEnum.FASTER_WHISPER:
                need_word_time_stamp = bool(cfg.faster_whisper_one_word.value)
            output_path = str(Path(file_path).parent / f"{file_name}.srt")

        use_asr_cache = cfg.use_asr_cache.value

        asr_cache_tag = "word" if need_word_time_stamp else "sentence"

        config = TranscribeConfig(
            transcribe_model=cfg.transcribe_model.value,
            transcribe_language=LANGUAGES[cfg.transcribe_language.value.value],
            use_asr_cache=use_asr_cache,
            asr_cache_tag=asr_cache_tag,
            need_word_time_stamp=need_word_time_stamp,
            # Whisper Cpp 配置
            whisper_model=cfg.whisper_model.value.value,
            # Whisper API 配置
            whisper_api_key=cfg.whisper_api_key.value,
            whisper_api_base=cfg.whisper_api_base.value,
            whisper_api_model=cfg.whisper_api_model.value,
            whisper_api_prompt=cfg.whisper_api_prompt.value,
            # Faster Whisper 配置
            faster_whisper_program=cfg.faster_whisper_program.value,
            faster_whisper_model=cfg.faster_whisper_model.value.value,
            faster_whisper_model_dir=str(MODEL_PATH),
            faster_whisper_device=cfg.faster_whisper_device.value,
            faster_whisper_vad_filter=cfg.faster_whisper_vad_filter.value,
            faster_whisper_vad_threshold=cfg.faster_whisper_vad_threshold.value,
            faster_whisper_vad_method=cfg.faster_whisper_vad_method.value.value,
            faster_whisper_ff_mdx_kim2=cfg.faster_whisper_ff_mdx_kim2.value,
            faster_whisper_one_word=cfg.faster_whisper_one_word.value,
            faster_whisper_prompt=cfg.faster_whisper_prompt.value,
        )

        return TranscribeTask(
            queued_at=datetime.datetime.now(),
            file_path=file_path,
            output_path=output_path,
            transcribe_config=config,
            need_next_task=need_next_task,
        )

    @staticmethod
    def create_subtitle_task(
        file_path: str, video_path: Optional[str] = None, need_next_task: bool = False
    ) -> SubtitleTask:
        """创建字幕任务"""
        output_name = (
            Path(file_path)
            .stem.replace("【原始字幕】", "")
            .replace(f"【下载字幕】", "")
        )
        output_name_safe = TaskFactory._safe_fs_name(output_name, max_len=72)
        # 只在需要翻译时添加翻译服务后缀
        suffix = (
            f"-{cfg.translator_service.value.value}" if cfg.need_translate.value else ""
        )

        if need_next_task:
            output_path = str(
                Path(file_path).parent / f"【样式字幕】{output_name_safe}{suffix}.ass"
            )
        else:
            output_path = str(
                Path(file_path).parent / f"【字幕】{output_name_safe}{suffix}.srt"
            )

        split_type_cfg = cfg.split_type.value
        if isinstance(split_type_cfg, SplitTypeEnum):
            split_type_enum = split_type_cfg
        elif split_type_cfg == SplitTypeEnum.SENTENCE.value:
            split_type_enum = SplitTypeEnum.SENTENCE
        else:
            split_type_enum = SplitTypeEnum.SEMANTIC

        if split_type_enum == SplitTypeEnum.SENTENCE:
            split_type = "sentence"
        else:
            split_type = "semantic"

        effective_need_split = bool(cfg.need_split.value or split_type == "semantic")

        # 根据当前选择的LLM服务获取对应的配置
        current_service = cfg.llm_service.value
        if current_service == LLMServiceEnum.OPENAI:
            base_url = cfg.openai_api_base.value
            api_key = cfg.openai_api_key.value
            llm_model = cfg.openai_model.value
        elif current_service == LLMServiceEnum.SILICON_CLOUD:
            base_url = cfg.silicon_cloud_api_base.value
            api_key = cfg.silicon_cloud_api_key.value
            llm_model = cfg.silicon_cloud_model.value
        elif current_service == LLMServiceEnum.DEEPSEEK:
            base_url = cfg.deepseek_api_base.value
            api_key = cfg.deepseek_api_key.value
            llm_model = cfg.deepseek_model.value
        elif current_service == LLMServiceEnum.OLLAMA:
            base_url = cfg.ollama_api_base.value
            api_key = cfg.ollama_api_key.value
            llm_model = cfg.ollama_model.value
        elif current_service == LLMServiceEnum.LM_STUDIO:
            base_url = cfg.lm_studio_api_base.value
            api_key = cfg.lm_studio_api_key.value
            llm_model = cfg.lm_studio_model.value
        elif current_service == LLMServiceEnum.GEMINI:
            base_url = cfg.gemini_api_base.value
            api_key = cfg.gemini_api_key.value
            llm_model = cfg.gemini_model.value
        elif current_service == LLMServiceEnum.CHATGLM:
            base_url = cfg.chatglm_api_base.value
            api_key = cfg.chatglm_api_key.value
            llm_model = cfg.chatglm_model.value
        elif current_service == LLMServiceEnum.PUBLIC:
            base_url = cfg.public_api_base.value
            api_key = cfg.public_api_key.value
            llm_model = cfg.public_model.value
        else:
            base_url = ""
            api_key = ""
            llm_model = ""

        config = SubtitleConfig(
            # 翻译配置
            base_url=base_url,
            api_key=api_key,
            llm_model=llm_model,
            deeplx_endpoint=cfg.deeplx_endpoint.value,
            # 翻译服务
            translator_service=cfg.translator_service.value,
            # 字幕处理
            split_type=split_type,
            need_reflect=cfg.need_reflect_translate.value,
            need_translate=cfg.need_translate.value,
            need_optimize=cfg.need_optimize.value,
            use_cache=cfg.use_subtitle_cache.value,
            use_processed_subtitle_cache=cfg.use_processed_subtitle_cache.value,
            thread_num=cfg.thread_num.value,
            batch_size=cfg.batch_size.value,
            # 字幕布局、样式
            subtitle_layout=cfg.subtitle_layout.value,
            subtitle_style=TaskFactory.get_subtitle_style(
                cfg.subtitle_style_name.value
            ),
            # 字幕分割
            max_word_count_cjk=cfg.max_word_count_cjk.value,
            max_word_count_english=cfg.max_word_count_english.value,
            need_split=effective_need_split,
            # 字幕翻译
            target_language=cfg.target_language.value.value,
            # 字幕优化
            need_remove_punctuation=cfg.needs_remove_punctuation.value,
            # 字幕提示
            custom_prompt_text=cfg.custom_prompt_text.value,
            subtitle_effect=cfg.subtitle_effect.value,
            subtitle_effect_duration=cfg.subtitle_effect_duration.value,
            subtitle_effect_intensity=cfg.subtitle_effect_intensity.value / 100,
            subtitle_rainbow_end_color=cfg.subtitle_rainbow_end_color.value,
            subtitle_style_preset=cfg.subtitle_style_preset.value,
            subtitle_motion_direction=cfg.subtitle_motion_direction.value,
            subtitle_motion_amplitude=cfg.subtitle_motion_amplitude.value / 100,
            subtitle_motion_easing=cfg.subtitle_motion_easing.value,
            subtitle_motion_jitter=cfg.subtitle_motion_jitter.value / 100,
            subtitle_motion_blur_strength=cfg.subtitle_motion_blur_strength.value,
            subtitle_karaoke_mode=cfg.subtitle_karaoke_mode.value,
            subtitle_karaoke_window_ms=cfg.subtitle_karaoke_window_ms.value,
            subtitle_auto_contrast=cfg.subtitle_auto_contrast.value,
            subtitle_anti_flicker=cfg.subtitle_anti_flicker.value,
            subtitle_gradient_mode=cfg.subtitle_gradient_mode.value,
            subtitle_gradient_color_1=cfg.subtitle_gradient_color_1.value,
            subtitle_gradient_color_2=cfg.subtitle_gradient_color_2.value,
            subtitle_safe_area_enabled=cfg.subtitle_safe_area_enabled.value,
            subtitle_safe_margin_x=cfg.subtitle_safe_margin_x.value,
            subtitle_safe_margin_y=cfg.subtitle_safe_margin_y.value,
            subtitle_speaker_color_mode=cfg.subtitle_speaker_color_mode.value,
        )

        return SubtitleTask(
            queued_at=datetime.datetime.now(),
            subtitle_path=file_path,
            video_path=video_path,
            output_path=output_path,
            subtitle_config=config,
            need_next_task=need_next_task,
        )

    @staticmethod
    def create_synthesis_task(
        video_path: str, subtitle_path: str, need_next_task: bool = False
    ) -> SynthesisTask:
        """创建视频合成任务"""
        if need_next_task:
            output_path = str(
                Path(video_path).parent / f"【卡卡】{Path(video_path).stem}.mp4"
            )
        else:
            output_path = str(
                Path(video_path).parent / f"【卡卡】{Path(video_path).stem}.mp4"
            )

        config = SynthesisConfig(
            need_video=cfg.need_video.value,
            soft_subtitle=cfg.soft_subtitle.value,
        )

        return SynthesisTask(
            queued_at=datetime.datetime.now(),
            video_path=video_path,
            subtitle_path=subtitle_path,
            output_path=output_path,
            synthesis_config=config,
            need_next_task=need_next_task,
        )

    @staticmethod
    def create_transcript_and_subtitle_task(
        file_path: str,
        output_path: Optional[str] = None,
        transcribe_config: Optional[TranscribeConfig] = None,
        subtitle_config: Optional[SubtitleConfig] = None,
    ) -> TranscriptAndSubtitleTask:
        """创建转录和字幕任务"""
        if output_path is None:
            output_path = str(
                Path(file_path).parent / f"{Path(file_path).stem}_processed.srt"
            )

        return TranscriptAndSubtitleTask(
            queued_at=datetime.datetime.now(),
            file_path=file_path,
            output_path=output_path,
        )

    @staticmethod
    def create_full_process_task(
        file_path: str,
        output_path: Optional[str] = None,
        transcribe_config: Optional[TranscribeConfig] = None,
        subtitle_config: Optional[SubtitleConfig] = None,
        synthesis_config: Optional[SynthesisConfig] = None,
    ) -> FullProcessTask:
        """创建完整处理任务（转录+字幕+合成）"""
        if output_path is None:
            output_path = str(
                Path(file_path).parent
                / f"{Path(file_path).stem}_final{Path(file_path).suffix}"
            )

        return FullProcessTask(
            queued_at=datetime.datetime.now(),
            file_path=file_path,
            output_path=output_path,
        )
