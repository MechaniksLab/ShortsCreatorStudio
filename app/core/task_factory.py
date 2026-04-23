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
    TranslatorServiceEnum,
    TranscribeConfig,
    TranscribeModelEnum,
    TranscribeTask,
    TranscriptAndSubtitleTask,
    VideoTranslateConfig,
    VideoTranslateTask,
)


class TaskFactory:
    """任务工厂类，用于创建各种类型的任务"""

    # Короткие русские префиксы для выходных файлов
    PREFIX_RAW_SUBTITLE = "исх"
    PREFIX_DOWNLOADED_SUBTITLE = "загр"
    PREFIX_STYLED_SUBTITLE = "стил"
    PREFIX_SUBTITLE = "суб"
    PREFIX_SPLIT_SUBTITLE = "разб"
    PREFIX_SMART_SPLIT_SUBTITLE = "умразб"
    PREFIX_RENDERED_VIDEO = "видео"
    PREFIX_TRANSLATED_VIDEO = "перевод"

    # Безопасные лимиты для Windows-путей
    MAX_FILENAME_LEN = 180
    MAX_PATH_LEN = 240

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
    def _strip_known_subtitle_prefixes(name: str) -> str:
        """Удаляет служебные префиксы у имён субтитров (старые и новые)."""
        value = (name or "").strip()
        known_prefixes = (
            "【原始字幕】",
            "【下载字幕】",
            "【样式字幕】",
            "【字幕】",
            "【断句字幕】",
            "【智能断句】",
            TaskFactory.PREFIX_RAW_SUBTITLE,
            TaskFactory.PREFIX_DOWNLOADED_SUBTITLE,
            TaskFactory.PREFIX_STYLED_SUBTITLE,
            TaskFactory.PREFIX_SUBTITLE,
            TaskFactory.PREFIX_SPLIT_SUBTITLE,
            TaskFactory.PREFIX_SMART_SPLIT_SUBTITLE,
            f"{TaskFactory.PREFIX_RAW_SUBTITLE}_",
            f"{TaskFactory.PREFIX_DOWNLOADED_SUBTITLE}_",
            f"{TaskFactory.PREFIX_STYLED_SUBTITLE}_",
            f"{TaskFactory.PREFIX_SUBTITLE}_",
            f"{TaskFactory.PREFIX_SPLIT_SUBTITLE}_",
            f"{TaskFactory.PREFIX_SMART_SPLIT_SUBTITLE}_",
        )
        for p in known_prefixes:
            if value.startswith(p):
                value = value[len(p) :]
        return value

    @staticmethod
    def _fit_filename_to_path_limit(parent: Path, filename: str) -> str:
        """Ограничивает имя файла так, чтобы полный путь был безопасен для Windows."""
        ext = Path(filename).suffix
        stem = Path(filename).stem
        parent_len = len(str(parent))
        max_stem_by_filename = max(8, TaskFactory.MAX_FILENAME_LEN - len(ext))
        max_stem_by_path = max(8, TaskFactory.MAX_PATH_LEN - parent_len - 1 - len(ext))
        max_stem = max(8, min(max_stem_by_filename, max_stem_by_path))

        if len(stem) <= max_stem and len(str(parent / filename)) <= TaskFactory.MAX_PATH_LEN:
            return filename

        digest = hashlib.md5(stem.encode("utf-8", errors="ignore")).hexdigest()[:8]
        head_len = max(4, max_stem - 9)
        short_stem = stem[:head_len].rstrip(" ._")
        if not short_stem:
            short_stem = "file"
        return f"{short_stem}_{digest}{ext}"

    @staticmethod
    def _build_output_path(
        parent: Path,
        base_name: str,
        ext: str,
        prefix: str = "",
        suffix: str = "",
        base_max_len: int = 72,
    ) -> str:
        """Собирает безопасный output path с коротким префиксом и контролем длины."""
        parent.mkdir(parents=True, exist_ok=True)
        safe_base = TaskFactory._safe_fs_name(base_name, max_len=base_max_len)
        safe_prefix = TaskFactory._safe_fs_name(prefix, max_len=16).replace(" ", "_") if prefix else ""
        ext = ext if ext.startswith(".") else f".{ext}"

        if safe_prefix:
            filename = f"{safe_prefix}_{safe_base}{suffix}{ext}"
        else:
            filename = f"{safe_base}{suffix}{ext}"

        filename = TaskFactory._fit_filename_to_path_limit(parent, filename)
        return str(parent / filename)

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
        safe_file_name = TaskFactory._safe_fs_name(file_name, max_len=48)

        # 构建输出路径
        if need_next_task:
            output_path = TaskFactory._build_output_path(
                parent=Path(cfg.work_dir.value) / safe_file_name / "subtitle",
                base_name=safe_file_name,
                ext=".srt",
                prefix=TaskFactory.PREFIX_RAW_SUBTITLE,
                suffix=f"-{cfg.transcribe_model.value.value}-{cfg.transcribe_language.value.value}",
                base_max_len=64,
            )
        else:
            if cfg.transcribe_model.value == TranscribeModelEnum.FASTER_WHISPER:
                need_word_time_stamp = bool(cfg.faster_whisper_one_word.value)
            output_path = TaskFactory._build_output_path(
                parent=Path(file_path).parent,
                base_name=file_name,
                ext=".srt",
                base_max_len=96,
            )

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
        output_name = TaskFactory._strip_known_subtitle_prefixes(Path(file_path).stem)
        output_name_safe = TaskFactory._safe_fs_name(output_name, max_len=72)
        # 只在需要翻译时添加翻译服务后缀
        suffix = (
            f"-{cfg.translator_service.value.value}" if cfg.need_translate.value else ""
        )

        if need_next_task:
            output_path = TaskFactory._build_output_path(
                parent=Path(file_path).parent,
                base_name=output_name_safe,
                ext=".ass",
                prefix=TaskFactory.PREFIX_STYLED_SUBTITLE,
                suffix=suffix,
            )
        else:
            output_path = TaskFactory._build_output_path(
                parent=Path(file_path).parent,
                base_name=output_name_safe,
                ext=".srt",
                prefix=TaskFactory.PREFIX_SUBTITLE,
                suffix=suffix,
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
        output_path = TaskFactory._build_output_path(
            parent=Path(video_path).parent,
            base_name=Path(video_path).stem,
            ext=".mp4",
            prefix=TaskFactory.PREFIX_RENDERED_VIDEO,
            base_max_len=96,
        )

        config = SynthesisConfig(
            need_video=cfg.need_video.value,
            soft_subtitle=cfg.soft_subtitle.value,
            fps_mode=str(cfg.batch_synthesis_fps_mode.value or "source"),
            resolution_mode=str(cfg.batch_synthesis_resolution_mode.value or "source"),
            resolution=str(cfg.batch_synthesis_resolution.value or "1080x1920"),
            quality_profile=str(cfg.batch_synthesis_quality_profile.value or "high"),
            render_backend=str(cfg.batch_synthesis_render_backend.value or "gpu"),
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

    @staticmethod
    def create_video_translate_task(video_path: str) -> VideoTranslateTask:
        """Создать задачу перевода видео с клонированием голоса."""
        safe_name = TaskFactory._safe_fs_name(Path(video_path).stem, max_len=80)
        output_path = TaskFactory._build_output_path(
            parent=Path(video_path).parent,
            base_name=safe_name,
            ext=".mp4",
            prefix=TaskFactory.PREFIX_TRANSLATED_VIDEO,
            suffix=f"-{cfg.video_translate_target_language.value.value}",
            base_max_len=96,
        )
        output_subtitle_path = TaskFactory._build_output_path(
            parent=Path(video_path).parent,
            base_name=safe_name,
            ext=".srt",
            prefix="перевод_суб",
            suffix=f"-{cfg.video_translate_target_language.value.value}",
            base_max_len=96,
        )

        # Переиспользуем текущие настройки ASR
        transcribe_task = TaskFactory.create_transcribe_task(video_path, need_next_task=False)
        # Для video translate используем отдельный флаг/namespace кэша ASR,
        # чтобы не смешивать его с обычными задачами транскрибации.
        transcribe_task.transcribe_config.use_asr_cache = bool(
            cfg.video_translate_use_asr_cache.value
        )
        transcribe_task.transcribe_config.transcribe_language = LANGUAGES[
            cfg.video_translate_source_language.value.value
        ]
        base_asr_tag = str(transcribe_task.transcribe_config.asr_cache_tag or "default")
        transcribe_task.transcribe_config.asr_cache_tag = f"video_translate:{base_asr_tag}"

        # Получаем LLM endpoint/key/model как в subtitle task
        current_service = cfg.llm_service.value
        if current_service == LLMServiceEnum.OPENAI:
            base_url, api_key, llm_model = (
                cfg.openai_api_base.value,
                cfg.openai_api_key.value,
                cfg.openai_model.value,
            )
        elif current_service == LLMServiceEnum.SILICON_CLOUD:
            base_url, api_key, llm_model = (
                cfg.silicon_cloud_api_base.value,
                cfg.silicon_cloud_api_key.value,
                cfg.silicon_cloud_model.value,
            )
        elif current_service == LLMServiceEnum.DEEPSEEK:
            base_url, api_key, llm_model = (
                cfg.deepseek_api_base.value,
                cfg.deepseek_api_key.value,
                cfg.deepseek_model.value,
            )
        elif current_service == LLMServiceEnum.OLLAMA:
            base_url, api_key, llm_model = (
                cfg.ollama_api_base.value,
                cfg.ollama_api_key.value,
                cfg.ollama_model.value,
            )
        elif current_service == LLMServiceEnum.LM_STUDIO:
            base_url, api_key, llm_model = (
                cfg.lm_studio_api_base.value,
                cfg.lm_studio_api_key.value,
                cfg.lm_studio_model.value,
            )
        elif current_service == LLMServiceEnum.GEMINI:
            base_url, api_key, llm_model = (
                cfg.gemini_api_base.value,
                cfg.gemini_api_key.value,
                cfg.gemini_model.value,
            )
        elif current_service == LLMServiceEnum.CHATGLM:
            base_url, api_key, llm_model = (
                cfg.chatglm_api_base.value,
                cfg.chatglm_api_key.value,
                cfg.chatglm_model.value,
            )
        else:
            base_url, api_key, llm_model = (
                cfg.public_api_base.value,
                cfg.public_api_key.value,
                cfg.public_model.value,
            )

        subtitle_cfg = SubtitleConfig(
            base_url=base_url,
            api_key=api_key,
            llm_model=llm_model,
            deeplx_endpoint=cfg.deeplx_endpoint.value,
            translator_service=cfg.translator_service.value,
            need_translate=True,
            need_optimize=False,
            need_reflect=cfg.need_reflect_translate.value,
            use_cache=cfg.video_translate_use_translation_cache.value,
            thread_num=cfg.thread_num.value,
            batch_size=cfg.batch_size.value,
            target_language=cfg.video_translate_target_language.value.value,
            custom_prompt_text=cfg.custom_prompt_text.value,
        )

        vt_cfg = VideoTranslateConfig(
            source_language=cfg.video_translate_source_language.value.value,
            target_language=cfg.video_translate_target_language.value.value,
            translator_service=cfg.translator_service.value,
            llm_base_url=base_url,
            llm_api_key=api_key,
            llm_model=llm_model,
            deeplx_endpoint=cfg.deeplx_endpoint.value,
            enable_diarization=cfg.video_translate_enable_diarization.value,
            expected_speaker_count=cfg.video_translate_expected_speaker_count.value,
            enable_source_separation=cfg.video_translate_enable_source_separation.value,
            source_separation_mode=cfg.video_translate_source_separation_mode.value,
            keep_background_music=cfg.video_translate_keep_background_music.value,
            enable_lipsync=cfg.video_translate_enable_lipsync.value,
            voice_clone_provider=cfg.video_translate_voice_provider.value,
            voice_clone_quality=cfg.video_translate_voice_quality.value,
            voice_reference_mode=cfg.video_translate_voice_reference_mode.value,
            manual_voice_map_json=cfg.video_translate_manual_voice_map_json.value,
            elevenlabs_api_key=cfg.video_translate_elevenlabs_api_key.value,
            azure_speech_key=cfg.video_translate_azure_speech_key.value,
            azure_speech_region=cfg.video_translate_azure_speech_region.value,
            cartesia_api_key=cfg.video_translate_cartesia_api_key.value,
            xtts_model_path=cfg.video_translate_xtts_model_path.value,
            openvoice_model_path=cfg.video_translate_openvoice_model_path.value,
            fish_speech_model_path=cfg.video_translate_fish_speech_model_path.value,
            rvc_runtime_python=cfg.video_translate_rvc_runtime_python.value,
            rvc_model_dir=cfg.video_translate_rvc_model_dir.value,
            rvc_default_model=cfg.video_translate_rvc_default_model.value,
            rvc_auto_male_models=cfg.video_translate_rvc_auto_male_models.value,
            rvc_auto_female_models=cfg.video_translate_rvc_auto_female_models.value,
            rvc_index_rate=cfg.video_translate_rvc_index_rate.value,
            rvc_protect=cfg.video_translate_rvc_protect.value,
            rvc_filter_radius=cfg.video_translate_rvc_filter_radius.value,
            rvc_male_f0_up_key=cfg.video_translate_rvc_male_f0_up_key.value,
            rvc_female_f0_up_key=cfg.video_translate_rvc_female_f0_up_key.value,
            local_tts_endpoint=cfg.video_translate_local_tts_endpoint.value,
            autonomous_mode=cfg.video_translate_autonomous_mode.value,
            auto_download_models=cfg.video_translate_auto_download_models.value,
            tts_parallel_workers=cfg.video_translate_tts_parallel_workers.value,
            use_translation_cache=cfg.video_translate_use_translation_cache.value,
            allow_speaker_overlap=cfg.video_translate_allow_speaker_overlap.value,
            overlap_aware_mix=cfg.video_translate_overlap_aware_mix.value,
            segment_qa_enabled=cfg.video_translate_segment_qa_enabled.value,
            segment_qa_retry_count=cfg.video_translate_segment_qa_retry_count.value,
            segment_min_duration_ms=cfg.video_translate_segment_min_duration_ms.value,
            segment_min_size_bytes=cfg.video_translate_segment_min_size_bytes.value,
            segment_min_mean_db=cfg.video_translate_segment_min_mean_db.value,
            segment_max_peak_db=cfg.video_translate_segment_max_peak_db.value,
            enable_background_ducking=cfg.video_translate_enable_background_ducking.value,
            preserve_background_loudness=cfg.video_translate_preserve_background_loudness.value,
            aggressive_vocal_suppression=cfg.video_translate_aggressive_vocal_suppression.value,
            reference_enhancement_enabled=cfg.video_translate_reference_enhancement_enabled.value,
            reference_min_mean_db=cfg.video_translate_reference_min_mean_db.value,
            reference_target_total_sec=cfg.video_translate_reference_target_total_sec.value,
            uvr_model_dir=cfg.video_translate_uvr_model_dir.value,
            uvr_inst_hq3_model_name=cfg.video_translate_uvr_inst_hq3_model_name.value,
            uvr_kim_vocal_model_name=cfg.video_translate_uvr_kim_vocal_model_name.value,
        )

        # Для автономного режима форсируем локальный ASR-маршрут
        if vt_cfg.autonomous_mode:
            transcribe_task.transcribe_config.transcribe_model = TranscribeModelEnum.FASTER_WHISPER
            transcribe_task.transcribe_config.faster_whisper_program = cfg.faster_whisper_program.value
            transcribe_task.transcribe_config.faster_whisper_model = cfg.faster_whisper_model.value.value
            transcribe_task.transcribe_config.faster_whisper_model_dir = str(MODEL_PATH)

        return VideoTranslateTask(
            queued_at=datetime.datetime.now(),
            video_path=video_path,
            output_path=output_path,
            output_subtitle_path=output_subtitle_path,
            transcribe_config=transcribe_task.transcribe_config,
            subtitle_config=subtitle_cfg,
            video_translate_config=vt_cfg,
        )
