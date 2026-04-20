import datetime
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Dict

from PyQt5.QtCore import QSettings, QThread, pyqtSignal

from app.common.config import cfg
from app.core.bk_asr.asr_data import ASRData, ASRDataSeg
from app.core.entities import SubtitleConfig, SubtitleTask, TranslatorServiceEnum
from app.core.subtitle_processor.split import SubtitleSplitter
from app.core.subtitle_processor.summarization import SubtitleSummarizer
from app.core.subtitle_processor.optimize import SubtitleOptimizer
from app.core.subtitle_processor.translate import TranslatorFactory, TranslatorType
from app.core.utils.logger import setup_logger
from app.core.utils.test_opanai import test_openai
from app.core.utils.get_subtitle_style import get_subtitle_style
from app.core.storage.cache_manager import ServiceUsageManager
from app.core.storage.database import DatabaseManager
from app.core.errors import AppError, ConfigError, map_exception
from app.core.task_factory import TaskFactory
from app.core.utils.io_utils import atomic_write_text
from app.config import CACHE_PATH, WORK_PATH

# 配置日志
logger = setup_logger("subtitle_optimization_thread")


class SubtitleThread(QThread):
    finished = pyqtSignal(str, str)
    progress = pyqtSignal(int, str)
    update = pyqtSignal(dict)
    update_all = pyqtSignal(dict)
    error = pyqtSignal(str)
    MAX_DAILY_LLM_CALLS = 30

    def __init__(self, task: SubtitleTask):
        super().__init__()
        self.task: SubtitleTask = task
        self.subtitle_length = 0
        self.finished_subtitle_length = 0
        self.custom_prompt_text = ""
        # 初始化数据库和服务使用管理器
        self.db_manager = DatabaseManager(CACHE_PATH)
        self.service_manager = ServiceUsageManager(self.db_manager)

    def set_custom_prompt_text(self, text: str):
        self.custom_prompt_text = text

    @staticmethod
    def _clone_asr_data(asr_data: ASRData) -> ASRData:
        """Небольшой безопасный clone через JSON-представление."""
        return ASRData.from_json(asr_data.to_json())

    @staticmethod
    def _count_units(text: str) -> int:
        """Простой счётчик длины фразы: слова + CJK-символы."""
        if not text:
            return 0
        cjk = len(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text))
        non_cjk = re.sub(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", " ", text)
        words = len(non_cjk.strip().split())
        return cjk + words

    @staticmethod
    def _collect_word_timestamps(segments: list[ASRDataSeg]) -> list[dict]:
        result: list[dict] = []
        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            if seg.word_timestamps:
                result.extend(seg.word_timestamps)
            else:
                result.append(
                    {
                        "text": text,
                        "start_time": seg.start_time,
                        "end_time": seg.end_time,
                    }
                )
        return result

    def _build_sentence_segments_from_word_level(
        self,
        asr_data: ASRData,
        subtitle_config: SubtitleConfig,
    ) -> ASRData:
        """Собирает предложения из word-level сегментов без LLM/оптимизации."""
        src = [s for s in asr_data.segments if (s.text or "").strip()]
        if not src:
            return asr_data

        max_units = max(6, int(subtitle_config.max_word_count_english or 18))
        max_gap_ms = 650
        sentence_end_re = re.compile(r"[.!?…。！？]$")

        groups: list[list[ASRDataSeg]] = []
        current: list[ASRDataSeg] = []

        for seg in src:
            if not current:
                current.append(seg)
                continue

            prev = current[-1]
            gap = int(seg.start_time - prev.end_time)
            current_text = " ".join((x.text or "").strip() for x in current).strip()
            cur_units = self._count_units(current_text)

            should_break = (
                gap > max_gap_ms
                or sentence_end_re.search((prev.text or "").strip()) is not None
                or cur_units >= max_units
            )

            if should_break:
                groups.append(current)
                current = [seg]
            else:
                current.append(seg)

        if current:
            groups.append(current)

        out_segments: list[ASRDataSeg] = []
        for group in groups:
            raw_text = " ".join((x.text or "").strip() for x in group).strip()
            # Нормализация пробелов перед пунктуацией
            text = re.sub(r"\s+([,.;:!?])", r"\1", raw_text)
            text = re.sub(r"\s+([\]\)»])", r"\1", text)
            text = re.sub(r"([\[\(«])\s+", r"\1", text)
            out_segments.append(
                ASRDataSeg(
                    text=text,
                    start_time=group[0].start_time,
                    end_time=group[-1].end_time,
                    word_timestamps=self._collect_word_timestamps(group),
                )
            )

        return ASRData(out_segments)

    def _processed_cache_path(self, subtitle_path: str, subtitle_config: SubtitleConfig) -> Path:
        source_bytes = Path(subtitle_path).read_bytes()
        source_hash = hashlib.md5(source_bytes).hexdigest()
        process_profile = {
            "cache_schema": 2,
            "source_hash": source_hash,
            "split": subtitle_config.need_split,
            "split_type": subtitle_config.split_type,
            "max_word_count_cjk": subtitle_config.max_word_count_cjk,
            "max_word_count_english": subtitle_config.max_word_count_english,
            "optimize": subtitle_config.need_optimize,
            "translate": subtitle_config.need_translate,
            "reflect": subtitle_config.need_reflect,
            "translator_service": str(subtitle_config.translator_service),
            "target_language": str(subtitle_config.target_language),
            "llm_model": subtitle_config.llm_model,
            "custom_prompt": subtitle_config.custom_prompt_text or "",
            "remove_punctuation": subtitle_config.need_remove_punctuation,
            # Для karaoke/word_highlight критично наличие word_timestamps,
            # поэтому учитываем эти режимы в ключе кэша.
            "karaoke_mode": subtitle_config.subtitle_karaoke_mode,
            "subtitle_effect": subtitle_config.subtitle_effect,
        }
        key = hashlib.md5(
            json.dumps(process_profile, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        cache_dir = Path(CACHE_PATH) / "processed_subtitles"
        return cache_dir / f"{key}.json"

    def _setup_api_config(self) -> SubtitleConfig:
        """设置API配置，返回SubtitleConfig"""
        public_base_url = "https://ddg.bkfeng.top/v1"
        if self.task.subtitle_config.base_url == public_base_url:
            # 检查是否可以使用服务

            if not self.service_manager.check_service_available(
                "llm", self.MAX_DAILY_LLM_CALLS
            ):
                raise Exception(
                    self.tr(
                        f"公益LLM服务已达到每日使用限制 {self.MAX_DAILY_LLM_CALLS} 次，建议使用自己的API"
                    )
                )
            self.task.subtitle_config.thread_num = 5
            self.task.subtitle_config.batch_size = 10
            return self.task.subtitle_config

        if self.task.subtitle_config.base_url and self.task.subtitle_config.api_key:
            if not test_openai(
                self.task.subtitle_config.base_url,
                self.task.subtitle_config.api_key,
                self.task.subtitle_config.llm_model,
            )[0]:
                raise Exception(
                    self.tr(
                        "（字幕断句或字幕修正需要大模型）\nOpenAI API 测试失败, 请检查LLM配置"
                    )
                )
            # 增加服务使用次数
            if self.task.subtitle_config.base_url == public_base_url:
                self.service_manager.increment_usage("llm", self.MAX_DAILY_LLM_CALLS)
            return self.task.subtitle_config
        else:
            raise Exception(
                self.tr(
                    "（字幕断句或字幕修正需要大模型）\nOpenAI API 未配置, 请检查LLM配置"
                )
            )

    def run(self):
        stage_metrics = {}
        qa_report = {}

        def _stage_start(name: str):
            stage_metrics[name] = {"start": time.perf_counter()}

        def _stage_end(name: str):
            if name in stage_metrics and "start" in stage_metrics[name]:
                stage_metrics[name]["seconds"] = round(
                    time.perf_counter() - stage_metrics[name]["start"], 3
                )

        try:
            logger.info(f"\n===========字幕处理任务开始===========")
            logger.info(f"时间：{datetime.datetime.now()}")

            # 字幕文件路径检查、对断句字幕路径进行定义
            subtitle_path = self.task.subtitle_path
            output_name = TaskFactory._strip_known_subtitle_prefixes(
                Path(subtitle_path).stem
            )
            split_path = TaskFactory._build_output_path(
                parent=Path(subtitle_path).parent,
                base_name=output_name,
                ext=".srt",
                prefix=TaskFactory.PREFIX_SPLIT_SUBTITLE,
            )
            assert subtitle_path is not None, self.tr("字幕文件路径为空")

            subtitle_config = self.task.subtitle_config
            # 同步最新UI配置，避免复用旧任务对象导致设置不生效
            subtitle_config.use_processed_subtitle_cache = bool(
                cfg.use_processed_subtitle_cache.value
            )
            subtitle_config.subtitle_layout = cfg.subtitle_layout.value
            subtitle_config.subtitle_style = get_subtitle_style(
                cfg.subtitle_style_name.value
            )
            subtitle_config.subtitle_effect = cfg.subtitle_effect.value
            subtitle_config.subtitle_effect_duration = cfg.subtitle_effect_duration.value
            subtitle_config.subtitle_effect_intensity = (
                cfg.subtitle_effect_intensity.value / 100
            )
            # split_type: "sentence" | "semantic"
            # В текущем UX "semantic" используется как режим "по словам".
            is_word_mode = subtitle_config.split_type == "semantic"
            is_sentence_mode = subtitle_config.split_type == "sentence"
            subtitle_config.subtitle_rainbow_end_color = (
                cfg.subtitle_rainbow_end_color.value
            )
            subtitle_config.subtitle_motion_direction = cfg.subtitle_motion_direction.value
            subtitle_config.subtitle_motion_amplitude = (
                cfg.subtitle_motion_amplitude.value / 100
            )
            subtitle_config.subtitle_motion_easing = cfg.subtitle_motion_easing.value
            subtitle_config.subtitle_motion_jitter = cfg.subtitle_motion_jitter.value / 100
            subtitle_config.subtitle_motion_blur_strength = (
                cfg.subtitle_motion_blur_strength.value
            )
            subtitle_config.subtitle_karaoke_mode = (
                cfg.subtitle_karaoke_mode.value and not is_word_mode
            )
            subtitle_config.subtitle_karaoke_window_ms = (
                cfg.subtitle_karaoke_window_ms.value
            )
            subtitle_config.subtitle_auto_contrast = cfg.subtitle_auto_contrast.value
            subtitle_config.subtitle_anti_flicker = cfg.subtitle_anti_flicker.value
            subtitle_config.subtitle_gradient_mode = cfg.subtitle_gradient_mode.value
            subtitle_config.subtitle_gradient_color_1 = cfg.subtitle_gradient_color_1.value
            subtitle_config.subtitle_gradient_color_2 = cfg.subtitle_gradient_color_2.value
            subtitle_config.subtitle_safe_area_enabled = cfg.subtitle_safe_area_enabled.value
            subtitle_config.subtitle_safe_margin_x = cfg.subtitle_safe_margin_x.value
            subtitle_config.subtitle_safe_margin_y = cfg.subtitle_safe_margin_y.value
            subtitle_config.subtitle_speaker_color_mode = (
                cfg.subtitle_speaker_color_mode.value
            )
            force_sentence_karaoke_timing = bool(
                is_sentence_mode
                and (
                    subtitle_config.subtitle_karaoke_mode
                    or subtitle_config.subtitle_effect == "word_highlight"
                )
            )

            asr_data = ASRData.from_subtitle_file(subtitle_path)
            word_level_asr_for_export: ASRData | None = None
            processed_cache_path = self._processed_cache_path(subtitle_path, subtitle_config)
            reused_processed_cache = False

            if subtitle_config.use_processed_subtitle_cache and processed_cache_path.exists():
                try:
                    cached_data = json.loads(processed_cache_path.read_text(encoding="utf-8"))
                    asr_data = ASRData.from_json(cached_data)
                    reused_processed_cache = True
                    logger.info(f"命中处理后字幕缓存：{processed_cache_path}")
                    self.update_all.emit(asr_data.to_json())
                except Exception as e:
                    logger.warning(f"读取处理后字幕缓存失败，回退到正常流程: {str(e)}")

            # 1. 分割成字词级时间戳
            # Для sentence+karaoke (или word_highlight) тоже форсируем word-level,
            # иначе получится «псевдо-караоке» без ожидания реальных слов.
            if (
                (not reused_processed_cache)
                and (subtitle_config.need_split or force_sentence_karaoke_timing)
                and not asr_data.is_word_timestamp()
            ):
                asr_data.split_to_word_segments()

            # 获取API配置，会先检查可用性（优先使用设置的API，其次使用自带的公益API）
            if (
                not reused_processed_cache
                and (
                    subtitle_config.need_optimize
                    or (
                        subtitle_config.need_split
                        and is_sentence_mode
                        and not force_sentence_karaoke_timing
                    )
                    or (
                        subtitle_config.need_translate
                        and subtitle_config.translator_service
                        not in [
                            TranslatorServiceEnum.DEEPLX,
                            TranslatorServiceEnum.BING,
                            TranslatorServiceEnum.GOOGLE,
                        ]
                    )
                )
            ):
                _stage_start("api_check")
                self.progress.emit(2, self.tr("开始验证API配置..."))
                subtitle_config = self._setup_api_config()
                _stage_end("api_check")

            # 2. Формирование сегментов-предложений
            # - для sentence+karaoke/word_highlight: из word-level локально (без LLM)
            # - обычный sentence-режим: штатный LLM splitter
            if (not reused_processed_cache) and (subtitle_config.need_split or force_sentence_karaoke_timing):
                if is_sentence_mode:
                    _stage_start("split")
                    self.progress.emit(5, self.tr("字幕断句..."))
                    if force_sentence_karaoke_timing:
                        logger.info("sentence+karoake: локальная сборка предложений из word-level сегментов")
                        word_level_asr_for_export = self._clone_asr_data(asr_data)
                        asr_data = self._build_sentence_segments_from_word_level(
                            asr_data,
                            subtitle_config,
                        )
                    else:
                        logger.info("正在字幕断句...")
                        splitter = SubtitleSplitter(
                            thread_num=subtitle_config.thread_num,
                            model=subtitle_config.llm_model,
                            use_cache=subtitle_config.use_cache,
                            openai_base_url=subtitle_config.base_url,
                            openai_api_key=subtitle_config.api_key,
                            temperature=0.3,
                            timeout=60,
                            retry_times=1,
                            split_type=subtitle_config.split_type,
                            max_word_count_cjk=subtitle_config.max_word_count_cjk,
                            max_word_count_english=subtitle_config.max_word_count_english,
                        )
                        asr_data = splitter.split_subtitle(asr_data)
                    asr_data.save(save_path=split_path)
                    self.update_all.emit(asr_data.to_json())
                    _stage_end("split")
                elif is_word_mode:
                    logger.info("Режим 'по словам': пропускаем LLM-разбиение, оставляем word-level сегменты")

            # 3. 优化字幕
            custom_prompt = subtitle_config.custom_prompt_text
            self.subtitle_length = len(asr_data.segments)

            if (not reused_processed_cache) and subtitle_config.need_optimize:
                _stage_start("optimize")
                self.progress.emit(0, self.tr("优化字幕..."))
                logger.info("正在优化字幕...")
                self.finished_subtitle_length = 0  # 重置计数器
                optimizer = SubtitleOptimizer(
                    custom_prompt=custom_prompt,
                    model=subtitle_config.llm_model,
                    use_cache=subtitle_config.use_cache,
                    batch_num=subtitle_config.batch_size,
                    thread_num=subtitle_config.thread_num,
                    openai_base_url=subtitle_config.base_url,
                    openai_api_key=subtitle_config.api_key,
                    update_callback=self.callback,
                )
                asr_data = optimizer.optimize_subtitle(asr_data)
                self.update_all.emit(asr_data.to_json())
                _stage_end("optimize")

            # 4. 翻译字幕
            translator_map = {
                TranslatorServiceEnum.OPENAI: TranslatorType.OPENAI,
                TranslatorServiceEnum.DEEPLX: TranslatorType.DEEPLX,
                TranslatorServiceEnum.BING: TranslatorType.BING,
                TranslatorServiceEnum.GOOGLE: TranslatorType.GOOGLE,
            }
            if (not reused_processed_cache) and subtitle_config.need_translate:
                _stage_start("translate")
                self.progress.emit(0, self.tr("翻译字幕..."))
                logger.info("正在翻译字幕...")
                self.finished_subtitle_length = 0  # 重置计数器
                translator = TranslatorFactory.create_translator(
                    translator_type=translator_map[subtitle_config.translator_service],
                    thread_num=subtitle_config.thread_num,
                    batch_num=subtitle_config.batch_size,
                    target_language=subtitle_config.target_language,
                    model=subtitle_config.llm_model,
                    custom_prompt=custom_prompt,
                    is_reflect=subtitle_config.need_reflect,
                    use_cache=subtitle_config.use_cache,
                    openai_base_url=subtitle_config.base_url,
                    openai_api_key=subtitle_config.api_key,
                    deeplx_endpoint=subtitle_config.deeplx_endpoint,
                    update_callback=self.callback,
                )
                asr_data = translator.translate_subtitle(asr_data)
                self.update_all.emit(asr_data.to_json())
                _stage_end("translate")
                # 保存翻译结果(单语、双语)
                if self.task.need_next_task and self.task.video_path:
                    for subtitle_layout in ["原文在上", "译文在上", "仅原文", "仅译文"]:
                        save_path = str(
                            Path(self.task.subtitle_path).parent
                            / f"{Path(self.task.video_path).stem}-{subtitle_layout}.srt"
                        )
                        asr_data.save(
                            save_path=save_path,
                            ass_style=subtitle_config.subtitle_style,
                            layout=subtitle_layout,
                            effect_type=subtitle_config.subtitle_effect,
                            effect_duration_ms=subtitle_config.subtitle_effect_duration,
                            effect_intensity=subtitle_config.subtitle_effect_intensity,
                            rainbow_end_color=subtitle_config.subtitle_rainbow_end_color,
                            motion_direction=subtitle_config.subtitle_motion_direction,
                            motion_amplitude=subtitle_config.subtitle_motion_amplitude,
                            motion_easing=subtitle_config.subtitle_motion_easing,
                            motion_jitter=subtitle_config.subtitle_motion_jitter,
                            karaoke_mode=subtitle_config.subtitle_karaoke_mode,
                            karaoke_window_ms=subtitle_config.subtitle_karaoke_window_ms,
                            auto_contrast=subtitle_config.subtitle_auto_contrast,
                            anti_flicker=subtitle_config.subtitle_anti_flicker,
                            gradient_mode=subtitle_config.subtitle_gradient_mode,
                            gradient_color_1=subtitle_config.subtitle_gradient_color_1,
                            gradient_color_2=subtitle_config.subtitle_gradient_color_2,
                            safe_area_enabled=subtitle_config.subtitle_safe_area_enabled,
                            safe_margin_x=subtitle_config.subtitle_safe_margin_x,
                            safe_margin_y=subtitle_config.subtitle_safe_margin_y,
                            speaker_color_mode=subtitle_config.subtitle_speaker_color_mode,
                            motion_blur_strength=subtitle_config.subtitle_motion_blur_strength,
                        )
                        logger.info(f"字幕保存到 {save_path}")

            # 统一移除末尾标点（不依赖是否翻译）
            if (not reused_processed_cache) and subtitle_config.need_remove_punctuation:
                asr_data.remove_punctuation()

            # 5. 可读性增强 + 时间轴修复 + QA
            _stage_start("quality_checks")
            if not reused_processed_cache:
                asr_data.apply_smart_line_break(
                    max_cjk_chars=max(8, min(20, subtitle_config.max_word_count_cjk)),
                    max_english_words=max(5, min(14, subtitle_config.max_word_count_english // 2)),
                )
                timing_fixes = asr_data.validate_and_fix_timing()
                qa_report = asr_data.build_qa_report(cps_limit=22.0)
                qa_report["timing_fixes"] = timing_fixes

                if subtitle_config.use_processed_subtitle_cache:
                    atomic_write_text(
                        str(processed_cache_path),
                        json.dumps(asr_data.to_json(), ensure_ascii=False),
                        encoding="utf-8",
                    )
                    logger.info(f"写入处理后字幕缓存：{processed_cache_path}")
            else:
                qa_report = asr_data.build_qa_report(cps_limit=22.0)
                qa_report["timing_fixes"] = {
                    "negative_duration": 0,
                    "too_short_duration": 0,
                    "overlap_fixed": 0,
                }
            _stage_end("quality_checks")

            # 6. 保存字幕
            _stage_start("save_outputs")
            asr_data.save(
                save_path=self.task.output_path,
                ass_style=subtitle_config.subtitle_style,
                layout=subtitle_config.subtitle_layout,
                effect_type=subtitle_config.subtitle_effect,
                effect_duration_ms=subtitle_config.subtitle_effect_duration,
                effect_intensity=subtitle_config.subtitle_effect_intensity,
                rainbow_end_color=subtitle_config.subtitle_rainbow_end_color,
                motion_direction=subtitle_config.subtitle_motion_direction,
                motion_amplitude=subtitle_config.subtitle_motion_amplitude,
                motion_easing=subtitle_config.subtitle_motion_easing,
                motion_jitter=subtitle_config.subtitle_motion_jitter,
                karaoke_mode=subtitle_config.subtitle_karaoke_mode,
                karaoke_window_ms=subtitle_config.subtitle_karaoke_window_ms,
                auto_contrast=subtitle_config.subtitle_auto_contrast,
                anti_flicker=subtitle_config.subtitle_anti_flicker,
                gradient_mode=subtitle_config.subtitle_gradient_mode,
                gradient_color_1=subtitle_config.subtitle_gradient_color_1,
                gradient_color_2=subtitle_config.subtitle_gradient_color_2,
                safe_area_enabled=subtitle_config.subtitle_safe_area_enabled,
                safe_margin_x=subtitle_config.subtitle_safe_margin_x,
                safe_margin_y=subtitle_config.subtitle_safe_margin_y,
                speaker_color_mode=subtitle_config.subtitle_speaker_color_mode,
                motion_blur_strength=subtitle_config.subtitle_motion_blur_strength,
            )
            logger.info(f"字幕保存到 {self.task.output_path}")
            _stage_end("save_outputs")

            # 7. 文件移动与清理
            if self.task.need_next_task and self.task.video_path:
                # 保存srt/ass文件到视频目录（对于全流程任务）
                save_srt_path = (
                    Path(self.task.video_path).parent
                    / f"{Path(self.task.video_path).stem}.srt"
                )
                asr_data.to_srt(
                    save_path=str(save_srt_path), layout=subtitle_config.subtitle_layout
                )
                # Дополнительно сохраняем word-level SRT для отладки karaoke-таймингов
                # в режиме sentence+karaoke/word_highlight.
                if word_level_asr_for_export is not None:
                    save_word_srt_path = (
                        Path(self.task.video_path).parent
                        / f"{Path(self.task.video_path).stem}-слова.srt"
                    )
                    word_level_asr_for_export.to_srt(
                        save_path=str(save_word_srt_path),
                        layout="仅原文",
                    )
                    logger.info(f"Сохранён word-level SRT: {save_word_srt_path}")
                # save_ass_path = (
                #     Path(self.task.video_path).parent
                #     / f"{Path(self.task.video_path).stem}.ass"
                # )
                # asr_data.to_ass(
                #     save_path=str(save_ass_path),
                #     layout=subtitle_config.subtitle_layout,
                #     style_str=subtitle_config.subtitle_style,
                # )
            else:
                # 删除断句文件（对于仅字幕任务）
                subtitle_parent = Path(self.task.subtitle_path).parent
                subtitle_stem = TaskFactory._strip_known_subtitle_prefixes(
                    Path(self.task.subtitle_path).stem
                )
                split_path = TaskFactory._build_output_path(
                    parent=subtitle_parent,
                    base_name=subtitle_stem,
                    ext=".srt",
                    prefix=TaskFactory.PREFIX_SMART_SPLIT_SUBTITLE,
                )
                split_path_stage = TaskFactory._build_output_path(
                    parent=subtitle_parent,
                    base_name=subtitle_stem,
                    ext=".srt",
                    prefix=TaskFactory.PREFIX_SPLIT_SUBTITLE,
                )

                legacy_split_paths = [
                    split_path,
                    split_path_stage,
                    str(subtitle_parent / f"【智能断句】{Path(self.task.subtitle_path).stem}.srt"),
                    str(subtitle_parent / f"【断句字幕】{subtitle_stem}.srt"),
                ]
                for path in legacy_split_paths:
                    if os.path.exists(path):
                        os.remove(path)

            # 8. 诊断报告写入 work-dir
            report_name = f"subtitle_diagnostics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_path = Path(WORK_PATH) / report_name
            diagnostics = {
                "task": {
                    "input_subtitle": self.task.subtitle_path,
                    "output_subtitle": self.task.output_path,
                    "video_path": self.task.video_path,
                },
                "stage_metrics": {
                    k: {"seconds": v.get("seconds", 0)} for k, v in stage_metrics.items()
                },
                "qa_report": qa_report,
            }
            atomic_write_text(
                str(report_path),
                json.dumps(diagnostics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info(f"诊断报告保存到 {report_path}")

            self.progress.emit(100, self.tr("优化完成"))
            logger.info("优化完成")
            self.finished.emit(self.task.video_path, self.task.output_path)
        except AppError as e:
            logger.exception(f"业务错误: {str(e)}")
            self.error.emit(str(e))
            self.progress.emit(100, self.tr("优化失败"))
        except Exception as e:
            mapped = map_exception(e)
            logger.exception(f"优化失败: {str(mapped)}")
            self.error.emit(str(mapped))
            self.progress.emit(100, self.tr("优化失败"))

    def callback(self, result: Dict):
        self.finished_subtitle_length += len(result)
        # 简单计算当前进度（0-100%）
        progress = min(
            int((self.finished_subtitle_length / self.subtitle_length) * 100), 100
        )
        self.progress.emit(progress, self.tr("{0}% 处理字幕").format(progress))
        self.update.emit(result)

    def stop(self):
        """停止所有处理"""
        try:
            # 先停止优化器
            if hasattr(self, "optimizer"):
                try:
                    self.optimizer.stop()
                except Exception as e:
                    logger.error(f"停止优化器时出错：{str(e)}")

            # 终止线程
            self.terminate()
            # 等待最多3秒
            if not self.wait(3000):
                logger.warning("线程未能在3秒内正常停止")

            # 发送进度信号
            self.progress.emit(100, self.tr("已终止"))

        except Exception as e:
            logger.error(f"停止线程时出错：{str(e)}")
            self.progress.emit(100, self.tr("终止时发生错误"))
