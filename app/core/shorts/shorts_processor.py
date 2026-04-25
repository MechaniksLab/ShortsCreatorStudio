import json
import os
import re
import signal
import subprocess
import tempfile
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter, sleep
from typing import Callable, Dict, List, Optional, Tuple

import openai

from app.config import LOG_PATH
from app.core.bk_asr.asr_data import ASRData
from app.core.utils.logger import setup_logger

logger = setup_logger("shorts_processor")

RENDER_DEBUG_LOG = LOG_PATH / "auto_shorts_render.log"
MAX_HEURISTIC_CANDIDATES = None
MAX_RENDER_CLIPS = None


def _append_render_debug(message: str):
    try:
        LOG_PATH.mkdir(parents=True, exist_ok=True)
        with RENDER_DEBUG_LOG.open("a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {message}\n")
    except Exception:
        pass


@dataclass
class ShortCandidate:
    start_ms: int
    end_ms: int
    score: float
    title: str
    reason: str
    excerpt: str
    speech_ranges: Optional[List[Tuple[int, int]]] = None
    speech_density: float = 0.0
    pause_ratio: float = 0.0
    hook_score: float = 0.0
    novelty_score: float = 0.0
    quality_score: float = 0.0
    quality_grade: str = "C"
    explainability_tags: Optional[List[str]] = None

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    def to_dict(self) -> Dict:
        return asdict(self)


class ShortsProcessor:
    def __init__(
        self,
        min_duration_s: int = 15,
        max_duration_s: int = 60,
        llm_base_url: str = "",
        llm_api_key: str = "",
        llm_model: str = "",
        repeat_similarity_threshold: float = 0.72,
        min_candidates: int = 8,
        max_candidates: int = 40,
        auto_filter_weak_candidates: bool = True,
        auto_filter_profile: str = "balanced",
        interest_threshold_percent: int = 52,
        llm_search_intensity: int = 3,
    ):
        self.min_duration_ms = int(min_duration_s * 1000)
        self.max_duration_ms = int(max_duration_s * 1000)
        self.llm_base_url = (llm_base_url or "").strip()
        self.llm_api_key = (llm_api_key or "").strip()
        self.llm_model = (llm_model or "").strip()
        self.repeat_similarity_threshold = max(0.40, min(0.98, float(repeat_similarity_threshold or 0.72)))
        self.min_candidates = max(1, int(min_candidates or 1))
        self.max_candidates = max(self.min_candidates, int(max_candidates or self.min_candidates))
        self.auto_filter_weak_candidates = bool(auto_filter_weak_candidates)
        profile = str(auto_filter_profile or "balanced").strip().lower()
        self.auto_filter_profile = profile if profile in {"soft", "balanced", "strict"} else "balanced"
        self.interest_threshold_percent = max(30, min(95, int(interest_threshold_percent or 52)))
        self.llm_search_intensity = max(1, min(5, int(llm_search_intensity or 3)))

    def find_candidates(self, asr_data: ASRData, progress_cb: Optional[Callable] = None) -> List[ShortCandidate]:
        if progress_cb:
            progress_cb(5, "Поиск интересных фрагментов...")

        llm_candidates: List[ShortCandidate] = []
        if self._llm_ready():
            if progress_cb:
                progress_cb(12, "AI Enterprise: семантический анализ эпизодов...")
            llm_candidates = self._build_enterprise_llm_candidates(asr_data, progress_cb=progress_cb)

        heuristic_candidates = self._build_heuristic_candidates(asr_data)
        if llm_candidates:
            # Не полагаемся только на LLM: домешиваем эвристику,
            # чтобы не получать 1-2 скучных кандидата на всём диапазоне.
            candidates = llm_candidates + heuristic_candidates[:260]
            candidates.sort(key=lambda x: x.score, reverse=True)
            candidates = self._deduplicate(candidates)
        else:
            candidates = heuristic_candidates

        # Если после всех фильтров кандидатов всё ещё мало,
        # делаем расширенный проход с более мягкими порогами.
        if len(candidates) < self.min_candidates:
            expanded = self._build_heuristic_candidates(asr_data, relaxed=True)
            candidates = self._deduplicate((candidates + expanded) if candidates else expanded)
            candidates.sort(key=lambda x: x.score, reverse=True)

        if progress_cb:
            progress_cb(55, f"Найдено кандидатов (эвристика): {len(candidates)}")

        # Важно: сохраняем пул ДО precision-pass.
        # Иначе при агрессивной фильтрации ниже по цепочке
        # не получится добрать до min_candidates даже если изначально
        # кандидатов было достаточно.
        pre_precision_candidates = list(candidates)

        # Pass-2 (precision): локально ужесточаем качество до LLM rerank.
        candidates = self._precision_pass(candidates)

        reranked = self._try_llm_rerank(candidates)
        reranked = self._diversify_by_timeline(reranked)

        interested_filtered = self._apply_interest_threshold(reranked)
        min_safe_interest = max(3, int(self.min_candidates * 0.55))
        if len(interested_filtered) >= min_safe_interest:
            reranked = interested_filtered
        elif interested_filtered:
            # Если порог интереса оказался слишком строгим —
            # делаем мягкий микс до min_candidates.
            reserve = sorted(reranked, key=lambda x: x.quality_score, reverse=True)
            used = {(c.start_ms, c.end_ms) for c in interested_filtered}
            mixed = list(interested_filtered)
            for c in reserve:
                key = (c.start_ms, c.end_ms)
                if key in used:
                    continue
                mixed.append(c)
                used.add(key)
                if len(mixed) >= self.min_candidates:
                    break
            reranked = mixed

        if self.auto_filter_weak_candidates:
            weak_filtered = self._auto_filter_weak_candidates(reranked)
            min_safe = max(3, int(self.min_candidates * 0.55))
            if len(weak_filtered) >= min_safe:
                reranked = weak_filtered
            elif weak_filtered:
                # Если фильтр слишком агрессивен — оставляем best effort микс.
                reserve = sorted(reranked, key=lambda x: x.quality_score, reverse=True)
                used = {(c.start_ms, c.end_ms) for c in weak_filtered}
                mixed = list(weak_filtered)
                for c in reserve:
                    key = (c.start_ms, c.end_ms)
                    if key in used:
                        continue
                    mixed.append(c)
                    used.add(key)
                    if len(mixed) >= self.min_candidates:
                        break
                reranked = mixed

        if len(reranked) < self.min_candidates:
            # Добираем до минимального числа из общего пула, сохраняя порядок по score
            reserve_pool = sorted(pre_precision_candidates, key=lambda x: x.score, reverse=True)
            seen = {(c.start_ms, c.end_ms) for c in reranked}
            for c in reserve_pool:
                key = (c.start_ms, c.end_ms)
                if key in seen:
                    continue
                reranked.append(c)
                seen.add(key)
                if len(reranked) >= self.min_candidates:
                    break

        # Если материала хватает, не "прижимаем" итог к минимуму:
        # стараемся отдать больше интересных моментов в пределах min..max.
        target_count = self._resolve_target_candidate_count(pre_precision_candidates, reranked)
        if len(reranked) < target_count:
            reserve_pool = sorted(pre_precision_candidates, key=lambda x: x.score, reverse=True)
            seen = {(c.start_ms, c.end_ms) for c in reranked}
            for c in reserve_pool:
                key = (c.start_ms, c.end_ms)
                if key in seen:
                    continue
                reranked.append(c)
                seen.add(key)
                if len(reranked) >= target_count:
                    break

        if len(reranked) > self.max_candidates:
            reranked = reranked[: self.max_candidates]

        if progress_cb:
            progress_cb(85, f"Кандидаты после ранжирования: {len(reranked)}")
        final_candidates = list(reranked)
        return final_candidates

    def _resolve_target_candidate_count(
        self,
        base_pool: List[ShortCandidate],
        current: List[ShortCandidate],
    ) -> int:
        """Определяет целевой размер финального списка в диапазоне [min_candidates, max_candidates]."""
        min_c = int(self.min_candidates)
        max_c = int(self.max_candidates)
        if max_c <= min_c:
            return min_c
        if not base_pool:
            return min_c

        # Оцениваем "глубину" пула: сколько кандидатов около/выше порога интереса.
        soft_threshold = max(30.0, float(self.interest_threshold_percent) - 6.0)
        strong_count = sum(
            1
            for c in base_pool
            if float(getattr(c, "quality_score", 0.0) or 0.0) >= soft_threshold
        )
        strong_ratio = strong_count / max(1, len(base_pool))

        # Интенсивность поиска тоже влияет на ожидаемую плотность выдачи.
        intensity_ratio = (int(self.llm_search_intensity) - 1) / 4.0
        intensity_ratio = max(0.0, min(1.0, intensity_ratio))

        # 0.35..1.0: при хорошем пуле и высокой интенсивности тянемся ближе к max.
        fill_ratio = 0.35 + 0.40 * strong_ratio + 0.25 * intensity_ratio
        fill_ratio = max(0.35, min(1.0, fill_ratio))

        target = min_c + int(round((max_c - min_c) * fill_ratio))
        target = max(min_c, min(max_c, target))

        # Если в текущем списке уже больше — не занижаем искусственно.
        return max(target, min(len(current), max_c))

    def _llm_ready(self) -> bool:
        return bool(self.llm_model and self.llm_base_url and self.llm_api_key)

    @staticmethod
    def _diversify_by_timeline(candidates: List[ShortCandidate]) -> List[ShortCandidate]:
        """Сохраняем разнообразие по таймлайну, чтобы список не состоял из соседних однотипных кусков."""
        if not candidates:
            return []
        # Сначала оставляем в каждой 20-сек корзине лучший кандидат.
        bucket_best: Dict[int, ShortCandidate] = {}
        for c in sorted(candidates, key=lambda x: x.score, reverse=True):
            bucket = int(c.start_ms // 20000)
            prev = bucket_best.get(bucket)
            if prev is None or c.score > prev.score:
                bucket_best[bucket] = c

        seeded = sorted(bucket_best.values(), key=lambda x: x.score, reverse=True)
        used = {(c.start_ms, c.end_ms) for c in seeded}

        # Затем добавляем оставшиеся сильные, но не слишком близкие к уже выбранным.
        extra: List[ShortCandidate] = []
        for c in sorted(candidates, key=lambda x: x.score, reverse=True):
            key = (c.start_ms, c.end_ms)
            if key in used:
                continue
            too_close = any(abs(c.start_ms - s.start_ms) < 10000 for s in seeded[:120])
            if too_close and c.score < 78:
                continue
            extra.append(c)

        out = seeded + extra
        out.sort(key=lambda x: x.score, reverse=True)
        return out

    def _precision_pass(self, candidates: List[ShortCandidate]) -> List[ShortCandidate]:
        if not candidates:
            return []
        passed: List[ShortCandidate] = []
        for c in candidates:
            quality = float(getattr(c, "quality_score", 0.0) or 0.0)
            hook = float(getattr(c, "hook_score", 0.0) or 0.0)
            pause_ratio = float(getattr(c, "pause_ratio", 1.0) or 1.0)
            if quality < 40 and hook < 3.0:
                continue
            if pause_ratio > 0.56 and quality < 55:
                continue
            passed.append(c)
        if len(passed) < max(3, int(self.min_candidates * 0.6)):
            return candidates
        return passed

    def _auto_filter_weak_candidates(self, candidates: List[ShortCandidate]) -> List[ShortCandidate]:
        if not candidates:
            return []

        if self.auto_filter_profile == "soft":
            min_quality = 44.0
            min_hook = 2.8
            max_pause = 0.56
            min_density = 0.62
            min_quality_for_low_density = 58.0
        elif self.auto_filter_profile == "strict":
            min_quality = 54.0
            min_hook = 3.8
            max_pause = 0.44
            min_density = 0.78
            min_quality_for_low_density = 68.0
        else:
            min_quality = 48.0
            min_hook = 3.2
            max_pause = 0.48
            min_density = 0.70
            min_quality_for_low_density = 62.0

        out: List[ShortCandidate] = []
        for c in candidates:
            quality = float(getattr(c, "quality_score", 0.0) or 0.0)
            hook = float(getattr(c, "hook_score", 0.0) or 0.0)
            pause_ratio = float(getattr(c, "pause_ratio", 1.0) or 1.0)
            speech_density = float(getattr(c, "speech_density", 0.0) or 0.0)
            if quality < min_quality:
                continue
            if hook < min_hook:
                continue
            if pause_ratio > max_pause:
                continue
            if speech_density < min_density and quality < min_quality_for_low_density:
                continue
            out.append(c)
        return out

    def _apply_interest_threshold(self, candidates: List[ShortCandidate]) -> List[ShortCandidate]:
        if not candidates:
            return []
        threshold = float(self.interest_threshold_percent)
        return [c for c in candidates if float(getattr(c, "quality_score", 0.0) or 0.0) >= threshold]

    def _build_enterprise_llm_candidates(
        self,
        asr_data: ASRData,
        progress_cb: Optional[Callable] = None,
    ) -> List[ShortCandidate]:
        segments = [s for s in asr_data.segments if s.text and s.text.strip()]
        if not segments:
            return []

        intensity = int(self.llm_search_intensity)
        packet_size_map = {1: 180, 2: 160, 3: 140, 4: 120, 5: 100}
        overlap_map = {1: 42, 2: 38, 3: 35, 4: 30, 5: 26}
        packet_size = packet_size_map.get(intensity, 140)
        overlap = overlap_map.get(intensity, 35)

        packets = self._build_segment_packets(segments, packet_size=packet_size, overlap=overlap)
        all_candidates: List[ShortCandidate] = []

        for i, (start_idx, end_idx, packet_segments) in enumerate(packets, 1):
            try:
                packet_candidates = self._llm_extract_candidates_from_packet(
                    packet_segments,
                    global_start_idx=start_idx,
                    packet_index=i,
                    total_packets=len(packets),
                )
                all_candidates.extend(packet_candidates)
            except Exception as e:
                logger.warning("Enterprise packet parse failed: %s", e)

            if progress_cb:
                p = 12 + int((i / max(1, len(packets))) * 36)
                progress_cb(min(52, p), f"AI Enterprise: пакет {i}/{len(packets)}")

        all_candidates.sort(key=lambda x: x.score, reverse=True)
        return self._deduplicate(all_candidates)

    @staticmethod
    def _build_segment_packets(segments: List, packet_size: int, overlap: int) -> List[Tuple[int, int, List]]:
        packets: List[Tuple[int, int, List]] = []
        if not segments:
            return packets
        step = max(1, packet_size - overlap)
        start = 0
        n = len(segments)
        while start < n:
            end = min(n, start + packet_size)
            packets.append((start, end - 1, segments[start:end]))
            if end == n:
                break
            start += step
        return packets

    def _llm_extract_candidates_from_packet(
        self,
        packet_segments: List,
        global_start_idx: int,
        packet_index: int = 1,
        total_packets: int = 1,
    ) -> List[ShortCandidate]:
        client = openai.OpenAI(base_url=self.llm_base_url, api_key=self.llm_api_key)

        intensity = int(self.llm_search_intensity)
        # Защита от переполнения контекста (особенно для локальных моделей в LM Studio).
        max_rows_map = {1: 64, 2: 72, 3: 84, 4: 96, 5: 110}
        max_text_chars_map = {1: 88, 2: 96, 3: 108, 4: 120, 5: 132}
        max_prompt_chars_map = {1: 6200, 2: 7000, 3: 7800, 4: 8600, 5: 9400}
        max_rows = max_rows_map.get(intensity, 84)
        max_text_chars = max_text_chars_map.get(intensity, 108)
        max_prompt_chars = max_prompt_chars_map.get(intensity, 7800)

        rows = []
        prompt_chars_used = 0
        for local_idx, seg in enumerate(packet_segments):
            if len(rows) >= max_rows:
                break
            abs_idx = global_start_idx + local_idx
            text = self._shorten((seg.text or "").strip(), max_text_chars)
            row = {
                "idx": abs_idx,
                "text": text,
            }
            # Ограничиваем общий размер user-пейлоада в символах.
            row_chars = len(json.dumps(row, ensure_ascii=False))
            if rows and (prompt_chars_used + row_chars) > max_prompt_chars:
                break
            prompt_chars_used += row_chars
            rows.append(
                row
            )

        min_s = max(8, int(self.min_duration_ms / 1000))
        max_s = max(min_s + 5, int(self.max_duration_ms / 1000))
        min_items = max(4, min(18, 4 + intensity * 2))
        max_items = max(min_items + 2, min(28, 10 + intensity * 4))
        max_output_tokens = max(220, min(640, 200 + max_items * 16))

        system = (
            "Ты enterprise-редактор YouTube Shorts. Найди лучшие моменты удержания. "
            "Критерии: hook в первые 2-5 секунд, эмоция, конфликт/неожиданность, панчлайн, кульминация, потенциал для шеринга. "
            "Верни СТРОГО JSON: "
            "{\"items\":[{\"start_idx\":int,\"end_idx\":int,\"score\":0-100,\"title\":str,\"reason\":str,"
            "\"hook\":0-10,\"emotion\":0-10,\"novelty\":0-10,\"shareability\":0-10}]}. "
            f"Длительность каждого фрагмента {min_s}-{max_s} секунд. "
            f"Верни от {min_items} до {max_items} кандидатов на этот пакет. "
            "Не придумывай таймкоды, используй только переданные idx."
        )
        user = (
            f"Пакет {packet_index}/{max(1, total_packets)}. "
            "Нужны разнообразные, неповторяющиеся моменты по смыслу и таймингу.\n"
            f"Сегменты:\n{json.dumps(rows, ensure_ascii=False)}"
        )

        rsp = client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.15,
            max_tokens=max_output_tokens,
            timeout=120,
        )
        content = rsp.choices[0].message.content or ""
        data = self._extract_json(content)
        items = data.get("items", []) if isinstance(data, dict) else []

        by_idx = {global_start_idx + i: seg for i, seg in enumerate(packet_segments)}
        result: List[ShortCandidate] = []
        for item in items:
            try:
                s_idx = int(item.get("start_idx"))
                e_idx = int(item.get("end_idx"))
                if s_idx not in by_idx or e_idx not in by_idx:
                    continue
                if e_idx < s_idx:
                    s_idx, e_idx = e_idx, s_idx

                start_ms = int(by_idx[s_idx].start_time)
                end_ms = int(by_idx[e_idx].end_time)
                duration = end_ms - start_ms
                if duration < self.min_duration_ms or duration > self.max_duration_ms:
                    continue

                base_score = float(item.get("score", 0))
                hook = float(item.get("hook", 0))
                emotion = float(item.get("emotion", 0))
                novelty = float(item.get("novelty", 0))
                shareability = float(item.get("shareability", 0))
                blended = min(100.0, 0.72 * base_score + 2.8 * (hook + emotion + novelty + shareability))

                excerpt = " ".join((by_idx[k].text or "").strip() for k in range(s_idx, e_idx + 1))
                speech_ranges = self._build_speech_ranges_from_segments([by_idx[k] for k in range(s_idx, e_idx + 1)])
                duration_s = max(0.8, duration / 1000.0)
                token_count = self._count_tokens(excerpt)
                speech_density = token_count / duration_s
                kept_ms = sum(max(0, b - a) for a, b in (speech_ranges or []))
                pause_ratio = max(0.0, min(1.0, 1.0 - (kept_ms / max(1, duration))))
                hook_score = max(0.0, min(10.0, float(hook)))
                novelty_score = max(0.0, min(10.0, float(novelty)))
                quality_score = self._compute_quality_score(
                    base_score=blended,
                    speech_density=speech_density,
                    pause_ratio=pause_ratio,
                    hook_score=hook_score,
                    novelty_score=novelty_score,
                )
                quality_grade = self._quality_grade(quality_score)
                explainability_tags = self._build_explainability_tags(
                    speech_density=speech_density,
                    pause_ratio=pause_ratio,
                    hook_score=hook_score,
                    novelty_score=novelty_score,
                    quality_grade=quality_grade,
                )
                result.append(
                    ShortCandidate(
                        start_ms=start_ms,
                        end_ms=end_ms,
                        score=round(blended, 2),
                        title=str(item.get("title", "")).strip() or self._build_title(excerpt),
                        reason=str(item.get("reason", "")).strip() or "AI Enterprise selection",
                        excerpt=self._shorten(excerpt, 220),
                        speech_ranges=speech_ranges,
                        speech_density=round(speech_density, 3),
                        pause_ratio=round(pause_ratio, 3),
                        hook_score=round(hook_score, 2),
                        novelty_score=round(novelty_score, 2),
                        quality_score=round(quality_score, 2),
                        quality_grade=quality_grade,
                        explainability_tags=explainability_tags,
                    )
                )
            except Exception:
                continue

        return result

    def _build_heuristic_candidates(self, asr_data: ASRData, relaxed: bool = False) -> List[ShortCandidate]:
        segments = [s for s in asr_data.segments if s.text and s.text.strip()]
        if not segments:
            return []

        prepared = []
        for seg in segments:
            clean = self._normalize_text(seg.text)
            token_count = self._count_tokens(clean)
            if token_count <= 0:
                continue
            filler_hits = self._count_filler_hits(clean)
            prepared.append(
                {
                    "seg": seg,
                    "text": seg.text.strip(),
                    "clean": clean,
                    "token_count": token_count,
                    "filler_hits": filler_hits,
                }
            )

        if not prepared:
            return []

        windows: List[ShortCandidate] = []
        n_prepared = len(prepared)
        if n_prepared > 900:
            start_step = 2
        else:
            start_step = 1

        if relaxed:
            start_step = max(1, start_step - 1)

        for i in range(0, n_prepared, start_step):
            start = prepared[i]["seg"].start_time
            text_parts = []
            end = start
            punch = 0.0
            speech_ms = 0
            pause_ms = 0
            token_sum = 0
            filler_sum = 0
            prev_end = start

            for j in range(i, len(prepared)):
                rec = prepared[j]
                seg = rec["seg"]
                if j - i > 180:
                    break
                end = seg.end_time
                text_parts.append(rec["text"])
                speech_ms += max(1, int(seg.end_time - seg.start_time))
                token_sum += int(rec["token_count"])
                filler_sum += int(rec["filler_hits"])
                if j > i:
                    pause_ms += max(0, int(seg.start_time - prev_end))
                prev_end = seg.end_time

                duration = end - start
                if duration > self.max_duration_ms:
                    break
                if duration < self.min_duration_ms:
                    continue

                speech_ratio = max(0.0, min(1.0, speech_ms / max(1, duration)))
                pause_ratio = max(0.0, min(1.0, pause_ms / max(1, duration)))
                filler_ratio = max(0.0, min(1.0, filler_sum / max(1, token_sum)))

                # Отсекаем "пустые"/скучные фрагменты: мало речи, много пауз, много филлеров.
                if relaxed:
                    skip_window = speech_ratio < 0.44 or pause_ratio > 0.42 or filler_ratio > 0.56
                else:
                    skip_window = speech_ratio < 0.54 or pause_ratio > 0.30 or filler_ratio > 0.45
                if skip_window:
                    continue

                joined = " ".join(text_parts)
                score = self._heuristic_score(
                    joined,
                    duration,
                    speech_ratio=speech_ratio,
                    pause_ratio=pause_ratio,
                    filler_ratio=filler_ratio,
                )
                if score <= 0:
                    continue
                if score < (36 if relaxed else 43):
                    continue

                duration_s = max(0.8, duration / 1000.0)
                speech_density = token_sum / duration_s
                unique_ratio = len(set(re.findall(r"[\w\u4e00-\u9fff]+", joined.lower()))) / max(1, token_sum)
                hook_score = self._estimate_hook_score(joined)
                novelty_score = self._estimate_novelty_score(joined, unique_ratio)
                quality_score = self._compute_quality_score(
                    base_score=score,
                    speech_density=speech_density,
                    pause_ratio=pause_ratio,
                    hook_score=hook_score,
                    novelty_score=novelty_score,
                )
                quality_grade = self._quality_grade(quality_score)
                explainability_tags = self._build_explainability_tags(
                    speech_density=speech_density,
                    pause_ratio=pause_ratio,
                    hook_score=hook_score,
                    novelty_score=novelty_score,
                    quality_grade=quality_grade,
                )

                punch = max(punch, score)
                excerpt = self._shorten(joined, 220)
                windows.append(
                    ShortCandidate(
                        start_ms=start,
                        end_ms=end,
                        score=score,
                        title=self._build_title(joined),
                        reason=self._build_reason(joined, score),
                        excerpt=excerpt,
                        speech_ranges=self._build_speech_ranges_from_segments(
                            [prepared[k]["seg"] for k in range(i, j + 1)]
                        ),
                        speech_density=round(speech_density, 3),
                        pause_ratio=round(pause_ratio, 3),
                        hook_score=round(hook_score, 2),
                        novelty_score=round(novelty_score, 2),
                        quality_score=round(quality_score, 2),
                        quality_grade=quality_grade,
                        explainability_tags=explainability_tags,
                    )
                )

                # если уже очень сильный фрагмент — можно не расширять дальше
                if punch > (98 if relaxed else 96):
                    break

        windows.sort(key=lambda x: x.score, reverse=True)
        return self._deduplicate(windows)

    def _heuristic_score(
        self,
        text: str,
        duration_ms: int,
        speech_ratio: float = 1.0,
        pause_ratio: float = 0.0,
        filler_ratio: float = 0.0,
    ) -> float:
        txt = (text or "").strip()
        if not txt:
            return 0

        words = txt.split()
        token_count = max(len(words), len(re.findall(r"[\u4e00-\u9fff]", txt)))
        if token_count == 0:
            return 0

        lowered = txt.lower()
        unique_ratio = len(set(re.findall(r"[\w\u4e00-\u9fff]+", lowered))) / max(1, token_count)
        if token_count >= 18 and unique_ratio < 0.42:
            return 0

        duration_s = max(duration_ms / 1000.0, 1.0)
        density = token_count / duration_s

        funny_kw = [
            "ха", "хаха", "lol", "смеш", "угар", "шок", "жесть", "пипец", "imagine", "wtf",
            "рж", "ору", "мем", "юмор", "lmao",
        ]
        hook_kw = [
            "смотри", "прикол", "история", "секрет", "факт", "топ", "не поверишь", "важно", "лайфхак",
            "подожди", "сейчас", "вот", "чел", "дальше",
        ]
        hype_kw = [
            "имба", "жёстко", "клатч", "тащит", "топ", "легенд", "финал", "камбэк",
            "нокаут", "ваншот", "ульт", "разнос", "фейл", "поворот", "драма", "пранк",
        ]

        funny_hits = sum(1 for k in funny_kw if k in lowered)
        hook_hits = sum(1 for k in hook_kw if k in lowered)
        hype_hits = sum(1 for k in hype_kw if k in lowered)
        punct_bonus = txt.count("!") * 2 + txt.count("?") * 1.5
        caps_bonus = min(8, len(re.findall(r"[A-ZА-ЯЁ]{3,}", txt)) * 2)
        digit_bonus = 4 if re.search(r"\d", txt) else 0
        quote_bonus = 4 if any(sym in txt for sym in ["—", "«", "»", "\"", "'"]) else 0
        early_text = " ".join(words[:10]).lower()
        early_hook_bonus = 7 if any(k in early_text for k in hook_kw + ["но", "вдруг", "прикинь", "короче"]) else 0

        duration_target_bonus = 14 if 16 <= duration_s <= 42 else (6 if 12 <= duration_s <= 55 else 0)
        density_bonus = max(0, min(22, (density - 1.45) * 10))
        anti_wall_penalty = -14 if density < 0.8 else 0
        pause_penalty = -18 * max(0.0, min(1.0, pause_ratio))
        filler_penalty = -22 * max(0.0, min(1.0, filler_ratio))
        low_speech_penalty = -20 * max(0.0, min(1.0, 1.0 - speech_ratio))
        diversity_bonus = max(0.0, min(8.0, (unique_ratio - 0.42) * 20))

        score = (
            34
            + density_bonus
            + duration_target_bonus
            + funny_hits * 8
            + hook_hits * 5
            + hype_hits * 6
            + punct_bonus
            + caps_bonus
            + digit_bonus
            + quote_bonus
            + early_hook_bonus
            + diversity_bonus
            + anti_wall_penalty
            + pause_penalty
            + filler_penalty
            + low_speech_penalty
        )
        if funny_hits == 0 and hook_hits == 0 and hype_hits == 0 and punct_bonus <= 0 and density < 1.2:
            score -= 8
        return round(min(100.0, score), 2)

    @staticmethod
    def _estimate_hook_score(text: str) -> float:
        txt = (text or "").strip()
        if not txt:
            return 0.0
        low = txt.lower()
        words = re.findall(r"[\w\u4e00-\u9fff]+", txt, flags=re.UNICODE)
        early_words = [w.lower() for w in words[:12]]
        early_text = " ".join(early_words)

        hook_kw = [
            "смотри", "прикол", "не поверишь", "вот", "сейчас", "история", "секрет", "факт", "подожди",
            "короче", "imagine", "wait", "look", "secret", "fact",
        ]
        filler_starts = ["ну", "эм", "ээ", "типа", "как бы", "uh", "um"]

        score = 2.8
        score += min(4.0, sum(1.2 for k in hook_kw if k in early_text))
        score += min(1.8, txt.count("!") * 0.35 + txt.count("?") * 0.25)
        if any(early_text.startswith(f) for f in filler_starts):
            score -= 1.6
        if len(words) >= 6:
            score += 0.8
        if len(words[:3]) < 2:
            score -= 1.0
        return round(max(0.0, min(10.0, score)), 2)

    @staticmethod
    def _estimate_novelty_score(text: str, unique_ratio: float) -> float:
        low = (text or "").lower()
        novelty_kw = [
            "внезапно", "неожидан", "поворот", "камбэк", "разрыв", "шок", "впервые", "новый",
            "twist", "unexpected", "plot twist", "suddenly", "new",
        ]
        novelty_hits = sum(1 for k in novelty_kw if k in low)
        score = 3.2 + min(3.5, novelty_hits * 1.25) + max(0.0, min(3.3, (unique_ratio - 0.42) * 10))
        return round(max(0.0, min(10.0, score)), 2)

    @staticmethod
    def _compute_quality_score(
        base_score: float,
        speech_density: float,
        pause_ratio: float,
        hook_score: float,
        novelty_score: float,
    ) -> float:
        density_bonus = max(0.0, min(12.0, (speech_density - 1.0) * 5.2))
        pause_penalty = max(0.0, min(18.0, pause_ratio * 26.0))
        hook_bonus = max(0.0, min(12.0, hook_score * 1.15))
        novelty_bonus = max(0.0, min(9.0, novelty_score * 0.85))
        quality = 0.56 * float(base_score) + density_bonus + hook_bonus + novelty_bonus - pause_penalty
        return round(max(0.0, min(100.0, quality)), 2)

    @staticmethod
    def _quality_grade(quality_score: float) -> str:
        q = float(quality_score or 0.0)
        if q >= 82:
            return "A"
        if q >= 68:
            return "B"
        if q >= 54:
            return "C"
        return "D"

    @staticmethod
    def _build_explainability_tags(
        speech_density: float,
        pause_ratio: float,
        hook_score: float,
        novelty_score: float,
        quality_grade: str,
    ) -> List[str]:
        tags: List[str] = [f"grade:{quality_grade}"]
        if hook_score >= 7.2:
            tags.append("сильный_хук")
        elif hook_score <= 3.0:
            tags.append("слабый_хук")

        if novelty_score >= 6.8:
            tags.append("неожиданность")

        if speech_density >= 1.9:
            tags.append("плотная_речь")
        elif speech_density < 1.0:
            tags.append("разреженная_речь")

        if pause_ratio <= 0.22:
            tags.append("мало_пауз")
        elif pause_ratio > 0.42:
            tags.append("много_пауз")
        return tags

    def _try_llm_rerank(self, candidates: List[ShortCandidate]) -> List[ShortCandidate]:
        if not candidates:
            return []

        if not self._llm_ready():
            return candidates

        try:
            # Rerank делаем компактным, чтобы не выбивать контекст на локальных LLM.
            rerank_top_map = {1: 24, 2: 30, 3: 36, 4: 42, 5: 50}
            base_top = rerank_top_map.get(int(self.llm_search_intensity), 44)
            # Важно: rerank не должен обрезать пул до околомин. значения.
            # Берём как минимум 2x max_candidates (в разумном лимите), чтобы
            # после фильтров оставалась возможность выйти за min_candidates.
            dynamic_top = max(base_top, min(120, int(self.max_candidates) * 2))
            top = candidates[:dynamic_top]
            payload = [
                {
                    "id": i,
                    "start_ms": c.start_ms,
                    "end_ms": c.end_ms,
                    "score": c.score,
                    "excerpt": self._shorten(c.excerpt, 130),
                }
                for i, c in enumerate(top)
            ]
            client = openai.OpenAI(base_url=self.llm_base_url, api_key=self.llm_api_key)
            system = (
                "Ты редактор YouTube Shorts. Выбери самые интересные, смешные, цепляющие моменты с максимальным удержанием. "
                "Отдавай приоритет моментам с хук-фразой, эмоцией, неожиданностью, панчлайном, кульминацией или мемным потенциалом. "
                "Верни JSON: {\"items\":[{\"id\":int,\"boost\":number,\"title\":str,\"reason\":str}]}."
            )
            user = f"Кандидаты: {json.dumps(payload, ensure_ascii=False)}"
            rsp = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                max_tokens=260,
                timeout=90,
            )
            content = rsp.choices[0].message.content or ""
            data = self._extract_json(content)
            items = data.get("items", []) if isinstance(data, dict) else []

            by_id = {i: c for i, c in enumerate(top)}
            reranked: List[ShortCandidate] = []
            used = set()
            for it in items:
                idx = it.get("id")
                if idx not in by_id:
                    continue
                c = by_id[idx]
                c.score = round(min(100.0, c.score + float(it.get("boost", 0))), 2)
                c.quality_score = self._compute_quality_score(
                    base_score=c.score,
                    speech_density=float(getattr(c, "speech_density", 0.0) or 0.0),
                    pause_ratio=float(getattr(c, "pause_ratio", 1.0) or 1.0),
                    hook_score=float(getattr(c, "hook_score", 0.0) or 0.0),
                    novelty_score=float(getattr(c, "novelty_score", 0.0) or 0.0),
                )
                c.quality_grade = self._quality_grade(c.quality_score)
                title = (it.get("title") or "").strip()
                reason = (it.get("reason") or "").strip()
                if title:
                    c.title = title
                if reason:
                    c.reason = reason
                reranked.append(c)
                used.add(idx)

            for i, c in enumerate(top):
                if i not in used:
                    reranked.append(c)

            # Хвост кандидатов вне rerank-окна возвращаем обратно,
            # чтобы этап rerank не терял сильные моменты из общего пула.
            if len(candidates) > len(top):
                reranked.extend(candidates[len(top) :])

            reranked.sort(key=lambda x: x.score, reverse=True)
            return self._deduplicate(reranked)
        except Exception as e:
            logger.warning(f"LLM rerank skipped: {e}")
            return candidates

    def _deduplicate(self, candidates: List[ShortCandidate]) -> List[ShortCandidate]:
        accepted: List[ShortCandidate] = []
        sim_strong = self.repeat_similarity_threshold
        sim_near = min(0.98, sim_strong + 0.14)
        for c in candidates:
            overlap = False
            c_text = f"{c.title} {c.excerpt}".strip()
            c_tokens = self._token_set(c_text)
            c_ngrams = self._char_ngram_set(c_text)
            c_bucket = int(c.start_ms // 20000)
            c_mid = int((c.start_ms + c.end_ms) / 2)
            for a in accepted:
                inter = max(0, min(c.end_ms, a.end_ms) - max(c.start_ms, a.start_ms))
                short = max(1, min(c.duration_ms, a.duration_ms))
                long = max(1, max(c.duration_ms, a.duration_ms))
                iou = inter / max(1, (c.duration_ms + a.duration_ms - inter))
                temporal_close = abs(c.start_ms - a.start_ms) < 3500 or abs(c.end_ms - a.end_ms) < 3500
                a_text = f"{a.title} {a.excerpt}".strip()
                a_tokens = self._token_set(a_text)
                a_ngrams = self._char_ngram_set(a_text)
                text_sim = max(self._jaccard(c_tokens, a_tokens), self._jaccard(c_ngrams, a_ngrams))
                mid_close = abs(c_mid - int((a.start_ms + a.end_ms) / 2)) < 18000
                same_bucket = c_bucket == int(a.start_ms // 20000)

                if inter / short > 0.72:
                    overlap = True
                    break
                if iou > 0.50 and text_sim > sim_strong:
                    overlap = True
                    break
                if temporal_close and text_sim > sim_near and inter / long > 0.22:
                    overlap = True
                    break
                # Для борьбы с повторяющимися моментами: близкие по времени и смыслу
                # фрагменты считаем дублями даже при небольшом перекрытии.
                if mid_close and text_sim > max(0.52, sim_strong - 0.10) and inter / long > 0.12:
                    overlap = True
                    break
                if same_bucket and text_sim > max(0.58, sim_strong - 0.08):
                    overlap = True
                    break
            if not overlap:
                accepted.append(c)
        return accepted

    @staticmethod
    def _normalize_text(text: str) -> str:
        t = (text or "").lower().strip()
        t = re.sub(r"\s+", " ", t)
        return t

    @staticmethod
    def _count_tokens(text: str) -> int:
        if not text:
            return 0
        words = re.findall(r"[\w\u4e00-\u9fff]+", text, flags=re.UNICODE)
        return len(words)

    @staticmethod
    def _count_filler_hits(text: str) -> int:
        if not text:
            return 0
        filler_kw = [
            "эм", "ээ", "мм", "ну", "типа", "короче", "как бы", "блин", "это самое",
            "uh", "um", "erm", "hmm", "like", "you know",
        ]
        low = text.lower()
        return sum(1 for k in filler_kw if k in low)

    @staticmethod
    def _token_set(text: str) -> set:
        if not text:
            return set()
        tokens = re.findall(r"[\w\u4e00-\u9fff]{2,}", text.lower(), flags=re.UNICODE)
        return set(tokens)

    @staticmethod
    def _char_ngram_set(text: str, n: int = 3) -> set:
        if not text:
            return set()
        t = re.sub(r"\s+", " ", text.lower()).strip()
        if len(t) < n:
            return {t} if t else set()
        return {t[i : i + n] for i in range(0, len(t) - n + 1)}

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        uni = len(a | b)
        return inter / max(1, uni)

    @staticmethod
    def _build_speech_ranges_from_segments(segments: List) -> List[Tuple[int, int]]:
        if not segments:
            return []
        ranges: List[Tuple[int, int]] = []
        for seg in segments:
            s = int(getattr(seg, "start_time", 0) or 0)
            e = int(getattr(seg, "end_time", 0) or 0)
            if e - s < 120:
                continue

            word_ts = getattr(seg, "word_timestamps", None) or []
            word_ranges: List[Tuple[int, int]] = []
            if isinstance(word_ts, list):
                for w in word_ts:
                    try:
                        ws = int(float(w.get("start_time", s)))
                        we = int(float(w.get("end_time", ws + 60)))
                    except Exception:
                        continue
                    ws = max(s, ws)
                    we = min(e, we)
                    if we - ws >= 80:
                        word_ranges.append((ws, we))

            if word_ranges:
                ranges.extend(word_ranges)
            else:
                # Без word timestamps немного сжимаем сегмент,
                # чтобы не тащить хвосты тишины по краям фраз.
                dur = e - s
                shrink = min(180, max(40, int(dur * 0.08)))
                ns = s + shrink
                ne = e - shrink
                if ne - ns >= 120:
                    ranges.append((ns, ne))
                else:
                    ranges.append((s, e))
        if not ranges:
            return []

        ranges.sort(key=lambda x: x[0])
        merged: List[Tuple[int, int]] = [ranges[0]]
        for s, e in ranges[1:]:
            ps, pe = merged[-1]
            if s - pe <= 140:
                merged[-1] = (ps, max(pe, e))
            else:
                merged.append((s, e))
        return merged

    @staticmethod
    def _extract_json(text: str) -> Dict:
        text = (text or "").strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", text)
            if not m:
                return {}
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}

    @staticmethod
    def _shorten(text: str, max_len: int) -> str:
        return text if len(text) <= max_len else text[: max_len - 1] + "…"

    @staticmethod
    def _build_title(text: str) -> str:
        txt = re.sub(r"\s+", " ", text).strip()
        return ShortsProcessor._shorten(txt, 72)

    @staticmethod
    def _build_reason(text: str, score: float) -> str:
        t = text.lower()
        tags = []
        if "!" in t or "?" in t:
            tags.append("эмоциональная подача")
        if any(k in t for k in ["смеш", "ха", "угар", "lol"]):
            tags.append("юмор")
        if any(k in t for k in ["факт", "секрет", "история", "прикол", "шок"]):
            tags.append("цепляющий хук")
        if not tags:
            tags.append("плотная речь")
        return f"Score {score}: " + ", ".join(tags)


def render_shorts(
    input_video: str,
    candidates: List[ShortCandidate],
    output_dir: str,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    vertical_resolution: str = "1080x1920",
    layout_template: Optional[Dict] = None,
    render_backend: str = "auto",
    render_options: Optional[Dict] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> List[str]:
    backend = (render_backend or "auto").strip().lower()
    if backend not in {"auto", "cpu", "gpu", "cuda"}:
        backend = "auto"
    _append_render_debug(
        f"START render_shorts input={input_video} candidates={len(candidates)} output_dir={output_dir} mode=single backend={backend}"
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_w, out_h = vertical_resolution.split("x")
    out_w_i, out_h_i = int(out_w), int(out_h)
    render_options = render_options or {}

    def _opt_int(key: str, default: int, min_v: int, max_v: int) -> int:
        try:
            val = int(render_options.get(key, default))
        except Exception:
            val = int(default)
        return max(min_v, min(max_v, val))

    clip_head_pad_ms = _opt_int("clip_head_pad_ms", 120, 0, 2500)
    clip_tail_pad_ms = _opt_int("clip_tail_pad_ms", 420, 0, 3500)
    speech_pre_pad_ms = _opt_int("speech_pre_pad_ms", 220, 0, 1800)
    speech_post_pad_ms = _opt_int("speech_post_pad_ms", 320, 0, 2200)
    speech_merge_gap_ms = _opt_int("speech_merge_gap_ms", 260, 60, 2200)
    speech_min_coverage_percent = _opt_int("speech_min_coverage_percent", 74, 30, 100)
    speech_min_coverage_ratio = speech_min_coverage_percent / 100.0

    def _even_size(v: int) -> int:
        v = max(2, int(v))
        return v if v % 2 == 0 else v - 1

    def _safe_filename(name: str, max_len: int = 72) -> str:
        raw = (name or "").strip()
        if not raw:
            return "без_названия"
        # Windows-safe: убираем запрещённые символы и управляющие коды,
        # но оставляем кириллицу для понятных русских имён.
        raw = re.sub(r'[<>:"/\\|?*]+', " ", raw)
        raw = re.sub(r"[\x00-\x1F]", "", raw)
        raw = re.sub(r"\s+", " ", raw).strip(" .")
        if not raw:
            raw = "без_названия"
        if len(raw) > max_len:
            raw = raw[:max_len].rstrip(" .")
        return raw

    def _fit_filename_to_path_limit(parent: Path, filename: str, max_path_len: int = 240) -> str:
        """Ограничивает имя файла, чтобы полный путь не превышал безопасную длину на Windows."""
        ext = Path(filename).suffix
        stem = Path(filename).stem
        full = str(parent / filename)
        if len(full) <= max_path_len and len(filename) <= 180:
            return filename

        max_stem_by_path = max(8, max_path_len - len(str(parent)) - 1 - len(ext))
        max_stem_by_name = max(8, 180 - len(ext))
        max_stem = max(8, min(max_stem_by_path, max_stem_by_name))
        digest = hashlib.md5(stem.encode("utf-8", errors="ignore")).hexdigest()[:8]
        head_len = max(4, max_stem - 9)
        short_stem = stem[:head_len].rstrip(" ._") or "шорт"
        return f"{short_stem}_{digest}{ext}"

    def _safe_int(v, default=0):
        try:
            return int(v)
        except Exception:
            return default

    use_dual = bool(layout_template and layout_template.get("enabled"))
    target_w_i, target_h_i = out_w_i, out_h_i

    sx = target_w_i / max(1, out_w_i)
    sy = target_h_i / max(1, out_h_i)

    def _scale_x(v: int) -> int:
        return max(0, int(round(v * sx)))

    def _scale_y(v: int) -> int:
        return max(0, int(round(v * sy)))

    def _probe_video_meta(path: str) -> Tuple[int, int, float]:
        try:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ]
            p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if p.returncode == 0:
                lines = [x.strip() for x in (p.stdout or "").splitlines() if x.strip()]
                if len(lines) >= 3:
                    w = max(2, int(lines[0]))
                    h = max(2, int(lines[1]))
                    fps_raw = lines[2]
                    fps = 30.0
                    if "/" in fps_raw:
                        a, b = fps_raw.split("/", 1)
                        fps = float(a) / max(1.0, float(b))
                    else:
                        fps = float(fps_raw)
                    return w, h, max(1.0, fps)
        except Exception:
            pass
        return 1920, 1080, 30.0

    src_w_i, src_h_i, src_fps = _probe_video_meta(input_video)

    resolution_mode = str(render_options.get("resolution_mode", "fixed") or "fixed").strip().lower()
    resolution_raw = str(render_options.get("resolution", f"{out_w_i}x{out_h_i}") or f"{out_w_i}x{out_h_i}").strip().lower()
    if resolution_mode == "source" or resolution_raw == "source":
        target_w_i, target_h_i = int(src_w_i), int(src_h_i)
    else:
        try:
            rw, rh = resolution_raw.replace("×", "x").split("x", 1)
            rw_i, rh_i = int(rw), int(rh)
            if rw_i >= 2 and rh_i >= 2:
                target_w_i, target_h_i = rw_i, rh_i
        except Exception:
            target_w_i, target_h_i = out_w_i, out_h_i

    # Для yuv420p/большинства кодеков размеры должны быть чётными,
    # иначе возможны ошибки рендера и файлы 0 байт.
    target_w_i = _even_size(target_w_i)
    target_h_i = _even_size(target_h_i)

    sx = target_w_i / max(1, out_w_i)
    sy = target_h_i / max(1, out_h_i)

    def _layer_fx_tail(layer_cfg: Dict) -> str:
        fx = layer_cfg if isinstance(layer_cfg, dict) else {}
        brightness = float(fx.get("brightness", 0.0) or 0.0)
        contrast = float(fx.get("contrast", 1.0) or 1.0)
        saturation = float(fx.get("saturation", 1.0) or 1.0)
        gamma = float(fx.get("gamma", 1.0) or 1.0)
        temperature = float(fx.get("temperature", 0.0) or 0.0)

        parts = []
        if (
            abs(brightness) > 1e-6
            or abs(contrast - 1.0) > 1e-6
            or abs(saturation - 1.0) > 1e-6
            or abs(gamma - 1.0) > 1e-6
        ):
            gamma = max(0.3, min(3.0, gamma))
            parts.append(
                f"eq=brightness={brightness:.3f}:contrast={contrast:.3f}:saturation={saturation:.3f}:gamma={gamma:.3f}"
            )
        if abs(temperature) > 1e-6:
            warm = max(-0.4, min(0.4, temperature))
            rr = max(0.55, min(1.45, 1.0 + warm * 0.55))
            bb = max(0.55, min(1.45, 1.0 - warm * 0.55))
            parts.append(f"colorchannelmixer=rr={rr:.3f}:bb={bb:.3f}")
        # Важно: unsharp на части Windows-сборок ffmpeg/nvenc может приводить
        # к падению процесса ffmpeg без stderr (rc=3221226356), из-за чего
        # шортсы не создаются (0 байтов). Поэтому в рендере Auto Shorts
        # оставляем безопасную цветокоррекцию через eq, а шарп в пайплайн
        # намеренно не добавляем.
        return ("," + ",".join(parts)) if parts else ""

    # Совместимость с возможной старой ссылкой в рантайме/кэше.
    _layer_ft_tail = _layer_fx_tail

    def _probe_has_audio(path: str) -> bool:
        try:
            p = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=index",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    path,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            return p.returncode == 0 and bool((p.stdout or "").strip())
        except Exception:
            return False

    input_has_audio = _probe_has_audio(input_video)
    fps_mode = str(render_options.get("fps_mode", "30") or "30").strip().lower()
    if fps_mode in {"source", "src", "original", "исходный"}:
        target_fps = int(max(1, round(src_fps)))
    elif fps_mode in {"60", "59.94"}:
        target_fps = 60
    else:
        target_fps = 30
    target_fps = max(1, min(120, target_fps))
    _append_render_debug(
        f"SOURCE meta={src_w_i}x{src_h_i}@{round(src_fps, 3)}fps target={target_w_i}x{target_h_i}@{target_fps}fps"
    )

    quality_profile = str(render_options.get("quality_profile", "balanced") or "balanced").strip().lower()

    def _cpu_codec_args_by_quality(profile: str) -> List[str]:
        if profile == "high":
            return ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-threads", "0"]
        if profile == "fast":
            return ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "24", "-threads", "0"]
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-threads", "0"]

    def _nvenc_codec_args_by_quality(profile: str) -> List[str]:
        if profile == "high":
            return [
                "-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "18", "-b:v", "0", "-bf", "2",
            ]
        if profile == "fast":
            return [
                "-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ll", "-rc", "vbr", "-cq", "25", "-b:v", "0", "-bf", "0",
            ]
        return [
            "-c:v", "h264_nvenc", "-preset", "p3", "-tune", "hq", "-rc", "vbr", "-cq", "21", "-b:v", "0", "-bf", "0",
        ]

    def _detect_best_encoder() -> Tuple[List[str], str]:
        cpu_args = _cpu_codec_args_by_quality(quality_profile)
        try:
            p = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            encoders_txt = (p.stdout or "") + "\n" + (p.stderr or "")
            if "h264_nvenc" in encoders_txt:
                nvenc_args = _nvenc_codec_args_by_quality(quality_profile)
                return (
                    nvenc_args,
                    "gpu/nvenc",
                )
            if "h264_qsv" in encoders_txt:
                return (["-c:v", "h264_qsv"], "gpu/qsv")
            if "h264_amf" in encoders_txt:
                return (["-c:v", "h264_amf"], "gpu/amf")
        except Exception:
            pass
        return cpu_args, "cpu/libx264"

    cpu_video_codec_args = _cpu_codec_args_by_quality(quality_profile)
    detected_video_codec_args, detected_encoder_label = _detect_best_encoder()

    if backend == "cpu":
        selected_video_codec_args, selected_encoder_label = cpu_video_codec_args, "cpu/libx264(forced)"
        use_nvenc_gpu_filters = False
    elif backend == "gpu":
        # Только аппаратный энкодер (без CUDA filter graph)
        if detected_encoder_label == "cpu/libx264":
            selected_video_codec_args, selected_encoder_label = cpu_video_codec_args, "cpu/libx264(fallback_no_gpu)"
        else:
            selected_video_codec_args, selected_encoder_label = detected_video_codec_args, f"{detected_encoder_label}(forced_gpu)"
        use_nvenc_gpu_filters = False
    elif backend == "cuda":
        # Полный CUDA path только при NVENC
        if detected_encoder_label == "gpu/nvenc":
            selected_video_codec_args, selected_encoder_label = detected_video_codec_args, "gpu/nvenc(forced_cuda)"
            use_nvenc_gpu_filters = True
        else:
            selected_video_codec_args, selected_encoder_label = cpu_video_codec_args, "cpu/libx264(fallback_no_nvenc)"
            use_nvenc_gpu_filters = False
    else:
        # auto
        selected_video_codec_args, selected_encoder_label = detected_video_codec_args, detected_encoder_label
        use_nvenc_gpu_filters = selected_encoder_label == "gpu/nvenc"

    # Прогоняем реальную проверку GPU montage-пайплайна на 0.2 сек.
    # Если не проходит, сразу отключаем CUDA-фильтры (иначе будет постоянный CPU fallback и потеря скорости).
    if use_nvenc_gpu_filters:
        probe_out = out_dir / "__gpu_probe__.mp4"
        probe_cmd = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            "0",
            "-t",
            "0.2",
            "-i",
            input_video,
            "-filter_complex",
            "color=c=black:s=1080x1920,format=nv12,hwupload_cuda[base];"
            "[0:v]format=nv12,hwupload_cuda,scale_cuda=1080:1920[v0];"
            "[base][v0]overlay_cuda=0:0[vout]",
            "-map",
            "[vout]",
            "-an",
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p1",
            "-y",
            str(probe_out),
        ]
        try:
            p_probe = subprocess.run(
                probe_cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=(
                    subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
                ),
            )
            if p_probe.returncode != 0:
                use_nvenc_gpu_filters = False
                _append_render_debug(
                    f"GPU_FILTERS_DISABLED probe_failed rc={p_probe.returncode} stderr={(p_probe.stderr or '')[-1200:]}"
                )
                if backend == "cuda":
                    raise RuntimeError(
                        "CUDA-монтаж недоступен на текущей сборке FFmpeg/драйвере (overlay_cuda path). "
                        "Выберите режим GPU (аппаратный энкодер + обычный filter graph) или обновите FFmpeg/CUDA драйвер."
                    )
            else:
                _append_render_debug("GPU_FILTERS_ENABLED probe=ok")
        except Exception as e:
            use_nvenc_gpu_filters = False
            _append_render_debug(f"GPU_FILTERS_DISABLED probe_exception={e}")
            if backend == "cuda":
                raise
        finally:
            try:
                if probe_out.exists():
                    probe_out.unlink()
            except Exception:
                pass
    _append_render_debug(f"ENCODER selected={selected_encoder_label}")

    # Рендерим выбранные пользователем кандидаты как есть, без скрытой дедупликации,
    # чтобы количество шортсов соответствовало количеству выбранных позиций.
    render_candidates = list(candidates)

    results: List[str] = []
    total = max(1, len(render_candidates))

    def _run_ffmpeg(cmd_line: List[str], clip_idx: int, clip_duration_s: float):
        progress_file = None
        cmd_for_run = list(cmd_line)
        # Реальный прогресс рендера из ffmpeg (-progress), чтобы полоса была не "фейковой".
        # Пишем в временный файл и периодически читаем out_time_*.
        try:
            fd, progress_file = tempfile.mkstemp(prefix="shorts_ffmpeg_progress_", suffix=".log")
            os.close(fd)
            if cmd_for_run and "ffmpeg" in Path(cmd_for_run[0]).name.lower():
                cmd_for_run = [cmd_for_run[0], "-progress", progress_file, "-nostats", *cmd_for_run[1:]]
        except Exception:
            progress_file = None

        creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        if os.name == "nt" and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
            creationflags |= subprocess.CREATE_NEW_PROCESS_GROUP
        p = subprocess.Popen(
            cmd_for_run,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
        )
        t_run = perf_counter()
        last_real_frac = 0.0

        def _force_stop_process_tree(proc: subprocess.Popen):
            try:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )
                else:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except Exception:
                        proc.kill()
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        def _read_real_progress_frac() -> Optional[float]:
            if not progress_file:
                return None
            try:
                data = Path(progress_file).read_text(encoding="utf-8", errors="replace")
                out_time_us_all = re.findall(r"out_time_us=(\d+)", data)
                if out_time_us_all:
                    sec = int(out_time_us_all[-1]) / 1_000_000.0
                    return max(0.0, min(1.0, sec / max(0.2, clip_duration_s)))

                out_time_ms_all = re.findall(r"out_time_ms=(\d+)", data)
                if out_time_ms_all:
                    raw = int(out_time_ms_all[-1])
                    # Совместимость с разными сборками ffmpeg:
                    # где-то это microseconds (исторически), где-то milliseconds.
                    sec = raw / (1_000_000.0 if raw > (clip_duration_s * 10_000) else 1000.0)
                    return max(0.0, min(1.0, sec / max(0.2, clip_duration_s)))
            except Exception:
                return None
            return None

        while True:
            if cancel_cb and cancel_cb():
                try:
                    p.terminate()
                except Exception:
                    pass
                try:
                    p.wait(timeout=2)
                except Exception:
                    _force_stop_process_tree(p)
                try:
                    if progress_file and Path(progress_file).exists():
                        Path(progress_file).unlink()
                except Exception:
                    pass
                return subprocess.CompletedProcess(cmd_for_run, 130, "", "Cancelled by user")

            rc = p.poll()
            if rc is not None:
                out, err = p.communicate()
                try:
                    if progress_file and Path(progress_file).exists():
                        Path(progress_file).unlink()
                except Exception:
                    pass
                return subprocess.CompletedProcess(cmd_for_run, rc, out or "", err or "")

            if progress_cb:
                real_frac = _read_real_progress_frac()
                progress_msg = f"Рендер шортсов: {clip_idx}/{total}"
                if real_frac is not None:
                    last_real_frac = max(last_real_frac, real_frac)
                    # На сложных графах ffmpeg может долго финализировать контейнер после ~99% out_time.
                    # Держим небольшой "хвост" под стадию завершения, чтобы полоса не выглядела зависшей.
                    if last_real_frac >= 0.985:
                        frac = 0.94
                        progress_msg = f"Финализация клипа {clip_idx}/{total} (аудио/контейнер)..."
                    else:
                        frac = min(0.94, last_real_frac)
                else:
                    # Fallback на оценку по времени, если ffmpeg ещё не успел записать прогресс.
                    est = max(2.0, clip_duration_s * 1.8)
                    frac = min(0.90, (perf_counter() - t_run) / est)
                global_frac = ((clip_idx - 1) + frac) / max(1, total)
                progress_value = min(99, int(global_frac * 100))
                if clip_idx == 1:
                    progress_value = max(1, progress_value)
                progress_cb(progress_value, progress_msg)
            sleep(0.35)

    for i, c in enumerate(render_candidates, 1):
        if cancel_cb and cancel_cb():
            _append_render_debug(f"CANCELLED before clip {i}/{total}")
            break

        clip_head_pad_s = clip_head_pad_ms / 1000.0
        clip_tail_pad_s = clip_tail_pad_ms / 1000.0
        start_s = max(0.0, (c.start_ms / 1000.0) - clip_head_pad_s)
        end_s = max(start_s + 0.2, (c.end_ms / 1000.0) + clip_tail_pad_s)
        duration_s = max(0.2, end_s - start_s)
        clip_start_ms = int(round(start_s * 1000.0))
        clip_end_ms = int(round(end_s * 1000.0))
        title_part = _safe_filename(c.title or c.excerpt or "short")
        out_name = f"шорт_{i:03d}_{title_part}_{int(start_s)}-{int(end_s)}с.mp4"
        out_name = _fit_filename_to_path_limit(out_dir, out_name)
        out_path = out_dir / out_name

        def _normalize_candidate_ranges() -> List[Tuple[int, int]]:
            raw = getattr(c, "speech_ranges", None) or []
            if not raw:
                return [(clip_start_ms, clip_end_ms)]
            norm: List[Tuple[int, int]] = []
            # Делаем склейку чуть более «чувствительной» к паузам,
            # чтобы чаще сохранять реальные вырезы неречевых кусков.
            effective_merge_gap_ms = max(80, int(speech_merge_gap_ms * 0.65))
            # Для внутреннего монтажа не раздуваем сильно края,
            # иначе соседние реплики быстро слипаются в один диапазон.
            effective_pre_pad_ms = min(speech_pre_pad_ms, 140)
            effective_post_pad_ms = min(speech_post_pad_ms, 200)
            # Небольшой контекст до/после речи, чтобы не рубить слова на стыках.
            # Это заметно смягчает "телепорт" между репликами.
            for it in raw:
                try:
                    s, e = int(it[0]), int(it[1])
                except Exception:
                    continue
                s = max(clip_start_ms, s - effective_pre_pad_ms)
                e = min(clip_end_ms, e + effective_post_pad_ms)
                if e - s >= 180:
                    norm.append((s, e))
            if not norm:
                return [(clip_start_ms, clip_end_ms)]
            norm.sort(key=lambda x: x[0])
            merged: List[Tuple[int, int]] = [norm[0]]
            for s, e in norm[1:]:
                ps, pe = merged[-1]
                # Чуть более мягкий merge после добавления контекста,
                # чтобы не плодить дробные микро-склейки.
                if s - pe <= effective_merge_gap_ms:
                    merged[-1] = (ps, max(pe, e))
                else:
                    merged.append((s, e))

            # Всегда оставляем небольшой хвост в конце последней фразы,
            # чтобы финальные слова не обрезались на полу-слоге.
            if merged:
                ls, le = merged[-1]
                tail_guard_ms = max(260, speech_post_pad_ms)
                merged[-1] = (ls, min(clip_end_ms, le + tail_guard_ms))

            kept_ms = sum(max(0, e - s) for s, e in merged)

            raw_total_ms = max(1, c.end_ms - c.start_ms)
            gaps = []
            for idx in range(1, len(merged)):
                prev_e = merged[idx - 1][1]
                cur_s = merged[idx][0]
                gaps.append(max(0, cur_s - prev_e))
            avg_gap_ms = (sum(gaps) / len(gaps)) if gaps else 0.0

            # Защита от "обрыва слов":
            # если после агрессивной чистки остаётся слишком мало покрытия
            # или разрывы между фразами в среднем маленькие,
            # лучше отдать цельный клип без внутренних склеек.
            kept_ratio = kept_ms / raw_total_ms
            # Раньше тут часто происходил ранний fallback к цельному клипу,
            # из-за чего монтаж почти не делал внутренних вырезов.
            # Теперь разрешаем 2+ диапазона при умеренном покрытии речи.
            if kept_ratio < speech_min_coverage_ratio and not (
                len(merged) >= 2 and kept_ratio >= max(0.45, speech_min_coverage_ratio - 0.22)
            ):
                return [(clip_start_ms, clip_end_ms)]
            if len(merged) >= 5 and avg_gap_ms < 190:
                return [(clip_start_ms, clip_end_ms)]

            # Если есть 2+ диапазона, это уже реальный сигнал для склейки
            # (даже если суммарная речь короткая).
            if len(merged) >= 2:
                return merged
            if kept_ms < 900:
                return [(clip_start_ms, clip_end_ms)]
            return merged

        candidate_ranges = _normalize_candidate_ranges()
        has_internal_cuts = not (
            len(candidate_ranges) == 1
            and abs(candidate_ranges[0][0] - clip_start_ms) <= 120
            and abs(candidate_ranges[0][1] - clip_end_ms) <= 120
        )

        src_ref = "0:v"
        audio_map_label = "0:a:0?"
        pre_cut = ""
        if has_internal_cuts:
            trim_parts = []
            audio_parts = []
            for idx_r, (rs, re_) in enumerate(candidate_ranges):
                rel_s = max(0.0, (rs - clip_start_ms) / 1000.0)
                rel_e = max(rel_s + 0.05, (re_ - clip_start_ms) / 1000.0)
                trim_parts.append(
                    f"[0:v]trim=start={rel_s:.3f}:end={rel_e:.3f},setpts=PTS-STARTPTS[vp{idx_r}]"
                )
                if input_has_audio:
                    audio_parts.append(
                        f"[0:a]atrim=start={rel_s:.3f}:end={rel_e:.3f},asetpts=PTS-STARTPTS[ap{idx_r}]"
                    )

            if len(trim_parts) == 1:
                pre_cut = trim_parts[0] + ";"
                src_ref = "vp0"
                if input_has_audio and audio_parts:
                    pre_cut += audio_parts[0] + ";"
                    audio_map_label = "[ap0]"
            else:
                concat_v_inputs = "".join([f"[vp{k}]" for k in range(len(trim_parts))])
                if input_has_audio and audio_parts:
                    # Для concat с v=1:a=1 порядок входов должен быть interleaved:
                    # [v0][a0][v1][a1]..., иначе ffmpeg получает media type mismatch.
                    concat_va_inputs = "".join([f"[vp{k}][ap{k}]" for k in range(len(trim_parts))])
                    pre_cut = (
                        ";".join(trim_parts + audio_parts)
                        + ";"
                        + f"{concat_va_inputs}concat=n={len(trim_parts)}:v=1:a=1[vsrc][asrc];"
                    )
                    src_ref = "vsrc"
                    audio_map_label = "[asrc]"
                else:
                    pre_cut = (
                        ";".join(trim_parts)
                        + ";"
                        + f"{concat_v_inputs}concat=n={len(trim_parts)}:v=1:a=0[vsrc];"
                    )
                    src_ref = "vsrc"

            kept_ms = sum(max(0, b - a) for a, b in candidate_ranges)
            _append_render_debug(
                f"CANDIDATE_MONTAGE clip={i}/{total} ranges={len(candidate_ranges)} raw_ms={int((c.end_ms-c.start_ms))} kept_ms={kept_ms}"
            )

        cmd = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_s:.3f}",
            "-t",
            f"{duration_s:.3f}",
            "-i",
            input_video,
        ]

        cpu_filter_complex = None
        cmd_cpu_base = None

        if use_dual:
            wc = layout_template.get("webcam", {})
            gm = layout_template.get("game", {})
            wc_fx = _layer_fx_tail(layout_template.get("webcam_fx", {}))
            gm_fx = _layer_fx_tail(layout_template.get("game_fx", {}))

            wc_crop_x = max(0, min(src_w_i - 2, _safe_int(wc.get("crop_x"), 0)))
            wc_crop_y = max(0, min(src_h_i - 2, _safe_int(wc.get("crop_y"), 0)))
            wc_crop_w = max(2, min(src_w_i - wc_crop_x, _safe_int(wc.get("crop_w"), src_w_i)))
            wc_crop_h = max(2, min(src_h_i - wc_crop_y, _safe_int(wc.get("crop_h"), int(src_h_i * 0.5))))
            wc_out_x = _safe_int(wc.get("out_x"), 0)
            wc_out_y = _safe_int(wc.get("out_y"), 0)
            wc_out_w = max(2, _safe_int(wc.get("out_w"), out_w_i))
            wc_out_h = max(2, _safe_int(wc.get("out_h"), int(out_h_i * 0.33)))
            wc_out_x, wc_out_y = _scale_x(wc_out_x), _scale_y(wc_out_y)
            wc_out_w, wc_out_h = _scale_x(wc_out_w), _scale_y(wc_out_h)

            gm_crop_x = max(0, min(src_w_i - 2, _safe_int(gm.get("crop_x"), 0)))
            gm_crop_y = max(0, min(src_h_i - 2, _safe_int(gm.get("crop_y"), int(src_h_i * 0.5))))
            gm_crop_w = max(2, min(src_w_i - gm_crop_x, _safe_int(gm.get("crop_w"), src_w_i)))
            gm_crop_h = max(2, min(src_h_i - gm_crop_y, _safe_int(gm.get("crop_h"), src_h_i)))
            gm_out_x = _safe_int(gm.get("out_x"), 0)
            gm_out_y = _safe_int(gm.get("out_y"), int(out_h_i * 0.33))
            gm_out_w = max(2, _safe_int(gm.get("out_w"), out_w_i))
            gm_out_h = max(2, _safe_int(gm.get("out_h"), int(out_h_i * 0.67)))
            gm_out_x, gm_out_y = _scale_x(gm_out_x), _scale_y(gm_out_y)
            gm_out_w, gm_out_h = _scale_x(gm_out_w), _scale_y(gm_out_h)

            # Быстрый путь: вертикальный top+bottom layout.
            # Делаем определение с "допуском", т.к. из UI часто приходят значения вроде 637+1280=1917.
            # В таком случае нормализуем высоты до target_h и используем vstack,
            # что заметно быстрее overlay-композита.
            stack_tol = max(8, int(target_h_i * 0.005))
            width_tol = max(8, int(target_w_i * 0.01))
            full_width = (
                abs(wc_out_x) <= width_tol
                and abs(gm_out_x) <= width_tol
                and abs(wc_out_w - target_w_i) <= width_tol
                and abs(gm_out_w - target_w_i) <= width_tol
            )
            top_bottom_ordered = wc_out_y <= gm_out_y and (wc_out_y + wc_out_h) <= (gm_out_y + stack_tol)
            starts_from_top = abs(wc_out_y) <= stack_tol
            ends_at_bottom = abs((gm_out_y + gm_out_h) - target_h_i) <= stack_tol
            near_full_height = abs((wc_out_h + gm_out_h) - target_h_i) <= max(stack_tol, int(target_h_i * 0.02))

            is_vertical_stack_layout = (
                full_width and top_bottom_ordered and starts_from_top and ends_at_bottom and near_full_height
            )

            wc_stack_h = wc_out_h
            gm_stack_h = gm_out_h
            if is_vertical_stack_layout:
                total_h = max(2, wc_out_h + gm_out_h)
                # Важно для стабильности ffmpeg на части Windows-сборок:
                # промежуточные высоты в vstack должны быть чётными, иначе
                # возможны падения процесса без stderr (0xc0000374 / 3221226356).
                wc_stack_h = _even_size(max(2, int(round(target_h_i * (wc_out_h / total_h)))))
                gm_stack_h = max(2, target_h_i - wc_stack_h)
                if gm_stack_h % 2 != 0:
                    gm_stack_h = max(2, gm_stack_h - 1)
                    wc_stack_h = max(2, target_h_i - gm_stack_h)

            if use_nvenc_gpu_filters:
                filter_complex = pre_cut + (
                    f"color=c=black:s={target_w_i}x{target_h_i},format=nv12,hwupload_cuda[base];"
                    f"[{src_ref}]fps={target_fps},split=2[src_cam][src_game];"
                    f"[src_cam]crop={wc_crop_w}:{wc_crop_h}:{wc_crop_x}:{wc_crop_y}{wc_fx},"
                    f"format=nv12,hwupload_cuda,scale_cuda={wc_out_w}:{wc_out_h}[cam];"
                    f"[src_game]crop={gm_crop_w}:{gm_crop_h}:{gm_crop_x}:{gm_crop_y}{gm_fx},"
                    f"format=nv12,hwupload_cuda,scale_cuda={gm_out_w}:{gm_out_h}[game];"
                    f"[base][cam]overlay_cuda={wc_out_x}:{wc_out_y}[tmp];"
                    f"[tmp][game]overlay_cuda={gm_out_x}:{gm_out_y}[vout]"
                )
            else:
                wc_scale = f"scale={wc_out_w}:{wc_out_h}:flags=fast_bilinear"
                gm_scale = f"scale={gm_out_w}:{gm_out_h}:flags=fast_bilinear"
                if is_vertical_stack_layout:
                    filter_complex = pre_cut + (
                        f"[{src_ref}]fps={target_fps},split=2[src_cam][src_game];"
                        f"[src_cam]crop={wc_crop_w}:{wc_crop_h}:{wc_crop_x}:{wc_crop_y}{wc_fx},"
                        f"scale={target_w_i}:{wc_stack_h}:flags=fast_bilinear[cam];"
                        f"[src_game]crop={gm_crop_w}:{gm_crop_h}:{gm_crop_x}:{gm_crop_y}{gm_fx},"
                        f"scale={target_w_i}:{gm_stack_h}:flags=fast_bilinear[game];"
                        f"[cam][game]vstack=inputs=2[vout]"
                    )
                else:
                    filter_complex = pre_cut + (
                        f"color=size={target_w_i}x{target_h_i}:color=black[base];"
                        f"[{src_ref}]fps={target_fps},split=2[src_cam][src_game];"
                        f"[src_cam]crop={wc_crop_w}:{wc_crop_h}:{wc_crop_x}:{wc_crop_y}{wc_fx},"
                        f"{wc_scale}[cam];"
                        f"[src_game]crop={gm_crop_w}:{gm_crop_h}:{gm_crop_x}:{gm_crop_y}{gm_fx},"
                        f"{gm_scale}[game];"
                        f"[base][cam]overlay={wc_out_x}:{wc_out_y}[tmp];"
                        f"[tmp][game]overlay={gm_out_x}:{gm_out_y}[vout]"
                    )

            # CPU-версия монтажного графа для корректного fallback без потери layout
            if is_vertical_stack_layout:
                cpu_filter_complex = pre_cut + (
                    f"[{src_ref}]fps={target_fps},split=2[src_cam][src_game];"
                    f"[src_cam]crop={wc_crop_w}:{wc_crop_h}:{wc_crop_x}:{wc_crop_y}{wc_fx},"
                    f"scale={target_w_i}:{wc_stack_h}:flags=fast_bilinear[cam];"
                    f"[src_game]crop={gm_crop_w}:{gm_crop_h}:{gm_crop_x}:{gm_crop_y}{gm_fx},"
                    f"scale={target_w_i}:{gm_stack_h}:flags=fast_bilinear[game];"
                    f"[cam][game]vstack=inputs=2[vout]"
                )
                _append_render_debug(
                    f"LAYOUT path=vstack normalized_h={wc_stack_h}+{gm_stack_h} target_h={target_h_i}"
                )
            else:
                cpu_filter_complex = pre_cut + (
                    f"color=size={target_w_i}x{target_h_i}:color=black[base];"
                    f"[{src_ref}]fps={target_fps},split=2[src_cam][src_game];"
                    f"[src_cam]crop={wc_crop_w}:{wc_crop_h}:{wc_crop_x}:{wc_crop_y}{wc_fx},"
                    f"scale={wc_out_w}:{wc_out_h}:flags=fast_bilinear[cam];"
                    f"[src_game]crop={gm_crop_w}:{gm_crop_h}:{gm_crop_x}:{gm_crop_y}{gm_fx},"
                    f"scale={gm_out_w}:{gm_out_h}:flags=fast_bilinear[game];"
                    f"[base][cam]overlay={wc_out_x}:{wc_out_y}[tmp];"
                    f"[tmp][game]overlay={gm_out_x}:{gm_out_y}[vout]"
                )
                _append_render_debug("LAYOUT path=overlay")

            cmd.extend(
                [
                    "-filter_complex",
                    filter_complex,
                    "-map",
                    "[vout]",
                ]
            )
            if audio_map_label:
                cmd.extend(["-map", audio_map_label])

            cmd_cpu_base = [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start_s:.3f}",
                "-t",
                f"{duration_s:.3f}",
                "-i",
                input_video,
                "-filter_complex",
                cpu_filter_complex,
                "-map",
                "[vout]",
            ]
            if audio_map_label:
                cmd_cpu_base.extend(["-map", audio_map_label])
        else:
            vf = (
                f"fps={target_fps},"
                f"scale={target_w_i}:{target_h_i}:force_original_aspect_ratio=increase:flags=fast_bilinear,"
                f"crop={target_w_i}:{target_h_i}"
            )
            if has_internal_cuts:
                filter_complex = pre_cut + f"[{src_ref}]{vf}[vout]"
                cmd.extend(["-filter_complex", filter_complex, "-map", "[vout]"])
                if audio_map_label:
                    cmd.extend(["-map", audio_map_label])
            else:
                cmd.extend(["-vf", vf])

            cmd_cpu_base = [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start_s:.3f}",
                "-t",
                f"{duration_s:.3f}",
                "-i",
                input_video,
            ]
            if has_internal_cuts:
                cmd_cpu_base.extend(["-filter_complex", filter_complex, "-map", "[vout]"])
                if audio_map_label:
                    cmd_cpu_base.extend(["-map", audio_map_label])
            else:
                cmd_cpu_base.extend(["-vf", vf])

        cmd_base = list(cmd)
        mux_args = [
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-shortest",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-y",
            str(out_path),
        ]
        cmd = cmd_base + selected_video_codec_args + mux_args

        logger.info("Render short: %s", " ".join(cmd))
        _append_render_debug(f"CMD [{out_name}] {' '.join(cmd)}")
        t0 = perf_counter()
        process = _run_ffmpeg(cmd, i, duration_s)

        if process.returncode == 130:
            _append_render_debug(f"CANCELLED during [{out_name}]")
            break

        if process.returncode == 0 and out_path.exists():
            results.append(str(out_path))
            _append_render_debug(
                f"OK [{out_name}] -> {out_path} elapsed={round(perf_counter() - t0, 3)}s"
            )
        else:
            # Если выбран GPU-кодек и он не отработал — повторяем этот же рендер на CPU
            if selected_encoder_label != "cpu/libx264":
                cmd_cpu = cmd_cpu_base + cpu_video_codec_args + mux_args
                _append_render_debug(f"RETRY_CPU [{out_name}] {' '.join(cmd_cpu)}")
                p_cpu = _run_ffmpeg(cmd_cpu, i, duration_s)
                if p_cpu.returncode == 130:
                    _append_render_debug(f"CANCELLED during RETRY_CPU [{out_name}]")
                    break
                if p_cpu.returncode == 0 and out_path.exists():
                    results.append(str(out_path))
                    _append_render_debug(
                        f"RETRY_CPU_OK [{out_name}] -> {out_path} elapsed={round(perf_counter() - t0, 3)}s"
                    )
                    if progress_cb:
                        progress = int((i / total) * 100)
                        progress_cb(progress, f"Рендер шортсов: {i}/{total}")
                    continue
                _append_render_debug(
                    f"RETRY_CPU_FAIL [{out_name}] rc={p_cpu.returncode} stderr={(p_cpu.stderr or '')[-3000:]}"
                )

            logger.error(
                "Short render failed [%s]. stderr: %s",
                out_name,
                (process.stderr or "")[-1500:],
            )
            _append_render_debug(
                f"FAIL [{out_name}] rc={process.returncode} stderr={(process.stderr or '')[-3000:]}"
            )
            # Fallback: рендер без монтажного dual-layer, чтобы процесс не стопорился полностью
            if use_dual:
                simple_cmd = cmd_cpu_base + [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "20",
                    "-threads",
                    "0",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "160k",
                    "-y",
                    str(out_path),
                ]
                p2 = _run_ffmpeg(simple_cmd, i, duration_s)
                if p2.returncode == 130:
                    _append_render_debug(f"CANCELLED during FALLBACK [{out_name}]")
                    break
                if p2.returncode == 0 and out_path.exists():
                    logger.warning("Fallback render succeeded for %s", out_name)
                    results.append(str(out_path))
                    _append_render_debug(
                        f"FALLBACK_OK [{out_name}] -> {out_path} elapsed={round(perf_counter() - t0, 3)}s"
                    )
                else:
                    logger.error("Fallback render failed [%s]: %s", out_name, (p2.stderr or "")[-1500:])
                    _append_render_debug(
                        f"FALLBACK_FAIL [{out_name}] rc={p2.returncode} stderr={(p2.stderr or '')[-3000:]}"
                    )

        if progress_cb:
            progress = int((i / total) * 100)
            progress_cb(progress, f"Рендер шортсов: {i}/{total}")

    _append_render_debug(f"END render_shorts produced={len(results)} log={RENDER_DEBUG_LOG}")
    return results
