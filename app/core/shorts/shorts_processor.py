import json
import re
import subprocess
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
    ):
        self.min_duration_ms = int(min_duration_s * 1000)
        self.max_duration_ms = int(max_duration_s * 1000)
        self.llm_base_url = (llm_base_url or "").strip()
        self.llm_api_key = (llm_api_key or "").strip()
        self.llm_model = (llm_model or "").strip()

    def find_candidates(self, asr_data: ASRData, progress_cb: Optional[Callable] = None) -> List[ShortCandidate]:
        if progress_cb:
            progress_cb(5, "Поиск интересных фрагментов...")

        candidates: List[ShortCandidate] = []
        if self._llm_ready():
            if progress_cb:
                progress_cb(12, "AI Enterprise: семантический анализ эпизодов...")
            candidates = self._build_enterprise_llm_candidates(asr_data, progress_cb=progress_cb)

        if not candidates:
            candidates = self._build_heuristic_candidates(asr_data)

        if progress_cb:
            progress_cb(55, f"Найдено кандидатов (эвристика): {len(candidates)}")

        reranked = self._try_llm_rerank(candidates)
        if progress_cb:
            progress_cb(85, f"Кандидаты после ранжирования: {len(reranked)}")

        return reranked

    def _llm_ready(self) -> bool:
        return bool(self.llm_model and self.llm_base_url and self.llm_api_key)

    def _build_enterprise_llm_candidates(
        self,
        asr_data: ASRData,
        progress_cb: Optional[Callable] = None,
    ) -> List[ShortCandidate]:
        segments = [s for s in asr_data.segments if s.text and s.text.strip()]
        if not segments:
            return []

        packets = self._build_segment_packets(segments, packet_size=140, overlap=35)
        all_candidates: List[ShortCandidate] = []

        for i, (start_idx, end_idx, packet_segments) in enumerate(packets, 1):
            try:
                packet_candidates = self._llm_extract_candidates_from_packet(
                    packet_segments,
                    global_start_idx=start_idx,
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

    def _llm_extract_candidates_from_packet(self, packet_segments: List, global_start_idx: int) -> List[ShortCandidate]:
        client = openai.OpenAI(base_url=self.llm_base_url, api_key=self.llm_api_key)

        rows = []
        for local_idx, seg in enumerate(packet_segments):
            abs_idx = global_start_idx + local_idx
            rows.append(
                {
                    "idx": abs_idx,
                    "start_ms": int(seg.start_time),
                    "end_ms": int(seg.end_time),
                    "text": (seg.text or "").strip(),
                }
            )

        system = (
            "Ты enterprise-редактор YouTube Shorts. Найди лучшие моменты удержания. "
            "Критерии: hook в первые 2-5 секунд, эмоция, конфликт/неожиданность, панчлайн, кульминация, потенциал для шеринга. "
            "Верни СТРОГО JSON: "
            "{\"items\":[{\"start_idx\":int,\"end_idx\":int,\"score\":0-100,\"title\":str,\"reason\":str,"
            "\"hook\":0-10,\"emotion\":0-10,\"novelty\":0-10,\"shareability\":0-10}]}. "
            "Длительность каждого фрагмента 15-60 секунд. Не придумывай таймкоды, используй только переданные idx."
        )
        user = f"Сегменты:\n{json.dumps(rows, ensure_ascii=False)}"

        rsp = client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.15,
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
                result.append(
                    ShortCandidate(
                        start_ms=start_ms,
                        end_ms=end_ms,
                        score=round(blended, 2),
                        title=str(item.get("title", "")).strip() or self._build_title(excerpt),
                        reason=str(item.get("reason", "")).strip() or "AI Enterprise selection",
                        excerpt=self._shorten(excerpt, 220),
                    )
                )
            except Exception:
                continue

        return result

    def _build_heuristic_candidates(self, asr_data: ASRData) -> List[ShortCandidate]:
        segments = [s for s in asr_data.segments if s.text and s.text.strip()]
        if not segments:
            return []

        windows: List[ShortCandidate] = []
        for i in range(len(segments)):
            start = segments[i].start_time
            text_parts = []
            end = start
            punch = 0.0

            for j in range(i, len(segments)):
                seg = segments[j]
                end = seg.end_time
                text_parts.append(seg.text.strip())
                duration = end - start
                if duration > self.max_duration_ms:
                    break
                if duration < self.min_duration_ms:
                    continue

                joined = " ".join(text_parts)
                score = self._heuristic_score(joined, duration)
                if score <= 0:
                    continue

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
                    )
                )

                # если уже очень сильный фрагмент — можно не расширять дальше
                if punch > 95:
                    break

        windows.sort(key=lambda x: x.score, reverse=True)
        return self._deduplicate(windows)

    def _heuristic_score(self, text: str, duration_ms: int) -> float:
        txt = (text or "").strip()
        if not txt:
            return 0

        words = txt.split()
        token_count = max(len(words), len(re.findall(r"[\u4e00-\u9fff]", txt)))
        if token_count == 0:
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
        hype_kw = ["имба", "жёстко", "клатч", "тащит", "топ", "легенд", "финал", "камбэк"]

        funny_hits = sum(1 for k in funny_kw if k in txt.lower())
        hook_hits = sum(1 for k in hook_kw if k in txt.lower())
        hype_hits = sum(1 for k in hype_kw if k in txt.lower())
        punct_bonus = txt.count("!") * 2 + txt.count("?") * 1.5
        caps_bonus = min(8, len(re.findall(r"[A-ZА-ЯЁ]{3,}", txt)) * 2)
        digit_bonus = 4 if re.search(r"\d", txt) else 0

        duration_target_bonus = 14 if 16 <= duration_s <= 42 else (6 if 12 <= duration_s <= 55 else 0)
        density_bonus = max(0, min(22, (density - 1.45) * 10))
        anti_wall_penalty = -10 if density < 0.8 else 0

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
            + anti_wall_penalty
        )
        return round(min(100.0, score), 2)

    def _try_llm_rerank(self, candidates: List[ShortCandidate]) -> List[ShortCandidate]:
        if not candidates:
            return []

        if not self._llm_ready():
            return candidates

        try:
            top = candidates[:40]
            payload = [
                {
                    "id": i,
                    "start_ms": c.start_ms,
                    "end_ms": c.end_ms,
                    "score": c.score,
                    "excerpt": c.excerpt,
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

            reranked.sort(key=lambda x: x.score, reverse=True)
            return self._deduplicate(reranked)
        except Exception as e:
            logger.warning(f"LLM rerank skipped: {e}")
            return candidates

    def _deduplicate(self, candidates: List[ShortCandidate]) -> List[ShortCandidate]:
        accepted: List[ShortCandidate] = []
        for c in candidates:
            overlap = False
            for a in accepted:
                inter = max(0, min(c.end_ms, a.end_ms) - max(c.start_ms, a.start_ms))
                short = max(1, min(c.duration_ms, a.duration_ms))
                if inter / short > 0.65:
                    overlap = True
                    break
            if not overlap:
                accepted.append(c)
        return accepted

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
    target_fps = int(min(60, max(24, round(src_fps))))
    _append_render_debug(f"SOURCE meta={src_w_i}x{src_h_i}@{round(src_fps, 3)}fps target_fps={target_fps}")

    def _detect_best_encoder() -> Tuple[List[str], str]:
        cpu_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-threads", "0"]
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
                nvenc_args = [
                    "-c:v",
                    "h264_nvenc",
                    "-preset",
                    "p1",
                    "-tune",
                    "ll",
                    "-rc",
                    "vbr",
                    "-cq",
                    "21",
                    "-b:v",
                    "0",
                    "-bf",
                    "0",
                ]
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

    cpu_video_codec_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-threads", "0"]
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

    results: List[str] = []
    total = max(1, len(candidates))

    def _run_ffmpeg(cmd_line: List[str], clip_idx: int, clip_duration_s: float):
        creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        p = subprocess.Popen(
            cmd_line,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
        )
        t_run = perf_counter()
        while True:
            if cancel_cb and cancel_cb():
                try:
                    p.terminate()
                except Exception:
                    pass
                try:
                    p.wait(timeout=2)
                except Exception:
                    try:
                        p.kill()
                    except Exception:
                        pass
                return subprocess.CompletedProcess(cmd_line, 130, "", "Cancelled by user")

            rc = p.poll()
            if rc is not None:
                out, err = p.communicate()
                return subprocess.CompletedProcess(cmd_line, rc, out or "", err or "")

            if progress_cb:
                base = int(((clip_idx - 1) / total) * 100)
                span = max(1, int(100 / total) - 1)
                est = max(2.0, clip_duration_s * 1.8)
                frac = min(0.97, (perf_counter() - t_run) / est)
                progress_cb(min(99, base + int(span * frac)), f"Рендер шортсов: {clip_idx}/{total}")
            sleep(0.35)

    for i, c in enumerate(candidates, 1):
        if cancel_cb and cancel_cb():
            _append_render_debug(f"CANCELLED before clip {i}/{total}")
            break

        start_s = max(0.0, c.start_ms / 1000.0)
        end_s = max(start_s + 0.2, c.end_ms / 1000.0)
        duration_s = max(0.2, end_s - start_s)
        out_name = f"short_{i:03d}_{int(start_s)}s_{int(end_s)}s.mp4"
        out_path = out_dir / out_name

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

            # Быстрый путь: если слои ровно вертикально состыкованы (top+bottom),
            # используем vstack вместо overlay-композита (меньше операций, быстрее рендер).
            is_vertical_stack_layout = (
                wc_out_x == 0
                and gm_out_x == 0
                and wc_out_y == 0
                and gm_out_y == wc_out_h
                and wc_out_w == target_w_i
                and gm_out_w == target_w_i
                and (wc_out_h + gm_out_h) == target_h_i
            )

            if use_nvenc_gpu_filters:
                filter_complex = (
                    f"color=c=black:s={target_w_i}x{target_h_i},format=nv12,hwupload_cuda[base];"
                    f"[0:v]fps={target_fps},split=2[src_cam][src_game];"
                    f"[src_cam]crop={wc_crop_w}:{wc_crop_h}:{wc_crop_x}:{wc_crop_y},"
                    f"format=nv12,hwupload_cuda,scale_cuda={wc_out_w}:{wc_out_h}[cam];"
                    f"[src_game]crop={gm_crop_w}:{gm_crop_h}:{gm_crop_x}:{gm_crop_y},"
                    f"format=nv12,hwupload_cuda,scale_cuda={gm_out_w}:{gm_out_h}[game];"
                    f"[base][cam]overlay_cuda={wc_out_x}:{wc_out_y}[tmp];"
                    f"[tmp][game]overlay_cuda={gm_out_x}:{gm_out_y}[vout]"
                )
            else:
                wc_scale = f"scale={wc_out_w}:{wc_out_h}:flags=fast_bilinear"
                gm_scale = f"scale={gm_out_w}:{gm_out_h}:flags=fast_bilinear"
                if is_vertical_stack_layout:
                    filter_complex = (
                        f"[0:v]fps={target_fps},split=2[src_cam][src_game];"
                        f"[src_cam]crop={wc_crop_w}:{wc_crop_h}:{wc_crop_x}:{wc_crop_y},"
                        f"{wc_scale}[cam];"
                        f"[src_game]crop={gm_crop_w}:{gm_crop_h}:{gm_crop_x}:{gm_crop_y},"
                        f"{gm_scale}[game];"
                        f"[cam][game]vstack=inputs=2[vout]"
                    )
                else:
                    filter_complex = (
                        f"color=size={target_w_i}x{target_h_i}:color=black[base];"
                        f"[0:v]fps={target_fps},split=2[src_cam][src_game];"
                        f"[src_cam]crop={wc_crop_w}:{wc_crop_h}:{wc_crop_x}:{wc_crop_y},"
                        f"{wc_scale}[cam];"
                        f"[src_game]crop={gm_crop_w}:{gm_crop_h}:{gm_crop_x}:{gm_crop_y},"
                        f"{gm_scale}[game];"
                        f"[base][cam]overlay={wc_out_x}:{wc_out_y}[tmp];"
                        f"[tmp][game]overlay={gm_out_x}:{gm_out_y}[vout]"
                    )

            # CPU-версия монтажного графа для корректного fallback без потери layout
            if is_vertical_stack_layout:
                cpu_filter_complex = (
                    f"[0:v]fps={target_fps},split=2[src_cam][src_game];"
                    f"[src_cam]crop={wc_crop_w}:{wc_crop_h}:{wc_crop_x}:{wc_crop_y},"
                    f"scale={wc_out_w}:{wc_out_h}:flags=fast_bilinear[cam];"
                    f"[src_game]crop={gm_crop_w}:{gm_crop_h}:{gm_crop_x}:{gm_crop_y},"
                    f"scale={gm_out_w}:{gm_out_h}:flags=fast_bilinear[game];"
                    f"[cam][game]vstack=inputs=2[vout]"
                )
                _append_render_debug("LAYOUT path=vstack")
            else:
                cpu_filter_complex = (
                    f"color=size={target_w_i}x{target_h_i}:color=black[base];"
                    f"[0:v]fps={target_fps},split=2[src_cam][src_game];"
                    f"[src_cam]crop={wc_crop_w}:{wc_crop_h}:{wc_crop_x}:{wc_crop_y},"
                    f"scale={wc_out_w}:{wc_out_h}:flags=fast_bilinear[cam];"
                    f"[src_game]crop={gm_crop_w}:{gm_crop_h}:{gm_crop_x}:{gm_crop_y},"
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
                    "-map",
                    "0:a?",
                ]
            )

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
                "-map",
                "0:a?",
            ]
        else:
            vf = (
                f"fps={target_fps},"
                f"scale={target_w_i}:{target_h_i}:force_original_aspect_ratio=increase:flags=fast_bilinear,"
                f"crop={target_w_i}:{target_h_i}"
            )
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
                "-vf",
                vf,
            ]

        cmd_base = list(cmd)
        mux_args = [
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
