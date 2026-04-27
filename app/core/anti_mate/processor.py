import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

try:
    import openai
except Exception:  # pragma: no cover
    openai = None

from app.core.utils.logger import setup_logger

logger = setup_logger("anti_mate_processor")


def is_llm_runtime_available() -> bool:
    return openai is not None


DEFAULT_FORBIDDEN_WORDS = {
    "бляд",
    "бля",
    "блять",
    "еб",
    "еба",
    "ёб",
    "пизд",
    "хуй",
    "нахуй",
    "хуе",
    "мудак",
    "мраз",
    "сука",
    "чмо",
    "гандон",
    "долбо",
    "уеб",
}


@dataclass
class CensorRegion:
    start_ms: int
    end_ms: int
    trigger_word: str
    source: str = "rule"
    enabled: bool = True
    mode: str = "beep"

    def to_dict(self) -> Dict:
        return asdict(self)


def _normalize_word(text: str) -> str:
    t = str(text or "").lower().replace("ё", "е")
    t = re.sub(r"[^\wа-я]+", "", t, flags=re.IGNORECASE)
    return t.strip()


def _contains_forbidden(normalized_word: str, forbidden: set[str]) -> bool:
    if not normalized_word:
        return False
    for f in forbidden:
        if not f:
            continue
        if normalized_word == f or normalized_word.startswith(f):
            return True
    return False


def _extract_words_with_timing(asr_json: Dict) -> List[Dict]:
    words: List[Dict] = []
    data = asr_json or {}
    for key in sorted(data.keys(), key=lambda x: int(x) if str(x).isdigit() else 0):
        seg = data.get(key) or {}
        seg_start = int(seg.get("start_time", 0) or 0)
        seg_end = int(seg.get("end_time", 0) or seg_start)
        raw_text = str(seg.get("original_subtitle", "") or "")
        wt = seg.get("word_timestamps") or []
        if wt:
            for w in wt:
                word = str(w.get("word", "") or "").strip()
                if not word:
                    continue
                start = int(w.get("start", w.get("start_time", seg_start)) or seg_start)
                end = int(w.get("end", w.get("end_time", start + 120)) or (start + 120))
                if end <= start:
                    end = start + 120
                words.append({"word": word, "start_ms": start, "end_ms": end})
            continue

        tokens = [t for t in re.findall(r"[\wа-яё]+", raw_text, flags=re.IGNORECASE) if t]
        if not tokens:
            continue
        duration = max(1, seg_end - seg_start)
        unit = max(80, duration // max(1, len(tokens)))
        cur = seg_start
        for tok in tokens:
            st = cur
            ed = min(seg_end, st + unit)
            if ed <= st:
                ed = st + 80
            words.append({"word": tok, "start_ms": st, "end_ms": ed})
            cur = ed

    words.sort(key=lambda x: int(x.get("start_ms", 0)))
    return words


def _extract_json_object(text: str) -> Dict:
    s = str(text or "")
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def _llm_detect_bad_words(
    transcript_text: str,
    llm_config: Optional[Dict[str, str]] = None,
    notify_cb: Optional[Callable[[str], None]] = None,
) -> set[str]:
    cfg = llm_config or {}
    base_url = str(cfg.get("base_url", "") or "").strip()
    api_key = str(cfg.get("api_key", "") or "").strip()
    model = str(cfg.get("model", "") or "").strip()
    if not (base_url and api_key and model and transcript_text.strip()):
        if notify_cb:
            notify_cb("LLM-этап пропущен: не заполнены Base URL / API key / Model.")
        return set()
    if openai is None:
        msg = "LLM-этап пропущен: пакет openai не установлен в текущем runtime."
        logger.warning(msg)
        if notify_cb:
            notify_cb(msg)
        return set()

    try:
        client = openai.OpenAI(base_url=base_url, api_key=api_key)
        sys_prompt = (
            "Ты модератор нецензурной брани. Верни СТРОГО JSON-объект: "
            '{"bad_words": ["слово1", "слово2"]}. '
            "Нужно извлечь только реально встречающиеся в тексте маты/оскорбления и их формы."
        )
        rsp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": transcript_text[:22000]},
            ],
        )
        content = ""
        if rsp and rsp.choices:
            content = str(rsp.choices[0].message.content or "")
        obj = _extract_json_object(content)
        values = obj.get("bad_words") if isinstance(obj, dict) else []
        if not isinstance(values, list):
            return set()
        return {_normalize_word(v) for v in values if _normalize_word(v)}
    except Exception as e:
        logger.warning("LLM profanity detect failed: %s", e)
        if notify_cb:
            notify_cb(f"LLM-этап завершился ошибкой и был пропущен: {e}")
        return set()


def detect_profanity_regions(
    asr_json: Dict,
    *,
    forced_forbidden_words: Optional[List[str]] = None,
    forced_allowed_words: Optional[List[str]] = None,
    use_llm: bool = True,
    llm_config: Optional[Dict[str, str]] = None,
    default_mode: str = "beep",
    pad_before_ms: int = 120,
    pad_after_ms: int = 180,
    merge_gap_ms: int = 220,
    min_region_ms: int = 120,
    max_region_ms: int = 2200,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    notify_cb: Optional[Callable[[str], None]] = None,
) -> List[CensorRegion]:
    words = _extract_words_with_timing(asr_json)
    if progress_cb:
        progress_cb(10, f"Слов с таймингами: {len(words)}")

    if forced_forbidden_words is None:
        forbidden = set(DEFAULT_FORBIDDEN_WORDS)
    else:
        forbidden = {_normalize_word(w) for w in (forced_forbidden_words or []) if _normalize_word(w)}
    allowed = {_normalize_word(w) for w in (forced_allowed_words or []) if _normalize_word(w)}
    if progress_cb:
        progress_cb(18, f"Словари: forbidden={len(forbidden)}, allowed={len(allowed)}")

    transcript = " ".join(str(w.get("word", "")) for w in words)
    if progress_cb:
        progress_cb(26, "Подготовка транскрипта для анализа")

    llm_bad_words: set[str] = set()
    if use_llm:
        if progress_cb:
            progress_cb(36, "LLM: отправка запроса")
        llm_bad_words = _llm_detect_bad_words(transcript, llm_config=llm_config, notify_cb=notify_cb)
        if progress_cb:
            progress_cb(52, "LLM: ответ получен, обработка")
    elif progress_cb:
        progress_cb(45, "LLM отключён пользователем: используется только словарный режим")
    if progress_cb:
        progress_cb(58, f"LLM-кандидатов: {len(llm_bad_words)}")

    raw_regions: List[CensorRegion] = []
    words_total = max(1, len(words))
    for idx, w in enumerate(words, start=1):
        raw_word = str(w.get("word", "") or "")
        n = _normalize_word(raw_word)
        if not n or n in allowed:
            if progress_cb and idx % 100 == 0:
                p = 60 + int((idx / words_total) * 20)
                progress_cb(min(82, p), f"Сканирование слов: {idx}/{words_total}")
            continue

        source = ""
        if _contains_forbidden(n, forbidden):
            source = "forced"
        elif _contains_forbidden(n, llm_bad_words):
            source = "llm"

        if not source:
            continue

        st = max(0, int(w.get("start_ms", 0)) - int(pad_before_ms))
        ed = max(st + 60, int(w.get("end_ms", st + 60)) + int(pad_after_ms))
        min_len = max(40, int(min_region_ms or 120))
        max_len = max(min_len, int(max_region_ms or 2200))
        if ed - st < min_len:
            ed = st + min_len
        if ed - st > max_len:
            ed = st + max_len
        raw_regions.append(
            CensorRegion(
                start_ms=st,
                end_ms=ed,
                trigger_word=raw_word,
                source=source,
                enabled=True,
                mode=default_mode,
            )
        )
        if progress_cb and idx % 100 == 0:
            p = 60 + int((idx / words_total) * 20)
            progress_cb(min(82, p), f"Сканирование слов: {idx}/{words_total}")

    if progress_cb:
        progress_cb(84, f"Найдено триггеров: {len(raw_regions)}")

    if not raw_regions:
        return []

    raw_regions.sort(key=lambda r: r.start_ms)
    merged: List[CensorRegion] = [raw_regions[0]]
    total_regions = max(1, len(raw_regions) - 1)
    for i, r in enumerate(raw_regions[1:], start=1):
        last = merged[-1]
        if r.start_ms <= last.end_ms + int(merge_gap_ms):
            last.end_ms = max(last.end_ms, r.end_ms)
            min_len = max(40, int(min_region_ms or 120))
            max_len = max(min_len, int(max_region_ms or 2200))
            if last.end_ms - last.start_ms < min_len:
                last.end_ms = last.start_ms + min_len
            if last.end_ms - last.start_ms > max_len:
                last.end_ms = last.start_ms + max_len
            if r.trigger_word and r.trigger_word not in last.trigger_word:
                last.trigger_word = f"{last.trigger_word}, {r.trigger_word}" if last.trigger_word else r.trigger_word
            if last.source != r.source:
                last.source = "mixed"
        else:
            merged.append(r)
        if progress_cb and (i % 40 == 0 or i == total_regions):
            p = 86 + int((i / total_regions) * 12)
            progress_cb(min(98, p), f"Объединение зон: {i}/{total_regions}")

    if progress_cb:
        progress_cb(100, f"Готово: областей цензуры {len(merged)}")
    return merged


def _probe_duration_sec(path: str) -> float:
    try:
        p = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=(subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0),
        )
        if p.returncode == 0:
            return max(0.1, float((p.stdout or "0").strip() or 0.0))
    except Exception:
        pass
    return 0.1


def _build_activity_expr(regions: List[CensorRegion], mode_filter: str = "") -> str:
    parts = []
    for r in regions:
        if not r.enabled:
            continue
        if mode_filter and str(getattr(r, "mode", "") or "").strip().lower() != mode_filter:
            continue
        a = max(0.0, r.start_ms / 1000.0)
        b = max(a + 0.01, r.end_ms / 1000.0)
        parts.append(f"between(t,{a:.3f},{b:.3f})")
    if not parts:
        return "0"
    return "+".join(parts)


def _build_beep_source_expr(profile: str, frequency: int, duration: float) -> str:
    d = max(0.1, float(duration or 0.1))
    f = max(120, int(frequency or 1000))
    p = str(profile or "classic").strip().lower()
    if p == "dual":
        f2 = int(f * 1.55)
        return f"aevalsrc=0.62*sin(2*PI*{f}*t)+0.38*sin(2*PI*{f2}*t):s=48000:d={d:.3f}"
    if p == "noise":
        return f"anoisesrc=color=white:amplitude=0.8:sample_rate=48000:d={d:.3f}"
    if p == "soft":
        return f"sine=frequency={max(120, int(f*0.82))}:sample_rate=48000:duration={d:.3f}"
    if p == "radio":
        return f"sine=frequency={max(180, int(f*1.25))}:sample_rate=48000:duration={d:.3f}"
    return f"sine=frequency={f}:sample_rate=48000:duration={d:.3f}"


def render_beep_preview(
    output_wav: str,
    *,
    profile: str = "classic",
    beep_frequency: int = 1000,
    beep_volume: float = 0.90,
    duration_sec: float = 1.8,
) -> str:
    out = Path(output_wav)
    out.parent.mkdir(parents=True, exist_ok=True)
    source = _build_beep_source_expr(profile, beep_frequency, duration_sec)
    p_norm = str(profile or "classic").strip().lower()
    post = ""
    if p_norm == "soft":
        post = ",lowpass=f=2200"
    elif p_norm == "radio":
        post = ",highpass=f=700,lowpass=f=3400"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        source,
        "-filter:a",
        f"volume={max(0.01, min(2.0, float(beep_volume))):.3f}{post}",
        "-t",
        f"{max(0.2, float(duration_sec)):.3f}",
        "-c:a",
        "pcm_s16le",
        str(out),
    ]
    p = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=(subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0),
    )
    if p.returncode != 0:
        err = (p.stderr or p.stdout or "").strip()
        raise RuntimeError(f"Не удалось создать предпрослушивание beep: {err[-900:]}")
    return str(out)


def render_censored_video(
    input_path: str,
    output_path: str,
    regions: List[CensorRegion],
    *,
    mode: str = "beep",
    render_device: str = "cpu",
    beep_frequency: int = 1000,
    beep_volume: float = 0.90,
    beep_profile: str = "classic",
    beep_duck_level: float = 0.08,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> str:
    src = Path(input_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    active = [r for r in (regions or []) if bool(getattr(r, "enabled", True))]
    if not active:
        shutil.copy2(src, out)
        if progress_cb:
            progress_cb(100, "Запрещённых фрагментов нет — файл скопирован")
        return str(out)

    duration = _probe_duration_sec(str(src))
    expr = _build_activity_expr(active)
    beep_expr_regions = _build_activity_expr(active, mode_filter="beep")
    mute_expr_regions = _build_activity_expr(active, mode_filter="mute")

    has_video = src.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".ts"}
    render_dev = str(render_device or "cpu").strip().lower()
    is_gpu = render_dev in {"cuda", "gpu", "nvenc"}
    # Для текущего пайплайна мы модифицируем только аудио-дорожку,
    # поэтому копирование видео-потока существенно быстрее и не ухудшает качество.
    # Режим CPU/GPU оставляем как индикатор/бэкенд рендера, но без принудительного
    # ре-энкода видео, чтобы не замедлять экспорт.
    vcodec = "copy" if has_video else ("h264_nvenc" if is_gpu else "libx264")

    if progress_cb:
        progress_cb(15, f"Рендер антимата: подготовка FFmpeg ({'GPU/CUDA' if is_gpu else 'CPU'})")

    cmd: List[str]
    mode_norm = str(mode).strip().lower()
    if mode_norm == "mute":
        audio_expr = f"if(gt({expr},0),0,1)"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-filter_complex",
            f"[0:a]volume='{audio_expr}':eval=frame[aout]",
            "-map",
            "0:v?",
            "-map",
            "[aout]",
            "-c:v",
            vcodec,
            "-c:a",
            "aac",
            str(out),
        ]
    elif mode_norm == "mixed":
        duck_level = max(0.0, min(1.0, float(beep_duck_level)))
        duck_expr = f"if(gt({beep_expr_regions},0),{duck_level:.3f},if(gt({mute_expr_regions},0),0,1))"
        beep_expr = f"if(gt({beep_expr_regions},0),{float(beep_volume):.3f},0)"
        source_expr = _build_beep_source_expr(beep_profile, beep_frequency, duration)
        profile_post = ""
        p_norm = str(beep_profile or "classic").strip().lower()
        if p_norm == "soft":
            profile_post = ",lowpass=f=2200"
        elif p_norm == "radio":
            profile_post = ",highpass=f=700,lowpass=f=3400"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-f",
            "lavfi",
            "-i",
            source_expr,
            "-filter_complex",
            (
                f"[0:a]volume='{duck_expr}':eval=frame[a0];"
                f"[1:a]volume='{beep_expr}':eval=frame{profile_post}[b];"
                "[a0][b]amix=inputs=2:normalize=0[aout]"
            ),
            "-map",
            "0:v?",
            "-map",
            "[aout]",
            "-c:v",
            vcodec,
            "-c:a",
            "aac",
            str(out),
        ]
    else:
        duck_level = max(0.0, min(1.0, float(beep_duck_level)))
        duck_expr = f"if(gt({expr},0),{duck_level:.3f},1)"
        beep_expr = f"if(gt({expr},0),{float(beep_volume):.3f},0)"
        source_expr = _build_beep_source_expr(beep_profile, beep_frequency, duration)
        profile_post = ""
        p_norm = str(beep_profile or "classic").strip().lower()
        if p_norm == "soft":
            profile_post = ",lowpass=f=2200"
        elif p_norm == "radio":
            profile_post = ",highpass=f=700,lowpass=f=3400"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-f",
            "lavfi",
            "-i",
            source_expr,
            "-filter_complex",
            (
                f"[0:a]volume='{duck_expr}':eval=frame[a0];"
                f"[1:a]volume='{beep_expr}':eval=frame{profile_post}[b];"
                "[a0][b]amix=inputs=2:normalize=0[aout]"
            ),
            "-map",
            "0:v?",
            "-map",
            "[aout]",
            "-c:v",
            vcodec,
            "-c:a",
            "aac",
            str(out),
        ]

    cmd = [
        cmd[0],
        "-progress",
        "pipe:1",
        "-nostats",
        *cmd[1:],
    ]

    if progress_cb:
        progress_cb(35, "FFmpeg: применение цензуры")

    with tempfile.TemporaryDirectory(prefix="anti_mate_"):
        creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
        )

        last_progress = 35
        stderr_text = ""
        try:
            while True:
                line = proc.stdout.readline() if proc.stdout else ""
                if not line:
                    if proc.poll() is not None:
                        break
                    continue
                line = line.strip()
                if not line or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key == "out_time_ms":
                    try:
                        out_sec = max(0.0, float(value) / 1_000_000.0)
                        ratio = min(1.0, out_sec / max(0.1, duration))
                        p = 35 + int(ratio * 63)
                        if p > last_progress and (p - last_progress >= 2 or p >= 98):
                            last_progress = p
                            if progress_cb:
                                progress_cb(min(98, p), f"FFmpeg: рендер {int(ratio * 100)}%")
                    except Exception:
                        pass
                elif key == "progress" and value == "end":
                    if progress_cb:
                        progress_cb(99, "FFmpeg: финализация")

            stderr_text = (proc.stderr.read() or "") if proc.stderr else ""
            return_code = proc.wait()
            if return_code != 0:
                err = (stderr_text or "").strip()
                raise RuntimeError(f"FFmpeg ошибка при рендере антимата: {err[-1200:]}")
        finally:
            try:
                if proc.stdout:
                    proc.stdout.close()
                if proc.stderr:
                    proc.stderr.close()
            except Exception:
                pass

    if progress_cb:
        progress_cb(100, f"Готово: {out}")
    return str(out)
