"""
Система управления эффектами для субтитров
"""

from dataclasses import dataclass
from enum import Enum
import re
from typing import Dict, Optional, List, Any


class SubtitleEffect(str, Enum):
    """Типы эффектов для субтитров"""
    NONE = "none"
    FADE_IN = "fade_in"
    FADE_OUT = "fade_out"
    FADE_IN_OUT = "fade_in_out"
    BOUNCE = "bounce"
    PULSE = "pulse"
    WAVE = "wave"
    SPIN = "spin"
    ZOOM_IN = "zoom_in"
    SWING = "swing"
    GLITCH = "glitch"
    TYPEWRITER = "typewriter"  # появление слов по одному
    TWINKLE = "twinkle"  # мерцание
    RAINBOW = "rainbow"  # радуга цветов
    SHINE = "shine"  # подсветка
    SLIDE_UP = "slide_up"
    SLIDE_LEFT = "slide_left"
    POP_ROTATE = "pop_rotate"
    SHAKE = "shake"
    NEON_FLICKER = "neon_flicker"
    WORD_HIGHLIGHT = "word_highlight"


@dataclass
class EffectConfig:
    """Конфигурация эффекта"""
    effect_type: SubtitleEffect
    duration: float = 1.0  # продолжительность эффекта в секундах
    intensity: float = 1.0  # интенсивность эффекта (от 0 до 1)
    color: Optional[str] = None  # цвет для эффектов, поддерживающих цвет


class EffectManager:
    """Менеджер эффектов субтитров"""

    # Единый реестр эффектов для UI и валидации.
    # format: (label, value, category)
    EFFECT_REGISTRY = [
        ("Без эффекта", SubtitleEffect.NONE.value, "basic"),
        ("Плавное появление", SubtitleEffect.FADE_IN.value, "fade"),
        ("Плавное исчезновение", SubtitleEffect.FADE_OUT.value, "fade"),
        ("Появление + исчезновение", SubtitleEffect.FADE_IN_OUT.value, "fade"),
        ("Прыжок", SubtitleEffect.BOUNCE.value, "motion"),
        ("Пульсация", SubtitleEffect.PULSE.value, "emphasis"),
        ("Волна", SubtitleEffect.WAVE.value, "motion"),
        ("Вращение", SubtitleEffect.SPIN.value, "motion"),
        ("Увеличение", SubtitleEffect.ZOOM_IN.value, "motion"),
        ("Качание", SubtitleEffect.SWING.value, "motion"),
        ("Скольжение снизу", SubtitleEffect.SLIDE_UP.value, "motion"),
        ("Скольжение слева", SubtitleEffect.SLIDE_LEFT.value, "motion"),
        ("Поп + поворот", SubtitleEffect.POP_ROTATE.value, "motion"),
        ("Дрожание", SubtitleEffect.SHAKE.value, "motion"),
        ("Неоновое мерцание", SubtitleEffect.NEON_FLICKER.value, "style"),
        ("Глитч", SubtitleEffect.GLITCH.value, "style"),
        ("Печатная машинка", SubtitleEffect.TYPEWRITER.value, "style"),
        ("Подсветка по словам", SubtitleEffect.WORD_HIGHLIGHT.value, "karaoke"),
        ("Мерцание", SubtitleEffect.TWINKLE.value, "style"),
        ("Радуга", SubtitleEffect.RAINBOW.value, "style"),
        ("Блик", SubtitleEffect.SHINE.value, "style"),
    ]
    
    def __init__(self):
        self.effects: Dict[str, EffectConfig] = {}
        
    def add_effect(self, subtitle_id: str, effect_config: EffectConfig):
        """Добавить эффект к конкретному субтитру"""
        self.effects[subtitle_id] = effect_config
        
    def get_effect(self, subtitle_id: str) -> Optional[EffectConfig]:
        """Получить эффект для субтитра"""
        return self.effects.get(subtitle_id)
        
    def remove_effect(self, subtitle_id: str):
        """Удалить эффект для субтитра"""
        if subtitle_id in self.effects:
            del self.effects[subtitle_id]
            
    def apply_effects_to_subtitle(self, subtitle_text: str, 
                                  effect_config: EffectConfig,
                                  time_position: float) -> str:
        """
        Применить эффект к тексту субтитра на основе временной позиции
        """
        if effect_config.effect_type == SubtitleEffect.NONE:
            return subtitle_text
            
        elif effect_config.effect_type == SubtitleEffect.FADE_IN:
            return self.generate_fade_in_effect(subtitle_text, time_position, effect_config)
            
        elif effect_config.effect_type == SubtitleEffect.FADE_OUT:
            return self.generate_fade_out_effect(subtitle_text, time_position, effect_config)
            
        elif effect_config.effect_type == SubtitleEffect.FADE_IN_OUT:
            return self.generate_fade_in_out_effect(subtitle_text, time_position, effect_config)
            
        elif effect_config.effect_type == SubtitleEffect.BOUNCE:
            return self.generate_bounce_effect(subtitle_text, time_position, effect_config)
            
        elif effect_config.effect_type == SubtitleEffect.PULSE:
            return self.generate_pulse_effect(subtitle_text, time_position, effect_config)
            
        elif effect_config.effect_type == SubtitleEffect.WAVE:
            return self.generate_wave_effect(subtitle_text, time_position, effect_config)
            
        elif effect_config.effect_type == SubtitleEffect.SPIN:
            return self.generate_spin_effect(subtitle_text, time_position, effect_config)
            
        elif effect_config.effect_type == SubtitleEffect.ZOOM_IN:
            return self.generate_zoom_in_effect(subtitle_text, time_position, effect_config)
            
        elif effect_config.effect_type == SubtitleEffect.SWING:
            return self.generate_swing_effect(subtitle_text, time_position, effect_config)
            
        elif effect_config.effect_type == SubtitleEffect.GLITCH:
            return self.generate_glitch_effect(subtitle_text, time_position, effect_config)
            
        elif effect_config.effect_type == SubtitleEffect.TYPEWRITER:
            return self.generate_typewriter_effect(subtitle_text, time_position, effect_config.duration)
            
        elif effect_config.effect_type == SubtitleEffect.TWINKLE:
            return self.generate_twinkle_effect(subtitle_text, time_position, effect_config)
            
        elif effect_config.effect_type == SubtitleEffect.RAINBOW:
            return self.generate_rainbow_effect(subtitle_text, time_position, effect_config)
            
        elif effect_config.effect_type == SubtitleEffect.SHINE:
            return self.generate_shine_effect(subtitle_text, time_position, effect_config)
            
        return subtitle_text
        
    def generate_typewriter_effect(self, text: str, time_position: float,
                                   total_duration: float) -> str:
        """
        Симуляция эффекта пишущей машинки - слова появляются по одному
        """
        # Разбиваем текст на слова
        words = text.split()
        
        # Рассчитываем, сколько слов должно быть видно в данный момент
        if total_duration <= 0:
            return ""
            
        words_per_second = len(words) / total_duration
        visible_words_count = int(time_position * words_per_second)
        
        # Ограничиваем количество видимых слов
        visible_words_count = max(0, min(visible_words_count, len(words)))
        
        # Возвращаем только те слова, которые должны быть видны
        return " ".join(words[:visible_words_count])
    
    def generate_fade_in_effect(self, text: str, time_position: float,
                                effect_config: EffectConfig) -> str:
        """
        Эффект плавного появления - прозрачность увеличивается от 0 до 1
        """
        progress = min(time_position / effect_config.duration, 1.0)
        opacity = progress * effect_config.intensity
        return f"<fade_in opacity='{opacity:.2f}'>{text}</fade_in>"
    
    def generate_fade_out_effect(self, text: str, time_position: float,
                                 effect_config: EffectConfig) -> str:
        """
        Эффект плавного исчезновения - прозрачность уменьшается от 1 до 0
        """
        progress = min(time_position / effect_config.duration, 1.0)
        opacity = 1.0 - (progress * effect_config.intensity)
        return f"<fade_out opacity='{opacity:.2f}'>{text}</fade_out>"
    
    def generate_fade_in_out_effect(self, text: str, time_position: float,
                                    effect_config: EffectConfig) -> str:
        """
        Комбинированный эффект - сначала появление, затем исчезновение
        """
        progress = min(time_position / effect_config.duration, 1.0)
        if progress < 0.5:
            opacity = progress * 2 * effect_config.intensity
        else:
            opacity = (1.0 - progress) * 2 * effect_config.intensity
        return f"<fade_in_out opacity='{max(0, min(1, opacity)):.2f}'>{text}</fade_in_out>"
    
    def generate_bounce_effect(self, text: str, time_position: float,
                               effect_config: EffectConfig) -> str:
        """
        Эффект отскока - текст подпрыгивает вверх и вниз
        """
        import math
        # Используем синусоидальную функцию для создания эффекта отскока
        bounce_offset = math.sin(time_position * 10) * effect_config.intensity * 20
        return f"<bounce offset_y='{bounce_offset:.1f}'>{text}</bounce>"
    
    def generate_pulse_effect(self, text: str, time_position: float,
                              effect_config: EffectConfig) -> str:
        """
        Эффект пульсации - размер текста плавно увеличивается и уменьшается
        """
        import math
        # Используем синусоиду для плавной пульсации
        scale = 1.0 + math.sin(time_position * 6) * effect_config.intensity * 0.2
        return f"<pulse scale='{scale:.2f}'>{text}</pulse>"
    
    def generate_wave_effect(self, text: str, time_position: float,
                             effect_config: EffectConfig) -> str:
        """
        Волновой эффект - буквы движутся вверх-вниз по волне
        """
        import math
        words = text.split()
        waved_words = []
        for i, word in enumerate(words):
            wave_offset = math.sin(time_position * 5 + i * 0.5) * effect_config.intensity * 10
            waved_words.append(f"<wave offset_y='{wave_offset:.1f}'>{word}</wave>")
        return " ".join(waved_words)
    
    def generate_spin_effect(self, text: str, time_position: float,
                             effect_config: EffectConfig) -> str:
        """
        Эффект вращения - текст вращается вокруг своей оси
        """
        import math
        rotation = (time_position * 360 / effect_config.duration) % 360
        return f"<spin rotation='{rotation:.1f}'>{text}</spin>"
    
    def generate_zoom_in_effect(self, text: str, time_position: float,
                                effect_config: EffectConfig) -> str:
        """
        Эффект приближения - текст увеличивается от 0 до полного размера
        """
        progress = min(time_position / effect_config.duration, 1.0)
        scale = progress * effect_config.intensity
        return f"<zoom_in scale='{scale:.2f}'>{text}</zoom_in>"
    
    def generate_swing_effect(self, text: str, time_position: float,
                              effect_config: EffectConfig) -> str:
        """
        Эффект качания - текст раскачивается влево-вправо
        """
        import math
        angle = math.sin(time_position * 4) * effect_config.intensity * 15
        return f"<swing rotation='{angle:.1f}'>{text}</swing>"
    
    def generate_glitch_effect(self, text: str, time_position: float,
                               effect_config: EffectConfig) -> str:
        """
        Эффект глитча - случайные искажения текста
        """
        import random
        if random.random() < effect_config.intensity * 0.3:
            glitched_text = list(text)
            for i in range(len(glitched_text)):
                if random.random() < effect_config.intensity * 0.2:
                    glitched_text[i] = chr(ord(glitched_text[i]) + int(random.gauss(0, 5)))
            return f"<glitch>{''.join(glitched_text)}</glitch>"
        return text
    
    def generate_twinkle_effect(self, text: str, time_position: float,
                                effect_config: EffectConfig) -> str:
        """
        Эффект мерцания - текст периодически мигает
        """
        import math
        opacity = 0.5 + 0.5 * math.sin(time_position * 12)
        return f"<twinkle opacity='{opacity:.2f}'>{text}</twinkle>"
    
    def generate_rainbow_effect(self, text: str, time_position: float,
                                effect_config: EffectConfig) -> str:
        """
        Радужный эффект - разные цвета для разных букв
        """
        colors = ['#FF0000', '#FF7F00', '#FFFF00', '#00FF00', '#0000FF', '#4B0082', '#9400D3']
        colored_chars = []
        for i, char in enumerate(text):
            color_idx = (i + int(time_position * 5)) % len(colors)
            colored_chars.append(f"<rainbow color='{colors[color_idx]}'>{char}</rainbow>")
        return "".join(colored_chars)
    
    def generate_shine_effect(self, text: str, time_position: float,
                              effect_config: EffectConfig) -> str:
        """
        Эффект подсветки - движущаяся полоса света через текст
        """
        shine_pos = (time_position * 100 / effect_config.duration) % 100
        return f"<shine position='{shine_pos:.1f}'>{text}</shine>"

    @staticmethod
    def get_effect_options() -> Dict[str, str]:
        """Опции эффектов для UI: {label: value}."""
        return {label: value for label, value, _ in EffectManager.EFFECT_REGISTRY}

    @staticmethod
    def get_effect_registry_duplicates() -> Dict[str, list[str]]:
        """Проверка дубликатов в реестре эффектов."""
        labels = [label for label, _, _ in EffectManager.EFFECT_REGISTRY]
        values = [value for _, value, _ in EffectManager.EFFECT_REGISTRY]

        dup_labels = sorted({x for x in labels if labels.count(x) > 1})
        dup_values = sorted({x for x in values if values.count(x) > 1})
        return {
            "labels": dup_labels,
            "values": dup_values,
        }

    @staticmethod
    def get_motion_effect_types() -> set[str]:
        """Эффекты, где используются direction/amplitude/easing/jitter."""
        return {
            SubtitleEffect.BOUNCE.value,
            SubtitleEffect.WAVE.value,
            SubtitleEffect.SLIDE_UP.value,
            SubtitleEffect.SLIDE_LEFT.value,
            SubtitleEffect.SHAKE.value,
            SubtitleEffect.ZOOM_IN.value,
        }

    @staticmethod
    def is_motion_customizable(effect_type: str) -> bool:
        return effect_type in EffectManager.get_motion_effect_types()

    @staticmethod
    def _hex_to_ass_bgr(hex_color: str) -> str:
        """#RRGGBB -> &H00BBGGRR"""
        if not hex_color:
            return "&H00FF0000"
        color = hex_color.strip().lstrip("#")
        if len(color) != 6:
            return "&H00FF0000"
        rr, gg, bb = color[0:2], color[2:4], color[4:6]
        return f"&H00{bb}{gg}{rr}".upper()

    @staticmethod
    def _easing_accel(easing: str) -> float:
        easing = (easing or "ease_out").lower()
        if easing == "ease_in":
            return 1.7
        if easing == "ease_in_out":
            return 1.2
        if easing == "linear":
            return 1.0
        return 0.7  # ease_out

    @staticmethod
    def _dir_to_offset(direction: str, distance: int) -> tuple[int, int]:
        direction = (direction or "up").lower()
        if direction == "down":
            return 0, -distance
        if direction == "left":
            return distance, 0
        if direction == "right":
            return -distance, 0
        return 0, distance  # up: появление снизу

    @staticmethod
    def _word_highlight_ass(text: str, effect_duration_ms: int) -> str:
        if not text:
            return text

        # Если строка уже содержит ASS override-теги, не дробим её повторно,
        # иначе можно получить «системные коды» в видимом тексте.
        if "{\\" in text:
            return text

        # Для языков с пробелами: подсветка по словам; иначе — по символам
        tokens = text.split()
        joiner = " "
        if len(tokens) <= 1:
            compact = text.strip()
            if not compact:
                return text
            tokens = list(compact)
            joiner = ""

        total_cs = max(1, int(effect_duration_ms / 10))
        per_cs = max(1, total_cs // max(1, len(tokens)))
        return joiner.join([f"{{\\k{per_cs}}}{token}" for token in tokens])

    @staticmethod
    def _hex_to_ass_primary(hex_color: str, fallback: str = "&H00FFFFFF&") -> str:
        """#RRGGBB -> &H00BBGGRR& для ASS \1c/\2c"""
        if not hex_color:
            return fallback
        color = hex_color.strip().lstrip("#")
        if len(color) != 6:
            return fallback
        rr, gg, bb = color[0:2], color[2:4], color[4:6]
        return f"&H00{bb}{gg}{rr}&".upper()

    @staticmethod
    def _apply_gradient(text: str, mode: str, color_1: str, color_2: str) -> str:
        """Простая символьная раскраска для градиентного вида."""
        if not text:
            return text
        mode = (mode or "off").lower()
        if mode == "off":
            return text

        if mode == "rainbow":
            palette = ["#FF4D4D", "#FF9F43", "#FFD93D", "#6BCB77", "#4D96FF", "#B980F0"]
        else:
            palette = [color_1 or "#FFFFFF", color_2 or "#66CCFF"]

        ass_palette = [EffectManager._hex_to_ass_primary(c) for c in palette]
        out = []
        color_idx = 0
        i = 0
        n = len(text)

        while i < n:
            ch = text[i]
            # Сохраняем ASS override-теги как есть: {...}
            if ch == "{":
                j = i + 1
                while j < n and text[j] != "}":
                    j += 1
                if j < n:
                    out.append(text[i : j + 1])
                    i = j + 1
                    continue
                # Невалидный хвост без закрывающей скобки
                out.append(text[i:])
                break

            if ch.isspace():
                out.append(ch)
            else:
                out.append(f"{{\\1c{ass_palette[color_idx % len(ass_palette)]}}}{ch}")
                color_idx += 1
            i += 1
        return "".join(out)

    @staticmethod
    def _choose_word_ts_transform(
        raw_pairs: List[tuple[float, float]],
        seg_start: int,
        seg_end: int,
    ):
        """Подбирает преобразование word timestamps (ms/sec, absolute/relative)."""
        duration = max(1, seg_end - seg_start)
        modes = {
            "ms_abs": lambda x: x,
            "sec_abs": lambda x: x * 1000.0,
            "ms_rel": lambda x: seg_start + x,
            "sec_rel": lambda x: seg_start + x * 1000.0,
        }

        def _score(transform) -> int:
            score = 0
            for s_raw, e_raw in raw_pairs[:40]:
                s = transform(s_raw)
                e = transform(e_raw)
                if e <= s:
                    continue
                score += 1
                # Попадание в окрестность сегмента
                if s <= seg_end + duration and e >= seg_start - duration:
                    score += 2
                if seg_start - 200 <= s <= seg_end + 200:
                    score += 1
                if seg_start - 200 <= e <= seg_end + 200:
                    score += 1
            return score

        best_name = "ms_abs"
        best_score = -1
        for name, fn in modes.items():
            sc = _score(fn)
            if sc > best_score:
                best_score = sc
                best_name = name
        return modes[best_name]

    @staticmethod
    def apply_ass_effect(
        text: str,
        effect_type: str,
        start_ms: int,
        end_ms: int,
        effect_duration_ms: int = 300,
        effect_intensity: float = 1.0,
        rainbow_end_color: str = "#0000FF",
        index: int = 0,
        motion_direction: str = "up",
        motion_amplitude: float = 1.0,
        motion_easing: str = "ease_out",
        motion_jitter: float = 0.0,
        play_res_x: int = 1280,
        play_res_y: int = 720,
        karaoke_mode: bool = False,
        karaoke_window_ms: int = 1200,
        auto_contrast: bool = False,
        anti_flicker: bool = False,
        gradient_mode: str = "off",
        gradient_color_1: str = "#FFFFFF",
        gradient_color_2: str = "#66CCFF",
        use_word_timestamps: bool = False,
        word_timestamps: Optional[List[Dict[str, Any]]] = None,
        segment_start_ms: Optional[int] = None,
        segment_end_ms: Optional[int] = None,
        anchor_x: Optional[int] = None,
        anchor_y: Optional[int] = None,
        safe_area_enabled: bool = True,
        safe_margin_x: int = 8,
        safe_margin_y: int = 10,
        speaker_color_mode: str = "off",
        motion_blur_strength: float = 0.0,
    ) -> str:
        """Преобразует текст в ASS override-теги для базовых анимаций."""
        if not text:
            return text

        duration = max(end_ms - start_ms, 1)

        effect_duration_ms = max(20, min(effect_duration_ms, duration))
        intensity = max(0.01, min(effect_intensity, 5.0))
        amp = max(0.01, min(motion_amplitude, 5.0))
        jitter = max(0.0, min(motion_jitter, 2.0))
        if anti_flicker:
            intensity *= 0.9
            jitter *= 0.45
            effect_duration_ms = max(120, effect_duration_ms)
        accel = EffectManager._easing_accel(motion_easing)
        j = int((6 * jitter) * (-1 if index % 2 else 1))
        safe_x = max(320, int(play_res_x or 1280))
        safe_y = max(180, int(play_res_y or 720))
        base_x = int(anchor_x) if anchor_x is not None else safe_x // 2
        base_y = int(anchor_y) if anchor_y is not None else int(safe_y * 0.9167)
        if safe_area_enabled:
            margin_x_px = int(max(0, min(40, safe_margin_x)) * safe_x / 100)
            margin_y_px = int(max(0, min(40, safe_margin_y)) * safe_y / 100)
            base_x = max(margin_x_px, min(safe_x - margin_x_px, base_x))
            base_y = max(margin_y_px, min(safe_y - margin_y_px, base_y))
        else:
            base_x = max(0, min(safe_x, base_x))
            base_y = max(0, min(safe_y, base_y))

        # Базовая обработка текста перед motion-эффектом
        processed_text = text
        wants_karaoke = karaoke_mode or effect_type == SubtitleEffect.WORD_HIGHLIGHT.value
        karaoke_active = False

        # Ключевая логика:
        # - если есть реальные word timestamps -> используем их
        # - если word timestamps нет -> делаем синтетический \k-каркас,
        #   чтобы пресет Karaoke визуально оставался "подсветкой по словам"
        if wants_karaoke and use_word_timestamps:
            wt = word_timestamps or []
            if wt:
                seg_start = start_ms if segment_start_ms is None else segment_start_ms
                seg_end = end_ms if segment_end_ms is None else segment_end_ms
                raw_pairs: List[tuple[float, float]] = []
                for item in wt:
                    try:
                        s_raw = float(item.get("start_time", 0))
                        e_raw = float(item.get("end_time", s_raw))
                        raw_pairs.append((s_raw, e_raw))
                    except (TypeError, ValueError):
                        continue

                ts_transform = EffectManager._choose_word_ts_transform(
                    raw_pairs,
                    int(seg_start),
                    int(seg_end),
                )

                tokens = []
                for item in wt:
                    t = str(item.get("text", "")).strip()
                    if not t:
                        continue
                    try:
                        ws_raw = float(item.get("start_time", seg_start))
                        we_raw = float(item.get("end_time", ws_raw))
                    except (TypeError, ValueError):
                        continue

                    ws = int(ts_transform(ws_raw))
                    we = int(ts_transform(we_raw))
                    ws = max(seg_start, ws)
                    we = min(seg_end, max(ws + 10, we))
                    k_cs = max(1, int((we - ws) / 10))
                    tokens.append((t, ws, we, k_cs))

                if tokens:
                    joined = []
                    cursor_ms = seg_start
                    for i, (token, ws, we, k_cs) in enumerate(tokens):
                        # 保留词与词之间的真实停顿，避免“按顺序提前高亮”
                        if ws > cursor_ms:
                            gap_cs = max(1, int((ws - cursor_ms) / 10))
                            joined.append(f"{{\\k{gap_cs}}}")

                        if i > 0:
                            prev = tokens[i - 1][0]
                            # Для языков с пробелами (включая кириллицу) сохраняем пробел
                            # между соседними словами, чтобы не склеивать текст.
                            if re.match(r"^[\w']+$", prev, flags=re.UNICODE) and re.match(
                                r"^[\w']+$", token, flags=re.UNICODE
                            ):
                                joined.append(" ")
                        joined.append(f"{{\\k{k_cs}}}{token}")
                        cursor_ms = max(cursor_ms, we)

                    # 尾部停顿也保留，避免最后一个词太早结束
                    if cursor_ms < seg_end:
                        tail_cs = max(1, int((seg_end - cursor_ms) / 10))
                        joined.append(f"{{\\k{tail_cs}}}")

                    processed_text = "".join(joined)
                    karaoke_active = True
            if not karaoke_active:
                # В режиме "настоящего" karaoke без валидных word timestamps
                # НЕ включаем синтетическую подсветку, чтобы не было псевдо-эффекта.
                karaoke_active = False
        elif wants_karaoke:
            # Если word timestamps не включены на уровне данных,
            # не имитируем karaoke искусственно.
            karaoke_active = False

        if karaoke_active:
            # \2c — неактивный текст, \1c — активное слово.
            # Цвета берём из пользовательских настроек UI.
            inactive_color_ass = EffectManager._hex_to_ass_primary(
                gradient_color_1 or "#B0B0B0",
                fallback="&H00B0B0B0&",
            )
            active_color_ass = EffectManager._hex_to_ass_primary(
                gradient_color_2 or "#FFFFFF",
                fallback="&H00FFFFFF&",
            )
            processed_text = (
                f"{{\\2c{inactive_color_ass}\\1c{active_color_ass}}}"
                f"{processed_text}"
            )

        if (not karaoke_active) and (speaker_color_mode or "off").lower() == "alternate":
            speaker_palette = ["#8FD3FF", "#FFD27A", "#B5F5B0", "#F6A9FF"]
            speaker_color = speaker_palette[index % len(speaker_palette)]
            processed_text = EffectManager._apply_gradient(
                processed_text,
                "two_color",
                speaker_color,
                speaker_color,
            )

        # В karaoke-режиме градиент даёт частые \1c-override на каждый символ,
        # что визуально "съедает" подсветку активного слова. Поэтому отключаем
        # градиент на время karaoke, чтобы эффект был читаемым и контрастным.
        if not karaoke_active:
            processed_text = EffectManager._apply_gradient(
                processed_text,
                gradient_mode,
                gradient_color_1,
                gradient_color_2,
            )

        if auto_contrast:
            processed_text = f"{{\\bord3\\shad1\\3c&H000000&\\blur1}}{processed_text}"

        motion_blur = max(0.0, min(20.0, float(motion_blur_strength or 0.0)))
        if motion_blur > 0:
            processed_text = f"{{\\blur{motion_blur:.1f}}}{processed_text}"

        if effect_type == SubtitleEffect.NONE.value:
            return processed_text

        if effect_type == SubtitleEffect.FADE_IN.value:
            return f"{{\\fad({min(effect_duration_ms, duration // 2)},0)}}{processed_text}"

        if effect_type == SubtitleEffect.FADE_OUT.value:
            return f"{{\\fad(0,{min(effect_duration_ms, duration // 2)})}}{processed_text}"

        if effect_type == SubtitleEffect.FADE_IN_OUT.value:
            fad = min(effect_duration_ms, duration // 3)
            return f"{{\\fad({fad},{fad})}}{processed_text}"

        if effect_type == SubtitleEffect.BOUNCE.value:
            jump = int(30 * intensity * amp)
            dx, dy = EffectManager._dir_to_offset(motion_direction, jump)
            sx, sy = base_x + dx + j, base_y + dy
            return f"{{\\move({sx},{sy},{base_x},{base_y},0,{min(effect_duration_ms, duration)})}}{processed_text}"

        if effect_type == SubtitleEffect.PULSE.value:
            scale = int(100 + 12 * intensity)
            return f"{{\\t(0,{min(effect_duration_ms, duration)},\\fscx{scale}\\fscy{scale})}}{processed_text}"

        if effect_type == SubtitleEffect.WAVE.value:
            wave_amp = int(8 * intensity * amp)
            y1 = base_y + (index % 2) * wave_amp
            y2 = base_y - 10 - (index % 2) * wave_amp
            return f"{{\\move({base_x+j},{y1},{base_x},{y2},0,{min(effect_duration_ms, duration)})}}{processed_text}"

        if effect_type == SubtitleEffect.SPIN.value:
            rot = int(360 * intensity)
            return f"{{\\frz0\\t(0,{min(effect_duration_ms, duration)},\\frz{rot})}}{processed_text}"

        if effect_type == SubtitleEffect.ZOOM_IN.value:
            start_scale = max(30, int(100 - 30 * intensity))
            distance = int(80 * amp)
            dx, dy = EffectManager._dir_to_offset(motion_direction, distance)
            sx, sy = base_x + dx + j, base_y + dy
            return (
                f"{{\\move({sx},{sy},{base_x},{base_y},0,{min(effect_duration_ms, duration)})"
                f"\\fscx{start_scale}\\fscy{start_scale}"
                f"\\t(0,{min(effect_duration_ms, duration)},{accel:.2f},\\fscx100\\fscy100)}}{processed_text}"
            )

        if effect_type == SubtitleEffect.SWING.value:
            return (
                f"{{\\frz0\\t(0,{min(effect_duration_ms//2, duration)},\\frz{int(8*intensity)})"
                f"\\t({min(effect_duration_ms//2, duration)},{min(effect_duration_ms, duration)},\\frz{-int(8*intensity)})}}{processed_text}"
            )

        if effect_type == SubtitleEffect.SLIDE_UP.value:
            distance = int(70 * intensity * amp)
            dx, dy = EffectManager._dir_to_offset(motion_direction, distance)
            sx, sy = base_x + dx + j, base_y + dy
            return f"{{\\move({sx},{sy},{base_x},{base_y},0,{min(effect_duration_ms, duration)})}}{processed_text}"

        if effect_type == SubtitleEffect.SLIDE_LEFT.value:
            distance = int(240 * amp)
            dx, dy = EffectManager._dir_to_offset(motion_direction, distance)
            sx, sy = base_x + dx + j, base_y + dy
            return f"{{\\move({sx},{sy},{base_x},{base_y},0,{min(effect_duration_ms, duration)})}}{processed_text}"

        if effect_type == SubtitleEffect.POP_ROTATE.value:
            start_scale = max(35, int(100 - 45 * intensity))
            rot = int(18 * intensity)
            return (
                f"{{\\fscx{start_scale}\\fscy{start_scale}\\frz{rot}"
                f"\\t(0,{min(effect_duration_ms, duration)},\\fscx100\\fscy100\\frz0)}}{processed_text}"
            )

        if effect_type == SubtitleEffect.SHAKE.value:
            shake_amp = max(2, int(6 * intensity * amp))
            t1 = min(effect_duration_ms // 3, duration)
            t2 = min((effect_duration_ms * 2) // 3, duration)
            t3 = min(effect_duration_ms, duration)
            return (
                f"{{\\pos({base_x},{base_y})"
                f"\\t(0,{t1},\\pos({base_x-shake_amp+j},{base_y+shake_amp}))"
                f"\\t({t1},{t2},\\pos({base_x+shake_amp+j},{base_y-shake_amp}))"
                f"\\t({t2},{t3},\\pos({base_x},{base_y}))}}{processed_text}"
            )

        if effect_type == SubtitleEffect.NEON_FLICKER.value:
            glow_alpha = max(0, min(255, int(110 - 35 * intensity)))
            return (
                f"{{\\bord3\\blur{max(1, int(2*intensity))}\\3a&H{glow_alpha:02X}&"
                f"\\t(0,{min(effect_duration_ms, duration)},\\3a&H00&)}}{processed_text}"
            )

        if effect_type == SubtitleEffect.TWINKLE.value:
            alpha = max(0, min(255, int(68 * intensity)))
            return f"{{\\alpha&H{alpha:02X}&\\t(0,{min(effect_duration_ms, duration)},\\alpha&H00&)}}{processed_text}"

        if effect_type == SubtitleEffect.SHINE.value:
            alpha = max(0, min(255, int(68 * intensity)))
            return f"{{\\1a&H{alpha:02X}&\\t(0,{min(effect_duration_ms, duration)},\\1a&H00&)}}{processed_text}"

        if effect_type == SubtitleEffect.TYPEWRITER.value:
            # Простая имитация: плавный fade-in + более быстрое появление
            return f"{{\\fad({min(effect_duration_ms, duration // 3)},0)}}{processed_text}"

        if effect_type == SubtitleEffect.WORD_HIGHLIGHT.value:
            return processed_text

        if effect_type == SubtitleEffect.RAINBOW.value:
            # ASS не поддерживает покадровую радугу без сложной генерации,
            # даем мягкую цветовую анимацию
            end_ass_color = EffectManager._hex_to_ass_bgr(rainbow_end_color)
            return f"{{\\1c&H00FFFFFF&\\t(0,{min(effect_duration_ms, duration)},\\1c{end_ass_color})}}{processed_text}"

        return processed_text
