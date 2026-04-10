"""
Система управления эффектами для субтитров
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


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
        return {
            "Без эффекта": SubtitleEffect.NONE.value,
            "Плавное появление": SubtitleEffect.FADE_IN.value,
            "Плавное исчезновение": SubtitleEffect.FADE_OUT.value,
            "Появление + исчезновение": SubtitleEffect.FADE_IN_OUT.value,
            "Прыжок": SubtitleEffect.BOUNCE.value,
            "Пульсация": SubtitleEffect.PULSE.value,
            "Волна": SubtitleEffect.WAVE.value,
            "Вращение": SubtitleEffect.SPIN.value,
            "Увеличение": SubtitleEffect.ZOOM_IN.value,
            "Качание": SubtitleEffect.SWING.value,
            "Скольжение снизу": SubtitleEffect.SLIDE_UP.value,
            "Скольжение слева": SubtitleEffect.SLIDE_LEFT.value,
            "Поп + поворот": SubtitleEffect.POP_ROTATE.value,
            "Дрожание": SubtitleEffect.SHAKE.value,
            "Неоновое мерцание": SubtitleEffect.NEON_FLICKER.value,
            "Глитч": SubtitleEffect.GLITCH.value,
            "Печатная машинка": SubtitleEffect.TYPEWRITER.value,
            "Подсветка по словам": SubtitleEffect.WORD_HIGHLIGHT.value,
            "Мерцание": SubtitleEffect.TWINKLE.value,
            "Радуга": SubtitleEffect.RAINBOW.value,
            "Блик": SubtitleEffect.SHINE.value,
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
    ) -> str:
        """Преобразует текст в ASS override-теги для базовых анимаций."""
        if not text:
            return text

        duration = max(end_ms - start_ms, 1)

        if effect_type == SubtitleEffect.NONE.value:
            return text

        effect_duration_ms = max(50, min(effect_duration_ms, duration))
        intensity = max(0.1, min(effect_intensity, 3.0))
        amp = max(0.2, min(motion_amplitude, 3.0))
        jitter = max(0.0, min(motion_jitter, 1.0))
        accel = EffectManager._easing_accel(motion_easing)
        j = int((6 * jitter) * (-1 if index % 2 else 1))

        if effect_type == SubtitleEffect.FADE_IN.value:
            return f"{{\\fad({min(effect_duration_ms, duration // 2)},0)}}{text}"

        if effect_type == SubtitleEffect.FADE_OUT.value:
            return f"{{\\fad(0,{min(effect_duration_ms, duration // 2)})}}{text}"

        if effect_type == SubtitleEffect.FADE_IN_OUT.value:
            fad = min(effect_duration_ms, duration // 3)
            return f"{{\\fad({fad},{fad})}}{text}"

        if effect_type == SubtitleEffect.BOUNCE.value:
            jump = int(30 * intensity * amp)
            dx, dy = EffectManager._dir_to_offset(motion_direction, jump)
            sx, sy = 640 + dx + j, 660 + dy
            return f"{{\\move({sx},{sy},640,660,0,{min(effect_duration_ms, duration)})}}{text}"

        if effect_type == SubtitleEffect.PULSE.value:
            scale = int(100 + 12 * intensity)
            return f"{{\\t(0,{min(effect_duration_ms, duration)},\\fscx{scale}\\fscy{scale})}}{text}"

        if effect_type == SubtitleEffect.WAVE.value:
            wave_amp = int(8 * intensity * amp)
            y1 = 650 + (index % 2) * wave_amp
            y2 = 640 - (index % 2) * wave_amp
            return f"{{\\move({640+j},{y1},640,{y2},0,{min(effect_duration_ms, duration)})}}{text}"

        if effect_type == SubtitleEffect.SPIN.value:
            rot = int(360 * intensity)
            return f"{{\\frz0\\t(0,{min(effect_duration_ms, duration)},\\frz{rot})}}{text}"

        if effect_type == SubtitleEffect.ZOOM_IN.value:
            start_scale = max(30, int(100 - 30 * intensity))
            distance = int(80 * amp)
            dx, dy = EffectManager._dir_to_offset(motion_direction, distance)
            sx, sy = 640 + dx + j, 660 + dy
            return (
                f"{{\\move({sx},{sy},640,660,0,{min(effect_duration_ms, duration)})"
                f"\\fscx{start_scale}\\fscy{start_scale}"
                f"\\t(0,{min(effect_duration_ms, duration)},{accel:.2f},\\fscx100\\fscy100)}}{text}"
            )

        if effect_type == SubtitleEffect.SWING.value:
            return (
                f"{{\\frz0\\t(0,{min(effect_duration_ms//2, duration)},\\frz{int(8*intensity)})"
                f"\\t({min(effect_duration_ms//2, duration)},{min(effect_duration_ms, duration)},\\frz{-int(8*intensity)})}}{text}"
            )

        if effect_type == SubtitleEffect.SLIDE_UP.value:
            distance = int(70 * intensity * amp)
            dx, dy = EffectManager._dir_to_offset(motion_direction, distance)
            sx, sy = 640 + dx + j, 660 + dy
            return f"{{\\move({sx},{sy},640,660,0,{min(effect_duration_ms, duration)})}}{text}"

        if effect_type == SubtitleEffect.SLIDE_LEFT.value:
            distance = int(240 * amp)
            dx, dy = EffectManager._dir_to_offset(motion_direction, distance)
            sx, sy = 640 + dx + j, 660 + dy
            return f"{{\\move({sx},{sy},640,660,0,{min(effect_duration_ms, duration)})}}{text}"

        if effect_type == SubtitleEffect.POP_ROTATE.value:
            start_scale = max(35, int(100 - 45 * intensity))
            rot = int(18 * intensity)
            return (
                f"{{\\fscx{start_scale}\\fscy{start_scale}\\frz{rot}"
                f"\\t(0,{min(effect_duration_ms, duration)},\\fscx100\\fscy100\\frz0)}}{text}"
            )

        if effect_type == SubtitleEffect.SHAKE.value:
            shake_amp = max(2, int(6 * intensity * amp))
            t1 = min(effect_duration_ms // 3, duration)
            t2 = min((effect_duration_ms * 2) // 3, duration)
            t3 = min(effect_duration_ms, duration)
            return (
                f"{{\\pos(640,660)"
                f"\\t(0,{t1},\\pos({640-shake_amp+j},{660+shake_amp}))"
                f"\\t({t1},{t2},\\pos({640+shake_amp+j},{660-shake_amp}))"
                f"\\t({t2},{t3},\\pos(640,660))}}{text}"
            )

        if effect_type == SubtitleEffect.NEON_FLICKER.value:
            glow_alpha = max(0, min(255, int(110 - 35 * intensity)))
            return (
                f"{{\\bord3\\blur{max(1, int(2*intensity))}\\3a&H{glow_alpha:02X}&"
                f"\\t(0,{min(effect_duration_ms, duration)},\\3a&H00&)}}{text}"
            )

        if effect_type == SubtitleEffect.TWINKLE.value:
            alpha = max(0, min(255, int(68 * intensity)))
            return f"{{\\alpha&H{alpha:02X}&\\t(0,{min(effect_duration_ms, duration)},\\alpha&H00&)}}{text}"

        if effect_type == SubtitleEffect.SHINE.value:
            alpha = max(0, min(255, int(68 * intensity)))
            return f"{{\\1a&H{alpha:02X}&\\t(0,{min(effect_duration_ms, duration)},\\1a&H00&)}}{text}"

        if effect_type == SubtitleEffect.TYPEWRITER.value:
            # Простая имитация: плавный fade-in + более быстрое появление
            return f"{{\\fad({min(effect_duration_ms, duration // 3)},0)}}{text}"

        if effect_type == SubtitleEffect.WORD_HIGHLIGHT.value:
            return EffectManager._word_highlight_ass(text, effect_duration_ms)

        if effect_type == SubtitleEffect.RAINBOW.value:
            # ASS не поддерживает покадровую радугу без сложной генерации,
            # даем мягкую цветовую анимацию
            end_ass_color = EffectManager._hex_to_ass_bgr(rainbow_end_color)
            return f"{{\\1c&H00FFFFFF&\\t(0,{min(effect_duration_ms, duration)},\\1c{end_ass_color})}}{text}"

        return text
