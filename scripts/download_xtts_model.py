#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Принудительно подготавливает XTTS модель в runtime.

Фактически делает первый запуск TTS("...xtts_v2"), что вызывает
скачивание/кэш модели через Coqui TTS.
"""

from __future__ import annotations


def main() -> int:
    print("[RUN] Подготовка XTTS v2...")
    from TTS.api import TTS  # type: ignore

    _ = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    print("[OK] XTTS v2 готов")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
