from dataclasses import dataclass
from typing import List, Optional

from app.core.bk_asr.asr_data import ASRData
from app.core.utils.logger import setup_logger

logger = setup_logger("video_translate_diarization")


@dataclass
class SpeakerTurn:
    start_ms: int
    end_ms: int
    speaker_id: str


class BaseDiarizer:
    def diarize(
        self,
        audio_path: str,
        asr_data: ASRData,
        expected_speaker_count: int = 0,
    ) -> List[SpeakerTurn]:
        raise NotImplementedError


class HeuristicPauseDiarizer(BaseDiarizer):
    """Fallback diarization: по паузам + циклические speaker id.

    Это безопасный оффлайн baseline, когда pyannote недоступен.
    """

    def __init__(self, pause_threshold_ms: int = 1200):
        self.pause_threshold_ms = int(pause_threshold_ms)

    def diarize(
        self,
        audio_path: str,
        asr_data: ASRData,
        expected_speaker_count: int = 0,
    ) -> List[SpeakerTurn]:
        _ = audio_path
        turns: List[SpeakerTurn] = []
        if not asr_data.segments:
            return turns

        speaker_count = int(expected_speaker_count or 0)
        if speaker_count <= 0:
            speaker_count = 2
        speaker_count = max(1, min(6, speaker_count))

        current_speaker = 0
        prev_end = None
        for seg in asr_data.segments:
            s = int(seg.start_time)
            e = int(seg.end_time)
            if prev_end is not None and (s - prev_end) >= self.pause_threshold_ms:
                current_speaker = (current_speaker + 1) % speaker_count
            turns.append(SpeakerTurn(start_ms=s, end_ms=e, speaker_id=f"spk_{current_speaker}"))
            prev_end = e
        return turns


class DiarizerFactory:
    @staticmethod
    def create_preferred() -> BaseDiarizer:
        # Production-safe: сначала пытаемся pyannote, при ошибке fallback.
        try:
            from pyannote.audio import Pipeline  # type: ignore

            class PyannoteDiarizer(BaseDiarizer):
                def diarize(
                    self,
                    audio_path: str,
                    asr_data: ASRData,
                    expected_speaker_count: int = 0,
                ) -> List[SpeakerTurn]:
                    # Без токена/HF модели может быть недоступно: пусть это обработает outer fallback
                    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
                    kwargs = {}
                    if expected_speaker_count and expected_speaker_count > 0:
                        kwargs["num_speakers"] = int(expected_speaker_count)
                    diarization = pipeline(audio_path, **kwargs)
                    turns: List[SpeakerTurn] = []
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        turns.append(
                            SpeakerTurn(
                                start_ms=int(turn.start * 1000),
                                end_ms=int(turn.end * 1000),
                                speaker_id=str(speaker),
                            )
                        )
                    return turns

            return PyannoteDiarizer()
        except Exception as e:
            logger.info("pyannote diarizer unavailable, fallback to heuristic: %s", e)
            return HeuristicPauseDiarizer()
