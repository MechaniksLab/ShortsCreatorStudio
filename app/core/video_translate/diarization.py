from dataclasses import dataclass
import json
import subprocess
from pathlib import Path
from typing import List, Optional

from app.config import PROJECT_ROOT
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
                    try:
                        import torch  # type: ignore

                        if bool(torch.cuda.is_available()) and hasattr(pipeline, "to"):
                            pipeline.to(torch.device("cuda"))
                            logger.info("pyannote diarizer moved to CUDA")
                    except Exception as e:
                        logger.info("pyannote CUDA move skipped: %s", e)
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
            logger.info("pyannote in-process unavailable, trying runtime diarizer: %s", e)

            class RuntimePyannoteDiarizer(BaseDiarizer):
                def diarize(
                    self,
                    audio_path: str,
                    asr_data: ASRData,
                    expected_speaker_count: int = 0,
                ) -> List[SpeakerTurn]:
                    _ = asr_data
                    runtime_py = PROJECT_ROOT / "runtime" / "python.exe"
                    script = PROJECT_ROOT / "scripts" / "pyannote_diarize.py"
                    if not runtime_py.exists() or not script.exists():
                        raise RuntimeError("runtime pyannote diarizer is not available")

                    out_json = Path(audio_path).with_suffix(".diarization.json")
                    cmd = [
                        str(runtime_py),
                        str(script),
                        "--audio",
                        str(audio_path),
                        "--out",
                        str(out_json),
                    ]
                    if expected_speaker_count and expected_speaker_count > 0:
                        cmd += ["--num-speakers", str(int(expected_speaker_count))]

                    p = subprocess.run(
                        cmd,
                        check=False,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        creationflags=(
                            subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
                        ),
                    )
                    if p.returncode != 0 or not out_json.exists():
                        err = (p.stderr or p.stdout or "").strip()
                        raise RuntimeError(f"runtime pyannote failed: {err[-500:]}")

                    data = json.loads(out_json.read_text(encoding="utf-8", errors="ignore") or "[]")
                    turns: List[SpeakerTurn] = []
                    for row in (data if isinstance(data, list) else []):
                        try:
                            turns.append(
                                SpeakerTurn(
                                    start_ms=int(row.get("start_ms", 0)),
                                    end_ms=int(row.get("end_ms", 0)),
                                    speaker_id=str(row.get("speaker_id") or "spk_0"),
                                )
                            )
                        except Exception:
                            continue
                    return turns

            try:
                return RuntimePyannoteDiarizer()
            except Exception:
                logger.info("runtime pyannote unavailable, fallback to heuristic")
                return HeuristicPauseDiarizer()
