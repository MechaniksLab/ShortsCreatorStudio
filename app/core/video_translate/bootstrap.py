from pathlib import Path
import json
import subprocess
import shutil
from typing import Dict, List

from app.config import MODEL_PATH, PROJECT_ROOT
from app.core.utils.logger import setup_logger
from app.core.video_translate.service_manager import VideoTranslateServiceManager

logger = setup_logger("video_translate_bootstrap")


class VideoTranslateBootstrap:
    """Подготовка локальных зависимостей для автономного video translate."""

    def __init__(self):
        self.model_root = Path(MODEL_PATH)
        self.model_root.mkdir(parents=True, exist_ok=True)

    def has_faster_whisper_model(self, model_name: str = "tiny") -> bool:
        target_dir = self.model_root / f"faster-whisper-{model_name}"
        model_bin = target_dir / "model.bin"
        if model_bin.exists():
            return True

        # Если уже есть любой faster-whisper model.bin — используем его как fallback
        existing = list(self.model_root.glob("faster-whisper-*/model.bin"))
        if existing:
            logger.info("Found existing whisper model: %s", existing[0])
            return True
        return False

    def ensure_faster_whisper_model(self, model_name: str = "tiny"):
        if self.has_faster_whisper_model(model_name):
            return True

        target_dir = self.model_root / f"faster-whisper-{model_name}"
        model_bin = target_dir / "model.bin"

        # Пробуем автозагрузку через ModelScope, если пакет доступен в runtime
        model_id = f"pengzhendong/faster-whisper-{model_name}"
        try:
            from modelscope.hub.snapshot_download import snapshot_download

            logger.info("Downloading ASR model from ModelScope: %s", model_id)
            snapshot_download(model_id, local_dir=str(target_dir))
            if model_bin.exists():
                return True
        except Exception as e:
            logger.warning("Model auto-download failed (%s): %s", model_id, e)

        logger.warning("FasterWhisper model is missing: %s", target_dir)
        return False

    @staticmethod
    def _runtime_torch_cuda_info() -> Dict:
        """Проверка torch/cuda через runtime/python.exe (стабильно для frozen-сценария)."""
        runtime_py = PROJECT_ROOT / "runtime" / "python.exe"
        if not runtime_py.exists():
            return {"cuda": False, "version": "", "ok": False, "reason": "runtime python missing"}
        try:
            p = subprocess.run(
                [
                    str(runtime_py),
                    "-c",
                    "import json,torch; print(json.dumps({'cuda': bool(torch.cuda.is_available()), 'version': getattr(torch, '__version__', '')}, ensure_ascii=False))",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=25,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
                ),
            )
            if p.returncode != 0:
                return {"cuda": False, "version": "", "ok": False, "reason": (p.stderr or p.stdout or "").strip()[:300]}
            line = ""
            for raw in (p.stdout or "").splitlines()[::-1]:
                s = raw.strip()
                if s.startswith("{") and s.endswith("}"):
                    line = s
                    break
            if not line:
                return {"cuda": False, "version": "", "ok": False, "reason": "no json output"}
            data = json.loads(line)
            return {
                "cuda": bool(data.get("cuda", False)),
                "version": str(data.get("version") or ""),
                "ok": True,
                "reason": "",
            }
        except Exception as e:
            return {"cuda": False, "version": "", "ok": False, "reason": str(e)}

    def readiness_report(self, model_name: str = "tiny", auto_download: bool = True) -> Dict:
        """Диагностика готовности production-пайплайна."""
        checks: List[Dict] = []

        ffmpeg_ok = bool(shutil.which("ffmpeg"))
        checks.append({"name": "ffmpeg", "ok": ffmpeg_ok, "hint": "Нужен для mux/аудио"})

        fw_bin_ok = bool(shutil.which("faster-whisper") or shutil.which("faster-whisper-xxl"))
        checks.append(
            {
                "name": "faster-whisper-bin",
                "ok": fw_bin_ok,
                "hint": "CLI ускоряет локальный ASR, но не обязателен (есть python fallback)",
            }
        )

        fw_py_ok = False
        try:
            import faster_whisper  # type: ignore

            _ = faster_whisper
            fw_py_ok = True
        except Exception:
            fw_py_ok = False
        checks.append(
            {
                "name": "faster-whisper-python",
                "ok": fw_py_ok,
                "hint": "Python-пакет faster-whisper нужен для fallback ASR",
            }
        )

        if auto_download:
            model_ok = self.ensure_faster_whisper_model(model_name)
        else:
            model_ok = self.has_faster_whisper_model(model_name)
        checks.append(
            {
                "name": f"faster-whisper-model:{model_name}",
                "ok": bool(model_ok),
                "hint": "Модель ASR должна быть в AppData/models",
            }
        )

        xtts_ok = False
        xtts_reason = ""
        try:
            from TTS.api import TTS  # type: ignore

            _ = TTS
            xtts_ok = True
        except Exception as e:
            xtts_ok = False
            xtts_reason = str(e)

        # Доп. проверка через локальный TTS server (как реально видит runtime в пайплайне)
        try:
            svc = VideoTranslateServiceManager.instance()
            endpoint = "http://127.0.0.1:8020"
            if svc.ensure_local_tts(endpoint):
                # Активная проверка (может подтвердить XTTS через runtime-subprocess,
                # даже если import TTS в основном процессе падает в frozen exe).
                tts_probe = svc.local_tts_probe_xtts(endpoint)
                if isinstance(tts_probe, dict):
                    probe_ok = bool(tts_probe.get("xtts_import_ok", False))
                    if probe_ok:
                        xtts_ok = True
                        xtts_reason = str(tts_probe.get("xtts_reason") or "ok(local-probe)")
                    elif tts_probe.get("xtts_reason"):
                        xtts_reason = str(tts_probe.get("xtts_reason"))

                # Быстрый health как дополнительный источник деталей
                tts_health = svc.local_tts_health_details(endpoint)
            else:
                tts_health = {}

            if isinstance(tts_health, dict):
                health_ok = bool(tts_health.get("xtts_import_ok", False))
                if health_ok:
                    xtts_ok = True
                elif tts_health.get("xtts_reason"):
                    xtts_reason = str(tts_health.get("xtts_reason"))
        except Exception as e:
            logger.warning("local tts xtts probe failed: %s", e)
        checks.append(
            {
                "name": "coqui-xtts",
                "ok": xtts_ok,
                "hint": (
                    "Для точного клонирования тембра рекомендуется установить пакет TTS"
                    + (f"; details: {xtts_reason}" if xtts_reason else "")
                ),
            }
        )

        pyannote_ok = False
        try:
            import pyannote.audio  # type: ignore

            _ = pyannote.audio
            pyannote_ok = True
        except Exception:
            pyannote_ok = False

        # fallback: проверяем pyannote в runtime/python.exe (актуально для frozen main-process)
        if not pyannote_ok:
            try:
                runtime_py = PROJECT_ROOT / "runtime" / "python.exe"
                if runtime_py.exists():
                    p = subprocess.run(
                        [str(runtime_py), "-c", "import pyannote.audio; print('ok')"],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=35,
                        creationflags=(
                            subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
                        ),
                    )
                    if p.returncode == 0:
                        pyannote_ok = True
            except Exception:
                pass
        checks.append(
            {
                "name": "pyannote-diarization",
                "ok": pyannote_ok,
                "hint": "Для максимально точного multi-speaker требуется pyannote.audio",
            }
        )

        torch_cuda_ok = False
        torch_version = ""
        try:
            import torch  # type: ignore

            torch_version = getattr(torch, "__version__", "") or ""
            torch_cuda_ok = bool(torch.cuda.is_available())
        except Exception:
            torch_cuda_ok = False

        # Если in-process не видит GPU (часто в frozen exe), сверяемся с runtime/python.exe.
        if not torch_cuda_ok:
            rt = self._runtime_torch_cuda_info()
            if bool(rt.get("cuda", False)):
                torch_cuda_ok = True
                if not torch_version:
                    torch_version = str(rt.get("version") or "")
            elif rt.get("reason"):
                logger.info("runtime torch cuda check: %s", rt.get("reason"))
        checks.append(
            {
                "name": "torch-cuda",
                "ok": torch_cuda_ok,
                "hint": "GPU-ускорение доступно, если установлен CUDA build torch и есть совместимый драйвер",
            }
        )

        asr_ready = bool(fw_bin_ok or fw_py_ok)
        all_ok = ffmpeg_ok and asr_ready and bool(model_ok)
        critical_failed = [
            c for c in checks if c.get("name") in {"ffmpeg", f"faster-whisper-model:{model_name}"} and not c.get("ok")
        ]
        if not asr_ready:
            critical_failed.append(
                {
                    "name": "faster-whisper-runtime",
                    "ok": False,
                    "hint": "Нужен либо faster-whisper CLI, либо python-пакет faster-whisper",
                }
            )
        clone_quality = "high" if xtts_ok else "basic-fallback"
        return {
            "ok": all_ok,
            "critical_failed": critical_failed,
            "clone_quality": clone_quality,
            "diarization_quality": "high" if pyannote_ok else "heuristic-fallback",
            "asr_runtime": "cli" if fw_bin_ok else ("python" if fw_py_ok else "missing"),
            "torch_version": torch_version,
            "torch_cuda": torch_cuda_ok,
            "checks": checks,
        }
