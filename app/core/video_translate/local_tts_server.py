import json
import os
import subprocess
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional

from app.config import PROJECT_ROOT
from app.core.utils.logger import setup_logger

logger = setup_logger("local_tts_server")

_XTTS_ENGINE = None
_XTTS_LAST_OK = None
_XTTS_LAST_REASON = "not_checked"
_XTTS_LAST_DEBUG_LOG = ""
_XTTS_RUNTIME_FAIL_COUNT = 0
_XTTS_RUNTIME_DISABLE_UNTIL = 0.0
_XTTS_RUNTIME_SERVER_PROC = None
_XTTS_RUNTIME_SERVER_URL = "http://127.0.0.1:8021"
_XTTS_RUNTIME_SERVER_LOG = PROJECT_ROOT / "AppData" / "logs" / "xtts_runtime_server.log"
_XTTS_RUNTIME_SERVER_LOCK = threading.Lock()


def _wait_http_health(url: str, timeout_sec: float = 12.0) -> bool:
    deadline = time.time() + max(1.0, float(timeout_sec))
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=1.5) as resp:
                if resp.status == 200:
                    body = resp.read().decode("utf-8", errors="ignore")
                    logger.info("XTTS runtime server health: %s", body[:400])
                    return True
        except Exception:
            pass
        time.sleep(0.35)
    return False


def _tail_text(path: Path, max_chars: int = 1200) -> str:
    try:
        if not path.exists():
            return ""
        txt = path.read_text(encoding="utf-8", errors="ignore")
        return txt[-max_chars:]
    except Exception:
        return ""


def _ensure_runtime_server(py: Path, server_script: Path) -> bool:
    global _XTTS_RUNTIME_SERVER_PROC
    with _XTTS_RUNTIME_SERVER_LOCK:
        if _XTTS_RUNTIME_SERVER_PROC is not None and _XTTS_RUNTIME_SERVER_PROC.poll() is None:
            return True

        _XTTS_RUNTIME_SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
        try:
            _XTTS_RUNTIME_SERVER_LOG.write_text("", encoding="utf-8")
        except Exception:
            pass

        logf = open(_XTTS_RUNTIME_SERVER_LOG, "a", encoding="utf-8", buffering=1)
        _XTTS_RUNTIME_SERVER_PROC = subprocess.Popen(
            [str(py), str(server_script), "--host", "127.0.0.1", "--port", "8021"],
            stdout=logf,
            stderr=logf,
            creationflags=(subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0),
        )

    # XTTS модель может грузиться долго (первый холодный старт), даём большой таймаут.
    ok = _wait_http_health(_XTTS_RUNTIME_SERVER_URL, timeout_sec=75.0)
    if not ok:
        p = _XTTS_RUNTIME_SERVER_PROC
        if p is not None and p.poll() is not None:
            tail = _tail_text(_XTTS_RUNTIME_SERVER_LOG)
            logger.warning(
                "XTTS runtime server exited early rc=%s; log=%s; tail=%s",
                p.returncode,
                _XTTS_RUNTIME_SERVER_LOG,
                tail,
            )
        else:
            logger.warning(
                "XTTS runtime server health timeout; log=%s",
                _XTTS_RUNTIME_SERVER_LOG,
            )
    return ok


def _map_lang(language_hint: str, for_sapi: bool = False) -> str:
    if for_sapi:
        lang_map = {
            "英语": "en",
            "English": "en",
            "俄语": "ru",
            "Russian": "ru",
            "日语": "ja",
            "日本語": "ja",
            "韩语": "ko",
            "Korean": "ko",
            "德语": "de",
            "German": "de",
            "法语": "fr",
            "French": "fr",
            "西班牙语": "es",
            "Spanish": "es",
            "葡萄牙语": "pt",
            "Portuguese": "pt",
            "土耳其语": "tr",
            "Turkish": "tr",
            "简体中文": "zh",
            "繁体中文": "zh",
            "中文": "zh",
            "Chinese": "zh",
        }
        return lang_map.get(language_hint, "")

    lang_map = {
        "英语": "en",
        "English": "en",
        "俄语": "ru",
        "Russian": "ru",
        "日语": "ja",
        "日本語": "ja",
        "韩语": "ko",
        "Korean": "ko",
        "德语": "de",
        "German": "de",
        "法语": "fr",
        "French": "fr",
        "西班牙语": "es",
        "Spanish": "es",
        "葡萄牙语": "pt",
        "Portuguese": "pt",
        "土耳其语": "tr",
        "Turkish": "tr",
        "简体中文": "zh-cn",
        "繁体中文": "zh-cn",
        "中文": "zh-cn",
        "Chinese": "zh-cn",
    }
    return lang_map.get(language_hint, "en")


def _runtime_python() -> Optional[Path]:
    p = PROJECT_ROOT / "runtime" / "python.exe"
    return p if p.exists() else None


def _xtts_subprocess_check() -> tuple[bool, str]:
    py = _runtime_python()
    if py is None:
        return False, "runtime/python.exe not found"
    cmd = [
        str(py),
        "-c",
        "from TTS.api import TTS; print('ok')",
    ]
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=35,
            creationflags=(subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0),
        )
        if r.returncode == 0:
            return True, "ok"
        return False, (r.stderr or r.stdout or "xtts import failed").strip()[:800]
    except Exception as e:
        return False, str(e)


def _try_xtts_via_runtime(text: str, out_wav: Path, language_hint: str, speaker_wav: Optional[Path]) -> bool:
    global _XTTS_RUNTIME_FAIL_COUNT, _XTTS_RUNTIME_DISABLE_UNTIL, _XTTS_LAST_DEBUG_LOG, _XTTS_RUNTIME_SERVER_PROC
    now = time.time()
    if _XTTS_RUNTIME_DISABLE_UNTIL > now:
        return False

    py = _runtime_python()
    script = PROJECT_ROOT / "scripts" / "xtts_synthesize.py"
    if py is None or not script.exists():
        return False
    # 1) Пытаемся использовать persistent runtime server (модель загружается 1 раз, быстрее и более GPU-oriented).
    server_script = PROJECT_ROOT / "scripts" / "xtts_runtime_server.py"
    if server_script.exists():
        try:
            if not _ensure_runtime_server(py, server_script):
                raise RuntimeError(
                    f"XTTS runtime server unavailable (log: {_XTTS_RUNTIME_SERVER_LOG})"
                )

            payload = {
                "text": text,
                "language": _map_lang(language_hint, for_sapi=False),
                "speaker_wav": str(speaker_wav) if speaker_wav and speaker_wav.exists() else "",
            }
            req = urllib.request.Request(
                _XTTS_RUNTIME_SERVER_URL.rstrip("/") + "/tts",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=70) as resp:
                data = resp.read()
                if resp.status == 200 and data:
                    out_wav.write_bytes(data)
                    _XTTS_RUNTIME_FAIL_COUNT = 0
                    return True
        except Exception as e:
            logger.warning("XTTS runtime server failed, fallback to one-shot subprocess: %s", e)

    cmd = [
        str(py),
        str(script),
        "--text",
        text,
        "--out",
        str(out_wav),
        "--lang",
        _map_lang(language_hint, for_sapi=False),
    ]
    debug_log = PROJECT_ROOT / "AppData" / "logs" / "xtts_runtime_last_error.log"
    debug_log.parent.mkdir(parents=True, exist_ok=True)
    cmd += ["--log-file", str(debug_log)]
    _XTTS_LAST_DEBUG_LOG = str(debug_log)
    if speaker_wav and speaker_wav.exists():
        cmd += ["--speaker-wav", str(speaker_wav)]
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=70,
            creationflags=(subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0),
        )
        if r.returncode == 0 and out_wav.exists() and out_wav.stat().st_size > 0:
            _XTTS_RUNTIME_FAIL_COUNT = 0
            return True
        _XTTS_RUNTIME_FAIL_COUNT += 1
        if _XTTS_RUNTIME_FAIL_COUNT >= 2:
            _XTTS_RUNTIME_DISABLE_UNTIL = time.time() + 600
        logger.warning(
            "XTTS runtime subprocess failed rc=%s; log=%s; stderr=%s",
            r.returncode,
            debug_log,
            (r.stderr or r.stdout or "").strip()[:1200],
        )
    except Exception as e:
        _XTTS_RUNTIME_FAIL_COUNT += 1
        if _XTTS_RUNTIME_FAIL_COUNT >= 2:
            _XTTS_RUNTIME_DISABLE_UNTIL = time.time() + 600
        logger.warning("XTTS runtime subprocess exception: %s; log=%s", e, debug_log)
    return False


def _xtts_import_check() -> tuple[bool, str]:
    global _XTTS_LAST_OK, _XTTS_LAST_REASON
    try:
        from TTS.api import TTS  # type: ignore

        _ = TTS
        _XTTS_LAST_OK = True
        _XTTS_LAST_REASON = "ok"
        return True, "ok"
    except Exception as e:
        # fallback-check в отдельном runtime (часто стабильнее, чем импорт в frozen-процессе)
        ok2, reason2 = _xtts_subprocess_check()
        _XTTS_LAST_OK = bool(ok2)
        _XTTS_LAST_REASON = str(reason2 if ok2 else e)
        if ok2:
            return True, "ok(runtime-subprocess)"
        return False, str(e)


def _try_xtts_to_wav(text: str, out_wav: Path, language_hint: str, speaker_wav: Optional[Path]) -> bool:
    """Пытается использовать Coqui XTTS v2 (реальное voice cloning)."""
    global _XTTS_ENGINE
    ok, reason = _xtts_import_check()
    if not ok:
        logger.info("XTTS in-process unavailable, try runtime subprocess: %s", reason)
        return _try_xtts_via_runtime(text, out_wav, language_hint, speaker_wav)

    # Если проверка прошла только через runtime-subprocess, не пытаемся
    # делать in-process import (он может падать в frozen exe с WinError 1114).
    if "runtime-subprocess" in str(reason):
        return _try_xtts_via_runtime(text, out_wav, language_hint, speaker_wav)

    try:
        from TTS.api import TTS  # type: ignore

        if _XTTS_ENGINE is None:
            _XTTS_ENGINE = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

        lang = _map_lang(language_hint, for_sapi=False)

        if speaker_wav and speaker_wav.exists():
            _XTTS_ENGINE.tts_to_file(
                text=text,
                speaker_wav=str(speaker_wav),
                language=lang,
                file_path=str(out_wav),
            )
            return out_wav.exists() and out_wav.stat().st_size > 0
    except Exception as e:
        logger.warning("XTTS fallback to SAPI: %s", e)
        # Если in-process XTTS упал (частый кейс в frozen exe), пробуем runtime-subprocess
        if _try_xtts_via_runtime(text, out_wav, language_hint, speaker_wav):
            return True
    return False


def _pick_voice_rate(speaker_tag: str) -> int:
    if not speaker_tag:
        return 0
    value = sum(ord(c) for c in speaker_tag)
    return (value % 7) - 3  # -3..+3


def _synthesize_sapi_to_wav(
    text: str,
    out_wav: Path,
    language_hint: str = "",
    speaker_tag: str = "",
    speaker_wav: Optional[Path] = None,
    require_xtts: bool = False,
    prefer_xtts: bool = True,
):
    """Полностью локальный TTS через Windows SAPI (без внешних API)."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    # 1) Пытаемся дать настоящее клонирование через XTTS (если установлен пакет TTS)
    if require_xtts:
        ok, reason = _xtts_import_check()
        if not ok:
            raise RuntimeError(f"XTTS import недоступен: {reason}")
        if _try_xtts_to_wav(text, out_wav, language_hint=language_hint, speaker_wav=speaker_wav):
            return
        raise RuntimeError("XTTS synthesis failed (strict xtts mode)")

    if prefer_xtts:
        if _try_xtts_to_wav(text, out_wav, language_hint=language_hint, speaker_wav=speaker_wav):
            return
    txt_file = out_wav.with_suffix(".txt")
    txt_file.write_text(text or " ", encoding="utf-8")

    # Грубое сопоставление языкового хинта к culture-префиксу голосов
    culture_prefix = _map_lang(language_hint, for_sapi=True)
    rate = _pick_voice_rate(speaker_tag)

    ps_script = out_wav.with_suffix(".ps1")
    ps_script.write_text(
        rf'''
Param(
  [string]$TextFile,
  [string]$OutFile,
  [string]$CulturePrefix,
  [int]$Rate = 0
)
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voices = $s.GetInstalledVoices() | ForEach-Object {{ $_.VoiceInfo }}
if ($CulturePrefix -and $CulturePrefix.Length -gt 0) {{
  $v = $voices | Where-Object {{ $_.Culture.Name.ToLower().StartsWith($CulturePrefix.ToLower()) }} | Select-Object -First 1
  if ($v) {{ $s.SelectVoice($v.Name) }}
}}
if ($Rate -lt -10) {{ $Rate = -10 }}
if ($Rate -gt 10) {{ $Rate = 10 }}
$s.Rate = $Rate
$text = Get-Content -Path $TextFile -Raw -Encoding UTF8
$s.SetOutputToWaveFile($OutFile)
$s.Speak($text)
$s.Dispose()
'''.strip(),
        encoding="utf-8",
    )

    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(ps_script),
        "-TextFile",
        str(txt_file),
        "-OutFile",
        str(out_wav),
        "-CulturePrefix",
        culture_prefix,
        "-Rate",
        str(rate),
    ]
    subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=(subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0),
    )


def _read_json_body(handler: BaseHTTPRequestHandler) -> Dict:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    raw = handler.rfile.read(length) if length > 0 else b"{}"
    try:
        return json.loads(raw.decode("utf-8", errors="ignore"))
    except Exception:
        return {}


class LocalTTSRequestHandler(BaseHTTPRequestHandler):
    server_version = "ShortsLocalTTS/1.0"

    def _send_json(self, obj: Dict, status: int = 200):
        payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/health":
            mode = "xtts" if _XTTS_ENGINE is not None else "sapi"
            # Делаем health максимально быстрым (без тяжёлых импортов),
            # чтобы ensure_local_tts не падал по timeout.
            xtts_ok = bool(_XTTS_LAST_OK) if _XTTS_LAST_OK is not None else False
            xtts_reason = _XTTS_LAST_REASON
            self._send_json(
                {
                    "ok": True,
                    "service": "local_tts",
                    "mode": mode,
                    "xtts_import_ok": xtts_ok,
                    "xtts_reason": xtts_reason,
                    "xtts_debug_log": _XTTS_LAST_DEBUG_LOG,
                }
            )
            return
        if parsed.path == "/probe_xtts":
            xtts_ok, xtts_reason = _xtts_import_check()
            self._send_json(
                {
                    "ok": True,
                    "xtts_import_ok": xtts_ok,
                    "xtts_reason": xtts_reason,
                    "xtts_debug_log": _XTTS_LAST_DEBUG_LOG,
                }
            )
            return
        if parsed.path == "/capabilities":
            self._send_json(
                {
                    "ok": True,
                    "supports_clone": True,
                    "engine": "xtts" if _XTTS_ENGINE is not None else "sapi-fallback",
                }
            )
            return
        self._send_json({"ok": False, "error": "not found"}, status=404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path not in {"/tts", "/clone_tts"}:
            self._send_json({"ok": False, "error": "not found"}, status=404)
            return

        data = _read_json_body(self)
        text = str(data.get("text") or "").strip()
        if not text:
            self._send_json({"ok": False, "error": "text is required"}, status=400)
            return

        language = str(data.get("language") or "")
        speaker_id = str(data.get("speaker_id") or data.get("speaker_tag") or "")
        require_xtts = bool(data.get("require_xtts", False))
        prefer_xtts = bool(data.get("prefer_xtts", True))

        tmp_dir = Path(tempfile.gettempdir()) / "shorts_local_tts"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        out_wav = tmp_dir / f"tts_{threading.get_ident()}_{os.getpid()}.wav"

        try:
            speaker_wav = None
            speaker_wav_raw = str(data.get("speaker_wav") or "").strip()
            if speaker_wav_raw:
                p = Path(speaker_wav_raw)
                if p.exists():
                    speaker_wav = p

            _synthesize_sapi_to_wav(
                text,
                out_wav,
                language_hint=language,
                speaker_tag=speaker_id,
                speaker_wav=speaker_wav,
                require_xtts=require_xtts,
                prefer_xtts=prefer_xtts,
            )
            content = out_wav.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            logger.exception("Local TTS error: %s", e)
            msg = str(e)
            if _XTTS_LAST_DEBUG_LOG:
                msg = f"{msg}. Debug log: {_XTTS_LAST_DEBUG_LOG}"
            self._send_json({"ok": False, "error": msg}, status=500)

    def log_message(self, format: str, *args):
        logger.info("local-tts: " + format, *args)


class LocalTTSServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8020):
        self.host = host
        self.port = int(port)
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._httpd = ThreadingHTTPServer((self.host, self.port), LocalTTSRequestHandler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        logger.info("Local TTS server started at http://%s:%s", self.host, self.port)

    def stop(self):
        global _XTTS_RUNTIME_SERVER_PROC
        if self._httpd:
            try:
                self._httpd.shutdown()
            except Exception:
                pass
            try:
                self._httpd.server_close()
            except Exception:
                pass
            self._httpd = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if _XTTS_RUNTIME_SERVER_PROC is not None and _XTTS_RUNTIME_SERVER_PROC.poll() is None:
            try:
                _XTTS_RUNTIME_SERVER_PROC.terminate()
            except Exception:
                pass
        _XTTS_RUNTIME_SERVER_PROC = None
        logger.info("Local TTS server stopped")
