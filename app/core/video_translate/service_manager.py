import threading
import json
import urllib.parse
import urllib.request
from typing import Optional

from app.core.utils.logger import setup_logger
from app.core.video_translate.local_tts_server import LocalTTSServer

logger = setup_logger("video_translate_service_manager")


class VideoTranslateServiceManager:
    """Менеджер встроенных локальных сервисов для автономного режима."""

    _instance: Optional["VideoTranslateServiceManager"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._local_tts_server: Optional[LocalTTSServer] = None
        self._local_tts_host: Optional[str] = None
        self._local_tts_port: Optional[int] = None

    @classmethod
    def instance(cls) -> "VideoTranslateServiceManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _health(self, endpoint: str, timeout: float = 1.2) -> bool:
        try:
            url = endpoint.rstrip("/") + "/health"
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return int(resp.status) < 300
        except Exception:
            return False

    def local_tts_health_details(self, endpoint: str, timeout: float = 1.8) -> dict:
        endpoint = (endpoint or "").strip() or "http://127.0.0.1:8020"
        url = endpoint.rstrip("/") + "/health"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            import json

            data = json.loads(raw) if raw else {}
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {}

    def local_tts_probe_xtts(self, endpoint: str, timeout: float = 12.0) -> dict:
        """Активная проверка XTTS через /probe_xtts (может быть тяжелее /health)."""
        endpoint = (endpoint or "").strip() or "http://127.0.0.1:8020"
        url = endpoint.rstrip("/") + "/probe_xtts"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw) if raw else {}
            if isinstance(data, dict):
                return data
        except Exception as e:
            logger.warning("local_tts_probe_xtts(%s) failed: %s", endpoint, e)
        return {}

    def ensure_local_tts(self, endpoint: str) -> bool:
        """Если endpoint локальный — поднимает встроенный сервер и проверяет health."""
        endpoint = (endpoint or "").strip() or "http://127.0.0.1:8020"
        parsed = urllib.parse.urlparse(endpoint)
        host = parsed.hostname or "127.0.0.1"
        port = int(parsed.port or 8020)

        # Для внешних endpoint ничего не поднимаем
        if host not in {"127.0.0.1", "localhost"}:
            return self._health(endpoint)

        if self._health(endpoint):
            return True

        need_restart = (
            self._local_tts_server is None
            or self._local_tts_host != host
            or self._local_tts_port != port
        )
        if need_restart:
            if self._local_tts_server is not None:
                try:
                    self._local_tts_server.stop()
                except Exception:
                    pass
            self._local_tts_server = LocalTTSServer(host=host, port=port)
            self._local_tts_host = host
            self._local_tts_port = port

        self._local_tts_server.start()
        ok = self._health(endpoint, timeout=2.5)
        logger.info("ensure_local_tts(%s) => %s", endpoint, ok)
        return ok

    def shutdown_all(self):
        if self._local_tts_server is not None:
            try:
                self._local_tts_server.stop()
            except Exception:
                pass
            self._local_tts_server = None
