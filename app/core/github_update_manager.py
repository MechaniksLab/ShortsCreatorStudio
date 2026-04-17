import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional

import requests

from app.common.config import cfg
from app.config import CACHE_PATH, ROOT_PATH, UPDATE_REPO_BRANCH, UPDATE_REPO_NAME, UPDATE_REPO_OWNER


class GitHubUpdateManager:
    """Проверка и применение обновлений из GitHub-репозитория."""

    def __init__(self):
        self.app_root = ROOT_PATH.parent
        self.updater_root = CACHE_PATH / "updater"
        self.updater_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _creation_flags() -> int:
        flags = 0
        if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
            flags |= subprocess.CREATE_NEW_PROCESS_GROUP
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            flags |= subprocess.CREATE_NO_WINDOW
        return flags

    def _repo(self):
        # Репозиторий фиксирован внутри приложения (без UI-настроек)
        return UPDATE_REPO_OWNER, UPDATE_REPO_NAME, UPDATE_REPO_BRANCH

    def fetch_latest_commit(self) -> Dict:
        owner, name, branch = self._repo()
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "ShortsCreatorStudio-Updater",
        }

        def _parse_commit(data: Dict, used_branch: str) -> Dict:
            return {
                "sha": str(data.get("sha") or "").strip(),
                "html_url": str(data.get("html_url") or "").strip(),
                "message": str(((data.get("commit") or {}).get("message") or "")).strip(),
                "branch": used_branch,
            }

        # 1) Пробуем ветку из конфига
        primary_url = f"https://api.github.com/repos/{owner}/{name}/commits/{branch}"
        r = requests.get(primary_url, timeout=15, headers=headers)
        if r.ok:
            return _parse_commit(r.json() or {}, branch)

        # 2) Если ref некорректен (часто 422), берем default_branch репозитория
        repo_url = f"https://api.github.com/repos/{owner}/{name}"
        rr = requests.get(repo_url, timeout=15, headers=headers)
        rr.raise_for_status()
        repo_info = rr.json() or {}
        default_branch = str(repo_info.get("default_branch") or branch).strip() or branch

        fallback_url = f"https://api.github.com/repos/{owner}/{name}/commits/{default_branch}"
        rf = requests.get(fallback_url, timeout=15, headers=headers)
        if rf.ok:
            return _parse_commit(rf.json() or {}, default_branch)

        # 3) Последний резерв: самый свежий коммит из списка
        list_url = f"https://api.github.com/repos/{owner}/{name}/commits?per_page=1"
        rl = requests.get(list_url, timeout=15, headers=headers)
        rl.raise_for_status()
        items = rl.json() or []
        if not items:
            raise requests.HTTPError("GitHub API returned no commits")
        data = items[0] or {}
        return {
            "sha": str(data.get("sha") or "").strip(),
            "html_url": str(data.get("html_url") or "").strip(),
            "message": str(((data.get("commit") or {}).get("message") or "")).strip(),
            "branch": default_branch,
        }

    def check_update(self) -> Dict:
        latest = self.fetch_latest_commit()
        latest_sha = latest.get("sha", "")
        known_sha = str(cfg.update_last_known_commit.value or "").strip()

        # Первый запуск: фиксируем baseline без навязчивого апдейта.
        if not known_sha and latest_sha:
            cfg.set(cfg.update_last_known_commit, latest_sha)
            return {"has_update": False, "latest": latest, "known": latest_sha, "baseline_initialized": True}

        return {
            "has_update": bool(latest_sha and known_sha and latest_sha != known_sha),
            "latest": latest,
            "known": known_sha,
            "baseline_initialized": False,
        }

    def apply_update_and_restart(self, progress_cb: Optional[Callable[[int, str], None]] = None) -> Dict:
        def report(percent: int, text: str):
            if callable(progress_cb):
                try:
                    progress_cb(int(percent), str(text))
                except Exception:
                    pass

        report(2, "Проверка последнего коммита...")
        latest = self.fetch_latest_commit()
        sha = latest.get("sha", "")
        if not sha:
            return {"ok": False, "error": "Не удалось получить SHA последнего коммита"}

        known_sha = str(cfg.update_last_known_commit.value or "").strip()

        owner, name, _ = self._repo()

        # Быстрый путь: применяем только изменённые файлы между known_sha и latest sha
        changed_files = self._fetch_changed_files(owner=owner, name=name, base_sha=known_sha, head_sha=sha)
        if changed_files and len(changed_files) <= 80:
            ok = self._apply_incremental_changes(owner=owner, name=name, sha=sha, files=changed_files, report=report)
            if ok:
                launcher = self._resolve_launcher()
                script_path = self._spawn_restart_only_script(launcher=launcher)
                cfg.set(cfg.update_last_known_commit, sha)
                report(100, "Обновление применено, перезапуск...")
                return {"ok": True, "sha": sha, "script": str(script_path), "mode": "incremental"}

        # fallback: полный снимок по SHA
        # Скачиваем по SHA коммита, чтобы не зависеть от названия ветки
        zip_url = f"https://api.github.com/repos/{owner}/{name}/zipball/{sha}"

        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        stage_dir = self.updater_root / f"stage_{stamp}"
        extract_dir = stage_dir / "extract"
        stage_dir.mkdir(parents=True, exist_ok=True)
        extract_dir.mkdir(parents=True, exist_ok=True)
        zip_path = stage_dir / "update.zip"

        report(8, "Скачивание обновления...")

        with requests.get(zip_url, timeout=45, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length") or 0)
            downloaded = 0
            with zip_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            # 8..72: этап скачивания
                            p = 8 + int((downloaded / total) * 64)
                            report(min(p, 72), "Скачивание обновления...")

        report(76, "Распаковка обновления...")

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        children = [p for p in extract_dir.iterdir() if p.is_dir()]
        if not children:
            return {"ok": False, "error": "Архив обновления пуст"}
        src_root = children[0]

        report(88, "Подготовка сценария обновления...")
        launcher = self._resolve_launcher()
        script_path = stage_dir / "apply_update.bat"
        script = self._build_update_script(src_root=src_root, dst_root=self.app_root, launcher=launcher)
        script_path.write_text(script, encoding="utf-8")

        report(95, "Запуск применения обновления...")
        subprocess.Popen(
            ["cmd", "/c", str(script_path)],
            creationflags=self._creation_flags(),
        )

        cfg.set(cfg.update_last_known_commit, sha)
        report(100, "Обновление запущено")
        return {"ok": True, "sha": sha, "script": str(script_path), "mode": "full"}

    def _fetch_changed_files(self, owner: str, name: str, base_sha: str, head_sha: str):
        if not base_sha or not head_sha or base_sha == head_sha:
            return []

        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "ShortsCreatorStudio-Updater",
        }
        url = f"https://api.github.com/repos/{owner}/{name}/compare/{base_sha}...{head_sha}"
        r = requests.get(url, timeout=20, headers=headers)
        if r.status_code in (404, 422):
            return []
        r.raise_for_status()
        data = r.json() or {}
        files = data.get("files") or []
        return files if isinstance(files, list) else []

    def _apply_incremental_changes(self, owner: str, name: str, sha: str, files: list, report: Callable[[int, str], None]) -> bool:
        try:
            headers = {
                "Accept": "application/vnd.github.raw",
                "User-Agent": "ShortsCreatorStudio-Updater",
            }
            total = len(files)
            if total <= 0:
                return False

            report(8, "Применение изменённых файлов...")
            for i, entry in enumerate(files, start=1):
                status = str((entry or {}).get("status") or "").strip().lower()
                filename = str((entry or {}).get("filename") or "").strip()
                prev_name = str((entry or {}).get("previous_filename") or "").strip()
                if not filename:
                    continue

                # Никогда не трогаем пользовательские директории
                norm = filename.replace("\\", "/")
                if norm.startswith("AppData/") or norm.startswith("runtime/") or norm.startswith("work-dir/"):
                    continue

                target = self.app_root / Path(*norm.split("/"))
                if status == "removed":
                    if target.exists() and target.is_file():
                        target.unlink(missing_ok=True)
                else:
                    raw_url = f"https://raw.githubusercontent.com/{owner}/{name}/{sha}/{norm}"
                    rr = requests.get(raw_url, timeout=25, headers=headers)
                    rr.raise_for_status()
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(rr.content)

                if status == "renamed" and prev_name:
                    prev_norm = prev_name.replace("\\", "/")
                    prev_target = self.app_root / Path(*prev_norm.split("/"))
                    if prev_target.exists() and prev_target.is_file() and prev_target != target:
                        prev_target.unlink(missing_ok=True)

                p = 8 + int((i / total) * 80)
                report(min(88, p), f"Применение файлов: {i}/{total}")

            return True
        except Exception:
            return False

    def _spawn_restart_only_script(self, launcher: str) -> Path:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        stage_dir = self.updater_root / f"restart_{stamp}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        script_path = stage_dir / "restart_only.bat"
        script = (
            "@echo off\n"
            "chcp 65001 >nul\n"
            "timeout /t 2 /nobreak >nul\n"
            f"start \"\" {launcher}\n"
        )
        script_path.write_text(script, encoding="utf-8")
        subprocess.Popen(
            ["cmd", "/c", str(script_path)],
            creationflags=self._creation_flags(),
        )
        return script_path

    def _resolve_launcher(self) -> str:
        # В режиме exe
        if getattr(sys, "frozen", False):
            return f'"{Path(sys.executable)}"'

        # Вне frozen: если рядом есть собранный GUI exe, предпочитаем его (без консоли)
        for exe_name in ["ShortsCreatorStudio.exe", "Shorts creator studio.exe"]:
            exe_path = self.app_root / exe_name
            if exe_path.exists():
                return f'"{exe_path}"'

        # В режиме разработки пробуем запускать текущий python + main.py (если есть)
        py = str(Path(sys.executable))
        for candidate in [self.app_root / "main.py", self.app_root / "run.py"]:
            if candidate.exists():
                return f'"{py}" "{candidate}"'

        # fallback: просто текущий python
        return f'"{py}"'

    @staticmethod
    def _build_update_script(src_root: Path, dst_root: Path, launcher: str) -> str:
        src = str(src_root)
        dst = str(dst_root)
        # /XD исключаем runtime и пользовательские данные.
        return (
            "@echo off\n"
            "chcp 65001 >nul\n"
            "setlocal\n"
            f"set SRC={src}\n"
            f"set DST={dst}\n"
            "timeout /t 4 /nobreak >nul\n"
            "robocopy \"%SRC%\" \"%DST%\" /E /R:2 /W:1 /XD \"%DST%\\runtime\" \"%DST%\\AppData\" \"%DST%\\work-dir\" >nul\n"
            f"start \"\" {launcher}\n"
            "endlocal\n"
        )
