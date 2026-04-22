import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from app.config import PROJECT_ROOT


@dataclass
class RVCModelInfo:
    slot: str
    slot_dir: Path
    params_path: Path
    name: str
    model_file: str
    index_file: str
    icon_file: str
    icon_path: Optional[Path]
    default_tune: int
    default_index_ratio: float
    default_protect: float
    sampling_rate: int
    f0: bool
    raw_params: Dict


def _pick(d: Dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _find_first_file(slot_dir: Path, suffixes: tuple[str, ...]) -> str:
    for p in sorted(slot_dir.iterdir() if slot_dir.exists() else []):
        if p.is_file() and p.suffix.lower() in suffixes:
            return p.name
    return ""


def _resolve_icon(slot_dir: Path, icon_file: str) -> Optional[Path]:
    if icon_file:
        p = Path(icon_file)
        if p.is_absolute() and p.exists():
            return p
        # MMVC-style path: model_dir\\31\\file.jpg
        if "model_dir" in icon_file.lower():
            tail = icon_file.replace("\\", "/").split("/")[-1]
            candidate = slot_dir / tail
            if candidate.exists():
                return candidate
        candidate = slot_dir / icon_file
        if candidate.exists():
            return candidate

    for p in sorted(slot_dir.iterdir() if slot_dir.exists() else []):
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            return p
    return None


def scan_rvc_models(model_root: Path) -> List[RVCModelInfo]:
    result: List[RVCModelInfo] = []
    if not model_root.exists():
        return result

    for slot_dir in sorted([p for p in model_root.iterdir() if p.is_dir()], key=lambda x: x.name):
        params_path = slot_dir / "params.json"
        raw: Dict = {}
        if params_path.exists():
            try:
                raw = json.loads(params_path.read_text(encoding="utf-8", errors="ignore") or "{}")
            except Exception:
                raw = {}

        model_file = str(_pick(raw, "modelFile", "model_file", default="") or "").strip()
        index_file = str(_pick(raw, "indexFile", "index_file", default="") or "").strip()
        icon_file = str(_pick(raw, "iconFile", "icon_file", default="") or "").strip()

        if not model_file:
            model_file = _find_first_file(slot_dir, (".pth", ".pt"))
        if not index_file:
            index_file = _find_first_file(slot_dir, (".index",))

        if not model_file:
            continue

        name = str(_pick(raw, "name", default=slot_dir.name) or slot_dir.name)
        default_tune = int(_pick(raw, "defaultTune", "pitch_shift", default=0) or 0)
        default_index_ratio = float(_pick(raw, "defaultIndexRatio", "index_ratio", default=0.0) or 0.0)
        default_protect = float(_pick(raw, "defaultProtect", "protect_ratio", default=0.5) or 0.5)
        sampling_rate = int(_pick(raw, "samplingRate", "sample_rate", default=40000) or 40000)
        f0 = bool(_pick(raw, "f0", "is_f0", default=True))

        icon_path = _resolve_icon(slot_dir, icon_file)
        result.append(
            RVCModelInfo(
                slot=slot_dir.name,
                slot_dir=slot_dir,
                params_path=params_path,
                name=name,
                model_file=model_file,
                index_file=index_file,
                icon_file=icon_file,
                icon_path=icon_path,
                default_tune=default_tune,
                default_index_ratio=default_index_ratio,
                default_protect=default_protect,
                sampling_rate=sampling_rate,
                f0=f0,
                raw_params=raw,
            )
        )

    return result


def default_rvc_model_root() -> Path:
    return PROJECT_ROOT / "AppData" / "models" / "rvc"


def save_rvc_model_params(model: RVCModelInfo, *, name: str, default_tune: int, default_index_ratio: float, default_protect: float):
    d = dict(model.raw_params or {})

    # Поддерживаем оба формата (camelCase и snake_case)
    if "name" in d or not d:
        d["name"] = name
    if "defaultTune" in d or "pitch_shift" not in d:
        d["defaultTune"] = int(default_tune)
    if "pitch_shift" in d:
        d["pitch_shift"] = int(default_tune)

    if "defaultIndexRatio" in d or "index_ratio" not in d:
        d["defaultIndexRatio"] = float(default_index_ratio)
    if "index_ratio" in d:
        d["index_ratio"] = float(default_index_ratio)

    if "defaultProtect" in d or "protect_ratio" not in d:
        d["defaultProtect"] = float(default_protect)
    if "protect_ratio" in d:
        d["protect_ratio"] = float(default_protect)

    model.params_path.write_text(json.dumps(d, ensure_ascii=False, indent=4), encoding="utf-8")
