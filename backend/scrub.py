"""Secure local residue wiping for ComfyUI working dirs."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Iterable, Optional

# Filenames we create or commonly produce for this app
OUR_PREFIXES = (
    "wan_in_",
    "wan_i2i",
    "wan_i2v",
    "wan_i2v14",
    "flux_i2i",
    "flux_kontext",
    "wan_klein",
    "ComfyUI_temp",
)


def secure_delete(path: Path, passes: int = 1) -> bool:
    """Overwrite file contents then unlink. Best-effort on Windows."""
    try:
        path = Path(path)
        if not path.is_file():
            return False
        # Make writable if needed
        try:
            path.chmod(stat.S_IWRITE | stat.S_IREAD)
        except OSError:
            pass

        size = path.stat().st_size
        if size > 0 and passes > 0:
            with open(path, "r+b", buffering=0) as fh:
                for _ in range(max(1, passes)):
                    fh.seek(0)
                    remaining = size
                    # Overwrite in chunks to limit peak memory
                    chunk = 1024 * 1024
                    while remaining > 0:
                        n = min(chunk, remaining)
                        fh.write(b"\x00" * n)
                        remaining -= n
                    fh.flush()
                    try:
                        os.fsync(fh.fileno())
                    except OSError:
                        pass
        path.unlink(missing_ok=True)
        # Kill common sidecar metadata if present
        for side in (path.with_suffix(path.suffix + ".json"), Path(str(path) + ".json")):
            if side.is_file():
                try:
                    side.unlink(missing_ok=True)
                except OSError:
                    pass
        return True
    except OSError:
        return False


def wipe_paths(paths: Iterable[Path], passes: int = 1) -> int:
    count = 0
    for p in paths:
        if secure_delete(Path(p), passes=passes):
            count += 1
    return count


def resolve_folder(root: Path, folder_type: str, subfolder: str = "") -> Path:
    """Map ComfyUI type names to directories under the install root."""
    root = Path(root)
    mapping = {
        "input": root / "input",
        "output": root / "output",
        "temp": root / "temp",
    }
    base = mapping.get(folder_type, root / folder_type)
    if subfolder:
        return base / subfolder
    return base


def path_for_media(
    root: Path,
    filename: str,
    subfolder: str = "",
    folder_type: str = "output",
) -> Path:
    return resolve_folder(root, folder_type, subfolder) / filename


def wipe_media_descriptor(
    root: Path,
    filename: str,
    subfolder: str = "",
    folder_type: str = "output",
    passes: int = 1,
) -> bool:
    return secure_delete(
        path_for_media(root, filename, subfolder, folder_type),
        passes=passes,
    )


def wipe_our_artifacts(root: Path, passes: int = 1) -> int:
    """
    Wipe app-tagged files under ComfyUI input/output/temp.
    Called after every job as a safety net.
    """
    root = Path(root)
    count = 0
    for folder_name in ("input", "output", "temp"):
        folder = root / folder_name
        if not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            if not path.is_file():
                continue
            name = path.name
            if name.startswith(OUR_PREFIXES) or name.startswith("wan_") or "wan_i" in name:
                if secure_delete(path, passes=passes):
                    count += 1
    return count


def wipe_listed_files(
    root: Path,
    descriptors: list[dict],
    passes: int = 1,
) -> int:
    """descriptors: {filename, subfolder, type} from ComfyUI history outputs."""
    n = 0
    for d in descriptors:
        if wipe_media_descriptor(
            root,
            d.get("filename") or "",
            d.get("subfolder") or "",
            d.get("type") or "output",
            passes=passes,
        ):
            n += 1
    return n


def detect_comfy_root_from_argv(argv0: Optional[str]) -> Optional[Path]:
    if not argv0:
        return None
    p = Path(argv0)
    # .../ComfyUI/main.py → ComfyUI/
    if p.name.lower() in ("main.py", "main.pyc"):
        return p.parent
    if (p.parent / "input").is_dir() or (p.parent / "models").is_dir():
        return p.parent
    return None
