"""The one local folder for media this app writes on a PC (tests, previews, Mongo pulls,
trainer datasets, thumbnails). `npm run wipe` empties it."""

from __future__ import annotations

from pathlib import Path

TEMP_ASSETS = Path(__file__).resolve().parents[1] / "temp_assets"


def temp_assets(*parts: str) -> Path:
    path = TEMP_ASSETS.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
