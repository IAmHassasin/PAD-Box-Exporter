from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BBox:
    """Pixel bounding box with top-left origin (OpenCV convention)."""

    x: int
    y: int
    w: int
    h: int

    def clamp(self, img_w: int, img_h: int) -> BBox:
        x = max(0, min(self.x, img_w - 1))
        y = max(0, min(self.y, img_h - 1))
        w = max(1, min(self.w, img_w - x))
        h = max(1, min(self.h, img_h - y))
        return BBox(x, y, w, h)


@dataclass
class LabelMatch:
    monster_id: int
    bbox: BBox
    ocr_confidence: float
    raw_text: str


@dataclass
class ClassificationResult:
    is_owned: bool
    confidence: float
    sat_mean: float
    white_ratio: float


@dataclass
class MonsterDetection:
    monster_id: int
    confidence: float
    is_owned: bool
    source_index: int
    sat_mean: float = 0.0
    white_ratio: float = 0.0


@dataclass
class BoxExport:
    version: int = 1
    exported_at: str = ""
    source: str = "monster_book_video"
    owned: list[int] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def now(cls, **kwargs: Any) -> BoxExport:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return cls(exported_at=ts, **kwargs)
