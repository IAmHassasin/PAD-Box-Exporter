from __future__ import annotations

import re
import sys
from typing import Literal

import cv2
import numpy as np
from PIL import Image

from pad_box_export.models import BBox, LabelMatch

LABEL_RE = re.compile(r"No[.:]?\s*(\d{1,5})", re.IGNORECASE)
ENTRIES_RE = re.compile(r"Entries\s+(\d+)", re.IGNORECASE)
OcrBackend = Literal["auto", "ocrmac", "easyocr"]

_easyocr_reader = None


def _resolve_backend(backend: OcrBackend) -> str:
    if backend != "auto":
        return backend
    if sys.platform == "darwin":
        try:
            import ocrmac  # noqa: F401

            return "ocrmac"
        except ImportError:
            pass
    try:
        import easyocr  # noqa: F401

        return "easyocr"
    except ImportError:
        raise RuntimeError(
            "No OCR backend available. Install: pip install ocrmac (macOS) or easyocr"
        )


def _norm_bbox_to_pixels(bbox: list[float], img_w: int, img_h: int) -> BBox:
    """Convert normalized Vision bbox (bottom-left origin) to pixel top-left."""
    x, y, w, h = bbox
    px = int(x * img_w)
    pw = max(1, int(w * img_w))
    ph = max(1, int(h * img_h))
    # Vision uses bottom-left origin; OpenCV uses top-left
    py = int((1.0 - y - h) * img_h)
    return BBox(px, py, pw, ph)


def _ocr_ocrmac(image_bgr: np.ndarray) -> list[tuple[str, float, BBox]]:
    from ocrmac import ocrmac

    img_h, img_w = image_bgr.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    annotations = ocrmac.OCR(pil).recognize()

    results: list[tuple[str, float, BBox]] = []
    for item in annotations:
        if len(item) < 3:
            continue
        text, conf, bbox = item[0], float(item[1]), item[2]
        if not text or not bbox:
            continue
        results.append((text.strip(), conf, _norm_bbox_to_pixels(bbox, img_w, img_h)))
    return results


def _get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr

        _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _easyocr_reader


def _ocr_easyocr(image_bgr: np.ndarray) -> list[tuple[str, float, BBox]]:
    reader = _get_easyocr_reader()
    results: list[tuple[str, float, BBox]] = []
    for bbox_pts, text, conf in reader.readtext(image_bgr):
        xs = [p[0] for p in bbox_pts]
        ys = [p[1] for p in bbox_pts]
        x, y = int(min(xs)), int(min(ys))
        w, h = int(max(xs) - x), int(max(ys) - y)
        results.append((text.strip(), float(conf), BBox(x, y, max(1, w), max(1, h))))
    return results


def run_ocr(image_bgr: np.ndarray, backend: OcrBackend = "auto") -> list[tuple[str, float, BBox]]:
    resolved = _resolve_backend(backend)
    if resolved == "ocrmac":
        return _ocr_ocrmac(image_bgr)
    if resolved == "easyocr":
        return _ocr_easyocr(image_bgr)
    raise ValueError(f"Unknown OCR backend: {resolved}")


def _parse_label(text: str) -> int | None:
    m = LABEL_RE.search(text.replace(" ", ""))
    if not m:
        m = LABEL_RE.search(text)
    if not m:
        return None
    return int(m.group(1))


def _merge_adjacent_labels(
    tokens: list[tuple[str, float, BBox]],
) -> list[tuple[str, float, BBox]]:
    """Join 'No:' and trailing digits when OCR splits them."""
    if not tokens:
        return []

    merged: list[tuple[str, float, BBox]] = []
    i = 0
    while i < len(tokens):
        text, conf, bbox = tokens[i]
        if re.match(r"No[.:]?\s*$", text, re.I) and i + 1 < len(tokens):
            nxt_text, nxt_conf, nxt_bbox = tokens[i + 1]
            if re.match(r"\d{1,5}", nxt_text):
                combined = f"{text}{nxt_text}"
                x0 = min(bbox.x, nxt_bbox.x)
                y0 = min(bbox.y, nxt_bbox.y)
                x1 = max(bbox.x + bbox.w, nxt_bbox.x + nxt_bbox.w)
                y1 = max(bbox.y + bbox.h, nxt_bbox.y + nxt_bbox.h)
                merged.append(
                    (combined, (conf + nxt_conf) / 2, BBox(x0, y0, x1 - x0, y1 - y0))
                )
                i += 2
                continue
        merged.append((text, conf, bbox))
        i += 1
    return merged


def find_labels(
    image_bgr: np.ndarray,
    backend: OcrBackend = "auto",
    min_id: int | None = None,
    max_id: int | None = None,
) -> list[LabelMatch]:
    img_h, img_w = image_bgr.shape[:2]
    tokens = run_ocr(image_bgr, backend=backend)
    tokens = sorted(tokens, key=lambda t: (t[2].y, t[2].x))
    tokens = _merge_adjacent_labels(tokens)

    matches: list[LabelMatch] = []
    seen_ids: set[int] = set()

    for text, conf, bbox in tokens:
        monster_id = _parse_label(text)
        if monster_id is None:
            continue
        if min_id is not None and monster_id < min_id:
            continue
        if max_id is not None and monster_id > max_id:
            continue

        # Skip header/footer UI (top 8%, bottom 12%)
        cy = bbox.y + bbox.h / 2
        if cy < img_h * 0.08 or cy > img_h * 0.88:
            continue

        if monster_id in seen_ids:
            continue
        seen_ids.add(monster_id)
        matches.append(LabelMatch(monster_id, bbox, conf, text))

    return matches


def find_entries_count(image_bgr: np.ndarray, backend: OcrBackend = "auto") -> int | None:
    tokens = run_ocr(image_bgr, backend=backend)
    for text, _, _ in tokens:
        m = ENTRIES_RE.search(text)
        if m:
            return int(m.group(1))
    return None
