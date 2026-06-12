from __future__ import annotations

import cv2
import numpy as np

from pad_box_export.models import BBox, ClassificationResult

DEFAULT_SAT_THRESHOLD = 42.0
DEFAULT_WHITE_THRESHOLD = 0.35
LOW_CONFIDENCE_MARGIN = 8.0

ICON_TOP_FACTOR = 4.5
ICON_HEIGHT_FACTOR = 4.0
ICON_WIDTH_FACTOR = 1.2


def crop_icon(image: np.ndarray, label: BBox) -> np.ndarray | None:
    img_h, img_w = image.shape[:2]
    icon_h = int(label.h * ICON_HEIGHT_FACTOR)
    icon_w = int(label.w * ICON_WIDTH_FACTOR)
    icon_top = int(label.y - label.h * ICON_TOP_FACTOR)
    icon_left = int(label.x + label.w / 2 - icon_w / 2)

    bbox = BBox(icon_left, icon_top, icon_w, icon_h).clamp(img_w, img_h)
    if bbox.h < 8 or bbox.w < 8:
        return None

    crop = image[bbox.y : bbox.y + bbox.h, bbox.x : bbox.x + bbox.w]
    return crop if crop.size else None


def classify_icon(
    icon_bgr: np.ndarray,
    sat_threshold: float = DEFAULT_SAT_THRESHOLD,
    white_threshold: float = DEFAULT_WHITE_THRESHOLD,
) -> ClassificationResult:
    if icon_bgr is None or icon_bgr.size == 0:
        return ClassificationResult(False, 0.0, 0.0, 1.0)

    hsv = cv2.cvtColor(icon_bgr, cv2.COLOR_BGR2HSV)
    sat_mean = float(hsv[:, :, 1].mean())
    val_mean = float(hsv[:, :, 2].mean())

    ch, cw = icon_bgr.shape[:2]
    cy0, cy1 = ch // 4, (3 * ch) // 4
    cx0, cx1 = cw // 4, (3 * cw) // 4
    center = icon_bgr[cy0:cy1, cx0:cx1]
    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    white_ratio = float((gray > 200).mean())

    is_owned = sat_mean > sat_threshold and white_ratio < white_threshold

    sat_margin = sat_mean - sat_threshold
    white_margin = white_threshold - white_ratio
    raw_conf = min(sat_margin, white_margin * 100.0)
    confidence = max(0.0, min(1.0, raw_conf / 30.0))

    if not is_owned and sat_mean > sat_threshold - LOW_CONFIDENCE_MARGIN:
        confidence = max(confidence, 0.35)

    # Very dark / flat icons are unlikely owned
    if val_mean < 35:
        is_owned = False
        confidence = min(confidence, 0.2)

    return ClassificationResult(is_owned, confidence, sat_mean, white_ratio)
