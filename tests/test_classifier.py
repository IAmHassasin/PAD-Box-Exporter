import numpy as np

from pad_box_export.classifier import classify_icon, crop_icon
from pad_box_export.models import BBox


def _solid_icon(bgr: tuple[int, int, int], size: int = 64) -> np.ndarray:
    icon = np.zeros((size, size, 3), dtype=np.uint8)
    icon[:, :] = bgr
    return icon


def test_owned_colorful_icon():
    icon = _solid_icon((20, 180, 220))
    result = classify_icon(icon, sat_threshold=42, white_threshold=0.35)
    assert result.is_owned
    assert result.sat_mean > 42


def test_seen_grayscale_icon():
    icon = _solid_icon((90, 90, 90))
    result = classify_icon(icon, sat_threshold=42, white_threshold=0.35)
    assert not result.is_owned
    assert result.sat_mean < 42


def test_unknown_white_center():
    icon = _solid_icon((70, 70, 70))
    ch, cw = icon.shape[:2]
    icon[ch // 4 : 3 * ch // 4, cw // 4 : 3 * cw // 4] = (240, 240, 240)
    result = classify_icon(icon, sat_threshold=42, white_threshold=0.35)
    assert not result.is_owned
    assert result.white_ratio >= 0.35


def test_crop_icon_from_label():
    image = np.zeros((800, 400, 3), dtype=np.uint8)
    label = BBox(50, 600, 60, 20)
    crop = crop_icon(image, label)
    assert crop is not None
    assert crop.shape[0] > 0 and crop.shape[1] > 0
