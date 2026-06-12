from __future__ import annotations

from pad_box_export.classifier import LOW_CONFIDENCE_MARGIN
from pad_box_export.models import MonsterDetection


def merge_detections(
    detections: list[MonsterDetection],
    sat_threshold: float,
) -> tuple[list[int], list[int], int]:
    """
    Merge detections by monster_id, keeping highest-confidence owned entry.

    Returns (owned_ids, low_confidence_ids, duplicates_dropped).
    """
    best: dict[int, MonsterDetection] = {}
    duplicates_dropped = 0

    for det in detections:
        if not det.is_owned:
            continue

        prev = best.get(det.monster_id)
        if prev is None:
            best[det.monster_id] = det
            continue

        duplicates_dropped += 1
        if det.confidence > prev.confidence:
            best[det.monster_id] = det

    owned_ids = sorted(best.keys())
    low_confidence: list[int] = []

    for mid, det in best.items():
        margin = det.sat_mean - sat_threshold
        if det.confidence < 0.5 or margin < LOW_CONFIDENCE_MARGIN:
            low_confidence.append(mid)

    return owned_ids, sorted(low_confidence), duplicates_dropped
