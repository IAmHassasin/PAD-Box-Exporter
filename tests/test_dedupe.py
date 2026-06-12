from pad_box_export.dedupe import merge_detections
from pad_box_export.models import MonsterDetection


def test_keeps_highest_confidence():
    dets = [
        MonsterDetection(100, 0.3, True, 0, sat_mean=50),
        MonsterDetection(100, 0.9, True, 1, sat_mean=55),
        MonsterDetection(200, 0.8, True, 0, sat_mean=60),
        MonsterDetection(300, 0.7, False, 0, sat_mean=20),
    ]
    owned, low, dupes = merge_detections(dets, sat_threshold=42)
    assert owned == [100, 200]
    assert dupes == 1
    assert 300 not in owned
