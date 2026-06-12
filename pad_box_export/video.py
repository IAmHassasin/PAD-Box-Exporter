from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _frame_sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _frame_diff_ratio(a: np.ndarray, b: np.ndarray) -> float:
    ga = cv2.cvtColor(cv2.resize(a, (160, 90)), cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(cv2.resize(b, (160, 90)), cv2.COLOR_BGR2GRAY)
    return float(np.mean(cv2.absdiff(ga, gb)) / 255.0)


def iter_video_frames(
    video_path: str | Path,
    interval_sec: float = 0.4,
    min_sharpness: float = 50.0,
    min_diff: float = 0.02,
    max_frames: int | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, bgr_frame) sampled from a screen recording."""
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * interval_sec)))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    prev_kept: np.ndarray | None = None
    yielded = 0
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step != 0:
                frame_idx += 1
                continue

            if _frame_sharpness(frame) < min_sharpness:
                frame_idx += 1
                continue

            if prev_kept is not None and _frame_diff_ratio(prev_kept, frame) < min_diff:
                frame_idx += 1
                continue

            yield frame_idx, frame
            prev_kept = frame
            yielded += 1
            frame_idx += 1

            if max_frames is not None and yielded >= max_frames:
                break
    finally:
        cap.release()

    if yielded == 0 and total > 0:
        cap = cv2.VideoCapture(str(path))
        ret, frame = cap.read()
        cap.release()
        if ret:
            yield 0, frame


def iter_image_folder(folder: str | Path) -> Iterator[tuple[int, np.ndarray]]:
    folder = Path(folder)
    paths = sorted(
        p for p in folder.iterdir() if p.is_file() and is_image(p)
    )
    for i, path in enumerate(paths):
        frame = cv2.imread(str(path))
        if frame is not None:
            yield i, frame


def load_single_image(path: str | Path) -> np.ndarray:
    frame = cv2.imread(str(path))
    if frame is None:
        raise OSError(f"Cannot read image: {path}")
    return frame
