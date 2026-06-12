from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import cv2
import numpy as np

from pad_box_export.classifier import (
    DEFAULT_SAT_THRESHOLD,
    DEFAULT_WHITE_THRESHOLD,
    classify_icon,
    crop_icon,
)
from pad_box_export.dedupe import merge_detections
from pad_box_export.models import BoxExport, MonsterDetection
from pad_box_export.ocr import OcrBackend, find_entries_count, find_labels
from pad_box_export.video import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    is_image,
    is_video,
    iter_image_folder,
    iter_video_frames,
    load_single_image,
)


def _iter_inputs(
    input_path: Path,
    frame_interval: float,
    max_frames: int | None,
) -> Iterator[tuple[int, np.ndarray]]:
    if input_path.is_dir():
        yield from iter_image_folder(input_path)
        return

    suffix = input_path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS or is_video(input_path):
        yield from iter_video_frames(
            input_path,
            interval_sec=frame_interval,
            max_frames=max_frames,
        )
        return

    if suffix in IMAGE_EXTENSIONS or is_image(input_path):
        yield 0, load_single_image(input_path)
        return

    raise ValueError(f"Unsupported input: {input_path}")


def process_frame(
    frame: np.ndarray,
    source_index: int,
    *,
    ocr_backend: OcrBackend = "auto",
    sat_threshold: float = DEFAULT_SAT_THRESHOLD,
    white_threshold: float = DEFAULT_WHITE_THRESHOLD,
    min_id: int | None = None,
    max_id: int | None = None,
    debug_crops_dir: Path | None = None,
) -> list[MonsterDetection]:
    labels = find_labels(
        frame,
        backend=ocr_backend,
        min_id=min_id,
        max_id=max_id,
    )
    detections: list[MonsterDetection] = []

    for label in labels:
        icon = crop_icon(frame, label.bbox)
        if icon is None:
            continue

        result = classify_icon(icon, sat_threshold, white_threshold)
        det = MonsterDetection(
            monster_id=label.monster_id,
            confidence=result.confidence,
            is_owned=result.is_owned,
            source_index=source_index,
            sat_mean=result.sat_mean,
            white_ratio=result.white_ratio,
        )
        detections.append(det)

        if debug_crops_dir is not None:
            tag = "owned" if result.is_owned else "skip"
            fname = f"{source_index:05d}_{label.monster_id}_{tag}.png"
            cv2.imwrite(str(debug_crops_dir / fname), icon)

    return detections


def run_pipeline(
    input_path: str | Path,
    *,
    ocr_backend: OcrBackend = "auto",
    sat_threshold: float = DEFAULT_SAT_THRESHOLD,
    white_threshold: float = DEFAULT_WHITE_THRESHOLD,
    min_id: int | None = None,
    max_id: int | None = None,
    frame_interval: float = 0.4,
    max_frames: int | None = None,
    debug_crops_dir: str | Path | None = None,
    verbose: bool = False,
    on_frame: Callable[[int, int], None] | None = None,
) -> BoxExport:
    input_path = Path(input_path)
    debug_dir = Path(debug_crops_dir) if debug_crops_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    all_detections: list[MonsterDetection] = []
    frames_processed = 0
    entries_footer: int | None = None
    source_kind = (
        "monster_book_video"
        if input_path.is_file() and is_video(input_path)
        else "monster_book_screenshots"
    )

    for source_index, frame in _iter_inputs(input_path, frame_interval, max_frames):
        frames_processed += 1
        if on_frame:
            on_frame(source_index, frames_processed)

        if verbose:
            print(f"Processing frame {source_index} (#{frames_processed})")

        dets = process_frame(
            frame,
            source_index,
            ocr_backend=ocr_backend,
            sat_threshold=sat_threshold,
            white_threshold=white_threshold,
            min_id=min_id,
            max_id=max_id,
            debug_crops_dir=debug_dir,
        )
        all_detections.extend(dets)

        if entries_footer is None:
            entries_footer = find_entries_count(frame, backend=ocr_backend)

    owned, low_confidence, dupes = merge_detections(all_detections, sat_threshold)

    meta = {
        "frames_processed": frames_processed,
        "owned_count": len(owned),
        "labels_seen": len(all_detections),
        "duplicates_dropped": dupes,
        "entries_footer": entries_footer,
        "low_confidence_ids": low_confidence,
        "sat_threshold": sat_threshold,
        "white_threshold": white_threshold,
        "frame_interval_sec": frame_interval,
    }

    return BoxExport.now(source=source_kind, owned=owned, meta=meta)


def write_report(export: BoxExport, path: str | Path) -> None:
    path = Path(path)
    meta = export.meta
    lines = [
        "PAD Box Export Report",
        "=====================",
        f"Exported at: {export.exported_at}",
        f"Source: {export.source}",
        f"Frames processed: {meta.get('frames_processed', 0)}",
        f"Labels detected: {meta.get('labels_seen', 0)}",
        f"Owned monsters: {meta.get('owned_count', 0)}",
        f"Duplicates dropped: {meta.get('duplicates_dropped', 0)}",
        f"Entries footer (OCR): {meta.get('entries_footer', 'n/a')}",
        "",
    ]

    footer = meta.get("entries_footer")
    owned_count = meta.get("owned_count", 0)
    if footer and owned_count:
        diff_pct = abs(owned_count - footer) / footer * 100
        lines.append(f"Footer vs owned delta: {diff_pct:.1f}%")
        if diff_pct > 2:
            lines.append("  Note: >2% gap — review scroll coverage or thresholds.")
        lines.append("")

    low = meta.get("low_confidence_ids") or []
    if low:
        lines.append(f"Low confidence IDs ({len(low)}): {low[:50]}")
        if len(low) > 50:
            lines.append(f"  ... and {len(low) - 50} more")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
