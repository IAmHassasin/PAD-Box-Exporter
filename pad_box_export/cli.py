from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pad_box_export.classifier import DEFAULT_SAT_THRESHOLD, DEFAULT_WHITE_THRESHOLD
from pad_box_export.pipeline import run_pipeline, write_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pad-box-export",
        description="Export owned PAD monsters from Monster Book screen recordings.",
    )
    p.add_argument(
        "input",
        help="Screen recording (.mp4, .mov) or folder of screenshots",
    )
    p.add_argument("-o", "--output", default="box.json", help="Output JSON path")
    p.add_argument("--report", help="Write text report to this path")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--min-id", type=int, help="Minimum monster ID to include")
    p.add_argument("--max-id", type=int, help="Maximum monster ID to include")
    p.add_argument(
        "--sat-threshold",
        type=float,
        default=DEFAULT_SAT_THRESHOLD,
        help=f"HSV saturation threshold for owned (default {DEFAULT_SAT_THRESHOLD})",
    )
    p.add_argument(
        "--white-threshold",
        type=float,
        default=DEFAULT_WHITE_THRESHOLD,
        help=f"Center white-pixel ratio threshold (default {DEFAULT_WHITE_THRESHOLD})",
    )
    p.add_argument(
        "--frame-interval",
        type=float,
        default=0.4,
        help="Seconds between sampled video frames (default 0.4)",
    )
    p.add_argument("--max-frames", type=int, help="Limit frames processed (debug)")
    p.add_argument("--debug-crops", metavar="DIR", help="Save icon crops for debugging")
    p.add_argument(
        "--ocr-backend",
        choices=["auto", "ocrmac", "easyocr"],
        default="auto",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: input not found: {input_path}", file=sys.stderr)
        return 1

    def on_frame(_idx: int, count: int) -> None:
        if args.verbose and count % 25 == 0:
            print(f"  ... {count} frames")

    try:
        export = run_pipeline(
            input_path,
            ocr_backend=args.ocr_backend,
            sat_threshold=args.sat_threshold,
            white_threshold=args.white_threshold,
            min_id=args.min_id,
            max_id=args.max_id,
            frame_interval=args.frame_interval,
            max_frames=args.max_frames,
            debug_crops_dir=args.debug_crops,
            verbose=args.verbose,
            on_frame=on_frame,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    export.save(args.output)
    print(f"Wrote {args.output} ({export.meta.get('owned_count', 0)} owned)")

    if args.report:
        write_report(export, args.report)
        print(f"Wrote {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
