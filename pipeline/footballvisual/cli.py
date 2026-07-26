"""Command line entry points.

    python -m footballvisual render      build the synthetic broadcast clip
    python -m footballvisual sprites     extract person cut-outs for the render
    python -m footballvisual track       run the vision pipeline over a video
    python -m footballvisual evaluate    score tracks.json against ground truth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "data"
DEFAULT_SPRITES = REPO_ROOT / "assets" / "sprites"


def _cmd_sprites(args: argparse.Namespace) -> int:
    """Cut segmented people out of ordinary photographs.

    These become the player billboards in the synthetic clip. Using real
    photographs is the point: it means the detector under test is looking at
    genuine photographic texture rather than at shapes this project drew for it.
    """
    import cv2
    import numpy as np
    from ultralytics import YOLO

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    saved = 0
    for source in args.images:
        image = cv2.imread(str(source))
        if image is None:
            print(f"  skipping unreadable image {source}", file=sys.stderr)
            continue
        h, w = image.shape[:2]
        result = model(image, classes=[0], conf=0.35, verbose=False)[0]
        if result.masks is None:
            continue
        for mask, box in zip(result.masks.data.cpu().numpy(), result.boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = (int(v) for v in box)
            full = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            crop = image[max(0, y1) : y2, max(0, x1) : x2]
            alpha = full[max(0, y1) : y2, max(0, x1) : x2]
            ch, cw = crop.shape[:2]
            if ch < 120 or cw < 40 or ch / max(cw, 1) < 1.4 or float((alpha > 0.5).mean()) < 0.25:
                continue
            alpha8 = cv2.GaussianBlur((np.clip(alpha, 0, 1) * 255).astype(np.uint8), (5, 5), 0)
            cv2.imwrite(str(out_dir / f"person_{saved:02d}.png"), np.dstack([crop, alpha8]))
            saved += 1

    print(f"wrote {saved} sprites to {out_dir}")
    return 0 if saved else 1


def _cmd_render(args: argparse.Namespace) -> int:
    from .scenario import Scenario
    from .synth import render

    scenario = Scenario(fps=args.fps, duration_s=args.duration)
    truth = render(
        out_dir=Path(args.out),
        sprite_dir=Path(args.sprites),
        scenario=scenario,
        width=args.width,
        height=args.height,
    )
    print(f"rendered {len(truth['frames'])} frames to {Path(args.out) / 'broadcast.mp4'}")
    return 0


def _cmd_track(args: argparse.Namespace) -> int:
    from .pipeline import Pipeline, PipelineConfig

    config = PipelineConfig(
        video=Path(args.video),
        out=Path(args.out),
        ground_truth=Path(args.ground_truth) if args.ground_truth else None,
        calibration=Path(args.calibration) if args.calibration else None,
        weights=args.weights,
        conf=args.conf,
        imgsz=args.imgsz,
        tiles=(args.tile_rows, args.tile_cols),
        device=args.device,
        max_frames=args.max_frames,
        smooth_window=args.smooth,
        verbose=not args.quiet,
        auto_calibrate=args.auto_calibrate,
        camera_side=args.camera_side,
        detect_cuts=not args.no_cut_detection,
    )
    payload = Pipeline(config).run()

    if args.ground_truth:
        from .evaluate import evaluate

        truth = json.loads(Path(args.ground_truth).read_text())
        result = evaluate(payload, truth)
        print()
        print(result.format())
        payload["meta"]["stats"]["evaluation"] = result.to_dict()
        Path(args.out).write_text(json.dumps(payload))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluate import evaluate_files

    result = evaluate_files(Path(args.tracks), Path(args.ground_truth))
    print(result.format())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="footballvisual", description="Football vision to tactical map pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sprites", help="extract segmented person cut-outs from photos")
    p.add_argument("images", nargs="+", type=Path)
    p.add_argument("--out", default=DEFAULT_SPRITES)
    p.add_argument("--weights", default="yolov8n-seg.pt")
    p.set_defaults(func=_cmd_sprites)

    p = sub.add_parser("render", help="render the synthetic broadcast clip")
    p.add_argument("--out", default=DEFAULT_DATA)
    p.add_argument("--sprites", default=DEFAULT_SPRITES)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--duration", type=float, default=10.0)
    p.set_defaults(func=_cmd_render)

    p = sub.add_parser("track", help="run the vision pipeline over a video")
    p.add_argument("--video", default=DEFAULT_DATA / "broadcast.mp4")
    p.add_argument("--out", default=DEFAULT_DATA / "tracks.json")
    p.add_argument("--ground-truth", default=None, help="score against this ground truth")
    p.add_argument("--calibration", default=None, help="JSON of clicked pitch landmarks")
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.15)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--tile-rows", type=int, default=2)
    p.add_argument("--tile-cols", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--smooth", type=int, default=9)
    p.add_argument(
        "--auto-calibrate",
        action="store_true",
        help="calibrate from the pitch markings instead of clicked landmarks",
    )
    p.add_argument(
        "--camera-side",
        default="minus_y",
        choices=["minus_y", "plus_y"],
        help="which touchline the camera is behind; resolves the pitch's mirror symmetry",
    )
    p.add_argument(
        "--no-cut-detection",
        action="store_true",
        help="do not look for shot changes (propagates the homography across cuts)",
    )
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=_cmd_track)

    p = sub.add_parser("evaluate", help="score tracks.json against ground truth")
    p.add_argument("--tracks", default=DEFAULT_DATA / "tracks.json")
    p.add_argument("--ground-truth", default=DEFAULT_DATA / "ground_truth.json")
    p.set_defaults(func=_cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
