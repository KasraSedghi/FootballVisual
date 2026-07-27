"""End-to-end: video in, tactical track data out.

The stages, in order, and why the order matters:

1. Detect players (YOLO, tiled) and the ball (classical) per frame.
2. Track players (ByteTrack) and the ball (coasting filter) to get identities.
3. Keep the homography valid as the camera pans, seeded from a calibration.
4. Project each track's foot point through the homography into pitch metres.
5. Cluster jersey colours into teams, once, over the whole clip.
6. Smooth the pitch trajectories and emit `tracks.json`.

Team clustering deliberately runs after tracking rather than per frame, because
it needs a track's whole colour history to vote on. Smoothing runs last,
because smoothing in pixels before projection would blur across the strongly
non-linear perspective map and bend straight runs.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import pitch
from .autocalibrate import calibrate_auto
from .calibrate import HomographyTracker, landmark_correspondences, pitch_error
from .cuts import CutDetector
from .detect import BallDetector, Detection, PlayerDetector, open_video
from .homography import calibrate_from_landmarks, image_point_for_player, invert, project
from .teams import TeamVoter, classify_officials
from .track import BallTracker, ByteTracker


@dataclass
class PipelineConfig:
    video: Path
    out: Path
    ground_truth: Path | None = None
    weights: str = "yolov8n.pt"
    conf: float = 0.15
    imgsz: int = 1280
    tiles: tuple[int, int] = (2, 2)
    device: str = "cpu"
    max_frames: int | None = None
    stride: int = 1
    calibration: Path | None = None
    click_noise_px: float = 2.0
    smooth_window: int = 9
    verbose: bool = True
    # Calibrate from the pitch markings instead of clicked landmarks.
    auto_calibrate: bool = False
    camera_side: str = "minus_y"
    # Detect shot changes and re-calibrate, rather than propagating a
    # homography across a cut that it cannot possibly still describe.
    detect_cuts: bool = True


@dataclass
class FrameRecord:
    frame: int
    time_s: float
    players: dict[int, tuple[float, float]] = field(default_factory=dict)
    boxes: dict[int, tuple[float, float, float, float]] = field(default_factory=dict)
    ball: tuple[float, float] | None = None
    homography: np.ndarray | None = None


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average with shrinking windows at the ends.

    Padding by edge replication would flatten the start and end of every run,
    which is where a tactical snapshot is most likely to be taken. Shrinking
    the window instead keeps the endpoints unbiased at the cost of noisier
    smoothing exactly where there is less data to smooth with.
    """
    n = len(values)
    if n == 0 or window <= 1:
        return values
    half = window // 2
    out = np.empty_like(values)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = values[lo:hi].mean(axis=0)
    return out


class Pipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.records: list[FrameRecord] = []
        self.stats: dict = {}

    # -- calibration -----------------------------------------------------

    def _seed_homography(self, width: int, height: int) -> np.ndarray:
        """Establish the frame-zero pitch-to-image homography.

        Production path: a calibration file of clicked landmark pixels. Demo
        path: derive those clicks from the synthetic clip's true homography and
        add noise, which represents a human clicking imprecisely rather than
        handing the pipeline a free perfect answer.
        """
        cfg = self.config
        if cfg.auto_calibrate:
            raise RuntimeError(
                "auto calibration is handled in run(), not _seed_homography"
            )
        if cfg.calibration is not None:
            data = json.loads(Path(cfg.calibration).read_text())
            correspondences = {k: tuple(v) for k, v in data["landmarks"].items()}
        elif cfg.ground_truth is not None:
            truth = json.loads(Path(cfg.ground_truth).read_text())
            h_true = np.array(truth["frames"][0]["homography"], dtype=np.float64).reshape(3, 3)
            correspondences = landmark_correspondences(
                h_true, width, height, noise_px=cfg.click_noise_px, seed=5
            )
        else:
            raise ValueError(
                "no calibration available: pass --calibration with clicked landmarks, "
                "or --ground-truth for a synthetic clip"
            )

        calib = calibrate_from_landmarks(correspondences)
        self.stats["calibration"] = calib.to_dict()
        if self.config.verbose:
            print(
                f"  calibrated on {len(correspondences)} landmarks, "
                f"mean reprojection {calib.mean_error_px:.2f}px, "
                f"rejected {len(calib.rejected())}"
            )
        return calib.h

    # -- main loop -------------------------------------------------------

    def run(self) -> dict:
        cfg = self.config
        cap, total, fps = open_video(cfg.video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        limit = min(total, cfg.max_frames) if cfg.max_frames else total

        if cfg.verbose:
            print(f"video {cfg.video} {width}x{height} {fps:.1f}fps, {limit} frames")

        detector = PlayerDetector(
            weights=cfg.weights, conf=cfg.conf, imgsz=cfg.imgsz,
            tiles=cfg.tiles, device=cfg.device,
        )
        ball_detector = BallDetector()
        tracker = ByteTracker()
        ball_tracker = BallTracker()
        voter = TeamVoter()

        cut_detector = CutDetector() if cfg.detect_cuts else None
        h: np.ndarray | None = None
        h_tracker: HomographyTracker | None = None
        if not cfg.auto_calibrate:
            h = self._seed_homography(width, height)
            h_tracker = HomographyTracker(h=h)
        self.stats["cuts"] = []
        self.stats["recalibrations"] = 0

        truth = None
        if cfg.ground_truth is not None:
            truth = json.loads(Path(cfg.ground_truth).read_text())

        drift: list[tuple[int, float, float]] = []
        last_ball_px: tuple[float, float] | None = None
        started = time.time()

        for index in range(limit):
            ok, frame = cap.read()
            if not ok:
                break
            if index % cfg.stride != 0:
                continue

            detections = detector(frame)
            boxes = [d.bbox for d in detections]

            # A cut invalidates the running homography outright. Propagating
            # across one produces a confidently wrong answer with no signal that
            # anything happened, so the estimate is rebuilt from the markings
            # instead, seeded with the last good fit to settle the pitch's
            # rotational symmetry.
            cut = cut_detector.update(frame, index) if cut_detector else False
            if cut:
                self.stats["cuts"].append(
                    {"frame": index, "correlation": round(cut_detector.last_correlation, 3)}
                )
                if cfg.verbose:
                    print(f"  shot change at frame {index}, re-calibrating")

            needs_calibration = h is None or cut
            if needs_calibration:
                result = calibrate_auto(
                    frame, camera_side=cfg.camera_side, prior_h=h
                )
                if result is not None and result.is_confident:
                    h = result.h
                    h_tracker = HomographyTracker(h=h)
                    h_tracker.start(frame, boxes)
                    self.stats["recalibrations"] += 1
                    if cfg.verbose:
                        print(
                            f"    auto calibrated: {result.score:.2f}px, "
                            f"{result.inlier_fraction:.0%} of the model on markings, "
                            f"{result.explained_fraction:.0%} of the markings explained"
                        )
                elif h is None:
                    # Nothing to fall back on yet, so this frame cannot be
                    # projected at all. Skip it rather than invent a homography.
                    if cfg.verbose and index % 25 == 0:
                        print(f"  frame {index}: no confident calibration yet")
                    continue
                else:
                    # Keep the old homography. It is wrong after a cut, but a
                    # low-confidence automatic fit is likely worse, and the
                    # count of unresolved cuts is reported either way.
                    if cfg.verbose:
                        print("    re-calibration was not confident, keeping previous")
                    h_tracker = HomographyTracker(h=h)
                    h_tracker.start(frame, boxes)
            elif index == 0 or h_tracker is None:
                h_tracker = HomographyTracker(h=h)
                h_tracker.start(frame, boxes)
            else:
                h = h_tracker.step(frame, boxes)

            ball_det = ball_detector(frame, boxes, last_ball_px)
            ball_px = ball_tracker.update(ball_det)
            last_ball_px = ball_px

            tracks = tracker.update(detections, frame_index=index)

            record = FrameRecord(frame=index, time_s=index / fps, homography=h.copy())
            h_inv = invert(h)

            for track in tracks:
                bbox = track.bbox
                voter.observe(track.track_id, frame, bbox)
                foot = image_point_for_player(bbox)
                xy = project(h_inv, np.array([foot]))[0]
                # Reject anything that lands implausibly far off the pitch. A
                # near-horizon detection projects to hundreds of metres, and one
                # such point would wreck every team-shape metric it entered.
                if not pitch.is_inside(float(xy[0]), float(xy[1]), margin=6.0):
                    continue
                record.players[track.track_id] = (float(xy[0]), float(xy[1]))
                record.boxes[track.track_id] = bbox

            if ball_px is not None:
                bxy = project(h_inv, np.array([ball_px]))[0]
                if pitch.is_inside(float(bxy[0]), float(bxy[1]), margin=4.0):
                    record.ball = (float(bxy[0]), float(bxy[1]))

            self.records.append(record)

            if truth is not None and index < len(truth["frames"]):
                h_true = np.array(
                    truth["frames"][index]["homography"], dtype=np.float64
                ).reshape(3, 3)
                mean_err, max_err = pitch_error(h, h_true, image_size=(width, height))
                drift.append((index, mean_err, max_err))

            if cfg.verbose and index % 25 == 0:
                elapsed = time.time() - started
                print(
                    f"  frame {index:4d}/{limit}  tracks={len(tracks):2d}  "
                    f"ball={'yes' if ball_px else ' no'}  {elapsed:.1f}s"
                )

        cap.release()

        assignment = voter.resolve()
        self.stats["runtimeS"] = round(time.time() - started, 2)
        self.stats["framesProcessed"] = len(self.records)
        self.stats["fps"] = fps
        self.stats["teamSamples"] = voter.sample_counts()

        if drift:
            arr = np.array([[d[1], d[2]] for d in drift])
            finite = arr[np.isfinite(arr).all(axis=1)]
            if len(finite):
                self.stats["homographyDrift"] = {
                    "meanErrorM": round(float(finite[:, 0].mean()), 3),
                    "finalMeanErrorM": round(float(finite[-1, 0]), 3),
                    "worstMeanErrorM": round(float(finite[:, 0].max()), 3),
                    "framesScored": int(len(finite)),
                }

        return self._export(assignment, fps, width, height)

    # -- export ----------------------------------------------------------

    def _export(self, assignment, fps: float, width: int, height: int) -> dict:
        cfg = self.config

        # Gather per-track series so they can be smoothed as trajectories.
        series: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
        for rec in self.records:
            for tid, (x, y) in rec.players.items():
                series[tid].append((rec.frame, x, y))

        # Colour puts keepers and officials in the same "other" bucket because
        # both wear kit unlike either team. Telling them apart needs pitch
        # positions, which only exist now that everything has been projected.
        trajectories = {
            tid: [(p[1], p[2]) for p in points] for tid, points in series.items()
        }
        ball_series = [r.ball for r in self.records if r.ball is not None]
        officials = classify_officials(
            candidate_ids=set(assignment.keeper_ids),
            trajectories=trajectories,
            ball=ball_series or None,
        )
        self.stats["officials"] = {
            str(tid): {"role": role, **officials.evidence.get(tid, {})}
            for tid, role in officials.roles.items()
        }

        smoothed: dict[int, dict[int, tuple[float, float]]] = {}
        for tid, points in series.items():
            arr = np.array([[p[1], p[2]] for p in points], dtype=np.float64)
            sm = _moving_average(arr, cfg.smooth_window)
            smoothed[tid] = {points[i][0]: (float(sm[i, 0]), float(sm[i, 1])) for i in range(len(points))}

        ball_points = [(r.frame, r.ball) for r in self.records if r.ball is not None]
        ball_smoothed: dict[int, tuple[float, float]] = {}
        if ball_points:
            arr = np.array([[b[1][0], b[1][1]] for b in ball_points], dtype=np.float64)
            # The ball is smoothed less than players: it genuinely does change
            # direction instantly when struck, and over-smoothing rounds off
            # exactly the pass events the tactics engine keys on.
            sm = _moving_average(arr, max(3, cfg.smooth_window // 3))
            ball_smoothed = {ball_points[i][0]: (float(sm[i, 0]), float(sm[i, 1])) for i in range(len(ball_points))}

        # Only keep tracks that persisted; a track seen for a handful of frames
        # is almost always a detector false positive and would appear in the
        # sandbox as a player who flickers in and out of existence.
        min_len = max(5, int(0.08 * max(1, len(self.records))))
        keep = {tid for tid, pts in series.items() if len(pts) >= min_len}

        def label_for(track_id: int) -> str:
            team = assignment.team_of(track_id)
            if team != "other":
                return team
            role = officials.role_of(track_id)
            return role if role in ("keeper", "referee") else "other"

        tracks_out = []
        for tid in sorted(keep):
            team = label_for(tid)
            tracks_out.append(
                {
                    "id": int(tid),
                    "team": team,
                    "teamConfidence": round(float(assignment.confidence.get(tid, 0.0)), 3),
                    "frames": len(series[tid]),
                }
            )

        frames_out = []
        for rec in self.records:
            players = []
            for tid, _ in rec.players.items():
                if tid not in keep:
                    continue
                x, y = smoothed[tid][rec.frame]
                players.append(
                    {
                        "id": int(tid),
                        "team": label_for(tid),
                        "x": round(x, 3),
                        "y": round(y, 3),
                    }
                )
            ball = ball_smoothed.get(rec.frame)
            frames_out.append(
                {
                    "frame": rec.frame,
                    "timeS": round(rec.time_s, 3),
                    "players": players,
                    "ball": [round(ball[0], 3), round(ball[1], 3)] if ball else None,
                }
            )

        payload = {
            "meta": {
                "source": str(cfg.video),
                "fps": fps,
                "width": width,
                "height": height,
                "generatedBy": "footballvisual.pipeline",
                "pitch": {"length": pitch.PITCH_LENGTH, "width": pitch.PITCH_WIDTH},
                "stats": self.stats,
            },
            "tracks": tracks_out,
            "frames": frames_out,
        }

        out_path = Path(cfg.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload))
        if cfg.verbose:
            print(f"wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")
        return payload
