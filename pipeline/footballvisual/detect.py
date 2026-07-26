"""Per-frame detection of players and the ball.

Two detectors live here because football needs two. A general object detector
handles people well and the ball badly: at broadcast framing the ball is a
handful of pixels, it is motion-blurred whenever it matters, and it looks like
every other small bright blob on the pitch. Measured on this project's own
clip, YOLOv8n finds roughly 85 to 90 percent of players and essentially zero
balls, which matches what the football-tracking literature reports. So players
go through YOLO and the ball goes through a purpose-built classical detector
that keys on the things that actually distinguish it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PERSON_CLASS = 0
SPORTS_BALL_CLASS = 32


@dataclass
class Detection:
    """One detection in image space."""

    bbox: tuple[float, float, float, float]
    score: float
    kind: str = "player"

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def centre(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def foot_point(self) -> tuple[float, float]:
        """Bottom-centre: where the object meets the ground plane.

        Correct for the ball as well as for players, and for the same reason.
        The homography maps the ground plane, so the point to project is where
        the object touches it. Projecting a ball's bbox *centre* treats a point
        one ball-radius up in the air as if it were on the grass, and at
        broadcast depth that reprojects metres up the pitch.
        """
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> np.ndarray:
    """Greedy non-maximum suppression, returning kept indices.

    Written out rather than pulled from OpenCV because the tiled detector feeds
    it boxes from overlapping crops, where a duplicate pair can have an IoU
    close to one, and having the comparison explicit here makes that behaviour
    easy to reason about.
    """
    if len(boxes) == 0:
        return np.empty(0, dtype=int)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou < iou_threshold]
    return np.asarray(keep, dtype=int)


class PlayerDetector:
    """YOLO person detection, optionally tiled for small-object recall.

    Tiling is the SAHI idea: cut the frame into overlapping crops and run the
    detector on each, so a player who is 30 pixels tall in the full frame is
    effectively 50 pixels tall by the time the network sees them. On this
    project's clip it lifts recall from 0.86 to 0.90 at the same confidence
    threshold. It costs one forward pass per tile, which is the trade being
    made.
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf: float = 0.15,
        imgsz: int = 1280,
        tiles: tuple[int, int] = (2, 2),
        tile_overlap: float = 0.2,
        device: str = "cpu",
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.tiles = tiles
        self.tile_overlap = tile_overlap
        self.device = device

    def _tile_windows(self, h: int, w: int) -> list[tuple[int, int, int, int]]:
        rows, cols = self.tiles
        if rows <= 1 and cols <= 1:
            return [(0, 0, w, h)]
        tw = int(round(w / (cols - (cols - 1) * self.tile_overlap)))
        th = int(round(h / (rows - (rows - 1) * self.tile_overlap)))
        step_x = int(round(tw * (1.0 - self.tile_overlap))) if cols > 1 else tw
        step_y = int(round(th * (1.0 - self.tile_overlap))) if rows > 1 else th

        windows = []
        for r in range(rows):
            for c in range(cols):
                x0 = min(c * step_x, max(0, w - tw))
                y0 = min(r * step_y, max(0, h - th))
                windows.append((x0, y0, min(w, x0 + tw), min(h, y0 + th)))
        return windows

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        boxes: list[list[float]] = []
        scores: list[float] = []

        for x0, y0, x1, y1 in self._tile_windows(h, w):
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            result = self.model(
                crop,
                classes=[PERSON_CLASS],
                conf=self.conf,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )[0]
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for b, s in zip(xyxy, confs):
                boxes.append([b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0])
                scores.append(float(s))

        if not boxes:
            return []

        box_arr = np.asarray(boxes, dtype=np.float64)
        score_arr = np.asarray(scores, dtype=np.float64)
        keep = _nms(box_arr, score_arr)

        out: list[Detection] = []
        for i in keep:
            x1, y1, x2, y2 = box_arr[i]
            bw, bh = x2 - x1, y2 - y1
            # Players are upright. A box wider than it is tall is a merged pair
            # or a piece of pitch furniture, and letting it through creates a
            # track that sits between two real players forever.
            if bh <= 0 or bw <= 0 or bh / bw < 1.1:
                continue
            out.append(Detection((float(x1), float(y1), float(x2), float(y2)), float(score_arr[i])))
        return out


class BallDetector:
    """Classical small-white-round-thing detector.

    The ball is separated from the many other bright blobs on a pitch by three
    properties that hold together only for it: it is small but not a single
    pixel, it is close to circular (pitch lines are not, and boot flashes are
    not), and it is not inside a player's bounding box (which removes socks,
    shorts, and the goalkeeper's gloves).

    A `previous` hint biases toward candidates near where the ball was last
    seen, because ball motion between frames is bounded and that single prior
    removes most of the remaining confusions.
    """

    def __init__(
        self,
        min_area: int = 4,
        max_area: int = 900,
        min_circularity: float = 0.55,
        search_radius: float = 180.0,
    ) -> None:
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.search_radius = search_radius

    def __call__(
        self,
        frame: np.ndarray,
        player_boxes: list[tuple[float, float, float, float]] | None = None,
        previous: tuple[float, float] | None = None,
    ) -> Detection | None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # White: bright and unsaturated. The pitch is strongly saturated green,
        # so this alone removes almost everything.
        mask = cv2.inRange(hsv, np.array([0, 0, 175]), np.array([180, 70, 255]))
        # Opening removes the thin pitch lines, which are white but only a
        # couple of pixels across; the ball survives because it is a disc.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

        if player_boxes:
            for x1, y1, x2, y2 in player_boxes:
                # Pad slightly: the ball at a player's feet should still be
                # findable, so only the body interior is suppressed.
                px1 = int(max(0, x1 + 0.15 * (x2 - x1)))
                px2 = int(min(frame.shape[1], x2 - 0.15 * (x2 - x1)))
                py1 = int(max(0, y1))
                py2 = int(min(frame.shape[0], y2 - 0.12 * (y2 - y1)))
                if px2 > px1 and py2 > py1:
                    mask[py1:py2, px1:px2] = 0

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best: Detection | None = None
        best_cost = float("inf")
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (self.min_area <= area <= self.max_area):
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 1e-6:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / max(1.0, h)
            if not (0.5 <= aspect <= 2.0):
                continue

            cx, cy = x + w / 2.0, y + h / 2.0
            # Cost prefers round things, and things near the last known ball.
            cost = (1.0 - circularity) * 100.0
            if previous is not None:
                dist = float(np.hypot(cx - previous[0], cy - previous[1]))
                if dist > self.search_radius:
                    cost += 500.0
                cost += dist * 0.4

            if cost < best_cost:
                best_cost = cost
                best = Detection(
                    (float(x), float(y), float(x + w), float(y + h)),
                    float(min(1.0, circularity)),
                    kind="ball",
                )
        return best


def open_video(path: str | Path) -> tuple[cv2.VideoCapture, int, float]:
    """Open a video, returning the capture, frame count and fps."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video {path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    return cap, count, fps
