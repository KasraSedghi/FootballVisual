"""Shot-change detection.

Homography propagation assumes the camera moved a little since the last frame.
A cut breaks that assumption completely, and it breaks it silently: optical flow
between two unrelated shots still returns matches, RANSAC still fits a
homography to them, and that homography gets composed onto the running estimate.
From then on every player is projected to a confidently wrong place, with
nothing in the output indicating anything went wrong.

So a cut has to be detected as a cut, not absorbed as fast motion. The signal
used here is the correlation between consecutive frames' colour histograms,
computed on a heavily downscaled frame. Within one shot that correlation stays
high even through a fast pan, because the scene is still the same grass, the
same kit and the same stands. Across a cut it collapses.

Histograms are used rather than pixel differences precisely because they ignore
where things are. A hard pan moves every pixel and would look like a cut to a
pixel-difference test, while barely moving the histogram.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class CutEvent:
    frame: int
    correlation: float


@dataclass
class CutDetector:
    """Flags frames where the shot changed.

    `threshold` is the histogram correlation below which a change of shot is
    declared. It is deliberately low: the cost of a missed cut is a permanently
    corrupted homography, but the cost of a false positive is only a needless
    re-calibration, so this is tuned to fire on real cuts rather than to catch
    every marginal one.

    `min_gap` suppresses repeat firings. Real broadcast cuts are followed by a
    dissolve or a settling camera, which can otherwise trip the detector several
    frames running and cause repeated re-calibration.
    """

    threshold: float = 0.55
    min_gap: int = 8
    _previous: np.ndarray | None = field(default=None, repr=False)
    _last_cut_frame: int = field(default=-10_000, repr=False)
    last_correlation: float = 1.0
    events: list[CutEvent] = field(default_factory=list)

    @staticmethod
    def _signature(frame: np.ndarray) -> np.ndarray:
        """A small, position-insensitive summary of a frame.

        A 2D hue/saturation histogram at coarse resolution. Hue and saturation
        rather than value so that stadium lighting changes and auto-exposure
        drift do not read as a cut.
        """
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 24], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def update(self, frame: np.ndarray, frame_index: int) -> bool:
        """Feed the next frame. Returns True when the shot changed."""
        signature = self._signature(frame)

        if self._previous is None:
            self._previous = signature
            return False

        correlation = float(
            cv2.compareHist(self._previous, signature, cv2.HISTCMP_CORREL)
        )
        self._previous = signature
        self.last_correlation = correlation

        if correlation >= self.threshold:
            return False
        if frame_index - self._last_cut_frame < self.min_gap:
            # Still inside the settling window of the cut just reported.
            return False

        self._last_cut_frame = frame_index
        self.events.append(CutEvent(frame=frame_index, correlation=correlation))
        return True

    def reset(self) -> None:
        self._previous = None
        self._last_cut_frame = -10_000
