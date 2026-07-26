"""A pinhole broadcast camera, and the ground-plane homography it induces.

The point of this module is that the homography used by the rest of the system
is not invented here, it *falls out* of a real camera. Given intrinsics K and
extrinsics [R|t], a world point on the pitch plane z=0 projects as

    x_img ~ K [R|t] (X, Y, 0, 1)^T = K [r1 r2 t] (X, Y, 1)^T

so `H = K [r1 r2 t]` exactly. That means the synthetic clip's ground-truth
homography is not an approximation of what a camera would do, it is what this
camera does, and the pipeline's estimate can be scored against it directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _normalise(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("cannot normalise a zero-length vector")
    return v / n


@dataclass(frozen=True)
class BroadcastCamera:
    """An elevated side-on camera of the sort a main broadcast feed uses.

    World coordinates are pitch metres with +z up, so `eye` and `target` are in
    the same frame as everything else in the project. Image coordinates are
    pixels with +y downward.
    """

    width: int
    height: int
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    vfov_deg: float = 24.0
    roll_deg: float = 0.0

    @property
    def focal_px(self) -> float:
        """Focal length in pixels from the vertical field of view."""
        return (self.height / 2.0) / np.tan(np.radians(self.vfov_deg) / 2.0)

    @property
    def K(self) -> np.ndarray:
        f = self.focal_px
        return np.array(
            [[f, 0.0, self.width / 2.0], [0.0, f, self.height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def Rt(self) -> tuple[np.ndarray, np.ndarray]:
        """World-to-camera rotation and translation.

        Camera axes follow the usual computer vision convention: +x right in
        the image, +y *down* in the image, +z along the viewing direction. The
        sign flip on the up vector is what converts the right-handed world into
        that image-space handedness, and forgetting it is why hand-rolled
        look-at code so often renders the scene upside down.
        """
        eye = np.asarray(self.eye, dtype=np.float64)
        target = np.asarray(self.target, dtype=np.float64)
        world_up = np.array([0.0, 0.0, 1.0])

        forward = _normalise(target - eye)
        right = _normalise(np.cross(forward, world_up))
        up = np.cross(right, forward)

        if abs(self.roll_deg) > 1e-9:
            c, s = np.cos(np.radians(self.roll_deg)), np.sin(np.radians(self.roll_deg))
            right, up = c * right + s * up, -s * right + c * up

        R = np.stack([right, -up, forward], axis=0)
        t = -R @ eye
        return R, t

    @property
    def P(self) -> np.ndarray:
        """Full 3x4 projection matrix, for points off the ground plane."""
        R, t = self.Rt
        return self.K @ np.hstack([R, t.reshape(3, 1)])

    @property
    def H(self) -> np.ndarray:
        """Pitch-plane (z=0) to image homography, normalised so h33 == 1."""
        R, t = self.Rt
        h = self.K @ np.column_stack([R[:, 0], R[:, 1], t])
        if abs(h[2, 2]) < 1e-12:  # pragma: no cover - would need a degenerate pose
            raise ValueError("camera pose yields a degenerate ground homography")
        return h / h[2, 2]

    def project_world(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project 3D world points.

        Returns `(image_points, depth)`. Depth is the camera-space z, which is
        both the visibility test (positive means in front of the camera) and the
        painter's-algorithm sort key the renderer needs.
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
        homogeneous = np.column_stack([pts, np.ones(len(pts))])
        projected = homogeneous @ self.P.T
        depth = projected[:, 2].copy()
        safe = np.where(np.abs(depth) < 1e-9, 1e-9, depth)
        return projected[:, :2] / safe.reshape(-1, 1), depth

    def player_pixel_height(self, x: float, y: float, height_m: float) -> float:
        """Pixel height of an upright object of `height_m` standing at (x, y).

        Computed by projecting the object's feet and head rather than by scaling
        a nominal size, so it stays correct across the frame where perspective
        foreshortening varies a lot between near and far touchlines.
        """
        pts, depth = self.project_world(
            np.array([[x, y, 0.0], [x, y, height_m]], dtype=np.float64)
        )
        if depth[0] <= 0.0 or depth[1] <= 0.0:
            return 0.0
        return float(abs(pts[1, 1] - pts[0, 1]))

    def horizon_row(self) -> float:
        """Image row of the horizon, i.e. where the ground plane vanishes.

        The renderer fills everything above this with stands instead of grass;
        without the cut, the inverse ground warp wraps around and paints a
        mirrored pitch into the sky.
        """
        R, t = self.Rt
        # The vanishing line of plane z=0 is H^-T (0,0,1)^T in image space, but
        # it is simpler and more robust to project a very distant ground point.
        f = self.K[0, 0]
        del f
        h = self.H
        # Rows where the projective denominator vanishes: h31*X + h32*Y + h33 = 0
        # maps to the image line l = H^-T (0,0,1). Take that line's row at the
        # image centre column.
        line = np.linalg.inv(h).T @ np.array([0.0, 0.0, 1.0])
        a, b, c = line
        if abs(b) < 1e-12:
            return 0.0
        return float(-(a * (self.width / 2.0) + c) / b)


def broadcast_pan(
    width: int,
    height: int,
    num_frames: int,
    start_x: float = 10.0,
    end_x: float = 24.0,
    eye_height: float = 17.0,
    eye_y: float = -70.0,
    vfov_deg: float = 24.0,
) -> list[BroadcastCamera]:
    """A camera that pans slowly to follow play, one entry per frame.

    A static camera would let the pipeline calibrate once and coast, which
    would dodge the hard part. Panning means the homography genuinely changes
    every frame and has to be tracked, which is the realistic problem. The pan
    uses a smoothstep so there is no velocity discontinuity at either end for
    the optical-flow tracker to trip over.
    """
    cams: list[BroadcastCamera] = []
    for i in range(num_frames):
        u = i / max(1, num_frames - 1)
        ease = u * u * (3.0 - 2.0 * u)
        cx = start_x + (end_x - start_x) * ease
        # The camera body barely moves; it is mostly a rotation, exactly like a
        # tripod-mounted broadcast camera.
        cams.append(
            BroadcastCamera(
                width=width,
                height=height,
                eye=(cx * 0.18, eye_y, eye_height),
                target=(cx, 1.0, 0.0),
                vfov_deg=vfov_deg,
            )
        )
    return cams
