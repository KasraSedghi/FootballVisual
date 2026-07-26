"""Render the scenario as a broadcast-style video.

Why this exists: the pipeline needs real photographic input to be an honest
test of a detector, but a licensed broadcast clip is not something this project
can ship, and an unlabelled clip could not be scored anyway. So the renderer
composites *real segmented photographs of people* onto a perspective-projected
pitch. YOLO then runs on genuine photographic pixels, while the exact pitch
coordinate of every player stays known.

The result is deliberately not photoreal. It is a test harness that is hard in
the ways that matter for this pipeline (perspective, scale variation with
depth, mutual occlusion, motion blur, a moving camera, jersey colours that have
to be clustered) and easy in the ways that do not (no crowd motion, no shadows,
no broadcast graphics).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import pitch
from .camera import BroadcastCamera, broadcast_pan
from .scenario import BALL_RADIUS_M, PLAYER_HEIGHT_M, Scenario

# Jersey colours as BGR, matching how OpenCV holds pixels. These are the
# ground truth the team-clustering step is trying to rediscover.
TEAM_COLOURS_BGR: dict[str, tuple[int, int, int]] = {
    "blue": (200, 70, 30),
    "red": (40, 40, 205),
}
KEEPER_COLOURS_BGR: dict[str, tuple[int, int, int]] = {
    "blue": (60, 210, 230),
    "red": (70, 200, 90),
}


@dataclass
class Sprite:
    """A segmented person photograph used as a player billboard."""

    bgr: np.ndarray
    alpha: np.ndarray

    @property
    def aspect(self) -> float:
        return self.bgr.shape[0] / max(1, self.bgr.shape[1])


def load_sprites(sprite_dir: Path) -> list[Sprite]:
    """Load RGBA person cut-outs produced by the segmentation step."""
    sprites: list[Sprite] = []
    for path in sorted(Path(sprite_dir).glob("*.png")):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None or img.shape[2] != 4:
            continue
        sprites.append(Sprite(bgr=img[:, :, :3].copy(), alpha=img[:, :, 3].copy()))
    if not sprites:
        raise FileNotFoundError(
            f"no RGBA sprites found in {sprite_dir}; run `make sprites` first"
        )
    return sprites


def tint_jersey(sprite: Sprite, colour_bgr: tuple[int, int, int]) -> Sprite:
    """Recolour a sprite's torso to a team colour, keeping photographic shading.

    The hue and saturation are replaced but the value channel is left alone, so
    folds, creases and lighting survive. Flat-filling the region with the team
    colour instead would produce a uniform blob that the jersey-colour clusterer
    could separate trivially, which would make the team-assignment step look far
    more reliable than it is.
    """
    h, w = sprite.bgr.shape[:2]
    top, bottom = int(0.16 * h), int(0.56 * h)
    out = sprite.bgr.copy()

    torso = out[top:bottom]
    if torso.size == 0:
        return Sprite(out, sprite.alpha.copy())

    target = np.uint8([[list(colour_bgr)]])
    target_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)[0, 0]

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = target_hsv[0]
    # Keep a little of the original saturation variation so the fabric does not
    # read as a solid colour swatch.
    hsv[:, :, 1] = np.clip(
        0.75 * float(target_hsv[1]) + 0.25 * hsv[:, :, 1].astype(np.float32), 0, 255
    ).astype(np.uint8)

    # Renormalise brightness into a band a real shirt occupies. Several source
    # photographs are of people in dark clothing, and preserving their original
    # value channel leaves the "jersey" nearly black, where hue carries almost
    # no information and the team clusterer has nothing to separate on. Real
    # shirts are plainly coloured under stadium light, so stretching the torso's
    # own contrast into a mid-to-bright band keeps the fabric shading while
    # making the colour legible. Without this the render is testing the
    # clusterer on a problem football does not actually pose.
    value = hsv[:, :, 2].astype(np.float32)
    lo, hi = float(value.min()), float(value.max())
    if hi - lo > 1e-3:
        value = (value - lo) / (hi - lo)
    else:
        value = np.zeros_like(value)
    hsv[:, :, 2] = np.clip(110.0 + value * 125.0, 0, 255).astype(np.uint8)
    out[top:bottom] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return Sprite(out, sprite.alpha.copy())


def _grass(cam: BroadcastCamera, rng: np.random.Generator) -> np.ndarray:
    """Paint the pitch by inverse-warping every pixel back to pitch metres.

    Going backwards from pixels avoids the gaps a forward warp leaves, and it
    makes the mow stripes automatically perspective-correct because the stripe
    index is decided in metres, not pixels.
    """
    h, w = cam.height, cam.width
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Stands above the horizon: a dark band with a little noise so the frame is
    # not a flat colour the detector could key on.
    horizon = cam.horizon_row()
    canvas[:, :] = (48, 42, 38)
    noise = rng.integers(-9, 10, size=(h, w, 1), dtype=np.int16)

    ys, xs = np.mgrid[0:h, 0:w]
    px = np.column_stack([xs.ravel().astype(np.float64), ys.ravel().astype(np.float64)])

    h_inv = np.linalg.inv(cam.H)
    hom = np.column_stack([px, np.ones(len(px))])
    world = hom @ h_inv.T
    wcomp = world[:, 2]
    valid = np.abs(wcomp) > 1e-9
    X = np.zeros(len(px))
    Y = np.zeros(len(px))
    X[valid] = world[valid, 0] / wcomp[valid]
    Y[valid] = world[valid, 1] / wcomp[valid]

    # Only pixels below the horizon that land inside a generous margin around
    # the pitch get grass; the rest stays stands.
    on_ground = (
        valid
        & (ys.ravel() > horizon + 1.0)
        & (np.abs(X) < pitch.HALF_LENGTH + 12.0)
        & (np.abs(Y) < pitch.HALF_WIDTH + 10.0)
    )

    stripe = (np.floor(X / 5.5).astype(np.int64) % 2).astype(np.float32)
    base = np.where(stripe > 0.5, 88.0, 74.0)
    # Darken with distance from the camera for a cheap depth cue.
    depth_fade = np.clip(1.0 - (Y + pitch.HALF_WIDTH) / 190.0, 0.75, 1.0)
    green = (base * depth_fade).astype(np.float32)

    flat = canvas.reshape(-1, 3)
    flat[on_ground, 0] = np.clip(green[on_ground] * 0.42, 0, 255)
    flat[on_ground, 1] = np.clip(green[on_ground] * 1.35, 0, 255)
    flat[on_ground, 2] = np.clip(green[on_ground] * 0.40, 0, 255)
    canvas = flat.reshape(h, w, 3)

    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return canvas


def _draw_lines(frame: np.ndarray, cam: BroadcastCamera) -> None:
    """Project the pitch markings and stroke them onto the frame."""
    h_mat = cam.H
    for poly in pitch.pitch_polylines():
        hom = np.column_stack([poly, np.ones(len(poly))])
        proj = hom @ h_mat.T
        w = proj[:, 2]
        ok = np.abs(w) > 1e-9
        if ok.sum() < 2:
            continue
        pts = np.zeros((len(poly), 2))
        pts[ok] = proj[ok, :2] / w[ok].reshape(-1, 1)

        # Split into runs of valid, on-screen-ish points so a polyline that
        # crosses the horizon is not stroked straight across the frame.
        run: list[tuple[int, int]] = []
        for i in range(len(poly)):
            if ok[i] and -4000 < pts[i, 0] < 4000 and -4000 < pts[i, 1] < 4000:
                run.append((int(round(pts[i, 0])), int(round(pts[i, 1]))))
            else:
                if len(run) >= 2:
                    cv2.polylines(frame, [np.array(run)], False, (235, 240, 235), 2, cv2.LINE_AA)
                run = []
        if len(run) >= 2:
            cv2.polylines(frame, [np.array(run)], False, (235, 240, 235), 2, cv2.LINE_AA)


def _composite(frame: np.ndarray, sprite: Sprite, cx: float, feet_y: float, height_px: float) -> tuple[float, float, float, float] | None:
    """Alpha-blend a sprite so its feet sit at `(cx, feet_y)`.

    Returns the drawn bounding box in image coordinates, which the harness
    records so detector output can be scored against where the player really
    was drawn rather than against a reprojected guess.
    """
    if height_px < 6.0:
        return None
    scale = height_px / sprite.bgr.shape[0]
    w = max(3, int(round(sprite.bgr.shape[1] * scale)))
    h = max(6, int(round(height_px)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    bgr = cv2.resize(sprite.bgr, (w, h), interpolation=interp)
    alpha = cv2.resize(sprite.alpha, (w, h), interpolation=interp)

    x0 = int(round(cx - w / 2.0))
    y0 = int(round(feet_y - h))
    x1, y1 = x0 + w, y0 + h

    fh, fw = frame.shape[:2]
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(fw, x1), min(fh, y1)
    if dx1 <= dx0 or dy1 <= dy0:
        return None

    sub_bgr = bgr[sy0 : sy0 + (dy1 - dy0), sx0 : sx0 + (dx1 - dx0)]
    sub_a = alpha[sy0 : sy0 + (dy1 - dy0), sx0 : sx0 + (dx1 - dx0)].astype(np.float32) / 255.0
    sub_a = sub_a[:, :, None]

    region = frame[dy0:dy1, dx0:dx1].astype(np.float32)
    frame[dy0:dy1, dx0:dx1] = np.clip(
        sub_bgr.astype(np.float32) * sub_a + region * (1.0 - sub_a), 0, 255
    ).astype(np.uint8)

    return (float(x0), float(y0), float(x1), float(y1))


def _draw_ball(frame: np.ndarray, cam: BroadcastCamera, x: float, y: float, z: float) -> tuple[float, float, float, float] | None:
    pts, depth = cam.project_world(np.array([[x, y, z]], dtype=np.float64))
    if depth[0] <= 0:
        return None
    cx, cy = pts[0]
    # Radius in pixels from the same projection, so the ball shrinks with depth
    # on exactly the same curve the players do.
    r = cam.player_pixel_height(x, y, BALL_RADIUS_M * 2.0) / 2.0
    # Floor the radius at three pixels. Geometrically the ball is smaller than
    # that at this range, but a real broadcast ball is the in-focus, high
    # contrast subject of the shot and survives compression far better than a
    # literal projection of a 22cm sphere would suggest. Rendering it at its
    # true sub-pixel size produces a ball that no detector could ever find,
    # which would be modelling the renderer's limits rather than the problem.
    r = float(np.clip(r, 3.0, 40.0))
    if not (-50 < cx < frame.shape[1] + 50 and -50 < cy < frame.shape[0] + 50):
        return None
    cv2.circle(frame, (int(round(cx)), int(round(cy))), int(round(r)), (250, 250, 250), -1, cv2.LINE_AA)
    cv2.circle(frame, (int(round(cx)), int(round(cy))), int(round(r)), (40, 40, 40), 1, cv2.LINE_AA)
    return (cx - r, cy - r, cx + r, cy + r)


def _ffmpeg_binary() -> str | None:
    """Locate an ffmpeg we can transcode with, if there is one.

    `imageio-ffmpeg` bundles a static build, which is the reliable path in a
    container that has no system ffmpeg. A system binary is preferred when
    present because it is usually newer.
    """
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover - optional dependency
        return None


def _run_ffmpeg(ffmpeg: str, args: list[str], destination: Path) -> bool:
    try:
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", *args], check=True, capture_output=True)
    except (subprocess.CalledProcessError, OSError):
        return False
    return destination.exists() and destination.stat().st_size > 0


def _transcode_for_browsers(source: Path, mp4_out: Path, webm_out: Path) -> tuple[bool, bool]:
    """Re-encode to the two codecs that between them cover every browser.

    Two separate problems are being solved here.

    OpenCV's writer cannot produce H.264 at all: asking for `avc1` or `H264`
    fails to open the writer, leaving `mp4v` (MPEG-4 Part 2) as the only option.
    That decodes fine in OpenCV and in desktop players, and in no browser, so
    the sandbox's video pane sits black beside a working map.

    H.264 alone is not enough either. It is patent-encumbered, so open-source
    Chromium builds ship without it, and on those `canPlayType` for avc1 returns
    empty while VP9 reports "probably". Emitting both and letting the browser
    pick via multiple `<source>` elements is the only combination that plays
    everywhere, and it is why the video element in the sandbox lists two files.
    """
    ffmpeg = _ffmpeg_binary()
    if ffmpeg is None:
        return False, False

    mp4_ok = _run_ffmpeg(
        ffmpeg,
        [
            "-i", str(source),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",  # required for broad browser decoding
            "-preset", "medium",
            "-crf", "23",
            "-movflags", "+faststart",  # playback can start before full download
            str(mp4_out),
        ],
        mp4_out,
    )

    webm_ok = _run_ffmpeg(
        ffmpeg,
        [
            "-i", str(source),
            "-c:v", "libvpx-vp9",
            "-pix_fmt", "yuv420p",
            # VP9 defaults are far too slow for a build step. `realtime` with a
            # high cpu-used trades a little compression for an encode that
            # finishes in seconds rather than minutes; this is a test fixture,
            # not a distribution master.
            "-deadline", "realtime",
            "-cpu-used", "5",
            "-b:v", "2M",
            "-row-mt", "1",
            str(webm_out),
        ],
        webm_out,
    )

    return mp4_ok, webm_ok


def render(
    out_dir: Path,
    sprite_dir: Path,
    scenario: Scenario | None = None,
    width: int = 1280,
    height: int = 720,
    seed: int = 3,
) -> dict:
    """Render the clip, and write the video plus its ground truth.

    Returns the ground-truth dictionary, which carries per-frame pitch
    positions, the true homography for every frame, and the drawn bounding
    boxes.
    """
    scenario = scenario or Scenario()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sprites = load_sprites(Path(sprite_dir))
    rng = np.random.default_rng(seed)

    # Give each shirt number a fixed sprite and a fixed tint, so a player looks
    # like the same person for the whole clip. Re-tinting per frame would hand
    # the tracker an appearance cue no real footage provides.
    tinted: dict[int, Sprite] = {}
    for spec in scenario.players:
        base = sprites[rng.integers(0, len(sprites))]
        palette = KEEPER_COLOURS_BGR if spec.is_keeper else TEAM_COLOURS_BGR
        tinted[spec.number] = tint_jersey(base, palette[spec.team])

    frames = scenario.frames()
    cams = broadcast_pan(width, height, len(frames))

    video_path = out_dir / "broadcast.mp4"
    # Written to a scratch file first, then transcoded to H.264 in place of the
    # real output. See `_transcode_to_h264` for why the intermediate exists.
    raw_path = out_dir / "broadcast.mp4v.mp4"
    writer = cv2.VideoWriter(
        str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), scenario.fps, (width, height)
    )
    if not writer.isOpened():  # pragma: no cover - codec availability
        raise RuntimeError(f"could not open a video writer for {raw_path}")

    truth = scenario.ground_truth()
    truth["camera"] = {"width": width, "height": height}

    specs = {p.number: p for p in scenario.players}

    for sf, cam in zip(frames, cams):
        frame = _grass(cam, rng)
        _draw_lines(frame, cam)

        # Painter's algorithm: farthest first, so near players occlude far ones
        # and the tracker has to survive real mutual occlusion.
        order = sorted(
            sf.positions.items(),
            key=lambda kv: cam.project_world(np.array([[kv[1][0], kv[1][1], 0.0]]))[1][0],
            reverse=True,
        )

        boxes: dict[str, list[float]] = {}
        for number, (px, py) in order:
            pts, depth = cam.project_world(np.array([[px, py, 0.0]], dtype=np.float64))
            if depth[0] <= 0:
                continue
            feet_x, feet_y = pts[0]
            hpx = cam.player_pixel_height(px, py, PLAYER_HEIGHT_M)
            if hpx <= 0:
                continue
            box = _composite(frame, tinted[number], feet_x, feet_y, hpx)
            if box is not None:
                boxes[str(number)] = [round(v, 2) for v in box]

        # Blur before the ball goes down, not after. The ball is only a few
        # pixels across, so blurring it averages it straight into the grass: at
        # this radius its centre pixel comes out green and no colour-keyed
        # detector can recover it. Compositing it after the blur mirrors how it
        # actually appears in a broadcast frame, where the camera is focused on
        # the ball and it stays crisp against a softer background.
        frame = cv2.GaussianBlur(frame, (3, 3), 0.6)
        ball_box = _draw_ball(frame, cam, sf.ball[0], sf.ball[1], sf.ball_height)

        grain = rng.integers(-5, 6, size=frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + grain, 0, 255).astype(np.uint8)

        writer.write(frame)

        record = truth["frames"][sf.frame]
        record["homography"] = [round(v, 8) for v in cam.H.ravel().tolist()]
        record["boxes"] = boxes
        record["ballBox"] = [round(v, 2) for v in ball_box] if ball_box else None
        record["teams"] = {str(n): specs[n].team for n in sf.positions}

    writer.release()

    mp4_ok, webm_ok = _transcode_for_browsers(
        raw_path, video_path, out_dir / "broadcast.webm"
    )
    if mp4_ok:
        raw_path.unlink(missing_ok=True)
    else:
        # No ffmpeg available. Keep the MPEG-4 Part 2 file so the pipeline still
        # has something to read, and say plainly that the browser pane will be
        # blank, rather than leaving that to be discovered in the UI.
        raw_path.replace(video_path)
        print(
            "  warning: no ffmpeg found, wrote MPEG-4 Part 2 instead of H.264. "
            "The pipeline will read this fine but browsers cannot play it, so "
            "the sandbox video pane will stay black. Install ffmpeg, or "
            "`pip install imageio-ffmpeg`, and re-render."
        )
    if not webm_ok:
        print(
            "  note: no VP9 WebM was produced. Browsers built without the "
            "patent-encumbered H.264 decoder, which includes most open-source "
            "Chromium builds, will not play the clip."
        )

    truth_path = out_dir / "ground_truth.json"
    truth_path.write_text(json.dumps(truth, indent=2))

    return truth
