# FootballVisual

Turn broadcast football video into an interactive top-down tactical map, then ask it questions.

The pipeline detects and tracks all the players and the ball, recovers the camera's
pitch-to-image homography, and projects everything onto a flat 2D pitch in metres. The
sandbox plays that back beside the source video. Pause it and the map locks: drag players
to pose a different shape, draw movement arrows, and the tactical analysis recomputes
instantly, including the answer to the question this project was built around, *where is
the open passing lane?*

```
broadcast video ──▶ detection ──▶ tracking ──▶ homography ──▶ pitch coordinates
                    (YOLOv8)      (ByteTrack)   (DLT+RANSAC)         │
                                                                     ▼
                                        interactive sandbox ◀── tracks.json
                                                 │
                                     deterministic tactical engine
                                                 │
                                     Claude narrates the numbers
```

## Quick start

```bash
make demo     # render the clip, run the pipeline, score it against ground truth
make web      # open http://localhost:3000
```

`make demo` takes about three minutes on CPU. No GPU required, no API key required.

## What each layer does

### 1. Detection (`pipeline/footballvisual/detect.py`)

YOLOv8 for players, run tiled: the frame is cut into overlapping crops so a player who
is 30 pixels tall in the full frame is effectively 50 by the time the network sees them.
On this project's clip that lifts recall from 0.86 to 0.90 at the same threshold.

The ball gets its own detector. A generic object detector finds essentially zero
footballs at broadcast framing, because the ball is a handful of pixels, motion-blurred
whenever it matters, and shaped like every other bright blob on the pitch. So it is
found classically instead, by the three properties that only hold together for a ball:
small but not a single pixel, close to circular, and not inside a player's box.

### 2. Tracking (`pipeline/footballvisual/track.py`)

ByteTrack, implemented from scratch: a constant-velocity Kalman filter over
(centre, aspect, height), Hungarian assignment on IoU, and the two-stage association
that gives ByteTrack its name. The low-confidence detections everyone throws away are
mostly real players who happen to be occluded, and recovering them is what keeps
identities stable through the crossings where a tracker would otherwise switch.

### 3. Homography (`pipeline/footballvisual/homography.py`, `calibrate.py`)

A football pitch is flat, so the map from pitch to image is a plane-to-plane projective
transform: an eight degree-of-freedom matrix. That is the whole reason this works
without recovering 3D camera pose. Estimated by normalised DLT inside RANSAC, written in
plain numpy so the maths is inspectable and testable.

Staying calibrated is the harder half. The camera pans, so a homography fitted on frame
zero is wrong by frame thirty. Sparse features are tracked on the grass between
consecutive frames, with players masked out, and the inter-frame homography they imply is
composed onto the running estimate.

Player positions are taken from the **bottom-centre** of each box, not the centre. The
homography maps the ground plane, and a player's centre floats a metre above it, which on
a shallow broadcast angle throws the position several metres up the pitch.

### 4. Team assignment (`pipeline/footballvisual/teams.py`)

Unsupervised, so it works on any fixture rather than only pre-configured ones. Shirt
colour is sampled from the torso (a bounding box is mostly grass), green pixels are
dropped before averaging, colours are compared in Lab, and each track votes over its
whole lifetime rather than per frame.

### 5. Tactical engine (`web/src/lib/tactics/`)

Deterministic, and the reason the analysis can be trusted.

The interesting piece is the **passing lane solver**. The obvious approach is to measure
how close the nearest defender is to the line between passer and receiver, and it is
wrong in both directions. A defender two metres off a 40 metre pass has all the time in
the world to step across it. A defender half a metre off a 6 metre pass cannot get a foot
to it before it arrives.

What actually decides a pass is a race, so that is what is modelled. For every point
along the lane, when does the ball get there, and when could the defender:

```
margin = min over the path of [ t_defender(s) − t_ball(s) ]
```

A positive margin is how many seconds the ball wins by. Defender momentum is included,
because a defender already sprinting the wrong way cannot simply stop. Two tests in
`lanes.test.ts` pin cases where distance and the interception race give *opposite*
answers, and the race is right.

Also computed: convex-hull team shape, block width and depth, the largest gap in the
defensive line, the offside line, who is between the lines, and a pitch-control field
from time-to-arrive (a soft logistic rather than a hard Voronoi, which would draw a crisp
border between two players a tenth of a second apart).

### 6. The analyst (`web/src/app/api/analyse/route.ts`)

Every tactical *fact* is computed before the model is involved. Claude receives the
numbers and turns them into the sentence a coach would say. It never sees raw
coordinates, so it cannot redo the geometry and reach a different answer to the one drawn
on screen, and it is never asked which lane is open.

That constraint is what makes the analysis reproducible, and it is why the deterministic
fallback is not a stub: with no `ANTHROPIC_API_KEY` set, the same facts go through a
rule-based writer and the answer is still specific and true, just less fluent. The UI
labels which path produced each answer.

Set `ANTHROPIC_API_KEY` in `web/.env.local` to enable the model.

## Accuracy

The synthetic clip knows where every player really was, so the pipeline can be scored
rather than demoed. `make demo` prints this:

```
position MAE          1.29 m
position p95          2.01 m
detection coverage    72.6%   (21 of 21 players matched by some track)
identity switches     7
team assignment       100.0%
ball coverage         96.4%
ball MAE              3.06 m
```

How to read these:

- **Position MAE 1.29 m** bounds everything above it. A passing-lane margin that turns on
  distances finer than about a metre is noise, which is why the verdict thresholds are set
  in *seconds* rather than centimetres.
- **Coverage 72.6%** counts frames, not players. Every one of the 21 on-screen players is
  followed by a track; the shortfall is frames where a player is missed and their track is
  coasting on prediction.
- **7 identity switches** over 250 frames. Each one corrupts a trajectory from that point
  on, so this is the number to watch when tuning.

Run `make evaluate` to re-score an existing `tracks.json`.

## The clip, and why it is synthetic

The pipeline works on any video. The one in the repo is generated, and that is a
deliberate choice rather than a shortcut.

A licensed broadcast clip cannot ship in an open repo, and an unlabelled clip could not
be *scored* anyway. So `synth.py` composites **real segmented photographs of people**
onto a perspective-projected pitch, driven by a real pinhole camera whose ground-plane
homography is exact by construction. YOLO therefore runs on genuine photographic texture,
while every player's true pitch coordinate stays known.

It is hard in the ways that matter here (perspective, scale with depth, mutual occlusion,
a moving camera, jersey colours to cluster) and easy in ways it does not model (no
shadows, no crowd, no broadcast graphics, no camera cuts). It is a measurable test
harness, not a claim of photorealism.

To run on real footage, supply your own landmark calibration:

```bash
python -m footballvisual track --video match.mp4 --calibration calib.json --out tracks.json
```

where `calib.json` maps landmark names from `pitch.py` to the pixel where each appears.

## Layout

```
pipeline/footballvisual/
  pitch.py        IFAB pitch geometry, the single source of truth
  camera.py       pinhole broadcast camera, and the homography it induces
  homography.py   normalised DLT + RANSAC, projection helpers
  calibrate.py    landmark calibration and frame-to-frame propagation
  detect.py       tiled YOLO player detection, classical ball detection
  track.py        ByteTrack, and a coasting single-target ball filter
  teams.py        jersey colour clustering with per-track voting
  scenario.py     ground-truth motion model for the synthetic clip
  synth.py        the renderer
  pipeline.py     orchestration and tracks.json export
  evaluate.py     scoring against ground truth
web/src/
  lib/tactics/    the deterministic tactical engine
  components/     pitch map, metrics, analyst panel
  app/api/analyse route handler: Claude narration + deterministic fallback
```

## Known limits

- **Calibration is seeded from named landmarks**, either a config file or, for the
  synthetic clip, ground truth plus click noise. Fully automatic pitch-line detection is
  not implemented; identifying *which* line is which robustly is its own project.
- **No camera-cut detection.** A hard cut breaks homography propagation, and the demo clip
  is a single continuous shot.
- **The ball is only tracked in 2D.** Height is not recovered, so a lofted pass is
  reported at its ground projection.
- **Two teams only.** Referees and keepers land in an "other" bucket rather than being
  identified as such.
- **Tiled inference costs one forward pass per tile**, roughly 0.5s per frame on CPU. Fine
  for offline analysis, not real time.

## Tests

```bash
make test
```

20 Python tests covering the homography (exact fit, degenerate inputs, RANSAC outlier
rejection, end-to-end calibration accuracy in metres) and the tracker. 17 TypeScript tests
covering the lane solver and scoring.

The tracker suite includes a regression test for a real bug found during development:
lost tracks were excluded from association, so an occluded player could never be
recovered *and* never aged out. Fixing it moved coverage from 43% to 76%.
