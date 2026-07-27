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
position MAE          0.78 m
position p95          1.76 m
detection coverage    74.6%   (21 of 21 players matched by some track)
identity switches     8
team assignment       100.0%
ball coverage         100.0%
ball MAE              1.85 m
```

How to read these:

- **Position MAE 0.78 m** bounds everything above it. A passing-lane margin that turns on
  distances finer than about a metre is noise, which is why the verdict thresholds are set
  in *seconds* rather than centimetres.
- **Coverage 74.6%** counts frames, not players. Every one of the 21 on-screen players is
  followed by a track; the shortfall is frames where a player is missed and their track is
  coasting on prediction.
- **8 identity switches** over 250 frames. Each one corrupts a trajectory from that point
  on, so this is the number to watch when tuning.
- **Ball MAE 1.85 m** is still the weakest number here, and it took three attempts to
  find out why, which is worth recording because two of them were wrong.

  Smoothing was not the cause: widening the window from 1 to 21 frames moved the error by
  0.01 m, so it was not random jitter. Ball height was not the cause either: in flight
  versus on the ground was 3.26 m against 2.96 m. What finally showed it was looking at
  the distribution instead of the mean. It was bimodal, a 1.8 m median and a 5.1 m 90th
  percentile, then a jump to 15.6 m at the 95th. About one frame in ten had locked onto
  the wrong white blob entirely.

  The cause was the motion gate, which allowed the ball to jump 220 pixels between
  frames when a driven pass moves it about 0.64 m, a few tens of pixels at this scale.
  Tightening it and scaling it with time-since-last-seen took ball error from 3.11 m down
  toward the current 1.85 m, with coverage at 100%.

Calibrating from the markings rather than from clicked landmarks is what moved position
MAE from 1.29 m down toward 0.78 m. That is not surprising in hindsight: the landmark path
simulates a human clicking with two pixels of error, and a line fit over hundreds of pixels
of evidence beats that. Run `make track-manual` to reproduce the clicked-landmark numbers.

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

To run on real footage, see **[docs/REAL_FOOTAGE.md](docs/REAL_FOOTAGE.md)**. It says what
a useful clip looks like, what is likely to break first (team clustering, if the two kits
are not visually distinct), and how to fall back to clicked landmarks if automatic
calibration cannot get a confident fit.

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

## Automatic calibration

Calibration can run from the pitch markings alone, with nobody clicking anything:

```bash
python -m footballvisual track --video match.mp4 --auto-calibrate --camera-side minus_y \
    --left-goal-side left
```

The hard part is not finding the lines, it is deciding *which* line each one is. A wrong
assignment still produces a perfectly self-consistent homography that puts every player in
the wrong half. So the approach is hypothesis and verify: group the detected lines into
their two vanishing-point families, enumerate assignments to model lines, solve from four
intersections, and score each candidate by reprojecting the *whole* pitch model against a
distance transform of the detected lines. A wrong assignment explains the four lines it
was fitted to and then puts the centre circle nowhere, so it scores badly.

Measured against the renderer's known homography, this recovers the camera to **0.09m**
mean pitch error in about 9 seconds per frame on CPU. On real broadcast footage it does not
work at all yet, which is documented in [docs/REAL_FOOTAGE.md](docs/REAL_FOOTAGE.md).

Scoring runs in both directions, and the second one exists because the first was not
enough. Reprojecting the model and asking whether it lands on detected line pixels says
nothing about the detected lines the fit ignored, so a homography that shrinks the pitch
onto a dense patch of the mask scores 0.85px at a 0.98 inlier fraction while sitting 94
metres from the truth. `explained_fraction` asks the converse, and separates those cleanly:
81% for a correct fit against 5% for that one.

### The symmetry problem, which is not solvable from geometry

A pitch is symmetric, and no amount of line finding escapes that:

- **Mirrored about the halfway line.** Explains the markings exactly as well as the truth
  while putting every player on the wrong side. Resolved by `--camera-side`, since knowing
  which touchline the camera is behind fixes the orientation. This is a required argument
  rather than an inferred one because it *cannot* be inferred.
- **Rotated 180 degrees.** Also maps every marking onto a marking, and unlike the mirror it
  preserves orientation, so the camera side does not settle it. The two fits score
  identically by construction.

The rotation is reported via `rotation_ambiguous` rather than guessed at. Passing a
`prior_h` resolves it, which is what makes re-calibration after a camera cut usable: settle
it once, then carry it across cuts. With a prior, every frame of the demo clip calibrates
to within 0.33m.

There is no prior on the very first calibration a video ever gets, though, and on the demo
clip the markings visible in frame are rotation-ambiguous throughout, so without more
information that first fit is confidently wrong end-for-end about half the time: score and
inlier fraction both pass, `is_confident` is true, and every player lands roughly 53m from
where they actually are. This is not hypothetical; it is what `make demo` produced before
`--left-goal-side` existed. That flag says which screen side shows the goal at pitch
x = `-HALF_LENGTH`, and is consulted only when there is no prior yet, exactly the
information an operator would confirm once at kickoff. See
`test_left_goal_side_resolves_the_rotation_ambiguity_with_no_prior` in
`test_autocalibrate.py`.

Getting the orientation sign backwards is a genuinely nasty bug, because it fails
silently: the mirrored homography reprojects onto the real markings perfectly. It cost a
debugging session here and is pinned by `test_camera_side_is_required_to_resolve_the_mirror`.

## Camera cuts

A cut invalidates the running homography completely, and it does so quietly: optical flow
between two unrelated shots still returns matches, RANSAC still fits a homography to them,
and that transform gets composed onto the running estimate. Every player is then projected
somewhere confidently wrong with nothing in the output indicating a problem.

Cuts are detected by the correlation between consecutive frames' colour histograms.
Histograms, not pixel differences, precisely because they ignore *where* things are: a hard
pan moves every pixel and would read as a cut to a pixel-difference test, while barely
moving the histogram. On the demo clip, which contains no cuts, the minimum correlation
across a full pan is 0.93 against a 0.55 threshold.

On a detected cut the pipeline re-calibrates from the markings, seeded with the last good
homography to settle the rotation ambiguity, rather than propagating across the cut.

## Known limits

- **The 180 degree rotation ambiguity needs external information.** Line markings do not
  encode which end is which. After the first calibration the pipeline resolves it from a
  prior; before that, `--left-goal-side` supplies the equivalent of an operator confirming
  it once at kickoff, or the direction of play in a real deployment.
- **The ball is only tracked in 2D.** Height is not recovered, so a lofted pass is
  reported at its ground projection. On the two real clips ball tracking appears to fail
  outright: it reports a position on every frame, and the thing it is following is
  stationary for half of them on one clip and travels outside the frame on the other.
- **Goalkeeper and referee separation is positional**, so it needs enough of a trajectory
  to judge. Tracks with under 8 observations are left `unknown` rather than guessed, since
  mislabelling a defender as a keeper would move the offside line.
- **Not real time.** See the throughput table below. Fine for offline analysis.
- **Automatic calibration does not work on real broadcast yet.** Two real Premier League
  clips were run through it and neither produces a usable homography: the fit locks onto
  advertising hoarding text and collapses the pitch into a corner. It is now correctly
  reported as low confidence rather than accepted, which it was not before those clips
  were tried. Detection, tracking, team clustering and cut detection all transfer;
  calibration is the layer that does not. See
  **[docs/REAL_FOOTAGE.md](docs/REAL_FOOTAGE.md)** for what was measured and the five
  approaches that were tried and rejected.
- **Every accuracy number here is measured on the synthetic clip**, which has no shadows,
  no crowd, no broadcast graphics and no cuts. Treat them as characterising the pipeline
  on a controlled input, not as a claim about real matches. Nothing in the real-clip work
  above produced an accuracy figure, because neither clip has ground truth.

## Throughput

Measured on this container's CPU (4 threads, no GPU), 1280x720 input, averaged over
8 frames after a warm-up:

| Stage | Cost | Note |
|---|---|---|
| Player detection, tiled 2x2 @1280 | 522 ms/frame | the default |
| Player detection, full frame @1280 | 149 ms/frame | 3.5x faster, recall 0.90 to 0.86 |
| Player detection, tiled 2x2 @960 | 293 ms/frame | middle option |
| Ball detection (classical) | 2.5 ms/frame | negligible |
| Homography flow step | 15.2 ms/frame | negligible |
| Automatic calibration | ~9 s/attempt | frame zero, and after each cut |

Detection dominates completely: everything else together is under 4% of a frame's cost.
So the only lever that matters is how much detector you buy, and tiling is the knob:

```bash
# Fastest, at a few points of recall on distant players
python -m footballvisual track --video match.mp4 --tile-rows 1 --tile-cols 1

# Middle ground
python -m footballvisual track --video match.mp4 --imgsz 960
```

Tiling costs 3.5x for about 4 percentage points of recall. That is worth it here because
a missed distant player is a missing member of the defensive block, and block shape is
what the tactical layer is measuring. On a workload that only cared about the ball and the
players near it, it would not be.

The honest summary is that a real-time version of this is a different engineering problem,
not a tuning exercise: it needs a GPU, batched tiles in one forward pass, and a smaller
model, and none of those are things this project has measured.

## Tests

```bash
make test
```

40 Python tests covering the homography (exact fit, degenerate inputs, RANSAC outlier
rejection, end-to-end calibration accuracy in metres), automatic calibration, and the
tracker. 26 TypeScript tests covering the lane solver and scoring.

Two of these pin real bugs found during development, which is most of the reason to have
them:

- Lost tracks were excluded from association, so an occluded player could never be
  recovered *and* never aged out. Fixing it moved coverage from 43% to 76%.
- Calibration scoring was one-sided, and accepted a fit 94 metres from the truth as
  confident.

The calibration suite also carries a note about its own limits. Every test in it used to
calibrate a frame of flat grass and clean lines, and a change that put the demo clip 79
metres out left all of them green. There is now a test against a fully rendered frame, and
its docstring says plainly that this was still not what caught the bug.
