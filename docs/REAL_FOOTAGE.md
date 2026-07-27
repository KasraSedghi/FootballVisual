# Running on real footage

Everything measured in this repo is measured on a synthetic clip. That clip is a genuine
test of detection, tracking and calibration, because it composites real photographs of
people onto a perspective-projected pitch with a known camera. It is not a test of whether
any of this survives real broadcast: shadows, crowd, motion blur, compression artefacts,
score bugs, and camera cuts are all absent or simplified.

This document says exactly what a useful real clip looks like, and what to expect.

## What to supply

A single file is enough to start.

| Property | What works | Why it matters |
|---|---|---|
| Length | 10 to 60 seconds | 30 seconds at 25fps is 750 frames, about 7 minutes of CPU |
| Shot | **Main tactical/wide camera**, one continuous shot if possible | Close-ups and replays contain no pitch lines to calibrate from |
| Visible markings | At least two lines from each direction: a touchline plus the halfway line or a penalty-area edge | Calibration solves from two lines per family. Fewer and it declines |
| Player size | Players **at least 40 pixels tall** in the frame | This is the real requirement; resolution only matters through it. Two 360p clips gave 17 pixel players and detection recovered a fifth of them |
| Resolution | 720p or better, given the above | The ball is a handful of pixels at 720p and fewer below it |
| Framing | Stable, no heavy zoom | Zoom changes the homography faster than flow propagation follows |
| Format | Anything ffmpeg reads | Transcoded internally as needed |

**Ideal:** a wide tactical-camera clip of a build-up phase, the kind used for analysis
rather than for broadcast, since those hold one continuous wide shot.

**Also useful, for different reasons:** a broadcast clip *with* cuts and replays. That
exercises cut detection and re-calibration, which currently have only synthetic tests.

## What to expect

Honest predictions, so a poor result is diagnosable rather than surprising. These were
written *before* any real clip was run; the next section records how they held up, and one
of them was flatly wrong.

- **Detection should transfer well.** YOLO is already looking at photographic pixels of
  real people. Real footage is likely *easier* than the synthetic clip in the near field
  and harder in the far field.
- **Team clustering is the most likely thing to break.** It assumes two visually distinct
  kit colours. Two dark kits, or a kit close to the grass tone, will collapse the two
  clusters into one.
- **Calibration will need `--camera-side` set correctly.** This is not a preference, it is
  the bit of information that resolves the pitch's mirror symmetry, and getting it wrong
  produces a fit that looks perfect and puts every player on the wrong side.
- **The 180 degree ambiguity will need settling once.** Without a prior, which end is which
  is a coin flip. Check the first frame's map against the video, and if the ends are
  swapped, that is the ambiguity rather than a bug.
- **Cuts to a close-up will fail to re-calibrate** and the pipeline will hold the previous
  homography, reporting that it did. That is the intended behaviour; frames inside a
  close-up have no pitch geometry to recover.

## What actually happened

Two real Premier League clips (Manchester City vs Tottenham, 640x360, 30fps, wide
broadcast camera, floodlit, score bug and advertising hoardings present) were run through
the pipeline. Neither is in the repository, per the licensing note at the bottom. There is
no ground truth for either, so nothing below is an accuracy figure. Everything here is
either a count, or a claim checked by looking at an overlay.

The predictions above were partly right and mostly not. Recording both.

### Calibration does not work on these clips

This is the headline, and it is a failure rather than a caveat. Neither clip produces a
usable homography. On the tighter of the two the automatic fit scores 0.80px with a 0.96
inlier fraction, which by the confidence rule *at the time* was a confident fit. Drawing
that homography's pitch back onto the frame shows it collapsed into a corner, locked onto
the text on the advertising hoardings, with the pitch model projecting off into the crowd.
The other clip fails the same way at 1.42px and 0.92.

That near miss is the most useful thing these clips produced, because it exposed a real
hole in the scoring rather than in the clips. The score only ever asked whether the
reprojected model lands on detected line pixels, never whether the detected lines are
explained by the fitted pitch, and a pitch shrunk onto a dense patch of the mask satisfies
the first completely. `explained_fraction` now asks the second question: the correct fit on
the demo clip explains 81% of its detected lines, and these two explain 9% and 6%. They are
now correctly rejected instead of silently trusted.

Five things were tried and did not fix it, each measured rather than assumed:

| Attempt | Result |
|---|---|
| Adaptive percentile threshold on brightness | Real clips improved, demo clip went from 0.09m to 79m |
| Otsu on the value channel | Fine on the render, takes 77% of a floodlit real frame |
| Tightening the grass colour bounds | Correctly stops the surface leaking into the stands, demo clip goes to 94m out |
| Cutting the mask at the grass horizon | Real clips improve to 21 to 23% explained, still short of 35%, demo clip goes to 94m |
| Upscaling 640x360 to 1280x720 before fitting | Worse, 1% explained. Not a resolution problem |

The top-hat mask that did ship is a genuine improvement, and it is not enough. The masks it
produces on these clips do contain the real markings, visibly so, alongside a large amount
of crowd and hoarding text that the line fitting then prefers. The remaining work is
filtering detected *lines* by whether they are plausibly pitch markings, not filtering mask
pixels, and that is not done.

Note the pattern in that table: every change that helps the real clips hurts the demo clip,
and the demo clip's sensitivity traces to a single structure. Removing the top of the
region, whether by erosion, by a tighter colour rule, or by a horizon cut, costs the far
touchline and takes the fit to the same 93.8m every time.

### Detection transfers, and scales with how much pitch is in shot

Confirmed roughly as predicted, with the caveat that the two clips differ a lot from each
other. Over 150 frames each:

| | Tighter shot | Wider shot |
|---|---|---|
| Confirmed tracks per frame, median | 12 | 3 |
| Distinct track ids | 38 | 7 |
| Ids alive for at least half the clip | 8 | 3 |

Against roughly 20 and 14 players visible.

**Read the tighter clip's number against the commercial baseline before calling it a
shortfall.** A professional broadcast tracking system detects a median of 13 players per
frame and a mean of 51% of the squad, measured over a full match of open SkillCorner data
(see the comparison section in the README, or run `make benchmark`). A median of 12 here
is in that range. The wider clip, at 3, genuinely is not.

So the tighter framing recovers about what the state of the art recovers, and the wider
one recovers a fifth of them. 38 ids for 12 concurrent tracks also
says identity churn is much higher than on the synthetic clip, where 21 players produced 7
switches over 250 frames.

The reason is apparent size, not anything about real pixels. Player boxes are a median of
**17 and 21 pixels tall** (10th percentile 13 on both). The synthetic clip is 1280x720 and
its players are several times that. Note that the wider clip has the *larger* median box
and finds far fewer players, because that median is taken over what was detected: the ones
it misses are the small ones, so missing more of them raises the median. This is what the 720p line in the table
above is really about, and these clips are 360p, so it is worth restating as a size
requirement rather than a resolution one: the pipeline wants players about 40 pixels tall,
and gets a fifth of them at 17.

### Team clustering did not break, which was the wrong prediction

The prediction above was that team clustering would fail first. It did not. Both clips
separate into two clusters with a large Lab distance between the centres (102 and 123) and
median assignment confidence of 0.69 and 0.86. Two bright kits under floodlights turn out
to be an easier clustering problem than expected.

What is off is the balance. The tighter clip assigns 7 tracks to one team and 23 to the
other, which cannot be right for an 11 a side match. That is a symptom of the identity
churn above rather than of the colour model: short spurious tracks all land in whichever
cluster is closest.

### The ball number is not a real number, and this is how that was caught

The naive readout says the ball is located on 150 frames out of 150 on both clips, which
would be better than on the synthetic clip. It is not true, and it is worth writing down
how it came apart, because a 100% figure on real broadcast should be distrusted on sight.

The counter was measuring `BallTracker.update` returning a position, which includes frames
where it is coasting on prediction rather than seeing anything. Separating raw detections
from coasted ones was the first check and did not settle it, since the classical detector
does fire on almost every frame. What settled it was asking whether the thing being tracked
moves like a ball. On the tighter clip the track spans **480 pixels vertically inside a 360
pixel tall frame**, so whatever it is following leaves the image, which no ball in shot
does. On the wider clip it is worse and in the opposite way: **stationary to within a pixel
on 53% of frames**, then a 670 pixel excursion. That is a white blob being sat on, and then
a jump to a different white blob.

So the honest statement is that ball tracking is unverified on real footage and the
evidence available points at it being wrong. It cannot be scored properly without ground
truth. The commercial baseline settles the direction of the error though: a professional
system sees the ball in **78.9%** of frames, so 100% was never a number to aim for. This is unsurprising for a detector built around small, near-circular, bright blobs
on a pitch where the players are 17 pixels tall.

### Cut detection works

Zero false positives on both clips, which contain no cuts. Minimum histogram correlation
0.978 and 0.994 against a 0.55 threshold, so there is a wide margin.

## Running it

```bash
# Automatic calibration from the markings
python -m footballvisual track \
    --video match.mp4 \
    --out data/tracks.json \
    --auto-calibrate --camera-side minus_y

# Faster, if you want a first look before committing CPU time
python -m footballvisual track --video match.mp4 --out data/tracks.json \
    --auto-calibrate --max-frames 100 --tile-rows 1 --tile-cols 1
```

Then copy the clip and `tracks.json` into `web/public/data/` and run `make web`.

`--camera-side` is `minus_y` for a camera on one touchline and `plus_y` for the other.
There is no way to tell which from the file, so if the map is mirrored, use the other one.

If calibration fails outright, check the reported reason: fewer than four detected lines
means the frame has too little pitch in it, and a low confidence score means the fit was
found but not trusted.

## Falling back to clicked landmarks

If automatic calibration cannot get a confident fit, supply the correspondences by hand.
Write a JSON file mapping landmark names from `pipeline/footballvisual/pitch.py` to the
pixel each appears at in the first frame:

```json
{
  "landmarks": {
    "halfway_top":            [612, 214],
    "halfway_bottom":         [588, 701],
    "right_pen_corner_top":   [1105, 268],
    "right_pen_corner_bottom":[1181, 623]
  }
}
```

Four is the minimum and more is better, since the fit is run through RANSAC and can then
reject a mis-click.

```bash
python -m footballvisual track --video match.mp4 --calibration calib.json --out tracks.json
```

## Licensing

Do not commit real broadcast footage to this repository. Keep it in `data/`, which is
gitignored, and share accuracy numbers rather than the clip.
