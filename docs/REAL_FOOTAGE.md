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
| Resolution | 720p or better | The ball is a handful of pixels at 720p and fewer below it |
| Framing | Stable, no heavy zoom | Zoom changes the homography faster than flow propagation follows |
| Format | Anything ffmpeg reads | Transcoded internally as needed |

**Ideal:** a wide tactical-camera clip of a build-up phase, the kind used for analysis
rather than for broadcast, since those hold one continuous wide shot.

**Also useful, for different reasons:** a broadcast clip *with* cuts and replays. That
exercises cut detection and re-calibration, which currently have only synthetic tests.

## What to expect

Honest predictions, so a poor result is diagnosable rather than surprising:

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
