# CLAUDE.md: FootballVisual

Orientation for any Claude Code session working in this repo. Read this before touching
the pipeline, the tactical engine, or the analyst route.

## What this is

A computer vision pipeline that turns broadcast football video into a top-down tactical
map, plus an interactive sandbox on top of it. See `README.md` for the architecture and
the accuracy numbers. Two runtimes:

- `pipeline/` — Python. Detection, tracking, homography, team clustering, export.
- `web/` — Next.js 16 + TypeScript + Tailwind v4. The sandbox and the tactical engine.

## The invariant this project is built around

**Deterministic geometry decides; the language model only writes.**

Every tactical claim — which lane is open, by how many seconds, how compact a block is,
where the gap is — is computed in `web/src/lib/tactics/`. The Claude call in
`web/src/app/api/analyse/route.ts` receives those computed numbers as facts and turns
them into prose. It never receives raw coordinates, so it cannot redo the geometry and
produce an answer that disagrees with what is drawn on screen.

Do not move a tactical decision into the prompt. If a new tactical concept is needed, add
it to the engine with tests, then expose it through `reportFacts()`.

The same rule governs clip retrieval (`web/src/lib/tactics/search.ts` and
`api/search/route.ts`). There the model's whole output is a structured `TacticalQuery`
against a fixed schema of thresholds, and the search runs in code over already-computed
measurements. The model picks filters; it never decides whether a frame qualifies. Do not
add a free-text field to that schema: anything the model cannot ground in a measured
threshold is something it would be inventing, and a retrieval tool that returns a passage
which does not have the property asked for is worse than no retrieval tool. The deterministic
fallback in the same route must stay genuinely useful, not a stub: it is what runs when
no `ANTHROPIC_API_KEY` is set, which is the default.

## Similarity retrieval has no model in it, and that is the point

`web/src/lib/tactics/embedding.ts` maps a frame to a vector so "moments like this one"
works without anyone naming a threshold. It runs entirely in the browser and never calls
Claude. Do not add a model to it: a shape is a thing you can compare arithmetically, and
the moment a model chooses which frames are similar, the result stops being reproducible.

`canonicalise` is the load-bearing part. It reflects x so the team in possession attacks
toward +x, and y so the ball sits in the +y half, which is what makes the same situation
at the other end or down the other wing match. Reflection is an isometry, so it cannot
distort the shape being measured. If you replace it with a rotation or a rescale, that
stops being true and every embedding measures a warped frame.

`attackingGoalX` on `TacticalReport` exists solely so the embedding takes direction of
play from the engine rather than recomputing it. Keep it that way, or the two can disagree
about which way a team is playing and the canonicalisation silently inverts.

The vector is handcrafted (a coarse occupancy grid per team plus the engine's scalars)
because ten matches of open tracking data cannot train a learned representation. That is a
documented limit, not a TODO to paper over. If a learned encoder ever becomes available,
it replaces `embedFrame` and nothing else.

Both equivariances and the negative case are pinned in `embedding.test.ts`. A change that
makes everything similar passes the first two tests and is worthless, which is what the
compact-versus-stretched test is there to catch.

## The three symmetries of a pitch

Automatic calibration (`autocalibrate.py`) exists now, and the thing to understand before
touching it is that a pitch is symmetric, so its markings under-determine the homography.

* **Mirror about the halfway line.** Explains the image exactly as well as the truth,
  with every player on the wrong side. Fixed by `camera_side`, which is a required
  argument because it cannot be inferred from the image.
* **180 degree rotation.** Also maps markings onto markings and *preserves orientation*,
  so `camera_side` does not help. Reported through `rotation_ambiguous` and resolved only
  by a `prior_h`, never guessed.

The orientation check in `_plausible` encodes the first of these. Image y points down, so
the correct homography yields a **negative** shoelace area for the projected pitch
corners. Getting that sign backwards does not fail loudly, it silently selects the
mirrored pitch, which reprojects onto the real lines perfectly. That bug cost a debugging
session and is pinned by `test_camera_side_is_required_to_resolve_the_mirror`.

Line families must be grouped by **vanishing point, not by angle**. Perspective makes
lines that are parallel on the grass span tens of degrees in the image, with members of
the other family sitting between them, so an angle split fails on exactly the wide views
where calibration matters.

## Calibration is scored in both directions, and that is not optional

`_score_homography` asks whether the reprojected model lands on detected line pixels.
`_explained_fraction` asks whether the detected lines are explained by the fitted pitch.
Only the second notices a homography that has shrunk the pitch onto a dense patch of the
mask, which scores 0.85px at a 0.98 inlier fraction while being 94 metres wrong. Do not
drop the second term, and do not tune its threshold up toward the values a correct fit
achieves: it sits at 0.35 against 0.81 for correct fits precisely so it rejects nonsense
without becoming brittle.

`_explained_fraction` runs once on the winning fit, never per hypothesis. It dilates over
the whole frame, so putting it in the search loop would cost far more than it is worth,
and it is a check on an answer rather than a way to find one.

## Real footage: calibration does not work there yet

Two real Premier League clips were run through the pipeline. Detection, tracking, team
clustering and cut detection all transfer; automatic calibration does not, on either clip.
The fit locks onto advertising hoarding text. Read `docs/REAL_FOOTAGE.md` before trying to
fix it: five approaches were tried and measured, and the recurring trap is that everything
which helps the real clips hurts the demo clip. Anything that removes the top of
`pitch_region`, whether erosion, tighter colour bounds, or a horizon cut, costs the far
touchline and takes the demo clip to the same 93.8m every time.

Filtering detected *lines* rather than mask pixels has since been done. `grass_support`
rejects any line without grass beside it, which is what separates a marking from hoarding
text, and on real frames the markings score 42% to 100% against 30% and below for
everything else. It preserves the demo clip, measurably improves the real ones, and does
**not** make them calibrate.

Do not raise `MIN_GRASS_SUPPORT` above 0.35 without re-checking the demo clip directly. A
touchline has grass on one side and stands on the other and scores 42.5%, so 0.5 looks
obviously safe and silently costs 75 metres.

The remaining work is a learned pitch-keypoint detector trained on
[SoccerNet](https://www.soccer-net.org/tasks/camera-calibration), which annotates 23 lines
and 3 circles per frame, rather than any further classical filtering.

## Non-obvious decisions, and why

Changing any of these without understanding the reason will regress something measurable.

- **Player position comes from the bottom-centre of the box, never the centre.** The
  homography maps the ground plane. A player's centre floats about a metre above it, and
  on a shallow broadcast angle that error projects to several metres up the pitch.
- **`ByteTracker.is_active` includes `LOST`.** Lost tracks must remain eligible for
  matching, or an occluded player can never be recovered and never ages out. This was a
  real bug; `test_lost_track_is_recovered_after_a_gap` pins it. Fixing it moved coverage
  from 43% to 76%.
- **`high_threshold` is 0.25, not the conventional 0.45.** Only high-score detections can
  start a track. Sweeping against ground truth showed detector precision holding at 0.98
  down to 0.25, while 0.45 discarded roughly a third of visible players. Distant players
  simply do not detect confidently.
- **The ball is drawn after the blur in `synth.py`, with a 3px floor.** At its true
  projected size, blurring averages the ball into the grass and its centre pixel comes out
  green, so no detector could ever find it.
- **`tint_jersey` renormalises brightness.** Several source photographs show people in
  dark clothing; preserving their value channel left "jerseys" near black, where hue
  carries no information. This single fix moved team assignment from 67% to 100%.
- **The clip is encoded twice, H.264 and VP9.** OpenCV cannot write H.264 at all, and
  H.264 alone is not enough because open-source Chromium builds ship without the decoder.
  The `<video>` element lists both sources.
- **Team assignment is resolved once at the end, not per frame.** Per-frame clustering
  lets a player change teams whenever they turn, and every shape metric recomputes around
  a different set of players.
- **Space control uses a soft logistic on time-to-arrive, not a Voronoi.** A hard nearest
  player rule draws a confident border between two players a tenth of a second apart.
- **Cuts are detected on colour histograms, not pixel differences.** A hard pan moves
  every pixel and reads as a cut to a pixel test, while barely moving the histogram.
  Propagating a homography across a cut fails silently: flow still matches, RANSAC still
  fits, and every player lands somewhere confidently wrong.
- **Keepers and referees are separated by position, not colour.** Both wear kit unlike
  either team, which is exactly why colour puts them in the same bucket. A keeper's x is
  extreme and its range small; a referee roams. This runs after projection because it
  needs pitch coordinates.
- **Keepers are excluded from both squads in the tactical engine.** One keeper in the
  defending set stretches measured block depth by the 40m they stand behind the line, and
  puts a goalkeeper in the list of passing options.
- **Automatic calibration is throttled** (`calibration_retry_frames`). Each attempt costs
  seconds, so retrying every frame on a clip it cannot solve turns a failure into an
  apparent hang.

## Conventions

- **No em dashes, en dashes, or `" - "` as sentence punctuation** anywhere a person reads
  it: prose, comments, UI copy, commit messages, this file. Rewrite the sentence, or use a
  comma, "and", or two sentences. The system prompt in `route.ts` says this explicitly to
  the model too; keep saying it if you add another prompt.
- **Everything downstream of the homography speaks pitch metres**, origin at the centre
  spot, +x toward the right-hand goal. Convert to screen space only at the moment of
  drawing.
- **Semantic colour comes from the tokens in `globals.css`, and a UI change updates them
  in the same commit.** In this app colour is an encoding, not decoration: green is the
  interception race's verdict, blue and red are the teams, amber is the offside line. That
  encoding has to appear in three languages at once, Tailwind utilities in the panels,
  `fill`/`stroke` on the SVG pitch, and raw RGB channels in the heatmap's `ImageData`, so
  a literal written into any one of them is a fourth source of truth waiting to drift.
  It already had: an open lane was drawn green-500 on the map and listed emerald-400 in
  the panel that is supposed to be the map's audit trail. Use `text-verdict-open` in a
  panel, `var(--color-verdict-open)` in SVG, and `readToken()` for canvas. Neutral chrome
  (slate) and app-level roles (emerald for the primary action, red for errors) stay on
  Tailwind's own scale, because those are not tactical claims and aliasing them would add
  indirection without preventing a bug.
- **`pitch.py` is the single source of truth for pitch geometry.** `web/src/lib/pitch.ts`
  mirrors those constants because the two runtimes cannot share a module. If you change
  one, change both.
- **Report accuracy in metres on the pitch, not pixels.** Two pixels near the far
  touchline is several metres; two pixels in the foreground is centimetres.
- Python: type hints, dataclasses for structured returns, numpy for maths. Comments
  explain *why*, not what.
- TypeScript: functional components, no default exports from `lib/`, engine code stays
  pure and free of React.

## Next.js 16 specifics

Read `web/AGENTS.md`. This is Next 16: Turbopack is the default for `dev` and `build`,
`middleware` is now `proxy`, and `experimental.turbopack` moved to a top-level
`turbopack` key. Check `node_modules/next/dist/docs/` before assuming an API from memory.

## Claude API specifics

The analyst route uses `claude-opus-5` with `output_config.format` for structured output.
Note that `output_format` is the deprecated spelling, `temperature`/`top_p`/`top_k` are
rejected on this model, and thinking is on by default. Load the `claude-api` skill before
editing that file rather than working from memory.

## Testing

```bash
make test        # both suites
make test-py     # pytest, in pipeline/
make test-web    # vitest, in web/
```

The Python tests need no network and no model weights. The TypeScript tests cover the
tactical engine only, which is where the correctness risk is.

**A green `test_autocalibrate.py` is weaker evidence than it looks.** Every test in it once
calibrated `render_lines_only`, flat grass and clean lines, and a `line_mask` change that
put the demo clip 79 metres out left all eight passing. There is now a
`render_full_frame` test with mow stripes, blur and grain, and that one still did not catch
it. If you change anything in `line_mask` or `pitch_region`, calibrate `data/broadcast.mp4`
frame 0 against `data/ground_truth.json` directly and check the error in metres. The suite
alone will not tell you.

When you change anything in the pipeline, re-run `make demo` and compare the printed
accuracy block against the numbers in `README.md`. Those numbers are a claim the repo
makes, so if they move, update the README in the same commit.

## Regenerating the demo

`data/` and `web/public/data/` are generated and gitignored. `make demo` rebuilds them:
sprites are cut from `assets/*.jpg` with YOLO segmentation, the clip is rendered from
`scenario.py`, then the pipeline runs and scores itself.

The scenario is deliberately a specific tactical situation, a build-up against a 4-4-2 mid
block ending in a pass into the space between the lines, so the tactical engine has a real
answer to find rather than noise to describe. Keep it that way if you edit `scenario.py`.

## Git workflow

Commit each logical change separately with a descriptive message. Work on a feature
branch. Verify `make test`, `npm run lint`, and `npm run build` before pushing.
