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
it to the engine with tests, then expose it through `reportFacts()`. The deterministic
fallback in the same route must stay genuinely useful, not a stub: it is what runs when
no `ANTHROPIC_API_KEY` is set, which is the default.

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

## Conventions

- **No em dashes, en dashes, or `" - "` as sentence punctuation** anywhere a person reads
  it: prose, comments, UI copy, commit messages, this file. Rewrite the sentence, or use a
  comma, "and", or two sentences. The system prompt in `route.ts` says this explicitly to
  the model too; keep saying it if you add another prompt.
- **Everything downstream of the homography speaks pitch metres**, origin at the centre
  spot, +x toward the right-hand goal. Convert to screen space only at the moment of
  drawing.
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
