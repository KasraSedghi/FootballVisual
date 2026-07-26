# FootballVisual — end to end demo build.
#
# `make demo` goes from an empty checkout to a running sandbox: extract player
# sprites, render the synthetic broadcast clip, run the vision pipeline over it,
# and score the result against ground truth.

VENV    := .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
DATA    := data
SPRITES := assets/sprites
export YOLO_CONFIG_DIR := /tmp/Ultralytics

.PHONY: help venv sprites render track track-manual evaluate demo test test-py test-web web clean

help:
	@echo "make demo      full pipeline: sprites, render, track, evaluate"
	@echo "make web       run the sandbox at http://localhost:3000"
	@echo "make test      python and typescript test suites"
	@echo "make clean     remove generated data"

venv:
	test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -e pipeline[dev]

sprites: venv
	$(PY) -m footballvisual sprites assets/*.jpg --out $(SPRITES)

render: venv
	$(PY) -m footballvisual render --out $(DATA) --sprites $(SPRITES)

# Calibrates from the pitch markings, so nothing has to be clicked first.
# --camera-side resolves the pitch's mirror symmetry, which cannot be inferred
# from the image; the renderer puts the camera on the negative-y touchline.
track: venv
	$(PY) -m footballvisual track \
		--video $(DATA)/broadcast.mp4 \
		--out $(DATA)/tracks.json \
		--ground-truth $(DATA)/ground_truth.json \
		--auto-calibrate --camera-side minus_y

# The original path, for comparison: a homography seeded from clicked pitch
# landmarks rather than found automatically.
track-manual: venv
	$(PY) -m footballvisual track \
		--video $(DATA)/broadcast.mp4 \
		--out $(DATA)/tracks.json \
		--ground-truth $(DATA)/ground_truth.json

evaluate: venv
	$(PY) -m footballvisual evaluate \
		--tracks $(DATA)/tracks.json \
		--ground-truth $(DATA)/ground_truth.json

# The sandbox reads from web/public, so the freshly generated clip and tracks
# are copied there as the last step of the demo build.
demo: render track
	mkdir -p web/public/data
	cp $(DATA)/tracks.json $(DATA)/broadcast.mp4 web/public/data/
	test -f $(DATA)/broadcast.webm && cp $(DATA)/broadcast.webm web/public/data/ || true
	@echo
	@echo "Done. Run 'make web' and open http://localhost:3000"

test: test-py test-web

test-py: venv
	cd pipeline && PYTHONPATH=. ../$(PY) -m pytest tests/ -q

test-web:
	cd web && npm test

web:
	cd web && npm run dev

clean:
	rm -rf $(DATA)/*.mp4 $(DATA)/*.webm $(DATA)/*.json web/public/data
