# Assets

## Source photographs

`bus.jpg` and `zidane.jpg` are the standard sample images from the
[Ultralytics repository](https://github.com/ultralytics/ultralytics), used there as
detection test fixtures. They are included because the synthetic renderer needs
photographs of real people: compositing genuine photographic texture onto the pitch is
what makes YOLO's job in the demo a real detection problem rather than a shape-matching
exercise. See the Ultralytics repository for their licence terms.

## Sprites

`sprites/*.png` are RGBA cut-outs of the people in those photographs, produced by running
YOLOv8 segmentation over them and keeping the mask as an alpha channel. They are checked
in so `make demo` does not have to re-download segmentation weights, and they are cheap to
regenerate:

```bash
make sprites
```

The renderer re-tints each sprite's torso to a team colour at render time, so a sprite's
original clothing colour does not matter. It does normalise torso brightness, because
several of these subjects wear dark clothing and a near-black jersey carries no usable
hue for the team clusterer.
