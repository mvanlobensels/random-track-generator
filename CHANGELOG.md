# Changelog

## [1.0.0] - 2026-02-28

### Breaking Changes

- `TrackGenerator` class removed — replaced by `generate_track()` function
- `Mode` and `SimType` no longer need to be imported separately; `mode` now also accepts plain strings (e.g. `"extend"`)

### Added

- `generate_track(n_points, n_regions, min_bound, max_bound, mode, seed)` — main entry point for track generation
- `load_track(name)` — load a preset track by name (e.g. `"FSG"`, `"FSI"`)
- `Track` dataclass with `as_tuple()` and `save(location, sim_type)` methods
- `seed` parameter in `generate_track()` for reproducible generation

### Migration Guide

```python
# Before
from track_generator import TrackGenerator
from utils import Mode, SimType

track_gen = TrackGenerator(
    n_points=60, n_regions=20, min_bound=0., max_bound=150.,
    mode=Mode.EXTEND, plot_track=True, create_output_file=True,
    output_location='/', sim_type=SimType.FSSIM
)
cones_left, cones_right = track_gen.generate()
```
```python
# After
from random_track_generator import generate_track

track = generate_track(n_points=60, n_regions=20, min_bound=0., max_bound=150., mode="extend")
cones_left, cones_right = track.as_tuple()
track.save("output/", sim_type="fssim")
```
