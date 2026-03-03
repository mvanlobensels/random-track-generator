# Random Track Generator

![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

<p float="middle">
  <img src="img/extend.png" width="45%" />
  <img src="img/expand.png" width="45%" />
</p>

Generate random, rules-compliant tracks for Formula Student Driverless competitions. Creates realistic tracks with proper track width, cone spacing, and corner radii using Voronoi diagram-based generation.

Developed by [Formula Student Team Delft](https://fsteamdelft.nl) for use with [FSSIM](https://github.com/AMZ-Driverless/fssim) and [FSDS](https://github.com/FS-Driverless/Formula-Student-Driverless-Simulator).

## How It Works

Tracks are generated using bounded Voronoi diagrams from uniformly sampled points. Regions are selected using one of three modes (**Expand**, **Extend**, or **Random**), then interpolated with cubic splines to create smooth, realistic racing lines.

<p float="middle">
  <img src="img/voronoi.png" width="50%" />
</p>

## Installation

```bash
pip install random-track-generator
```

## Usage

### Generate a random track

```python
from random_track_generator import generate_track

# Use preset parameters
track = generate_track("small")
track = generate_track("medium")
track = generate_track("large", seed=42)

# Or set parameters manually
track = generate_track(
    n_points=60,       # Voronoi points
    n_regions=20,      # Regions to select
    min_bound=0.,      # Minimum x/y bound [m]
    max_bound=150.,    # Maximum x/y bound [m]
    mode="extend",     # Generation mode
    seed=42            # Optional: for reproducibility
)

cones_left, cones_right = track.as_tuple()
```

#### Generation Modes

- **`"expand"`** - Selects nearest neighbors for roundish tracks
- **`"extend"`** - Selects regions along a random line for elongated tracks
- **`"random"`** - Randomly selects regions for large, irregular tracks


> **Note:** Not all parameter combinations produce stable results. Experiment with settings if generation fails.

### Load a preset track

```python
from random_track_generator import load_track

track = load_track("FSG")                       # FSG and FSI available
cones_left, cones_right = track.as_tuple()
```

### Save to file

```python
track.save("output/", sim_type="fssim")         # YAML for FSSIM
track.save("output/", sim_type="fsds")          # CSV for FSDS
track.save("output/", sim_type="gpx", 
           lat_offset=51.19, lon_offset=5.32)   # GPX with coordinates
```

## Credits

Based on Ian Hudson's [Race-Track-Generator](https://github.com/I-Hudson/Race-Track-Generator).
