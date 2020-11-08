from track_generator import TrackGenerator
from utils import Mode

# Input parameters
n_points = 100
n_regions = 10
mode = Mode.EXTEND

# Output options
plot_track = True
output_file = False

# Generate track
track_gen = TrackGenerator(n_regions, n_points, mode, plot_track, output_file)
track_gen.create_track()