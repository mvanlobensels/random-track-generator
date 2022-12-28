from track_generator import TrackGenerator, SimType
from utils import Mode

# Input parameters
n_points = 60
n_regions = 20
min_bound = 0.
max_bound = 150.
mode = Mode.EXTEND

# Output options
plot_track = True
visualise_voronoi = True
create_output_file = True
output_location = '/'

# Generate track
track_gen = TrackGenerator(n_points, n_regions, min_bound, max_bound, mode, plot_track, visualise_voronoi, create_output_file, output_location, lat_offset=51.197682, lon_offset=5.323411, simtype=SimType.GPX)
track_gen.create_track()
