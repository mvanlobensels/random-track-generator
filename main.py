from track_generator import TrackGenerator
from utils import Mode

mode = Mode.EXTEND
n_points = 100
n_regions = 10

track_gen = TrackGenerator(n_regions, n_points, mode)
track_gen.create_track()