import yaml
import numpy as np
from pathlib import Path
from scipy import signal, spatial, interpolate
from shapely.geometry.polygon import Point, LineString, Polygon

from .geometry import closest_node, clockwise_sort, curvature, arc_length, transformation_matrix
from .track import Track, Mode

# Track parameters
TRACK_WIDTH = 3.                   # [m]
CONE_SPACING = 5.                  # [m]
LENGTH_START_AREA = 6.             # [m]
CURVATURE_THRESHOLD = 1. / 3.75    # [m^-1]
STRAIGHT_THRESHOLD = 1. / 100.     # [m^-1]

def _bounded_voronoi(input_points, bounding_box):
    """
    Creates a Voronoi diagram bounded by the bounding box.
    Mirror input points at edges of the bounding box.
    Then create Voronoi diagram using all five sets of points.
    This prevents having a Voronoi diagram with edges going off to infinity.
    
    Args:
        input_points (numpy.ndarray): Coordinates of input points for Voronoi diagram.
        bounding_box (numpy.ndarray): Specifies the boundaries of the Voronoi diagram, [x_min, x_max, y_min, y_max].
    
    Returns:
        scipy.spatial.qhull.Voronoi: Voronoi diagram object.
    """
    
    def _mirror(boundary, axis):
        mirrored = np.copy(points_center)
        mirrored[:, axis] = 2 * boundary - mirrored[:, axis]
        return mirrored
    
    x_min, x_max, y_min, y_max = bounding_box
    
    # Mirror points around each boundary
    points_center = input_points
    points_left = _mirror(x_min, axis=0) 
    points_right = _mirror(x_max, axis=0) 
    points_down = _mirror(y_min, axis=1)
    points_up = _mirror(y_max, axis=1)
    points = np.concatenate([points_center, points_left, points_right, points_down, points_up])
    
    # Compute Voronoi
    vor = spatial.Voronoi(points)
    
    # We only need the section of the Voronoi diagram that is inside the bounding box
    vor.filtered_points = points_center
    vor.filtered_regions = np.array(vor.regions, dtype=object)[vor.point_region[:vor.npoints//5]]
    return vor

def _create_track(n_points: int, 
                  n_regions: int, 
                  min_bound: float, 
                  max_bound: float, 
                  mode: Mode | str = Mode.EXPAND,
                  seed: int | None = None) -> Track:
    """
    Creates a track from the vertices of a Voronoi diagram.
    1.  Create bounded Voronoi diagram.
    2.  Select regions of Voronoi diagram based on selection mode.
    3.  Get the vertices belonging to the regions and sort them clockwise.
    4.  Interpolate between vertices.
    5.  Calculate curvature of track to check wether the curvature threshold is exceeded.
    6.  If curvature threshold is exceeded, remove vertice where the curvature is the highest from its set.
        Repeat steps 4-6 until curvature is within limimts.
    7.  Check if track does not cross itself. If so, go to step 2 and reiterate.
    8.  Find long enough straight section to place start line and start position.
    9.  Translate and rotate track to origin.

    Args:
        seed (int | None): seed for random number generator.
    
    Returns:
        Track: generated track object.
    """
    rng = np.random.default_rng(seed)

    # Create bounded Voronoi diagram
    input_points = rng.uniform(min_bound, max_bound, (n_points, 2))
    bounding_box = np.array([min_bound, max_bound] * 2)
    vor = _bounded_voronoi(input_points, bounding_box)

    while True:
        
        if mode == Mode.EXPAND:
            # Pick a random point and find its n closest neighbours
            random_index = rng.integers(0, n_points)
            random_point_indices = [random_index]
            random_point = input_points[random_index]
            
            for i in range(n_regions - 1):
                closest_point_index = closest_node(random_point, input_points, k=i+1)
                random_point_indices.append(closest_point_index)
                
        elif mode == Mode.EXTEND:
            # Pick a random point, create a line extending from this point and find other points close to this line
            random_index = rng.integers(0, n_points)
            random_heading = rng.uniform(0, np.pi/2)
            random_point = input_points[random_index]
            
            start = (random_point[0] - 1./2. * max_bound * np.cos(random_heading), random_point[1] - 1./2. * max_bound * np.sin(random_heading))
            end = (random_point[0] + 1./2. * max_bound * np.cos(random_heading), random_point[1] + 1./2. * max_bound * np.sin(random_heading))
            line = LineString([start, end])
            distances = [Point(p).distance(line) for p in input_points]
            random_point_indices = np.argpartition(distances, n_regions)[:n_regions]
            
        elif mode == Mode.RANDOM:
            # Select regions randomly
            random_point_indices = rng.integers(0, n_points, n_regions)
        
        # From the Voronoi regions, get the regions belonging to the randomly selected points
        regions = np.array([np.array(region) for region in vor.regions], dtype=object)
        random_region_indices = vor.point_region[random_point_indices]
        random_regions = np.concatenate(regions[random_region_indices])
        
        # Get the vertices belonging to the random regions
        random_vertices = np.unique(vor.vertices[random_regions], axis=0)
        
        # Sort vertices
        sorted_vertices = clockwise_sort(random_vertices)
        sorted_vertices = np.vstack([sorted_vertices, sorted_vertices[0]])
        
        while True:
    
            # Interpolate
            tck, _ = interpolate.splprep([sorted_vertices[:,0], sorted_vertices[:,1]], s=0, per=True)
            t = np.linspace(0, 1, 1000)
            x, y = interpolate.splev(t, tck, der=0)
            dx_dt, dy_dt = interpolate.splev(t, tck, der=1)
            d2x_dt2, d2y_dt2 = interpolate.splev(t, tck, der=2)
            
            # Calculate curvature
            k = curvature(dx_dt, d2x_dt2, dy_dt, d2y_dt2)
            abs_curvature = np.abs(k)
            
            # Check if curvature exceeds threshold
            peaks, _ = signal.find_peaks(abs_curvature)
            exceeded_peaks = abs_curvature[peaks] > CURVATURE_THRESHOLD
            max_peak_index = abs_curvature[peaks].argmax()
            is_curvature_exceeded = exceeded_peaks[max_peak_index]
            
            if is_curvature_exceeded:
                # Find vertice where curvature is exceeded and delete vertice from sorted vertices. Reiterate
                max_peak = peaks[max_peak_index]
                peak_coordinate = (x[max_peak], y[max_peak])
                vertice = closest_node(peak_coordinate, sorted_vertices, k=0)
                sorted_vertices = np.delete(sorted_vertices, vertice, axis=0)
                
                # Make sure that first and last coordinate are the same for periodic interpolation
                if not np.array_equal(sorted_vertices[0], sorted_vertices[-1]):
                    sorted_vertices = np.vstack([sorted_vertices, sorted_vertices[0]])
            else:
                break
        
        # Create track boundaries
        track = Polygon(zip(x, y))
        track_left = track.buffer(TRACK_WIDTH / 2)
        track_right = track.buffer(-TRACK_WIDTH / 2)
        
        # Check if track does not cross itself
        if track.is_valid and track_left.is_valid and track_right.is_valid:
            if track.geom_type == track_left.geom_type == track_right.geom_type == 'Polygon':
                break

    # Calculate cone spacing        
    cone_spacing_left = np.linspace(0, track_left.length, np.ceil(track_left.length / TRACK_WIDTH).astype(int) + 1)[:-1]
    cone_spacing_right= np.linspace(0, track_right.length, np.ceil(track_right.length / TRACK_WIDTH).astype(int) + 1)[:-1]
        
    # Determine coordinates of cones
    cones_left = np.asarray([np.asarray(track_left.exterior.interpolate(sp).xy).flatten() for sp in cone_spacing_left])
    cones_right = np.asarray([np.asarray(track_right.exterior.interpolate(sp).xy).flatten() for sp in cone_spacing_right])

    # Find straight section in track that is at least the length of the start area
    # If such a section cannot be found, adjust the straight_threshold and length_start_area variables
    # There is only a chance of this happening if n_regions == 1 
    straight_threshold = STRAIGHT_THRESHOLD if abs_curvature.min() < STRAIGHT_THRESHOLD else abs_curvature.min() + 0.1
    straight_sections = abs_curvature[:-1] <= straight_threshold
    distances = arc_length(x, y, 1 / abs_curvature)
    length_straights = distances * straight_sections

    # Find cumulative length of straight sections
    for i in range(1, len(length_straights)):
        if length_straights[i]:
            length_straights[i] += length_straights[i-1]
            
    # Find start line and start pose
    length_start_area = LENGTH_START_AREA if length_straights.max() > LENGTH_START_AREA else length_straights.max()
    try:
        start_line_index = np.where(length_straights > length_start_area)[0][0]
    except IndexError:
        raise Exception("Unable to find suitable starting position. Try to decrease the length of the starting area or different input parameters.")
    start_line = np.array([x[start_line_index], y[start_line_index]])
    start_position = np.asarray(track.exterior.interpolate(np.sum(distances[:start_line_index]) - length_start_area)).flatten()
    start_position = np.array([start_position[0].x, start_position[0].y]) 
    start_heading = float(np.arctan2(*(start_line - start_position)))

    # Translate and rotate track to origin
    M = transformation_matrix(-start_position, start_heading - np.pi/2)
    cones_left = M.dot(np.c_[cones_left, np.ones(len(cones_left))].T)[:-1].T
    cones_right = M.dot(np.c_[cones_right, np.ones(len(cones_right))].T)[:-1].T
    track = Track(cones_left, cones_right)

    return track

def generate_track(n_points: int, 
                   n_regions: int, 
                   min_bound: float, 
                   max_bound: float, 
                   mode: Mode | str = Mode.EXPAND,
                   seed: int | None = None) -> Track:
    """
    Generates a track from the vertices of a Voronoi diagram.

    Args: 
        n_points: Number of points to generate for the Voronoi diagram.
        n_regions: Number of regions in the Voronoi diagram.
        min_bound: Minimum boundary value for the track.
        max_bound: Maximum boundary value for the track.
        mode: Mode of generation.
        seed: Random seed for reproducibility.
    """
    mode = Mode[mode.upper()] if isinstance(mode, str) else Mode(mode)

    while True:
        try:
            track = _create_track(n_points, n_regions, min_bound, max_bound, mode, seed)
            return track
        except:
            continue

def load_track(name: str) -> Track:
    """
    Loads a track from the tracks folder.

    Args:
        name: Name of track to load. Must be either "FSG" or "FSI".
    
    Returns:
        Track: loaded track object.
    """
    try:
        data = yaml.safe_load(open(Path(__file__).parent / "tracks" / f"{name}.yaml"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Track {name} not found.")
    return Track(np.array(data["cones_left"]), np.array(data["cones_right"]))