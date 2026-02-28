import gpxpy
import yaml
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

class Mode(Enum):
    """ 
    Possible modes for how Voronoi regions are selected.
    
    1. Expand:
        Find closest nodes around starting node.
        Results in roundish track shapes.
    
    2. Extend:
        Find nodes closest to line extending from starting node.
        Results in elongated track shapes.
        
    3. Random:
        Select all regions randomly.
        Results in large track shapes.
    """
    EXPAND = 1
    EXTEND = 2
    RANDOM = 3

class SimType(Enum):
    """ Selection between output format for different simulators.

    1. FSSIM:
        Output FSSIM compatible .yaml file.
    2. FSDS:
        Output FSDS compatible .csv file 
    """
    FSSIM = 1
    FSDS = 2
    GPX = 3

@dataclass
class Track:
    """ Track dataclass

    Attributes:
         cones_left (np.ndarray): Left cones of track.
         cones_right (np.ndarray): Right cones of track.
    """
    cones_left: np.ndarray
    cones_right: np.ndarray

    def as_tuple(self):
        """ Returns cones as tuple of left and right cones.
        """
        return self.cones_left, self.cones_right

    def save(self, location: str | Path, sim_type: SimType | str, *,
             lat_offset: float = 0.0, lon_offset: float = 0.0, z_offset: float = 0.0):
        """ Saves track in specified format for use in different simulators.

        Args:
            location: Location to save track to.
            sim_type: Format to save track in. Must be either "fssim", "fsds" or "gpx".
            lat_offset: Latitude offset for GPX output format, in degrees.
            lon_offset: Longitude offset for GPX output format, in degrees.
            z_offset: Altitude offset for GPX output format, in meters.
        """
        sim_type = SimType[sim_type.upper()] if isinstance(sim_type, str) else SimType(sim_type)
        path = Path(location)

        if sim_type == SimType.FSSIM:
            with open(path / "random_track.yaml", 'w') as f:
                yaml.dump({
                    'cones_left': self.cones_left.tolist(),
                    'cones_right': self.cones_right.tolist(),
                    'cones_orange': [],
                    'cones_orange_big': [[4.7, 2.5], [4.7, -2.5], [7.3, 2.5], [7.3, -2.5]],
                    'starting_pose_cg': [0., 0., 0.],
                    'tk_device': [[6., 3.], [6., -3.]],
                }, f)

        elif sim_type == SimType.FSDS:
            out = path / "random_track.csv"
            with open(out, 'w') as f:
                for cone in self.cones_left:
                    f.write(f"blue,{cone[0]},{cone[1]},0,0.01,0.01,0\n")
                for cone in self.cones_right:
                    f.write(f"yellow,{cone[0]},{cone[1]},0,0.01,0.01,0\n")
                f.write("big_orange,4.7,2.2,0,0.01,0.01,0\n")
                f.write("big_orange,4.7,-2.2,0,0.01,0.01,0\n")
                f.write("big_orange,7.3,2.2,0,0.01,0.01,0\n")
                f.write("big_orange,7.3,-2.2,0,0.01,0.01,0\n")

        elif sim_type == SimType.GPX:
            gpx = gpxpy.gpx.GPX()
            gpx.tracks.append(gpxpy.gpx.GPXTrack())

            deg_per_m_lat = np.degrees(1 / 6378100)
            deg_per_m_lon = np.degrees(1 / 6378100) / np.cos(np.radians(lat_offset))

            for cone in np.vstack([self.cones_left, self.cones_right]):
                lat = lat_offset + cone[1] * deg_per_m_lat
                lon = lon_offset + cone[0] * deg_per_m_lon
                gpx.waypoints.append(gpxpy.gpx.GPXWaypoint(latitude=lat, longitude=lon, elevation=z_offset))

            (path / "random_track.gpx").write_text(gpx.to_xml())
