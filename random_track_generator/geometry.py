import numpy as np

def closest_node(node: np.ndarray, nodes: np.ndarray, k: int) -> int:
    """
    Returns the index of the k-th closest node.
    
    Args:
        node: Node to find k-th closest node to.
        nodes: Available nodes.
        k: Number which determines which closest node to return.
    
    Returns:
        Index of k-th closest node.
    """
    deltas = nodes - node
    distance = np.einsum('ij,ij->i', deltas, deltas)
    return np.argpartition(distance, k)[k]

def clockwise_sort(p: np.ndarray) -> np.ndarray:
    """
    Sorts nodes in clockwise order.
    
    Args:
        p: Points to sort.
    
    Returns:
        Clockwise sorted points.
    """
    d = p - np.mean(p, axis=0)
    s = np.arctan2(d[:,0], d[:,1])
    return p[np.argsort(s),:]

def curvature(dx_dt: np.ndarray, d2x_dt2: np.ndarray, dy_dt: np.ndarray, d2y_dt2: np.ndarray) -> np.ndarray:
    """
    Calculates the curvature along a line.
    
    Args:
        dx_dt: First derivative of x.
        d2x_dt2: Second derivative of x.
        dy_dt: First derivative of y.
        d2y_dt2: Second derivative of y.
    
    Returns:
        Curvature along line.
    """
    return (dx_dt**2 + dy_dt**2)**-1.5 * (dx_dt * d2y_dt2 - dy_dt * d2x_dt2)

def arc_length(x: np.ndarray, y: np.ndarray, R: np.ndarray) -> float:
    """
    Calculates the arc length between to points based on the radius of curvature of the path segment.
    
    Args:
        x: x-coordinates.
        y: y-coordinates.
        R: Radius of curvature of track segment in meters.
    Returns:
        Arc length in meters.
    """
    x0, x1 = x[:-1], x[1:]
    y0, y1 = y[:-1], y[1:]   
    R = R[:-1]
    
    distance = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    theta = 2 * np.arcsin(0.5 * distance / R)
    arc_length = R * theta
    return arc_length

def transformation_matrix(displacement: tuple, angle: float):
    """
    Translate, then rotate around origin.
    
    Args:
        displacement: Distance to translate along both axes.
        angle: Angle in radians to rotate.
    
    Returns:
        3x3 transformation matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([
        [c, -s],
        [s,  c]
    ])
    tx, ty = R @ displacement
    
    return np.array([
        [c, -s, tx],
        [s,  c, ty],
        [0,  0,  1]
    ])