"""
Cloud processing utilities for LiDAR point clouds.
"""

import numpy as np


def remove_distant_points(points: np.ndarray, max_distance: float = 50.0) -> np.ndarray:
    """
    Remove points that are too distant from the origin from a LiDAR point cloud.
    
    This function filters out points whose Euclidean distance from the origin
    exceeds the specified maximum distance threshold. This is useful for removing
    far-away objects (e.g., buildings, mountains) that may interfere with local
    feature matching in LiDAR odometry applications.
    
    Parameters
    ----------
    points : np.ndarray
        Input point cloud of shape (N, 3) where each row is a [x, y, z] coordinate.
    max_distance : float, optional
        Maximum distance from the origin to keep points. Default is 50.0 meters.
        
    Returns
    -------
    np.ndarray
        Filtered point cloud containing only points within max_distance from origin.
    """
    if points is None or len(points) == 0:
        return np.array([])
    
    # Compute Euclidean distance from origin for each point
    norms = np.linalg.norm(points, axis=1)
    
    # Filter points within the maximum distance threshold
    mask = norms <= max_distance
    filtered_points = points[mask]
    
    return filtered_points
