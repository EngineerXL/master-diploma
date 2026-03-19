import numpy as np
from typing import Dict
from .voxelize import get_voxel
from scipy.spatial.transform import RigidTransform


class VoxelMap:
    """Voxel map for storing LiDAR points in a voxel-based structure."""

    def __init__(self, voxel_size : float, r_max: float, N_max: int):
        """Initialize the voxel map.

        Args:
            r_max: Maximum radius of the voxel map (in meters).
            N_max: Maximum number of points per voxel.
        """
        self.voxel_size = voxel_size
        self.r_max = r_max
        self.N_max = N_max
        self.points_map = {}

    def add_point(self, point: np.ndarray):
        voxel = get_voxel(point, self.voxel_size)
        voxel_bytes = voxel.tobytes()
        if voxel_bytes not in self.points_map:
            self.points_map[voxel_bytes] = []

        if len(self.points_map[voxel_bytes]) < self.N_max:
            self.points_map[voxel_bytes].append(point)

    def add_points(self, points: np.ndarray) -> None:
        """Add points to the voxel map.

        Args:
            points: Nx3 numpy array of 3D points.

        Points are grouped by voxel and each voxel stores at most N_max points.
        """

        # Group points by voxel and limit to N_max per voxel
        np.apply_along_axis(self.add_point, arr=points, axis = 1)

    def get_points(self) -> np.ndarray:
        """Get all points stored in the map.

        Returns:
            Nx3 numpy array containing all points from all voxels.
        """
        all_points = []
        for voxel, points in self.points_map.items():
            all_points.extend(points)
        return np.array(all_points)

    def remove_distant_voxels(self, transform : RigidTransform):
        pose, rot = transform.as_components()
        to_remove = set()
        for voxel_bytes, points in self.points_map.items():
            point = points[0]
            if (np.linalg.norm(pose - point) > self.r_max):
                to_remove.add(voxel_bytes)
        for voxel_bytes in to_remove:
            self.points_map.pop(voxel_bytes)
