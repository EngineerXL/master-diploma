"""
LiDAR Odometry Actor Class

Provides functionality for processing LiDAR point clouds and retrieving current velocities.
"""

import numpy as np
from .voxelize import voxelize
from .icp import align_point_clouds_icp
from scipy.spatial.transform import RigidTransform as Tf
from .icp_kabsch import align_points_geman_mcclure
from .cloud_processing import remove_distant_points


class LidarOdometryActor:
    """

    Provides two main functions:
    1. process_lidar_cloud - Process incoming LiDAR point cloud data
    2. get_current_velocities - Retrieve current linear and angular velocities

    Attributes:
    -----------
    _point_cloud : numpy.ndarray
        Stored processed point cloud (N, 3) shape
    _velocities : numpy.ndarray
        Current velocities [vx, vy, vz, wx, wy, wz]
    _timestamp : float
        Timestamp of the last update
    _odometry_state : dict
        Additional odometry state information
    """

    def __init__(self, config: dict = None):
        self._config = config
        self._algo = self._config["algo"]
        self._prev_cloud = None
        self._prev_stamp = None
        self._initial_guess = None
        self._velocities = np.zeros(6)
        self._state = None

    def process_lidar_cloud(
        self,
        points: np.ndarray,
        timestamp: float,
    ):
        processed_points = self.preprocess_cloud(points)
        if self._prev_cloud is None:
            self._prev_cloud = processed_points
            self._prev_stamp = timestamp
            return

        prev_voxelized = voxelize(self._prev_cloud, self._config["voxel_size"])
        cur_voxelized = voxelize(processed_points, self._config["voxel_size"])
        if self._algo == "icp_geman_mcclure":
            transform, info = align_points_geman_mcclure(
                cur_voxelized,
                prev_voxelized,
                init_transform=self.get_initial_guess(),
                max_iterations=self._config["icp_max_iterations"],
            )
        elif self._algo == "icp":
            transform, info = align_point_clouds_icp(
                cur_voxelized,
                prev_voxelized,
                init_transform=self.get_initial_guess(),
                max_iterations=self._config["icp_max_iterations"],
                w_func="geman-mcclure",
            )
        else:
            raise RuntimeError(f"Unknown algorithm: {self._algo}")
        self._state = {"icp_info": info}

        self.save_initial_guess(transform)
        dt = timestamp - self._prev_stamp

        self._velocities = transform.as_exp_coords() / dt

        self._prev_cloud = processed_points
        self._prev_stamp = timestamp

    # RigidTransform from scipy store angular velocity first
    def swap_v_and_w(self, velocities: np.ndarray) -> np.ndarray:
        return np.concat([velocities[3:6], velocities[0:3]])

    def preprocess_cloud(self, points: np.ndarray):
        return remove_distant_points(points, self._config["points_max_distance"])

    def get_initial_guess(self):
        return self._initial_guess

    def save_initial_guess(self, transform: Tf):
        self._initial_guess = transform

    def set_currect_velocities(self, velocities: np.ndarray):
        self._velocities = self.swap_v_and_w(velocities)

    def get_current_velocities(self) -> np.ndarray:
        return self.swap_v_and_w(self._velocities)

    def get_state(self) -> dict:
        return self._state
