"""
LiDAR Odometry Actor Class

Processes LiDAR point clouds and retrieves current velocities using ICP-based odometry.
"""

import numpy as np
from .voxelize import voxelize
from .icp import align_point_clouds_icp
from scipy.spatial.transform import RigidTransform
from .cloud_processing import remove_distant_points


class LidarOdometryActor:
    """
    LiDAR Odometry Actor for ICP-based pose estimation.

    Processes incoming LiDAR point clouds by filtering distant points, voxelizing
    for efficient matching, aligning to a local map via ICP, and computing velocities.
    """

    def __init__(self, config: dict, initial_velocities: np.ndarray = np.zeros(6)):
        self.parse_config(config)
        self._local_map = None
        self._prev_stamp = None
        self._initial_guess = None
        self._state = None
        self.set_current_velocities(initial_velocities)
        initial_transform = RigidTransform.from_exp_coords(self._velocities)
        self.update_initial_guess(initial_transform)
        # Set initial deviation
        self.deviations_2_sum = np.square(
            self.initial_max_correspondance_distance_tau_0
        )
        self.deviations_count = 1

    def parse_config(self, config: dict):
        self.voxel_size = config["voxel_size"]
        self.icp_max_iterations = config["icp_max_iterations"]
        self.points_max_distance_r_max = config["points_max_distance_r_max"]
        self.min_deviation_distance_delta_min = config[
            "min_deviation_distance_delta_min"
        ]
        self.tolerance_gamma = config["tolerance_gamma"]
        self.initial_max_correspondance_distance_tau_0 = config[
            "initial_max_correspondance_distance_tau_0"
        ]
        self.factor_voxel_size_map_merge = config["factor_voxel_size_map_merge"]
        self.factor_voxel_size_registration = config["factor_voxel_size_registration"]

    def process_lidar_cloud(self, points: np.ndarray, timestamp: float) -> None:
        """Process a LiDAR frame.

        First frame registers as initial map. Subsequent frames are filtered,
        voxelized, aligned via ICP, and velocities computed from the transform delta.
        Original (unfiltered) points are registered to grow the local map.

        Parameters
        ----------
        points : np.ndarray
            Incoming point cloud (N, 3).
        timestamp : float
            Frame timestamp in seconds.
        """
        if self._local_map is None:
            # First frame: register as initial map and return early
            self.register_points_to_map(points)
            self._prev_stamp = timestamp
            return

        # Pre-filter current points by removing distant outliers
        preprocessed_points = self.preprocess_cloud(points)

        # Voxelize current points for efficient ICP matching
        points_merge = voxelize(
            preprocessed_points, self.factor_voxel_size_map_merge * self.voxel_size
        )
        points_reduced = voxelize(
            points_merge, self.factor_voxel_size_registration * self.voxel_size
        )

        # Run ICP alignment to compute transformation between frames
        transform, info = align_point_clouds_icp(
            points_reduced,
            self.get_local_map(),
            init_transform=self.get_initial_guess(),
            max_iterations=self.icp_max_iterations,
            displacement_deviation_sigma_t=self.get_deviation(),
            tolerance_gamma=self.tolerance_gamma,
        )

        # Store ICP information in state
        self._state = {"icp_info": info}
        # Save the computed transformation as initial guess for next iteration
        self.update_initial_guess(transform)
        self.update_deviation(transform)

        # Compute velocity from transformation delta and time difference
        dt = timestamp - self._prev_stamp
        self._velocities = transform.as_exp_coords() / dt

        # Register original (unfiltered) points to the local map
        self.register_points_to_map(points)
        self._prev_stamp = timestamp

    def get_local_map(self) -> np.ndarray:
        """Get the local map point cloud."""
        return self._local_map

    def register_points_to_map(self, points: np.ndarray) -> None:
        """Register a point cloud as the new local map."""
        self._local_map = points

    def swap_v_and_w(self, velocities: np.ndarray) -> np.ndarray:
        """Reorder velocities from scipy [wx, wy, wz, vx, vy, vz] to [vx, vy, vz, wx, wy, wz]."""
        return np.concatenate([velocities[3:6], velocities[0:3]])

    def preprocess_cloud(self, points: np.ndarray) -> np.ndarray:
        """Remove distant outliers from a point cloud."""
        return remove_distant_points(points, self.points_max_distance_r_max)

    def get_initial_guess(self):
        return self.initial_guess

    def update_initial_guess(self, transform: RigidTransform):
        self.initial_guess = transform

    def get_deviation(self):
        return np.sqrt(self.deviations_2_sum / self.deviations_count)

    def calculate_deviations_from_tf(self, transform: RigidTransform):
        t, rot = transform.as_components()
        delta_t = np.linalg.norm(t)
        theta = np.arccos((np.trace(rot.as_matrix()) - 1) / 2)
        delta_rot = 2 * self.points_max_distance_r_max * np.sin(theta / 2)
        return delta_t, delta_rot

    def update_deviation(self, transform: RigidTransform):
        delta_t, delta_rot = self.calculate_deviations_from_tf(transform)
        cur_delta_upper_bound = delta_t + delta_rot
        if cur_delta_upper_bound >= self.min_deviation_distance_delta_min:
            self.deviations_2_sum += np.square(cur_delta_upper_bound)
            self.deviations_count += 1

    def set_current_velocities(self, velocities: np.ndarray) -> None:
        """Set velocities from an external source."""
        self._velocities = self.swap_v_and_w(velocities)

    def get_current_velocities(self) -> np.ndarray:
        """Get current velocity estimates [vx, vy, vz, wx, wy, wz]."""
        return self.swap_v_and_w(self._velocities)

    def get_state(self) -> dict | None:
        """Get the state dictionary."""
        return self._state
