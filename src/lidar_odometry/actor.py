"""
LiDAR Odometry Actor Class

Processes LiDAR point clouds and retrieves current velocities using ICP-based odometry.
"""

import numpy as np
from .voxelize import voxelize
from .icp import align_point_clouds_icp
from scipy.spatial.transform import RigidTransform
from .local_map import VoxelMap
from .kalman_filter import ConstVelocityKalmanFilter


class LidarOdometryActor:
    """
    LiDAR Odometry Actor for ICP-based pose estimation.

    Processes incoming LiDAR point clouds by filtering distant points, voxelizing
    for efficient matching, aligning to a local map via ICP, and computing velocities.
    """

    def __init__(self, config: dict, initial_velocities: np.ndarray = np.zeros(6)):
        self.parse_config(config)
        self._initial_guess = None
        self.state = {"transforms": []}
        self.local_map = VoxelMap(
            self.voxel_size,
            self.points_max_distance_r_max,
            self.max_points_per_voxel_N_max,
        )
        if self.use_kalman_filter:
            self.kalman_filter = ConstVelocityKalmanFilter(
                x_init=initial_velocities,
                linear_process_noise_covariance=self.linear_process_noise_covariance,
                angular_process_noise_covariance=self.angular_process_noise_covariance,
                linear_measurement_noise_covariance=self.linear_measurement_noise_covariance,
                angular_measurement_noise_covariance=self.angular_measurement_noise_covariance,
            )
        self.update_velocities(self.swap_v_and_w(initial_velocities))
        # Set initial deviation
        self.deviations_2_sum = 0
        self.deviations_count = 0

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
        self.max_points_per_voxel_N_max = config["max_points_per_voxel_N_max"]
        self.use_local_map = config["use_local_map"]

        kalman_config = config["kalman_filter_settings"]
        self.use_kalman_filter = kalman_config["use"]
        if self.use_kalman_filter:
            self.linear_process_noise_covariance = kalman_config[
                "linear_process_noise_covariance"
            ]
            self.angular_process_noise_covariance = kalman_config[
                "angular_process_noise_covariance"
            ]
            self.linear_measurement_noise_covariance = kalman_config[
                "linear_measurement_noise_covariance"
            ]
            self.angular_measurement_noise_covariance = kalman_config[
                "angular_measurement_noise_covariance"
            ]

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
        # Voxelize current points for efficient ICP matching
        points_merge = voxelize(
            points, self.factor_voxel_size_map_merge * self.voxel_size
        )

        if len(self.state["transforms"]) == 0:
            # First frame: register as initial map and return early
            self.state["prev_timestamp"] = timestamp
            self.state["transforms"].append(RigidTransform.identity())
            self.update_local_map(points_merge, self.get_transform(-1))
            return

        points_reduced = voxelize(
            points_merge, self.factor_voxel_size_registration * self.voxel_size
        )

        dt = timestamp - self.state["prev_timestamp"]
        t_prev = self.get_transform(-1)
        t_pred = self.get_initial_guess(dt)
        points_global = (t_prev * t_pred).apply(points_reduced)

        displacement_deviation_sigma_t, max_correspondence_distance_tau_t = (
            self.get_deviation()
        )

        # Run ICP alignment to compute transformation between frames
        t_icp, info = align_point_clouds_icp(
            points_global,
            self.get_local_map_points(),
            max_iterations=self.icp_max_iterations,
            displacement_deviation_sigma_t=displacement_deviation_sigma_t,
            max_correspondence_distance_tau_t=max_correspondence_distance_tau_t,
            tolerance_gamma=self.tolerance_gamma,
        )

        t_cur = t_icp * (t_prev * t_pred)
        pose_deviation_delta_t = (t_prev * t_pred).inv() * t_icp * (t_prev * t_pred)

        # Store ICP information in state
        self.state["icp_info"] = info

        # Save the computed transformation as initial guess for next iteration
        self.update_initial_guess(t_cur)
        self.update_deviation(pose_deviation_delta_t)
        self.update_local_map(points_merge, t_cur)

        self.update_velocities(self.calculate_velocities(dt))
        self.state["prev_timestamp"] = timestamp

    def swap_v_and_w(self, velocities: np.ndarray) -> np.ndarray:
        """Reorder velocities from scipy [wx, wy, wz, vx, vy, vz] to [vx, vy, vz, wx, wy, wz]."""
        return np.concatenate([velocities[3:6], velocities[0:3]])

    def calculate_velocities(self, dt: float):
        # Compute velocity from relative transformation and time difference
        t_relative = self.get_transform(-2).inv() * self.get_transform(-1)
        return t_relative.as_exp_coords() / dt

    def get_initial_guess(self, dt: float):
        return RigidTransform.from_exp_coords(self.state["model_velocities"] * dt)

    def update_initial_guess(self, transform: RigidTransform):
        self.state["transforms"].append(transform)

    def get_local_map_points(self):
        if self.use_local_map:
            return self.local_map.get_points()
        else:
            return self.prev_points

    def update_local_map(self, points: np.ndarray, pose: RigidTransform):
        if self.use_local_map:
            self.local_map.add_points(pose.apply(points))
            self.local_map.remove_distant_voxels(pose)
        else:
            self.prev_points = pose.apply(points)

    def get_transform(self, index=-1):
        return self.state["transforms"][index]

    def get_deviation(self):
        # 3 sigma rule
        if self.deviations_count > 0:
            sigma = np.sqrt(self.deviations_2_sum / self.deviations_count)
            return sigma, 3 * sigma
        else:
            tau = self.initial_max_correspondance_distance_tau_0
            return tau / 3, tau

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

    def update_velocities(self, velocities: np.ndarray) -> None:
        if self.use_kalman_filter:
            self.kalman_filter.predict()
            self.kalman_filter.update(self.swap_v_and_w(velocities))
            self.state["model_velocities"] = self.swap_v_and_w(
                self.kalman_filter.get_estimate()
            )
        else:
            self.state["model_velocities"] = velocities
        self.state["output_velocities"] = self.swap_v_and_w(velocities)

    def get_current_velocities(self) -> np.ndarray:
        """Get current velocity estimates [vx, vy, vz, wx, wy, wz]."""
        return self.state["output_velocities"]

    def get_state(self) -> dict:
        """Get the state dictionary."""
        return self.state
