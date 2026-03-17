"""
LiDAR Odometry Actor Class

Processes LiDAR point clouds and retrieves current velocities using ICP-based odometry.
"""

import numpy as np
from .voxelize import voxelize
from .icp import align_point_clouds_icp
from scipy.spatial.transform import RigidTransform as Tf
from .cloud_processing import remove_distant_points


class LidarOdometryActor:
    """
    LiDAR Odometry Actor for ICP-based pose estimation.

    Processes incoming LiDAR point clouds by filtering distant points, voxelizing
    for efficient matching, aligning to a local map via ICP, and computing velocities.

    Attributes:
    ----------
    _config : dict
        Algorithm parameters: "algo", "voxel_size", "points_max_distance", "icp_max_iterations"
    _local_map : np.ndarray or None
        Accumulated local map (N, 3). None initially.
    _prev_stamp : float or None
        Timestamp of last processed frame for velocity computation.
    _initial_guess : RigidTransform or None
        Initial ICP transformation guess. Updated after each alignment.
    _velocities : np.ndarray
        Current velocities [vx, vy, vz, wx, wy, wz] in m/s and rad/s.
    _state : dict or None
        State info, e.g., {"icp_info": ...}
    """

    def __init__(
        self, config: dict = None, initial_velocities: np.ndarray | None = None
    ):
        """Initialize the actor.

        Parameters
        ----------
        config : dict
            Required keys: "algo", "voxel_size", "points_max_distance", "icp_max_iterations"
        """
        self._config = config
        self._algo = self._config["algo"]
        self._local_map = None
        self._prev_stamp = None
        self._initial_guess = None
        self._velocities = np.zeros(6)
        self._state = None
        if initial_velocities is not None:
            self.set_current_velocities(initial_velocities)

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
        processed_points = self.preprocess_cloud(points)

        # Voxelize current points for efficient ICP matching
        cur_voxelized = voxelize(processed_points, self._config["voxel_size"])

        # Run ICP alignment to compute transformation between frames
        transform, info = align_point_clouds_icp(
            cur_voxelized,
            self.get_local_map(),
            init_transform=self.get_initial_guess(),
            max_iterations=self._config["icp_max_iterations"],
            w_func="geman-mcclure",
        )

        # Store ICP information in state
        self._state = {"icp_info": info}
        # Save the computed transformation as initial guess for next iteration
        self.save_initial_guess(transform)

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
        return remove_distant_points(points, self._config["points_max_distance"])

    def get_initial_guess(self) -> Tf | None:
        """Get the initial ICP transformation guess."""
        return self._initial_guess

    def save_initial_guess(self, transform: Tf) -> None:
        """Save a transformation as the initial guess for next ICP iteration."""
        self._initial_guess = transform

    def set_current_velocities(self, velocities: np.ndarray) -> None:
        """Set velocities from an external source."""
        self._velocities = self.swap_v_and_w(velocities)

    def get_current_velocities(self) -> np.ndarray:
        """Get current velocity estimates [vx, vy, vz, wx, wy, wz]."""
        return self.swap_v_and_w(self._velocities)

    def get_state(self) -> dict | None:
        """Get the state dictionary."""
        return self._state
