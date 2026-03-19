from math import pi
from pyboreas import BoreasDataset
import numpy as np
from src.geometry.add_obstacle import (
    remove_points_by_obstacle_bbox_2d,
    DEFAULT_TRUCK_OFFSET,
    DEFAULT_FORD_TRANSIT_VAN_DIMS,
)

DATASET_ROOT = "/data/boreas"


class LidarOdometryWrapper:
    """
    Wrapper class for LiDAR odometry data.

    Provides methods to access rotated LiDAR point clouds and velocities
    by frame index. Rotation is applied by default around the z-axis.
    """

    def __init__(
        self,
        ride_id: str,
        dataset_root: str = DATASET_ROOT,
        rotation_angle: float = pi / 4,
        truck_dims: np.ndarray = DEFAULT_FORD_TRANSIT_VAN_DIMS,
        truck_offset: np.ndarray = DEFAULT_TRUCK_OFFSET,
        truck_initial_x: float = -7.5,  # truck will traverse vx * frames * 0.1 meters forward
        truck_overtake_vx: float = 1,
        simulate_truck: bool = True,
    ) -> None:
        """
        Initialize the LidarOdometryWrapper.

        Parameters:
        -------
        ride_id : str
            ID of the ride/sequence to load
        dataset_root : str
            Root path to the BOREAS dataset (default: DATASET_ROOT)
        rotation_angle : float
            Rotation angle in radians around z-axis (default: pi/4)
        simulate_truck : bool
            Whether to add truck obstacle to lidar point cloud (default: True)
        """
        self.dataset = BoreasDataset(dataset_root)
        if ride_id not in self.dataset.seqDict.keys():
            raise RuntimeError(f"Unknown ride {ride_id}")

        seq_idx = self.dataset.seqDict[ride_id]
        self.seq = self.dataset.sequences[seq_idx]

        # Rotation matrix for rotation around z-axis
        self.rotation_angle = rotation_angle
        cos_theta = np.cos(self.rotation_angle)
        sin_theta = np.sin(self.rotation_angle)
        self._rotation_matrix = np.array(
            [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]]
        )

        self.truck_dims = truck_dims
        self.truck_offset = truck_offset
        self.truck_initial_x = truck_initial_x
        self.truck_overtake_vx = truck_overtake_vx
        self.simulate_truck = simulate_truck
        self.first_timestamp = None

    def set_first_frame(self, index: int):
        first_lidar_frame = self.seq.get_lidar(index)
        self.first_timestamp = first_lidar_frame.timestamp

    def rotate_points(self, points: np.ndarray) -> np.ndarray:
        """
        Apply rotation to a point cloud.

        Parameters:
        -----------
        points : numpy.ndarray of shape (N, 3)
            Array of points with x, y, z coordinates

        Returns:
        --------
        rotated_points : numpy.ndarray of shape (N, 3)
            Rotated points
        """

        # Apply rotation to each point
        rotated_points = points @ self._rotation_matrix
        return rotated_points

    def get_frames_count(self):
        return len(self.seq.lidar_frames)

    def get_frame(self, index: int) -> tuple:
        lidar_frame = self.seq.get_lidar(index)

        timestamp = lidar_frame.timestamp

        # Get raw points (first 3 columns: x, y, z)
        raw_points = lidar_frame.points[:, :3]

        rotated_points = self.rotate_points(raw_points)

        if self.simulate_truck:
            frame_offset = DEFAULT_TRUCK_OFFSET
            if self.first_timestamp is None:
                raise RuntimeError(
                    "First frame index should be set before getting frame!"
                )
            dt = timestamp - self.first_timestamp
            frame_offset[0] = self.truck_initial_x + self.truck_overtake_vx * dt
            output_points = remove_points_by_obstacle_bbox_2d(
                rotated_points, frame_offset
            )
        else:
            output_points = rotated_points

        # Access velocity data from the sensor frame
        # varpi contains [v_se_in_s; w_se_in_s] - velocities in sensor frame
        varpi = (
            lidar_frame.body_rate
        )  # 6x1 vel in sensor frame [vx, vy, vz, wx, wy, wz]

        # Apply rotation to linear velocities
        v_rotated = self._rotation_matrix.T @ varpi[:3]
        # Apply rotation to angular velocities
        w_rotated = self._rotation_matrix.T @ varpi[3:]

        velocities = np.concatenate([v_rotated, w_rotated])

        # Unload to free memory
        lidar_frame.unload_data()

        return timestamp, output_points, velocities.reshape(6)

    def get_current_rotation_matrix(self) -> np.ndarray:
        """
        Get the current rotation matrix.

        Returns:
        --------
        matrix : numpy.ndarray of shape (3, 3)
            Rotation matrix around z-axis
        """
        if self._rotation_matrix is None:
            # Default identity matrix
            return np.eye(3)
        return self._rotation_matrix
