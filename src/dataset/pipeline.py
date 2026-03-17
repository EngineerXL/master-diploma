"""
LiDAR Odometry Pipeline Module

Provides LidarOdometryPipeline class for processing LiDAR point clouds
and storing velocity estimates with ground truth values.
"""

import os
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm

from src.dataset.utils import LidarOdometryWrapper
from src.lidar_odometry.actor import LidarOdometryActor

DATA_OUTPUT_ROOT = "/data/boreas/output"


class LidarOdometryPipeline:
    """
    Pipeline for running LiDAR odometry actor and storing velocity estimates.

    The pipeline processes LiDAR point clouds using the LidarOdometryActor,
    collects both ground truth and estimated velocities, and saves results
    to a specified folder.

    Attributes:
    -----------
    config : dict
        Configuration dictionary with structure:
        {
            'ride_info': {
                'ride_id': str,
                'ride_frame_st': int,
                'ride_frame_en': int
            },
            'actor_settings': dict,  # actor_cfg_dict
            'config_name': str
        }
    """

    def __init__(self, config: Dict):
        """
        Initialize the LidarOdometryPipeline.

        Parameters:
        -----------
        config : dict
            Configuration dictionary with structure:
            {
                'ride_info': {
                    'ride_id': str,
                    'ride_frame_st': int,
                    'ride_frame_en': int
                },
                'actor_settings': dict,  # actor_cfg_dict
                'config_name': str
            }
        """
        self.ride_info = config["ride_info"]
        self.actor_settings = config["actor_settings"]
        self.config_name = config["config_name"]

        # Initialize the LidarOdometryActor with actor settings
        self.actor = LidarOdometryActor(config=self.actor_settings)

        # Results storage
        self.results: List[Dict] = []

        ride_frame_st = self.ride_info["ride_frame_st"]
        ride_frame_en = self.ride_info["ride_frame_en"]
        suffix = self.config_name
        self.pipeline_name = f"{ride_frame_st}-{ride_frame_en}-{suffix}"

    def _get_wrapper(self) -> LidarOdometryWrapper:
        wrapper = LidarOdometryWrapper(ride_id=self.ride_info["ride_id"])
        return wrapper

    def velocity_do_dict(self, velocities: np.ndarray) -> Dict:
        return {
            "vx": velocities[0].astype(float),
            "vy": velocities[1].astype(float),
            "vz": velocities[2].astype(float),
            "wx": velocities[3].astype(float),
            "wy": velocities[4].astype(float),
            "wz": velocities[5].astype(float),
        }

    def process_ride(self) -> List[Dict]:
        wrapper = self._get_wrapper()

        ride_frame_st = self.ride_info["ride_frame_st"]
        ride_frame_en = self.ride_info["ride_frame_en"]

        _, _, first_frame_velocities = wrapper.get_frame(ride_frame_st)
        self.actor.set_current_velocities(first_frame_velocities)

        # Process each frame in the ride range
        for i in tqdm(
            range(ride_frame_st, ride_frame_en),
            desc=f"Running '{self.ride_info["ride_id"]}/{self.pipeline_name}' pipeline...",
        ):
            # Get rotated LiDAR point cloud from the wrapper
            timestamp, points, gt_velocities = wrapper.get_frame(i)

            # Process the lidar cloud through the actor
            self.actor.process_lidar_cloud(points, timestamp)

            # Get estimated velocities from the actor
            estimated_velocities = self.actor.get_current_velocities()

            # Create result dictionary
            result = {
                "timestamp": timestamp,
                "gt": {"velocities": self.velocity_do_dict(gt_velocities)},
                "estimate": {
                    "velocities": self.velocity_do_dict(estimated_velocities),
                },
            }

            actor_state = self.actor.get_state()
            if actor_state is not None:
                result["estimate"]["iterations"] = actor_state["icp_info"]["iterations"]

            self.results.append(result)

        return self.results

    def save_results(self, output_root: str = DATA_OUTPUT_ROOT) -> str:
        output_folder = os.path.join(output_root, self.ride_info["ride_id"])

        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        fname = f"{self.pipeline_name}.json"
        # Save results as JSON file
        output_path = os.path.join(output_folder, fname)
        import json

        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)

        return output_path
