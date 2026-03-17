import numpy as np
import json
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from src.dataset.pipeline import DATA_OUTPUT_ROOT

VELOCITIES_FIELDS = ["vx", "vy", "vz", "wx", "wy", "wz"]


class PostprocessingWrapper:
    """
    Wrapper class for post-processing and visualizing LiDAR odometry results.

    Takes a config JSON file path (same structure as pipeline_runner.py),
    loads processed ride results from the corresponding output folders,
    provides methods to access velocity data and generate plots comparing
    estimated vs ground truth velocities.

    Attributes:
    -----------
    config : dict
        Configuration dictionary loaded from JSON file with structure:
        - actor_settings: dict
        - wrapper_settings: dict
        - config_name: str
        - ride_segments: list of segment objects
    results : List[Dict]
        List of result dictionaries from the pipeline, each containing:
        - timestamp: float
        - gt: dict with velocities (vx, vy, vz, wx, wy, wz)
        - estimate: dict with velocities and optionally iterations
    """

    def __init__(self, config_json_path: str):
        """
        Initialize the PostprocessingWrapper.

        Parameters:
        -----------
        config_json_path : str
            Path to the JSON config file (same structure as pipeline_runner.py loads).
            Expected structure: {actor_settings, wrapper_settings, config_name, ride_segments}
        """
        # Load config without validation - assume correct structure
        with open(config_json_path, "r") as f:
            self.config = json.load(f)

        # Extract config components
        self.actor_settings = self.config["actor_settings"]
        self.wrapper_settings = self.config["wrapper_settings"]
        self.config_name = self.config["config_name"]
        self.ride_segments = self.config["ride_segments"]

        # Load results from all ride segments using get_folder_and_fname pattern
        # Pattern: {ride_id}/{frame_st}-{frame_en}-{config_name}.json
        self.bag_paths = []
        for segment in self.ride_segments:
            ride_id = segment["ride_id"]
            frame_st = segment["ride_frame_st"]
            frame_en = segment["ride_frame_en"]

            # Use get_folder_and_fname pattern directly from pipeline.py
            folder_and_fname = f"{ride_id}/{frame_st}-{frame_en}-{self.config_name}"
            result_path = f"{DATA_OUTPUT_ROOT}/{folder_and_fname}.json"

            self.bag_paths.append(result_path)

    def load_bag(self, ride_segment_id: int):
        with open(self.bag_paths[ride_segment_id], "r") as f:
            bag_data = json.load(f)
        return bag_data

    def get_velocity_data(self, ride_segment_id: int) -> Dict[str, List[float]]:
        """
        Extract all velocity data from results into organized lists.

        Returns:
        --------
        Dict[str, List[float]]
            Dictionary with keys: timestamp, vx_gt, vy_gt, vz_gt, wx_gt, wy_gt, wz_gt,
            vx_est, vy_est, vz_est, wx_est, wy_est, wz_est
        """
        data = {
            "timestamp": [],
        }

        for field in VELOCITIES_FIELDS:
            data[f"{field}_gt"] = []
            data[f"{field}_est"] = []

        bag = self.load_bag(ride_segment_id)
        for result in bag:
            data["timestamp"].append(result["timestamp"])

            # Ground truth velocities
            gt = result["gt"]["velocities"]

            # Estimated velocities
            est = result["estimate"]["velocities"]

            for field in VELOCITIES_FIELDS:
                data[f"{field}_gt"].append(gt[field])
                data[f"{field}_est"].append(est[field])

        return data

    def plot_velocities(
        self,
        ride_segment_id: int,
        frame_st: Optional[int] = None,
        frame_en: Optional[int] = None,
        figsize: tuple = (14, 10),
        height: int = 3,
        width: int = 2,
        save_path: Optional[str] = None,
    ):
        data_to_plot = self.get_velocity_data(ride_segment_id)
        if frame_st is None:
            frame_st = 0
        if frame_en is None:
            frame_en = len(data_to_plot["timestamp"])

        # Create 3x2 figure layout for estimated vs ground truth velocities
        fig, axes = plt.subplots(height, width, figsize=figsize)

        for i, vel_field in enumerate(VELOCITIES_FIELDS):
            col = i // height
            row = i % height
            ax = axes[row, col]

            t = (
                np.array(data_to_plot["timestamp"][frame_st:frame_en])
                - data_to_plot["timestamp"][0]
            )
            ax.plot(
                t,
                data_to_plot[f"{vel_field}_gt"][frame_st:frame_en],
                "r--",
                linewidth=1.5,
                label="GT",
            )
            ax.plot(
                t,
                data_to_plot[f"{vel_field}_est"][frame_st:frame_en],
                "b-",
                linewidth=1.5,
                label="Estimated",
            )
            plot_name = (
                "Linear Velocity " if col == 0 else "Angular Velocity "
            ) + vel_field[-1]
            ax.set_title(f"{plot_name} ({vel_field}) - Estimated vs Ground Truth")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Velocity (m/s)" if col == 0 else "Angular Velocity (rad/s)")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to {save_path}")

        return fig
