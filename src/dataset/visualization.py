import plotly.graph_objects as go
import numpy as np
import json
from typing import Dict, List, Optional
import matplotlib.pyplot as plt


def draw_lidar_cloud(
    input_points: np.ndarray,
    n_points: int = 10000,
    plot_name: str = "LiDAR Points",
    title: str = "LiDAR Point Cloud Visualization",
):
    n_points = min(len(input_points), n_points)
    points_downsampled = input_points[
        np.random.choice(input_points.shape[0], n_points, replace=False)
    ]

    # Create a 3D scatter plot for the LiDAR point cloud
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points_downsampled[:, 0],
                y=points_downsampled[:, 1],
                z=points_downsampled[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=points_downsampled[:, 2],  # Color points by height
                    colorscale="Viridis",
                    opacity=0.7,
                ),
                name=plot_name,
            )
        ]
    )

    # Update layout with custom styling
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, family="Arial")),
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            aspectmode="data",
        ),
        height=900,
        width=1200,
    )
    return fig


VELOCITIES_FIELDS = ["vx", "vy", "vz", "wx", "wy", "wz"]

class PostprocessingWrapper:
    """
    Wrapper class for post-processing and visualizing LiDAR odometry results.

    Takes a JSON file path containing processed ride results from the pipeline,
    provides methods to access velocity data and generate plots comparing
    estimated vs ground truth velocities.

    Attributes:
    -----------
    results : List[Dict]
        List of result dictionaries from the pipeline, each containing:
        - timestamp: float
        - gt: dict with velocities (vx, vy, vz, wx, wy, wz)
        - estimate: dict with velocities and optionally iterations
    """

    def __init__(self, input_file_path: str):
        """
        Initialize the PostprocessingWrapper.

        Parameters:
        -----------
        input_file_path : str
            Path to the JSON file containing processed ride results from pipeline.
        """
        self.input_file_path = input_file_path

        # Load results from JSON file
        with open(input_file_path, "r") as f:
            self.bag = json.load(f)
        self.data = self.get_velocity_data()

    def get_velocity_data(self) -> Dict[str, List[float]]:
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

        for result in self.bag:
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
        frame_st: Optional[int] = None,
        frame_en: Optional[int] = None,
        figsize: tuple = (14, 10),
        height:int = 3,
        width:int = 2,
        save_path: Optional[str] = None,
    ):
        if (frame_st is None):
            frame_st = 0
        if (frame_en is None):
            frame_en = len(self.data["timestamp"])

        # Create 3x2 figure layout for estimated vs ground truth velocities
        fig, axes = plt.subplots(height, width, figsize=figsize)

        for i, vel_field in enumerate(VELOCITIES_FIELDS):
            col = i // height
            row = i % height
            ax = axes[row, col]

            t = np.array( self.data["timestamp"][frame_st:frame_en]) - self.data["timestamp"][0]
            ax.plot(t, self.data[f"{vel_field}_gt"][frame_st : frame_en], "r--", linewidth=1.5, label="GT")
            ax.plot(t, self.data[f"{vel_field}_est"][frame_st : frame_en], "b-", linewidth=1.5, label="Estimated")
            plot_name = ("Linear Velocity " if col == 0 else "Angular Velocity ") + vel_field[-1]
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
