import numpy as np
import pandas as pd
import json
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from src.dataset.pipeline import DATA_OUTPUT_ROOT
from scipy.stats import ttest_rel, normaltest


VELOCITIES_FIELDS = ["vx", "vy", "vz", "wx", "wy", "wz"]
METRICS_NAMES = ["mean", "std", "MSE", "MAE", "q95_abs", "RMSE"]


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
        try:
            with open(self.bag_paths[ride_segment_id], "r") as f:
                bag_data = json.load(f)
            return bag_data
        except:
            print(
                f"Error opening file {ride_segment_id}: {self.bag_paths[ride_segment_id]}"
            )
            raise

    def get_velocity_data(self, ride_segment_id: int) -> Dict[str, np.ndarray]:
        """
        Extract all velocity data from results into organized lists.

        Returns:
        --------
        Dict[str, np.ndarray]
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

        data_np = {x: np.array(y) for x, y in data.items()}
        return data_np

    def get_velocity_errors(
        self,
        ride_segment_id: int | None = None,
        outliers_threshold: int | None = None,
    ) -> Dict[str, np.ndarray]:
        segments = range(len(self)) if ride_segment_id is None else [ride_segment_id]
        errors = {field: np.array([]) for field in VELOCITIES_FIELDS}
        for segment in segments:
            data = self.get_velocity_data(segment)
            for field in VELOCITIES_FIELDS:
                err_part = data[f"{field}_est"] - data[f"{field}_gt"]
                errors[field] = np.concatenate([errors[field], err_part])
        if outliers_threshold is not None:
            return {
                field: self.remove_outliers_zscore(errors[field], outliers_threshold)
                for field in VELOCITIES_FIELDS
            }
        return errors

    def find_velocity_outlier_ride(self, field: str, threshold: float) -> list:
        outliers = []
        for id, path in enumerate(self.bag_paths):
            errors = self.get_velocity_errors(id)
            max_abs_err = np.max(np.abs(errors[field]))
            if max_abs_err > threshold:
                outliers.append((id, path))
        return outliers

    def __len__(self):
        """Returns the number of bags in the postprocessing wrapper."""
        return len(self.bag_paths)

    def get_ride_metrics(self, ride_segment_id: int | None = None) -> dict:
        segments = range(len(self)) if ride_segment_id is None else [ride_segment_id]
        metrics = dict()
        for field in VELOCITIES_FIELDS:
            metrics[field] = {key: [] for key in METRICS_NAMES}

        for segment in segments:
            # Get raw errors (estimated - ground truth) - already combined by get_velocity_errors
            errors = self.get_velocity_errors(segment)

            # Calculate metrics for each velocity component
            for field in VELOCITIES_FIELDS:
                # Calculate metrics
                metrics[field]["mean"].append(np.mean(errors[field]))
                metrics[field]["std"].append(np.std(errors[field]))
                metrics[field]["MSE"].append(np.mean(np.square(errors[field])))
                metrics[field]["MAE"].append(np.mean(np.abs(errors[field])))
                metrics[field]["q95_abs"].append(
                    np.quantile(np.abs(errors[field]), 0.95)
                )
                metrics[field]["RMSE"].append(
                    np.sqrt(np.mean(np.square(errors[field])))
                )

        result = dict()
        for field in VELOCITIES_FIELDS:
            result[field] = {
                key: np.array(metrics[field][key]) for key in METRICS_NAMES
            }
        return result

    def get_ride_avg_metrics_df(
        self, ride_segment_id: int | None = None
    ) -> pd.DataFrame:
        metrics = self.get_ride_metrics(ride_segment_id)
        avg_metrics = {}
        for field in VELOCITIES_FIELDS:
            avg_metrics[field] = {
                key: np.mean(metrics[field][key]) for key in METRICS_NAMES
            }
        # Create DataFrame with metric types as rows and velocity components as columns
        df = pd.DataFrame(avg_metrics).T
        return df

    def plot_velocities(
        self,
        ride_segment_id: int,
        frame_st: Optional[int] = None,
        frame_en: Optional[int] = None,
        figsize: tuple = (14, 10),
        height: int = 3,
        width: int = 2,
        save_path: Optional[str] = None,
        compare_against=None,
    ):
        data_to_plot = self.get_velocity_data(ride_segment_id)
        if frame_st is None:
            frame_st = 0
        if frame_en is None:
            frame_en = len(data_to_plot["timestamp"])

        if compare_against is not None:
            data_to_compare = compare_against.get_velocity_data(ride_segment_id)

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
            if compare_against is not None:
                ax.plot(
                    t,
                    data_to_compare[f"{vel_field}_est"][frame_st:frame_en],
                    "g-",
                    linewidth=1.5,
                    label="Estimate B",
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

    def remove_outliers_zscore(self, data: np.ndarray, threshold=3):
        # Calculate the mean and standard deviation
        mean = np.mean(data)
        std = np.std(data)

        # Calculate Z-scores for each data point
        z_scores = np.abs((data - mean) / std)

        # Filter out data points where the absolute Z-score is greater than the threshold
        cleaned_data = data[z_scores <= threshold]
        return cleaned_data

    def plot_velocity_errors(
        self,
        ride_segment_id: int | None = None,
        figsize: tuple = (14, 10),
        bins: int = 50,
        height: int = 3,
        width: int = 2,
        save_path: Optional[str] = None,
        outliers_threshold: int | None = None,
        with_normal_distribution: bool = False,
    ):
        """
        Plot error histograms for every velocity component.

        Parameters:
        ----------
        ride_segment_id : int or None
            Ride segment ID to process. If None, processes all segments.
        figsize : tuple
            Figure size (width, height).
        bins : int
            Number of histogram bins.
        save_path : str or None
            Path to save the figure. If None, figure is not saved.

        Returns:
        --------
        matplotlib.figure.Figure
            The created figure object.
        """
        errors = self.get_velocity_errors(ride_segment_id, outliers_threshold)

        # Create figure with 2 columns (linear and angular velocities) and 3 rows
        fig, axes = plt.subplots(height, width, figsize=figsize)

        for i, field in enumerate(VELOCITIES_FIELDS):
            col = i // height
            row = i % height
            ax = axes[row, col]

            # Plot histogram for this velocity component error
            ax.hist(
                errors[field],
                bins=bins,
                alpha=0.7,
                color="steelblue",
                edgecolor="black",
                linewidth=0.5,
                density=True,
            )

            if with_normal_distribution:
                # Calculate mean and std for normal distribution curve
                mean = np.mean(errors[field])
                std = np.std(errors[field])

                # Create x values spanning the histogram range
                x_min, x_max = ax.get_xlim()
                x = np.linspace(x_min, x_max, 100)

                # Calculate y values using normal distribution formula
                y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(
                    -0.5 * ((x - mean) ** 2) / (std**2)
                )

                # Plot the normal distribution curve as red line
                ax.plot(x, y, "r-", linewidth=2, label="Normal Distribution")

            # Add vertical line at zero error
            ax.axvline(
                x=0, color="red", linestyle="--", linewidth=1.0, label="Zero Error"
            )

            # Set title and labels
            plot_name = (
                "Linear Velocity Errors" if col == 0 else "Angular Velocity Errors"
            )
            ax.set_title(f"{plot_name} ({field})")
            ax.set_xlabel("Error")
            ax.set_ylabel("Frequency")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to {save_path}")

        return fig


def ttest_related_ride_metrics(
    config_A: str,
    config_B: str,
    alternative: str = "greater",
    ride_segment_id: int | None = None,
    confidence_alpha: float = 0.05,
) -> pd.DataFrame:
    wrapper_A = PostprocessingWrapper(config_A)
    wrapper_B = PostprocessingWrapper(config_B)
    if wrapper_A.ride_segments != wrapper_B.ride_segments:
        raise RuntimeError(
            "Unable to perform related T-test due to different ride segment in configs!"
        )
    metrics_A = wrapper_A.get_ride_metrics(ride_segment_id)
    metrics_B = wrapper_B.get_ride_metrics(ride_segment_id)

    result = dict()
    for field in VELOCITIES_FIELDS:
        result[f"{field}_avg_A"] = {
            key: np.mean(metrics_A[field][key]) for key in METRICS_NAMES
        }
        result[f"{field}_avg_B"] = {
            key: np.mean(metrics_B[field][key]) for key in METRICS_NAMES
        }
        result[f"{field}_abs_diff"] = {
            key: np.mean(metrics_A[field][key] - metrics_B[field][key])
            for key in METRICS_NAMES
        }
        result[f"{field}_rel_diff"] = {
            key: 100
            * (np.mean(metrics_A[field][key]) - np.mean(metrics_B[field][key]))
            / np.mean(metrics_A[field][key])
            for key in METRICS_NAMES
        }
        result[f"{field}_p_value"] = {
            key: ttest_rel(
                metrics_A[field][key],
                metrics_B[field][key],
                alternative=alternative,
            ).pvalue
            for key in METRICS_NAMES
        }

    df = pd.DataFrame(result).T

    P_VALUE_DEFAULT_STYLE = "background-color: black;"

    # Color the p_value row based on confidence_alpha
    def color_pvalue(val):
        result = []
        for elem in val:
            style = P_VALUE_DEFAULT_STYLE
            if elem < confidence_alpha:
                style += "color: green;"
            if alternative != "two-sided" and elem > 1 - confidence_alpha:
                style += "color: red;"
            result.append(style)
        return result

    p_value_rows_subset = (
        [f"{field}_p_value" for field in VELOCITIES_FIELDS],
        slice(None),
    )
    styled_df = df.style.apply(color_pvalue, axis="columns", subset=p_value_rows_subset)

    return styled_df
