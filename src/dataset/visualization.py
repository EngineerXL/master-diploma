import plotly.graph_objects as go
import numpy as np
import io
from typing import List
import matplotlib.pyplot as plt
from PIL import Image


def plot_lidar_clouds_interactive(
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


def plot_lidar_clouds_animation(
    input_clouds: List[np.ndarray],
    max_points_per_plot: int = 10000,
    title: str = "Voxelized LiDAR Point Cloud",
    gif_path: str = "voxelized_lidar_animation.gif",
    frame_duration_ms: int = 500,  # 0.5 seconds per frame
    fig_width: int = 18,
    fig_height: int = 9,
    dpi: int = 300,
    point_size_3d: int = 11,
    point_size_2d: int = 16,
    x_low: float = -40.0,
    x_high: float = 40.0,
    y_low: float = -40.0,
    y_high: float = 40.0,
    z_low: float = -2.5,
    z_high: float = 7.5,
    v_low: float = -2.5,
    v_high: float = 5.0,
    cmap: str = "inferno",
    alpha: float = 0.8,
):
    frames = []
    for i, points in enumerate(input_clouds):
        mask = (
            (x_low <= points[:, 0])
            & (points[:, 0] <= x_high)
            & (y_low <= points[:, 1])
            & (points[:, 1] <= y_high)
            & (z_low <= points[:, 2])
            & (points[:, 2] <= z_high)
        )
        filtered_points = points[mask]

        n_points_to_show = min(len(filtered_points), max_points_per_plot)
        points_downsampled = filtered_points[
            np.random.choice(filtered_points.shape[0], n_points_to_show, replace=False)
        ]
        points_sorted = points_downsampled[points_downsampled[:, 2].argsort()]

        xs, ys, zs = (
            points_sorted[:, 0],
            points_sorted[:, 1],
            points_sorted[:, 2],
        )

        fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
        fig.suptitle(f"{title} - Frame {i+1}")

        # Create 3D scatter plot using matplotlib
        ax3d = fig.add_subplot(1, 3, (1, 2), projection="3d")
        ax3d.scatter(
            xs,
            ys,
            zs,
            c=zs,
            vmin=v_low,
            vmax=v_high,
            cmap=cmap,
            s=point_size_3d,
            alpha=alpha,
            marker="o",
            linewidth=0,
        )

        # Set labels and title
        ax3d.set_xlabel("X (m)")
        ax3d.set_ylabel("Y (m)")
        ax3d.set_zlabel("Z (m)")

        # Set aspect ratio
        ax3d.set_box_aspect((x_high - x_low, y_high - y_low, z_high - z_low))

        ax3d.set_xlim3d(x_low, x_high)
        ax3d.set_ylim3d(y_low, y_high)
        ax3d.set_zlim3d(z_low, z_high)

        # Adjust view angle for better visualization
        ax3d.view_init(elev=30, azim=165)

        # 2D Swap x and y
        ax2d = fig.add_subplot(1, 3, 3)
        ax2d.scatter(
            ys,
            xs,
            c=zs,
            vmin=v_low,
            vmax=v_high,
            cmap=cmap,
            s=point_size_2d,
            alpha=alpha,
            marker="o",
            linewidth=0,
        )

        # Set labels and title
        ax2d.set_xlabel("Y (m)")
        ax2d.set_ylabel("X (m)")

        # Set aspect ratio
        ax2d.set_aspect("equal")

        ax2d.set_xlim(y_low, y_high)
        ax2d.set_ylim(x_low, x_high)

        # Convert the figure to a PIL Image using BytesIO
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        img = Image.open(buf)
        frames.append(img)

    # Save as GIF with Pillow
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,  # Loop infinitely
    )
