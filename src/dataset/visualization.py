import plotly.graph_objects as go
import numpy as np


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
