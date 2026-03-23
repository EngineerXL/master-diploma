import numpy as np


def get_voxel(point: np.ndarray, voxel_size: float, dtype=np.int64) -> np.ndarray:
    """Convert a 3D point to discrete voxel coordinates.

    Args:
        point: A 3D point with shape (3,).
        voxel_size: Size of each voxel in meters.
        dtype: Data type for the voxel indices.

    Returns:
        Array of voxel indices for the given point.
    """
    return np.floor(point / voxel_size).astype(dtype)


def voxelize(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Downsample point cloud by keeping one representative point per voxel.

    Args:
        points: Point cloud coordinates with shape (N, 3).
        voxel_size: Size of each voxel in meters.

    Returns:
        Downsampled point cloud with at most one point per voxel.
    """
    voxels = get_voxel(points, voxel_size)
    unique_voxels, indices = np.unique(voxels, axis=0, return_index=True, sorted=False)
    return points[indices]
