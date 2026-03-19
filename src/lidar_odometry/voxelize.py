import numpy as np


def get_voxel(point: np.ndarray, voxel_size: float, dtype=np.int64) -> np.ndarray:
    """Convert 3D point to discrete voxel coordinates."""
    return np.floor(point / voxel_size).astype(dtype)


def voxelize(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Filter points keeping only one representative per voxel."""
    voxels = get_voxel(points, voxel_size)
    unique_voxels, indices = np.unique(voxels, axis=0, return_index=True, sorted=False)
    return points[indices]
