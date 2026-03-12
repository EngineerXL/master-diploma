import numpy as np


def get_voxel(point: np.ndarray, voxel_size: float) -> tuple[int, int, int]:
    """Convert 3D point to discrete voxel coordinates."""
    return tuple(int(np.floor(el / voxel_size)) for el in point)


def voxelize(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Filter points keeping only one representative per voxel."""
    voxels: set[tuple[int, int, int]] = set()
    result: list[np.ndarray] = []
    for point in points:
        p_voxel = get_voxel(point, voxel_size)
        if p_voxel not in voxels:
            voxels.add(p_voxel)
            result.append(point)
    return np.array(result)
