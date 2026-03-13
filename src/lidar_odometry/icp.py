from typing import Optional, Dict
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, RigidTransform


def geman_mcclure_weights(r: np.ndarray, c: float) -> np.ndarray:
    # IRLS weights for Geman-McClure (scaling doesn't change argmin):
    # w(r) = c^2 / (r^2 + c^2)^2
    r2 = r * r
    c2 = c * c
    return c2 / (r2 + c2) ** 2


def rigid_kabsch(
    source: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> RigidTransform:
    # Compute centroids
    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)

    # Center the point clouds
    source_centered = source - source_centroid
    target_centered = target - target_centroid

    # argmin w * || target - R * source ||2
    rot, rssd = Rotation.align_vectors(target_centered, source_centered, weights)
    # target = R * source + T
    translation = target_centroid - rot.apply(source_centroid)
    return RigidTransform.from_components(translation, rot)


def align_point_clouds_icp(
    source: np.ndarray,
    target: np.ndarray,
    init_transform=None,
    max_iterations: int = 50,
    tolerance: float = 1e-7,
    max_correspondence_distance: float | None = None,
    w_func: str | None = None,
):
    """
    Iterative Closest Point (ICP) algorithm using scipy RigidTransform.

    Parameters
    ----------
    source : np.ndarray
        Source point cloud (N, 3)
    target : np.ndarray
        Target point cloud (M, 3)
    init_transform : scipy.spatial.transform.RigidTransform or None
        Initial transformation. If None, identity transform is used.
    max_iterations : int
        Maximum number of ICP iterations (default: 50)
    tolerance : float
        Convergence tolerance for RMS error (default: 1e-7)
    max_correspondence_distance : float or None
        Maximum distance for point correspondence. If None, all points are matched.

    Returns
    -------
    transform : scipy.spatial.transform.RigidTransform
        The computed rigid transformation
    info : dict
        Information dictionary with {"iterations": iterations}
    """
    # Ensure inputs are float64 for numerical stability
    source = source.astype(np.float64)
    target = target.astype(np.float64)

    # Initialize transform
    if init_transform is None:
        transform = RigidTransform.from_exp_coords(np.zeros(6))
    else:
        transform = init_transform

    # Build KD-tree on target for efficient nearest neighbor search
    tree = cKDTree(target)

    iterations = 0
    converged = False

    for iteration in range(max_iterations):
        iterations += 1

        aligned_source = transform.apply(source)

        # Find corresponding points in target
        distances, indices = tree.query(aligned_source, k=1)

        # Apply correspondence distance filter if specified
        if max_correspondence_distance is not None:
            mask = distances <= max_correspondence_distance
            if not mask.any():
                # No valid correspondences found
                break
            aligned_source = aligned_source[mask]
            distances = distances[mask]
            target_filtered = target[indices[mask]]
        else:
            aligned_source = aligned_source
            distances = distances
            target_filtered = target[indices]

        if w_func == "geman-mcclure":
            c_from_median_scale = 2
            med = float(np.median(distances))
            c_it = max(1e-12, c_from_median_scale * med)

            # IRLS weights + weighted Kabsch
            weights = geman_mcclure_weights(distances, c_it)
        else:
            weights = np.ones(target_filtered.shape[0])

        # Compute optimal rotation and translation using Kabsch algorithm
        delta_transform = rigid_kabsch(aligned_source, target_filtered, weights)

        transform = delta_transform * transform

    return transform, {"iterations": iterations, "converged": converged}
