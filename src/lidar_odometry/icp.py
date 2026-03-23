from typing import Optional, Dict
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, RigidTransform


def geman_mcclure_weights(distances_e: np.ndarray, sigma: float) -> np.ndarray:
    """Compute Geman-McClure robust weights for outlier rejection.

    Args:
        distances_e: Euclidean distances between corresponding points.
        sigma: Scale parameter controlling weight decay with distance.

    Returns:
        Array of weights where larger distances receive smaller weights.
    """
    # ro(e) = (0.5 * e^2) / (sigma / 3 + e ^ 2)
    distances_e_2 = np.square(distances_e)
    return 0.5 / (sigma / 3 + distances_e_2)


def rigid_kabsch(
    source: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> RigidTransform:
    """Compute optimal rigid transform using Kabsch algorithm with weighted points.

    Args:
        source: Source point cloud coordinates (N, 3).
        target: Target point cloud coordinates (M, 3).
        weights: Weights for each correspondence.

    Returns:
        RigidTransform containing the optimal rotation and translation.
    """
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


def build_correspondances(
    source: np.ndarray,
    target: np.ndarray,
    target_tree: cKDTree,
    max_correspondence_distance_tau_t: float,
):
    """Find corresponding points in the target point cloud.

    Args:
        source: Source point cloud coordinates (N, 3).
        target: Target point cloud coordinates (M, 3).
        target_tree: Pre-built KD-tree for the target point cloud.
        max_correspondence_distance_tau_t: Maximum allowed distance for valid correspondences.

    Returns:
        Tuple of (source_corr, target_corr, distances) for valid correspondences.
    """
    # Find corresponding points in target
    distances, indices = target_tree.query(source, k=1)

    # Apply correspondence distance filter if specified
    mask = distances <= max_correspondence_distance_tau_t
    if not mask.any():
        raise RuntimeError("No valid correspondences found")
    source_corr = source[mask]
    distances = distances[mask]
    target_corr = target[indices[mask]]
    return source_corr, target_corr, distances


def align_point_clouds_icp(
    source: np.ndarray,
    target: np.ndarray,
    displacement_deviation_sigma_t: float,
    max_correspondence_distance_tau_t: float,
    init_transform: RigidTransform = RigidTransform.identity(),
    max_iterations: int = 50,
    tolerance_gamma: float = 1e-4,
):
    """Iteratively align source point cloud to target using ICP.

    Args:
        source: Source point cloud coordinates (N, 3).
        target: Target point cloud coordinates (M, 3).
        displacement_deviation_sigma_t: Sigma parameter for Geman-McClure weights.
        max_correspondence_distance_tau_t: Maximum correspondence distance threshold.
        init_transform: Initial rigid transform (rotation + translation).
        max_iterations: Maximum number of ICP iterations.
        tolerance_gamma: Convergence tolerance for rotation/translation change.

    Returns:
        Tuple of (final_transform, info_dict) where info_dict contains:
            - iterations: Number of iterations performed.
            - converged: Whether convergence was achieved.
            - tau_t: The max_correspondence_distance_tau_t used.
    """
    # Ensure inputs are float64 for numerical stability
    source = source.astype(np.float64)
    target = target.astype(np.float64)

    # Initialize transform
    transform = init_transform

    # Build KD-tree on target for efficient nearest neighbor search
    target_tree = cKDTree(target)

    iterations = 0
    converged = False

    aligned_source = transform.apply(source)

    for iteration in range(max_iterations):
        aligned_source_corr, target_corr, distances = build_correspondances(
            aligned_source, target, target_tree, max_correspondence_distance_tau_t
        )
        weights = geman_mcclure_weights(distances, displacement_deviation_sigma_t)

        # Compute optimal rotation and translation using Kabsch algorithm
        delta_transform = rigid_kabsch(aligned_source_corr, target_corr, weights)

        aligned_source = delta_transform.apply(aligned_source)
        transform = delta_transform * transform
        iterations += 1

        if icp_convergence_criterion_met(delta_transform, tolerance_gamma):
            converged = True
            break

    return transform, {
        "iterations": iterations,
        "converged": bool(converged),
        "tau_t": max_correspondence_distance_tau_t,
    }


def icp_convergence_criterion_met(delta_transform: RigidTransform, tolerance_gamma):
    """Check if ICP has converged based on transform change.

    Args:
        delta_transform: The computed transform for the current iteration.
        tolerance_gamma: Threshold for considering convergence.

    Returns:
        True if the norm of the exponential coordinates is below tolerance.
    """
    velocities = delta_transform.as_exp_coords()
    return np.linalg.norm(velocities) < tolerance_gamma
