from typing import Optional, Dict
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, RigidTransform


def geman_mcclure_weights(distances_e: np.ndarray, sigma: float) -> np.ndarray:
    # ro(e) = (0.5 * e^2) / (sigma / 3 + e ^ 2)
    distances_e_2 = np.square(distances_e)
    return 0.5 / (sigma / 3 + distances_e_2)


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


def build_correspondances(
    source: np.ndarray,
    target: np.ndarray,
    target_tree: cKDTree,
    max_correspondence_distance_tau_t: float,
):
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
    velocities = delta_transform.as_exp_coords()
    return np.linalg.norm(velocities) < tolerance_gamma
