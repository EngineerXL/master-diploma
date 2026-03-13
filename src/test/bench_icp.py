import os
import sys

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
from scipy.spatial.transform import Rotation, RigidTransform
from lidar_odometry.icp import align_point_clouds_icp

def _make_synthetic_data(
    n_inliers=2000,
    n_outliers=400,
    noise_sigma=0.01,
    outlier_box=2.5,
    seed=0,
):
    rng = np.random.default_rng(seed)

    # Source: points on a slightly non-symmetric shape (ellipsoid-ish) to avoid degeneracy
    # Sample random directions and radii
    X = rng.normal(size=(n_inliers, 3))
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    radii = np.array([1.0, 0.6, 0.3])
    source = X * radii

    # Ground-truth transform
    rot_gt = Rotation.from_euler("zyx", [25, -10, 15], degrees=True)
    R_gt = rot_gt.as_matrix()
    t_gt = np.array([10.3, -0.2, 0.15])

    RR = RigidTransform.from_exp_coords([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    # target_clean = source @ R_gt.T + t_gt
    print(source)
    target_clean = RR.apply(source)

    target = target_clean + rng.normal(scale=noise_sigma, size=target_clean.shape)

    # Add outliers to target
    outliers = rng.uniform(low=-outlier_box, high=outlier_box, size=(n_outliers, 3))
    target_with_outliers = np.vstack([target, outliers])

    return source, target_with_outliers, (R_gt, t_gt)


def _rotation_angle_error_deg(R_est, R_gt):
    R_rel = R_est @ R_gt.T
    # numerical safety
    cosang = (np.trace(R_rel) - 1.0) / 2.0
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def main():
    np.set_printoptions(precision=4, suppress=True)

    source, target, (R_gt, t_gt) = _make_synthetic_data(
        n_inliers=2500,
        n_outliers=800,
        noise_sigma=0.01,
        outlier_box=3.0,
        seed=42,
    )

    # Initial guess (slightly off)
    rot0 = Rotation.from_euler("zyx", [10, 5, -5], degrees=True)
    t0 = np.array([0, 0, 0])
    T0 = RigidTransform.from_components(t0, rot0)

    T_est, info = align_point_clouds_icp(
        source,
        target,
        init_transform=T0,
        max_iterations=100,
        tolerance=1e-9,
        max_correspondence_distance=1.0,  # helps reject far matches initially
        w_func="geman-mcclure",
    )

    R_est = T_est.rotation.as_matrix()
    t_est = np.asarray(T_est.translation)
    T_mat = T_est.as_matrix()

    print("=== Ground truth ===")
    print("R_gt:\n", R_gt)
    print("t_gt:", t_gt)

    print("\n=== Estimated ===")
    print("R_est:\n", R_est)
    print("t_est:", t_est)
    print("T_est:\n", T_mat)
    print("Velocities!!!: \n", RigidTransform.from_matrix(T_mat).as_exp_coords())

    ang_err = _rotation_angle_error_deg(R_est, R_gt)
    t_err = float(np.linalg.norm(t_est - t_gt))
    print("\n=== Errors ===")
    print(f"Rotation error (deg): {ang_err:.6f}")
    print(f"Translation L2 error: {t_err:.6f}")
    print("Iterations:", info["iterations"])


if __name__ == "__main__":
    main()
