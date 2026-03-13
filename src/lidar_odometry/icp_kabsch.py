#!/usr/bin/env python3
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

try:
    from scipy.spatial.transform import (
        RigidTransform,
    )  # SciPy may or may not have it depending on version

    _HAVE_RIGIDTRANSFORM = True
except Exception:
    RigidTransform = None
    _HAVE_RIGIDTRANSFORM = False


def _to_rigid_transform(Rm: np.ndarray, t: np.ndarray):
    rot = Rotation.from_matrix(Rm)
    t = np.asarray(t, dtype=float).reshape(3)

    if _HAVE_RIGIDTRANSFORM:
        # return RigidTransform.from_rotation_translation(rot, t)
        return RigidTransform.from_exp_coords(np.concat([rot.as_rotvec(), t]))
    else:
        T = np.eye(4)
        T[:3, :3] = Rm
        T[:3, 3] = t
        return {"rotation": rot, "translation": t, "matrix": T}


def _apply_transform(points: np.ndarray, Rm: np.ndarray, t: np.ndarray) -> np.ndarray:
    return points @ Rm.T + t


def _geman_mcclure_weights(r: np.ndarray, c: float) -> np.ndarray:
    # IRLS weights for Geman-McClure (scaling doesn't change argmin):
    # w(r) = c^2 / (r^2 + c^2)^2
    r2 = r * r
    c2 = c * c
    return c2 / (r2 + c2) ** 2


def _weighted_kabsch(P: np.ndarray, Q: np.ndarray, w: np.ndarray):
    """
    Weighted rigid alignment: min sum_i w_i ||R P_i + t - Q_i||^2
    """
    w = np.asarray(w, dtype=float).reshape(-1, 1)
    wsum = float(np.sum(w))
    if wsum <= 0:
        raise ValueError("Sum of weights must be > 0")

    muP = np.sum(w * P, axis=0) / wsum
    muQ = np.sum(w * Q, axis=0) / wsum

    X = P - muP
    Y = Q - muQ

    H = X.T @ (w * Y)
    U, _, Vt = np.linalg.svd(H)

    Rm = Vt.T @ U.T
    if np.linalg.det(Rm) < 0:
        Vt[-1, :] *= -1
        Rm = Vt.T @ U.T

    t = muQ - Rm @ muP
    return Rm, t


def align_points_geman_mcclure(
    source: np.ndarray,
    target: np.ndarray,
    init_transform=None,
    max_iterations: int = 50,
    tolerance: float = 1e-7,
    max_correspondence_distance: float | None = None,
    c: float | None = None,
    c_from_median_scale: float = 2.0,
    min_pairs: int = 6,
    return_history: bool = False,
):
    """
    Point-to-point ICP with robust Geman-McClure loss via IRLS.
    Finds transform mapping source -> target.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if (
        source.ndim != 2
        or source.shape[1] != 3
        or target.ndim != 2
        or target.shape[1] != 3
    ):
        raise ValueError("source and target must be Nx3 arrays")

    # init (R,t)
    Rm = np.eye(3)
    t = np.zeros(3)

    if init_transform is not None:
        if _HAVE_RIGIDTRANSFORM and isinstance(init_transform, RigidTransform):
            Rm = init_transform.rotation.as_matrix()
            t = np.asarray(init_transform.translation, dtype=float).reshape(3)
        elif isinstance(init_transform, dict) and "matrix" in init_transform:
            T0 = np.asarray(init_transform["matrix"], dtype=float)
            Rm, t = T0[:3, :3], T0[:3, 3]
        else:
            T0 = np.asarray(init_transform, dtype=float)
            if T0.shape != (4, 4):
                raise ValueError(
                    "init_transform must be SciPy RigidTransform, dict with 'matrix', or 4x4"
                )
            Rm, t = T0[:3, :3], T0[:3, 3]

    tree = cKDTree(target)

    prev_error = np.inf

    iterations = 0
    for it in range(max_iterations):
        src_x = _apply_transform(source, Rm, t)

        dists, idx = tree.query(src_x, k=1)
        tgt_nn = target[idx]

        if max_correspondence_distance is not None:
            mask = dists <= float(max_correspondence_distance)
            P = src_x[mask]
            Q = tgt_nn[mask]
            r = dists[mask]
        else:
            P, Q, r = src_x, tgt_nn, dists

        if P.shape[0] < min_pairs:
            break

        # robust parameter
        if c is None:
            med = float(np.median(r))
            c_it = max(1e-12, c_from_median_scale * med)
        else:
            c_it = float(c)

        # IRLS weights + weighted Kabsch
        w = _geman_mcclure_weights(r, c_it)
        dR, dt = _weighted_kabsch(P, Q, w)

        # compose update: new = (dR,dt) o old
        Rm = dR @ Rm
        t = dR @ t + dt

        iterations += 1

        # robust objective (mean rho)
        rho = (r * r) / (r * r + c_it * c_it)
        err = float(np.mean(rho))

        if abs(prev_error - err) < tolerance:
            prev_error = err
            break
        prev_error = err

    transform = _to_rigid_transform(Rm, t)
    info = {"iterations": iterations}
    return transform, info


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
        n_outliers=0,
        noise_sigma=0.01,
        outlier_box=3.0,
        seed=42,
    )

    # Initial guess (slightly off)
    rot0 = Rotation.from_euler("zyx", [10, 5, -5], degrees=True).as_matrix()
    t0 = np.array([0, 0, 0])
    T0 = np.eye(4)
    T0[:3, :3] = rot0
    T0[:3, 3] = t0

    T_est, info, hist = align_points_geman_mcclure(
        source,
        target,
        init_transform=T0,
        max_iterations=100,
        tolerance=1e-9,
        max_correspondence_distance=1.3,  # helps reject far matches initially
        c=None,  # auto from median each iter
        c_from_median_scale=2.0,
        return_history=True,
    )

    if _HAVE_RIGIDTRANSFORM and isinstance(T_est, RigidTransform):
        R_est = T_est.rotation.as_matrix()
        t_est = np.asarray(T_est.translation)
        T_mat = T_est.as_matrix()
    else:
        R_est = T_est["rotation"].as_matrix()
        t_est = T_est["translation"]
        T_mat = T_est["matrix"]

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
    if hist:
        print("Final stats:", hist[-1])

    # Optional: show convergence (last few)
    print("\nLast 5 iterations:")
    for h in hist[-5:]:
        print(h)


if __name__ == "__main__":
    main()
