from skspatial.objects import Plane, LineSegment, Line
import numpy as np


# https://www.dimensions.com/element/ford-transit-van
DEFAULT_FORD_TRANSIT_VAN_DIMS = np.array([5.59, 2.07, 2.09])


def check_segment_line_intersection(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
):
    #             A
    #            /
    #           /
    # ----C----+----D----
    #         /
    #        /
    #       B
    return np.cross(a - c, d - c) * np.cross(b - c, d - c) <= 0


def check_segment_segment_intersection(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
):
    #         A
    #        /
    #       /
    # C----+----D
    #     /
    #    /
    #   B
    return check_segment_line_intersection(
        a, b, c, d
    ) & check_segment_line_intersection(c, d, a, b)


def remove_points_by_obstacle_bbox_2d(
    points: np.ndarray,
    bbox_center: np.ndarray,
    bbox_dims: np.ndarray = DEFAULT_FORD_TRANSIT_VAN_DIMS,
    origin: np.ndarray = np.array([0.0, 0.0, 0.0]),
) -> np.ndarray:
    # Calculate half-dimensions from full dimensions
    half_dims = bbox_dims / 2.0

    # Face corners relative to center
    x_min = bbox_center[0] - half_dims[0]
    x_max = bbox_center[0] + half_dims[0]
    y_min = bbox_center[1] - half_dims[1]
    y_max = bbox_center[1] + half_dims[1]

    # A------B
    # |      |
    # |      |
    # D------C
    a = np.array([x_min, y_min])
    b = np.array([x_min, y_max])
    c = np.array([x_max, y_max])
    d = np.array([x_max, y_min])

    # Define the 4 lines
    all_segments = [(a, b), (b, c), (c, d), (d, a)]

    bad_points_mask = np.zeros(points.shape[0], dtype=bool)
    origin_2d = origin[:2]
    points_2d = points[:, :2]
    for a, b in all_segments:
        mask = check_segment_segment_intersection(a, b, origin_2d, points_2d)
        bad_points_mask |= mask

    # Return filtered points
    return points[~bad_points_mask]
