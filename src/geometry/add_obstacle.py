from skspatial.objects import Plane, LineSegment, Line
import numpy as np


# Rectangle
# B-----.
# |     |
# A-----C
# Segment
# P-----Q
def find_rectangle_segment_intersection(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, p: np.ndarray, q: np.ndarray
) -> np.ndarray | None:
    # 1. Define the Plane (point on plane and normal vector)
    plane = Plane.from_points(a, b, c)

    # 2. Define the Segment (two endpoints)
    line = Line.from_points(p, q)

    # 3. Find the intersection
    # The intersect_plane method handles the logic of checking if the intersection point
    # lies within the finite bounds of the segment.
    try:
        intersection = plane.intersect_line(line)
    except ValueError:
        return None

    segment = LineSegment(p, q)
    if not segment.contains_point(intersection):
        return None

    on_rect = True
    v = intersection - a
    for proj_end in (b, c):
        basis_vec = proj_end - a
        on_rect &= 0 <= basis_vec.dot(v) <= basis_vec.dot(basis_vec)

    return intersection if on_rect else None


# https://www.dimensions.com/element/ford-transit-van
DEFAULT_FORD_TRANSIT_VAN_DIMS = np.array([5.59, 2.07, 2.09])


# This is slow as fuck
def remove_points_by_obstacle_bbox_3d(
    points: np.ndarray,
    bbox_center: np.ndarray,
    bbox_dims: np.ndarray = DEFAULT_FORD_TRANSIT_VAN_DIMS,
    origin: np.ndarray = np.array([0.0, 0.0, 0.0]),
) -> np.ndarray:
    """
    Filter point cloud by removing points whose segment from origin intersects an obstacle bbox.

    Parameters
    ----------
    points : numpy.ndarray of shape (N, 3)
        Array of points with x, y, z coordinates
    bbox_center : numpy.ndarray of shape (3,)
        Center of the bounding box [x, y, z]
    bbox_dims : numpy.ndarray of shape (3,)
        Dimensions of the bounding box [width, height, depth]
    origin : numpy.ndarray of shape (3,), optional
        Starting point for ray casting (default: [0, 0, 0])

    Returns
    -------
    filtered_points : numpy.ndarray of shape (M, 3) where M <= N
        Point cloud with points intersecting the obstacle bbox removed
    """
    # Calculate half-dimensions from full dimensions
    half_dims = bbox_dims / 2.0

    # Define the 6 faces of the bounding box
    # Each face is defined by 3 points (a, b, c)
    # We'll use the center and half-dims to define each face corner

    # Face corners relative to center
    x_min = bbox_center[0] - half_dims[0]
    x_max = bbox_center[0] + half_dims[0]
    y_min = bbox_center[1] - half_dims[1]
    y_max = bbox_center[1] + half_dims[1]
    z_min = bbox_center[2] - half_dims[2]
    z_max = bbox_center[2] + half_dims[2]

    # A------B
    # |\     |\
    # | \    | \
    # |  D---+--C
    # |  |   |  |
    # |  |   |  |
    # |  |   |  |
    # E--+---F  |
    #  \ |    \ |
    #   \|     \|
    #    H------G
    a = np.array([x_min, y_min, z_max])
    b = np.array([x_min, y_max, z_max])
    c = np.array([x_max, y_max, z_max])
    d = np.array([x_max, y_min, z_max])

    e = np.array([x_min, y_min, z_min])
    f = np.array([x_min, y_max, z_min])
    g = np.array([x_max, y_max, z_min])
    h = np.array([x_max, y_min, z_min])

    # Define the 6 faces (each face has 3 corners)
    all_faces = [(a, b, c), (e, f, g), (c, d, h), (a, b, f), (d, a, e), (b, c, g)]

    # Create segments from origin to each point in the cloud
    # For each point, check if segment from origin to that point intersects any face
    filtered_indexes = []

    for i, point in enumerate(points):
        obscured_by_obstacle = False

        # Check intersection with each face
        for a, b, c in all_faces:
            # Check for intersection
            intersection = find_rectangle_segment_intersection(a, b, c, origin, point)

            if intersection is not None:
                obscured_by_obstacle = True
                break
        if not obscured_by_obstacle:
            filtered_indexes.append(i)

    # Return filtered points
    return points[filtered_indexes]


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

    bad_points_mask = None
    origin_2d = origin[:2]
    points_2d = points[:, :2]
    for a, b in all_segments:
        #     A
        #     |
        #     |
        # C---+---D
        #     |
        #     |
        #     |
        #     B
        # It could a nice code with for loop and a function
        # But it was also slow as fuck
        # This thing should be vectorized by numpy and run in C
        mask = (
            np.cross(origin_2d - a, b - a) * np.cross(points_2d - a, b - a) <= 0
        ) & (
            np.cross(a - origin_2d, points_2d - origin_2d)
            * np.cross(b - origin_2d, points_2d - origin_2d)
            <= 0
        )

        if bad_points_mask is None:
            bad_points_mask = mask
        else:
            bad_points_mask |= mask

    # Return filtered points
    return points[~bad_points_mask]
