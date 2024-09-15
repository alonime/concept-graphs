import numpy as np
import open3d as o3d


def _rotate_point_cloud(point_cloud, rotation_matrix):
    return np.dot(point_cloud, rotation_matrix.T)


def _get_rotation_matrix(rx):
    rotation_matrices = np.array(
        [
            [np.cos(rx), -np.sin(rx), np.zeros_like(rx)],
            [np.sin(rx), np.cos(rx), np.zeros_like(rx)],
            [np.zeros_like(rx), np.zeros_like(rx), np.ones_like(rx)],
        ]
    )
    return rotation_matrices


def _compute_score(pc, angle):
    rpc = _rotate_point_cloud(pc, _get_rotation_matrix(np.pi * angle / 180))
    x_max, y_max, z_max = np.max(rpc, axis=0)
    x_min, y_min, z_min = np.min(rpc, axis=0)
    return (x_max - x_min) * (y_max - y_min) * (z_max - z_min)


def _compute_scores(pc, low, high, res):
    angles = np.arange(low, high + res, res) % 360
    scores = []
    for angle in angles:
        scores.append(_compute_score(pc, angle))
    top_indices = np.argsort(scores)[:2]
    i, j = top_indices.tolist()
    if i == 0 and j == len(angles) - 1:
        j = len(angles) // 2
    if j == 0 and i == len(angles) - 1:
        i = len(angles) // 2
    low, high = np.sort([angles[i], angles[j]])
    return (low + 180) % 360 - 180, (high + 180) % 360 - 180


def estimate_rotated_bbox(
    pc: np.ndarray, max_angle: int = 45, step: int = 1, res: int = 15, dec: int = 3
) -> o3d.geometry.OrientedBoundingBox:
    pc = pc[::dec]
    low, high = -max_angle, max_angle
    while high - low > step:
        low, high = _compute_scores(pc, low, high, res=res)
        res = max(res // 2, 1)
    best_angle = low
    rx = np.pi * best_angle / 180
    best_rmatrix = _get_rotation_matrix(rx)
    rpcs = _rotate_point_cloud(pc, best_rmatrix)
    x_max, y_max, z_max = np.max(pc, axis=0)
    x_min, y_min, z_min = np.min(pc, axis=0)
    bb = o3d.geometry.OrientedBoundingBox()
    bb.center = (x_max / 2 + x_min / 2, y_max / 2 + y_min / 2, z_max / 2 + z_min / 2)
    x_max, y_max, z_max = np.max(rpcs, axis=0)
    x_min, y_min, z_min = np.min(rpcs, axis=0)
    bb.extent = (x_max - x_min, y_max - y_min, z_max - z_min)
    bb.R = best_rmatrix.T
    return bb


def transform_points(points: np.ndarray, transform_mat: np.ndarray):
    assert (
        transform_mat.shape == (3, 3) or transform_mat.shape == (4, 4) or transform_mat.shape == (3, 4)
    ), "expected transformation matrix of size (3,3), (4,4) or (3,4)"
    return np.matmul(points, transform_mat[:3, :3].T) + transform_mat[:3, -1]


def get_rotation_matrix_from_plane(plane, up_dir=None):
    # Normal vector of the original plane
    normal_vector = plane[:3]

    # Define the up-axis
    up_dir = np.array([0, 1, 0]) if up_dir is None else np.asarray(up_dir)

    # Calculate the angle of rotation
    cos_angle = np.dot(normal_vector, up_dir) / (np.linalg.norm(normal_vector) * np.linalg.norm(up_dir))
    angle = np.arccos(cos_angle)

    # Calculate the axis of rotation
    rotation_axis = np.cross(normal_vector, up_dir)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    # Create the rotation matrix
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    x, y, z = rotation_axis
    rotation_matrix = np.array(
        [
            [t * x * x + c, t * x * y - z * s, t * x * z + y * s],
            [t * x * y + z * s, t * y * y + c, t * y * z - x * s],
            [t * x * z - y * s, t * y * z + x * s, t * z * z + c],
        ]
    )
    return rotation_matrix.T


def get_calib_from_floor_plane(plane_model, forward_dir=None):
    height = np.abs(plane_model[-1]) / np.power(plane_model[:-1], 2).sum()

    # Define the left-axis
    forward_dir = np.array([0, 0, 1]) if forward_dir is None else np.asarray(forward_dir)
    dp = np.dot(forward_dir, plane_model[:3]) / np.linalg.norm(plane_model[:3]) / np.linalg.norm(forward_dir)
    pitch = np.arcsin(dp) / np.pi * 180
    return height, pitch
