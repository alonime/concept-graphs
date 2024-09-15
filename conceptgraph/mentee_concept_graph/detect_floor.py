import numpy as np
import open3d


DISTANCE_THRESHOLD = 0.2


def detect_floor(
    points: np.ndarray,
    distance_threshold: float = DISTANCE_THRESHOLD,
    pre_horizontal_threshold: float = None,
    post_horiztonal_threshold: float = 0.8,
    apply_pca: bool = True,
):
    # Filter points before PCA using normal
    pc = _create_pointcloud(points)
    if pre_horizontal_threshold is not None:
        is_horizontal = _check_horz(pc, pre_horizontal_threshold)
        pc = pc.select_by_index(np.where(is_horizontal)[0])

    # Use PCA to align axis (the UP direction should be the last component)
    if apply_pca:
        mu, cov = pc.compute_mean_and_covariance()
        eig_mag, eig_vecs = np.linalg.eig(cov)
        order = np.argsort(-eig_mag)
        comps = np.eye(4)
        comps[:3, :3] = eig_vecs[:, order].T
        pc_t = open3d.geometry.PointCloud(pc).translate(-mu)
        pc_t = pc_t.transform(comps)
        # open3d.visualization.draw_geometries([pc_t])
    else:
        pc_t = pc

    # Estimate normals to only consider vertical planes
    if post_horiztonal_threshold is not None:
        is_horizontal = _check_horz(pc_t, post_horiztonal_threshold)
        pc_t = pc_t.select_by_index(np.where(is_horizontal)[0])

    if len(pc_t.points) < 10:
        return np.zeros(points.shape[0], dtype=bool), None

    # Fit floor plane using ransac
    plane_model_t = _estimate_ground_plane(pc_t, distance_threshold)

    # points_t = np.asarray(pc_t.points)
    # is_floor = classify_floor(points_t, plane_model_t, distance_threshold)
    # pc_t.colors = open3d.utility.Vector3dVector(is_floor[:, None]*[-1, 1, 0]+[[1, 0, 0]])
    # ax, ay = np.floor(points_t.min(0)[:2])
    # bx, by = np.ceil(points_t.max(0)[:2])
    # x, y = np.meshgrid(np.arange(ax, bx+0.1, 1), np.arange(ay, by+0.1, 1))
    # z = -(plane_model_t[0] * x + plane_model_t[1] * y + plane_model_t[3])/plane_model_t[2]
    # xyz = np.stack((x, y, z), 0)
    # xyz = xyz.reshape(3,-1).T
    # meshes = []
    # for p in xyz:
    #     box = open3d.geometry.AxisAlignedBoundingBox(p - 0.15, p + 0.15)
    #     mesh = open3d.geometry.TriangleMesh.create_from_oriented_bounding_box(box.get_oriented_bounding_box())
    #     mesh.paint_uniform_color([0, 0, 1])
    #     meshes.append(mesh)
    # open3d.visualization.draw_geometries([pc_t] + meshes)

    # Fix plane model to real world cooridnate system
    if apply_pca:
        plane_model = _fix_plane_model(plane_model_t, mu, comps)
    else:
        plane_model = plane_model_t

    # Classify original pointcloud using plane model
    is_floor = classify_floor(points, plane_model, distance_threshold)

    # pc = open3d.geometry.PointCloud()
    # pc.points = open3d.utility.Vector3dVector(points)
    # pc.colors = open3d.utility.Vector3dVector(is_floor[:, None]*[-1, 1, 0]+[[1, 0, 0]])
    # ax, ay = np.floor(points.min(0)[:2])
    # bx, by = np.ceil(points.max(0)[:2])
    # x, y = np.meshgrid(np.arange(ax, bx+0.1, 1), np.arange(ay, by+0.1, 1))
    # z = -(plane_model_t[0] * x + plane_model_t[1] * y + plane_model_t[3])/plane_model_t[2]
    # xyz = np.stack((x, y, z), 0)
    # xyz = xyz.reshape(3,-1).T
    # meshes = []
    # for p in xyz:
    #     box = open3d.geometry.AxisAlignedBoundingBox(p - 0.15, p + 0.15)
    #     mesh = open3d.geometry.TriangleMesh.create_from_oriented_bounding_box(box.get_oriented_bounding_box())
    #     mesh.paint_uniform_color([0, 0, 1])
    #     meshes.append(mesh)
    # open3d.visualization.draw_geometries([pc] + meshes)

    return is_floor, plane_model


def _create_pointcloud(points_t):
    points_t = points_t.astype(np.float64)  # speeds up Vector3dVector()
    pc_t = open3d.geometry.PointCloud()
    pc_t.points = open3d.utility.Vector3dVector(points_t)
    return pc_t


def _fix_plane_model(plane_model_t, mu, comps):
    plane_model = plane_model_t.copy()
    plane_model[:3] = plane_model[:3] @ comps[:3, :3]
    plane_model[3] -= np.dot(plane_model[:3], mu)

    if plane_model[2] < 0:
        plane_model = -plane_model

    return plane_model


def classify_floor(points_t, plane_model_t, distance_threshold: float = DISTANCE_THRESHOLD):
    proj = points_t @ plane_model_t[:3] + plane_model_t[3]
    is_floor = proj < distance_threshold

    # # DEBUG:
    # floor_points = pc.select_by_index(np.where(is_floor)[0])
    # floor_points.paint_uniform_color([0, 1, 0])
    # other_points = pc.select_by_index(np.where(~is_floor)[0])
    # other_points.paint_uniform_color([1, 0, 0])
    # open3d.visualization.draw_geometries([floor_points.voxel_down_sample(0.1), other_points.voxel_down_sample(0.1)])

    return is_floor


def _estimate_ground_plane(pc_t, distance_threshold):
    plane_model_t, inliers = pc_t.segment_plane(distance_threshold=distance_threshold, ransac_n=3, num_iterations=1000)
    return plane_model_t


def _check_horz(pc, horiztonal_threshold):
    search_param = open3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=15)
    pc.estimate_normals(search_param=search_param, fast_normal_computation=True)
    is_horizontal = np.abs(np.asarray(pc.normals)[:, 2]) > horiztonal_threshold

    # # DEBUG:
    # floor_points = pc.select_by_index(np.where(is_horizontal)[0])
    # floor_points.paint_uniform_color([0, 1, 0])
    # other_points = pc.select_by_index(np.where(~is_horizontal)[0])
    # other_points.paint_uniform_color([1, 0, 0])
    # open3d.visualization.draw_geometries([floor_points.voxel_down_sample(0.1), other_points.voxel_down_sample(0.1)])

    return is_horizontal
