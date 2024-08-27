import plotly.graph_objects as go
from omegaconf import OmegaConf
import shutil
import numpy as np
from PIL import Image
import cv2
import open3d
from pathlib import Path
import pycolmap
import h5py
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from hloc import (
    extract_features,
    match_features,
    pairs_from_exhaustive,
    reconstruction,
    visualization,
    pairs_from_retrieval,
)
from hloc.visualization import plot_images, read_image

from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import copy

import json

from conceptgraph.mentee_concept_graph.generate_pc import generate_pointcloud
from conceptgraph.mentee_concept_graph.show_pc import show_pointcloud

import matplotlib
matplotlib.use('TkAgg')


def geodesic_distance(R1, R2):
    """
    Calculate the geodesic distance (in radians) between two rotation matrices.
    
    Parameters:
    - R1: First rotation matrix (3x3)
    - R2: Second rotation matrix (3x3)
    
    Returns:
    - Geodesic distance (angle in radians) between the two rotation matrices.
    """
    # Ensure the input matrices are numpy arrays
    R1 = np.array(R1)
    R2 = np.array(R2)
    
    # Compute the rotation matrix that rotates R1 to R2
    R = np.dot(R1.T, R2)
    
    # Compute the trace of the rotation matrix
    trace_R = np.trace(R)
    
    # Calculate the angle using the formula
    angle = np.arccos((trace_R - 1) / 2)
    
    # Ensure the angle is within the valid range of arccos
    angle = np.clip(angle, 0, np.pi)
    
    return angle

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    open3d.visualization.draw_geometries([source_temp, target_temp])


def estimate_translation_ransac(P1, P2, min_samples=3, residual_threshold=0.01, max_trials=1000):
    # Create an empty list to store the translation vectors for each axis (x, y, z)
    translation_vector = []

    # Fit RANSAC for each axis (x, y, z)
    for i in range(3):  # Iterate over x, y, z coordinates
        model = RANSACRegressor(base_estimator=LinearRegression(),
                                min_samples=min_samples,
                                residual_threshold=residual_threshold,
                                max_trials=max_trials)
        
        # Fit the model for the i-th coordinate
        model.fit(P1, P2[:, i])
        
        # The translation is given by the intercept of the model
        translation_vector.append(model.estimator_.intercept_)

    # Combine the translations into a single translation vector
    return np.array(translation_vector)



def estimate_rigid_transformation_ransac(P1, P2, min_samples=100, residual_threshold=0.01, max_trials=100):
    best_inliers = 0
    best_translation = None
    best_rotation = None

    n_points = P1.shape[0]

    for _ in range(max_trials):
        # Step 1: Randomly sample a subset of points
        sample_indices = np.random.choice(n_points, min_samples, replace=False)
        P1_sample = P1[sample_indices]
        P2_sample = P2[sample_indices]
        
        # Step 2: Estimate the transformation (rotation and translation) using SVD
        translation, rotation = estimate_rigid_transformation(P1_sample, P2_sample)
        
        # Step 3: Apply the transformation to the full point set
        P1_transformed = (rotation @ P1.T).T + translation
        
        # Step 4: Calculate distances and count inliers
        distances = np.linalg.norm(P1_transformed - P2, axis=1)
        inliers = np.sum(distances < residual_threshold)
        
        # Step 5: Update the best model if current model has more inliers
        if inliers > best_inliers:
            best_inliers = inliers
            best_translation = translation
            best_rotation = rotation
    
    return best_translation, best_rotation

def estimate_rigid_transformation(P1, P2):
    """
    Estimate the rigid transformation (rotation and translation) between two sets of corresponding 3D points
    using the SVD approach.
    """
    # Step 1: Compute the centroids of each point set
    centroid_P1 = np.mean(P1, axis=0)
    centroid_P2 = np.mean(P2, axis=0)
    
    # Step 2: Center the points by subtracting the centroids
    P1_centered = P1 - centroid_P1
    P2_centered = P2 - centroid_P2
    
    # Step 3: Compute the covariance matrix
    H = P1_centered.T @ P2_centered
    
    # Step 4: Compute the Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(H)
    
    # Step 5: Compute the rotation matrix
    rotation = Vt.T @ U.T
    
    # Ensure a proper rotation matrix (det = 1)
    if np.linalg.det(rotation) < 0:
        Vt[-1, :] *= -1
        rotation = Vt.T @ U.T
    
    # Step 6: Compute the translation vector
    translation = centroid_P2 - rotation @ centroid_P1
    
    return translation, rotation

def get_pc(camera_params, depth, rot=None, t=None):
    intrinsic = np.asarray([[camera_params['params'][0], 0, camera_params['params'][2]], 
                            [0, camera_params['params'][1], camera_params['params'][3]], 
                            [0, 0, 1]])
    intrinsic_inv = np.linalg.inv(intrinsic)

    width = camera_params['width']
    height = camera_params['height']

    grid = np.stack(
        np.meshgrid(
            np.arange(width, dtype=np.float32) + 0.5,  # middle of the pixel
            np.arange(height, dtype=np.float32) + 0.5,  # middle of the pixel
            np.ones(1, dtype=np.float32),
        ),
        -1,
    ).reshape(-1, 3)
    cam_grid = np.matmul(intrinsic_inv, grid.T).T

    cam_pc = cam_grid * depth.reshape(-1, 1)
    cam_pc = cam_pc * [1, -1, -1]  # x: right, y: up, z: backward
    if rot is not None and t is not None:
        cam_pc = cam_pc @ rot.T + t
    return cam_pc.reshape(height, width, -1)


def load_depth(path, clip_detstance=3):
    depth = np.load(path) / 1000
    depth[depth < 0] = np.nan

    if not clip_detstance == 0:
        depth[depth > clip_detstance] = np.nan

    return depth


def load_features(feature_path, image_name):
    """Load features for a specific image from the feature file."""
    with h5py.File(feature_path, 'r') as f:
        keypoints = f[image_name]['keypoints'].__array__()
        descriptors = f[image_name]['descriptors'].__array__()
    return keypoints, descriptors

def load_matches(matches_path, image1_name, image2_name):
    """Load matches between two images from the match file."""
    with h5py.File(matches_path, 'r') as f:
        matches = f[f"{image1_name}/{image2_name}"]['matches0'].__array__()
    return matches

def estimate_pose(keypoints1, keypoints2, matches, camera_params, plot_debug=False):
    """Estimate the relative pose between two images."""
    # Select matched keypoints
    ref_idx = np.where(matches != -1)[0]
    matched_keypoints1 = keypoints1[ref_idx, :]
    matched_keypoints2 = keypoints2[matches[ref_idx]]

    if plot_debug:
        img1 = cv2.imread(output_perent / "tmp/ref.png")  # Query image
        img2 = cv2.imread(output_perent / "tmp/target.png")   
        concatenated_img = np.hstack((img1, img2))
        fig = plt.figure()
        plt.imshow(concatenated_img)
        offset = img1.shape[1]  # Width of the first image

        points2_offset = matched_keypoints2 + np.array([offset, 0])

        # Plot the matching points
        for (pt1, pt2) in zip(matched_keypoints1[200:210,:], points2_offset[200:210,:]):
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='yellow', marker='o', markersize=5, linestyle='-', linewidth=1)
        plt.axis('off')
        plt.show()


    # Use COLMAP's essential matrix estimation
    results = pycolmap.essential_matrix_estimation(matched_keypoints1, matched_keypoints2, camera_params, camera_params)

    assert not results == None, "estimation failed"

    r = R.from_quat(results['cam2_from_cam1'].rotation.quat).as_matrix()
    t = results['cam2_from_cam1'].translation
    inliers = results['inliers']
    
    return r, t, inliers


def scale_transition_between_sections(kp1, kp2, 
                                      image1, image2,
                                      depth1, depth2, 
                                      t,
                                      r,
                                      inliers,
                                      matches_pairmatchess_idx, 
                                      camera_params,
                                      ref_pose=np.eye(4),
                                      debug_plot=False):
    
    ref_idx = np.where(matches_pairmatchess_idx != -1)[0]
    matched_keypoints1 = kp1[ref_idx, :]
    matched_keypoints2 = kp2[matches_pairmatchess_idx[ref_idx]]
    matched_keypoints1 = matched_keypoints1[inliers].astype(np.int64)
    matched_keypoints2 = matched_keypoints2[inliers].astype(np.int64)

    pc1 = get_pc(camera_params, depth1)
    pc2 = get_pc(camera_params, depth2, rot=r, t=t)
    
    kp1_pos = pc1[matched_keypoints1[:, 1], matched_keypoints1[:, 0]]
    kp2_pos = pc2[matched_keypoints2[:, 1], matched_keypoints2[:, 0]]
    valid_points = np.vstack([(~np.isnan(kp1_pos)).all(axis=1), (~np.isnan(kp2_pos)).all(axis=1)]).all(axis=0)
    kp1_pos = kp1_pos[valid_points, :]
    kp2_pos = kp2_pos[valid_points, :]

    # Option 1
    translation, rotation = estimate_rigid_transformation_ransac(kp1_pos, kp2_pos)

    threshold = 0.02
    trans_init = np.eye(4)
    trans_init[:3, :3] = r.T @ rotation
    trans_init[:3, -1] = t - translation

    pc1 = pc1.reshape(-1, 3)
    pc2 = get_pc(camera_params, depth2).reshape(-1, 3)
    pc1_valid = ~np.isnan(pc1).any(axis=1)
    pc1_clean = pc1[pc1_valid, :]
    open3d_pc1 = open3d.geometry.PointCloud()
    open3d_pc1.points = open3d.utility.Vector3dVector(pc1_clean)
    open3d_pc1.colors = open3d.utility.Vector3dVector(image1.reshape(-1, 3)[pc1_valid,:] / 255)

    pc2_valid = ~np.isnan(pc2).any(axis=1)
    pc2_clean = pc2[pc2_valid, :]
    open3d_pc2 = open3d.geometry.PointCloud()
    open3d_pc2.points = open3d.utility.Vector3dVector(pc2_clean)
    open3d_pc2.colors = open3d.utility.Vector3dVector(image2.reshape(-1, 3)[pc2_valid,:]/ 255)


    # draw_registration_result(open3d_pc1, open3d_pc1, trans_init)
    # print("Initial alignment")
    # evaluation = open3d.pipelines.registration.evaluate_registration(open3d_pc1, open3d_pc2, threshold, trans_init)
    # print(evaluation)

    reg_p2p = open3d.pipelines.registration.registration_icp(open3d_pc2, open3d_pc1, threshold, trans_init, 
                                                             open3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                             open3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
    # print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    # draw_registration_result(open3d_pc1, open3d_pc2, reg_p2p.transformation)


    # open3d_pc1.estimate_normals()
    # open3d_pc2.estimate_normals()
    # reg_p2l = open3d.pipelines.registration.registration_icp(open3d_pc1, open3d_pc2, threshold, trans_init, open3d.pipelines.registration.TransformationEstimationPointToPlane())
    # print(reg_p2l)
    
    # np.asanyarray(reg_p2p.correspondence_set)
   
    if debug_plot:
        draw_registration_result(open3d_pc2, open3d_pc1, reg_p2p.transformation)
        draw_registration_result(open3d_pc2, open3d_pc1, np.eye(4))

        pc1 = get_pc(camera_params, depth1)
        pc2 = get_pc(camera_params, depth2)
        # tmp_rotation = r.T @ rotation
        # ref_rot = np.eye(3)
        ref_rot = ref_pose[:3, :3]
        tmp_rotation =  reg_p2p.transformation[:3, :3] @ ref_pose[:3, :3]
        # tmp_rotation =  reg_p2p.transformation[:3, :3] 

        
        # tmp_trans = t - translation
        # ref_trans = np.zeros(3)
        ref_trans = ref_pose[:3, -1]
        # tmp_trans = reg_p2p.transformation[:3, -1]
        tmp_trans = ref_pose[:3, -1] + ref_pose[:3, :3] @ reg_p2p.transformation[:3, -1].T

        diist_trans = reg_p2p.transformation[:3, -1]
        print(f"estimated: {np.linalg.norm(reg_p2p.transformation[:3, -1])} | global: {np.linalg.norm(ref_pose[:3, -1] - tmp_trans)}")
        geodesic_distance(ref_pose[:3, :3],  tmp_rotation)
        geodesic_distance(np.eye(3),  reg_p2p.transformation[:3, :3])



        pc1_clean = pc1.reshape(-1, 3)
        pc1_valid = ~np.isnan(pc1_clean).any(axis=1)
        pc1_clean = pc1_clean[pc1_valid, :] @ ref_rot.T + ref_trans
        
        pc2_clean = pc2.reshape(-1, 3)
        pc2_valid = ~np.isnan(pc2_clean).any(axis=1)
        pc2_clean = pc2_clean[pc2_valid, :] @ tmp_rotation.T + tmp_trans

        open3d_pc1 = open3d.geometry.PointCloud()
        open3d_pc1.points = open3d.utility.Vector3dVector(pc1_clean)
        open3d_pc1.colors = open3d.utility.Vector3dVector((image1.reshape(-1, 3)[pc1_valid, :] / 255)[:,[2, 1, 0]])

        open3d_pc2 = open3d.geometry.PointCloud()
        open3d_pc2.points = open3d.utility.Vector3dVector(pc2_clean)
        open3d_pc2.colors = open3d.utility.Vector3dVector((image2.reshape(-1, 3)[pc2_valid, :] / 255)[:,[2, 1, 0]])

        visualizer = open3d.visualization.Visualizer()
        visualizer.create_window()

        visualizer.add_geometry(open3d_pc1)
        visualizer.add_geometry(open3d_pc2)

        visualizer.run()
        visualizer.destroy_window()

        fig, axs = plt.subplots(2,2)
        axs[0,0].imshow(depth1, cmap='viridis')
        axs[0,1].imshow(image1)
        axs[1,0].imshow(depth2, cmap='viridis')
        axs[1,1].imshow(image2)
        fig.show()


    # return reg_p2l.transformation
    # return trans_init
    return reg_p2p.transformation


def refine_transform(init_transform, 
                     camera_params,
                     last_position,
                     curr_image, curr_depth, 
                     last_image, last_depth,
                     max_iter = 200,
                     debug_plot=False):
    
    if last_image is None or last_depth is None:
        print("No last data...")
        return init_transform

    assert init_transform.shape == (4,4)
    
    threshold = 0.02

    pc1 = get_pc(camera_params, last_depth, rot=last_position[:3, :3], t=last_position[:3, -1]).reshape(-1, 3)
    pc2 = get_pc(camera_params, curr_depth).reshape(-1, 3)

    pc1_valid = ~np.isnan(pc1).any(axis=1)
    pc1_clean = pc1[pc1_valid, :]
    open3d_pc1 = open3d.geometry.PointCloud()
    open3d_pc1.points = open3d.utility.Vector3dVector(pc1_clean)
    open3d_pc1.colors = open3d.utility.Vector3dVector(curr_image.reshape(-1, 3)[pc1_valid,:] / 255)

    pc2_valid = ~np.isnan(pc2).any(axis=1)
    pc2_clean = pc2[pc2_valid, :]
    open3d_pc2 = open3d.geometry.PointCloud()
    open3d_pc2.points = open3d.utility.Vector3dVector(pc2_clean)
    open3d_pc2.colors = open3d.utility.Vector3dVector(last_image.reshape(-1, 3)[pc2_valid,:]/ 255)


    # draw_registration_result(open3d_pc2, open3d_pc1, np.eye(4))
    # print("Initial alignment")
    # evaluation = open3d.pipelines.registration.evaluate_registration(open3d_pc2, open3d_pc1, threshold, init_transform)
    # print(evaluation)

    reg_p2p = open3d.pipelines.registration.registration_icp(open3d_pc2, open3d_pc1, threshold, init_transform, 
                                                             open3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                             open3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))
    # print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    # draw_registration_result(open3d_pc1, open3d_pc2, reg_p2p.transformation)


    # open3d_pc1.estimate_normals()
    # open3d_pc2.estimate_normals()
    # reg_p2l = open3d.pipelines.registration.registration_icp(open3d_pc1, open3d_pc2, threshold, trans_init, open3d.pipelines.registration.TransformationEstimationPointToPlane())
    # print(reg_p2l)
    mask = np.zeros_like(pc2)
    mask = np.asanyarray(reg_p2p.correspondence_set)
   
    if debug_plot:
        draw_registration_result(open3d_pc2, open3d_pc1, reg_p2p.transformation)

        pc2_tmp = get_pc(camera_params, curr_depth).reshape(-1, 3)
        # pc2_tmp = np.copy(pc2)


        t_vec =  reg_p2p.transformation[:3, -1]
        r_mat =  reg_p2p.transformation[:3, :3]


        pc1_clean = pc1.reshape(-1, 3)
        pc1_valid = ~np.isnan(pc1_clean).any(axis=1)
        pc1_clean = pc1_clean[pc1_valid, :]
        
        pc2_clean = pc2_tmp.reshape(-1, 3)
        pc2_valid = ~np.isnan(pc2_clean).any(axis=1)
        pc2_clean = pc2_clean[pc2_valid, :] @ r_mat.T + t_vec

        open3d_pc1 = open3d.geometry.PointCloud()
        open3d_pc1.points = open3d.utility.Vector3dVector(pc1_clean)
        open3d_pc1.colors = open3d.utility.Vector3dVector((last_image.reshape(-1, 3)[pc1_valid, :] / 255)[:,[2, 1, 0]])

        open3d_pc2 = open3d.geometry.PointCloud()
        open3d_pc2.points = open3d.utility.Vector3dVector(pc2_clean)
        open3d_pc2.colors = open3d.utility.Vector3dVector((curr_image.reshape(-1, 3)[pc2_valid, :] / 255)[:,[2, 1, 0]])

        visualizer = open3d.visualization.Visualizer()
        visualizer.create_window()

        visualizer.add_geometry(open3d_pc1)
        visualizer.add_geometry(open3d_pc2)

        visualizer.run()
        visualizer.destroy_window()

    return reg_p2p.transformation






output_perent = Path("/home/liora/Lior/Datasets/svo/global/reconstruction")
merge_path = Path("/home/liora/Lior/Datasets/svo/global/merge")
base_dir = Path("/home/liora/Lior/Datasets/svo/global/records")
sections_dirs = ["global_1_skip_8", 
                "global_2_skip_8",
                 "global_3_skip_8",]
                #  "global_4_skip_8",
                # "global_5_skip_8",
                # "global_6_skip_8",
                # "global_7_skip_8", 
                # "global_8_skip_8",
                # "global_9_skip_8",
                # "global_10_skip_8",
                # "global_11_skip_8"]


outputs = output_perent / "tmp"
images_between = outputs / "images"
depth_between =  outputs / "depth"
sfm_pairs = outputs / 'pairs-sfm.txt'
loc_pairs = outputs / 'pairs-loc.txt'
sfm_dir = outputs / 'sfm'
features = outputs / 'features.h5'
matches = outputs / 'matches.h5'



feature_conf = extract_features.confs['disk'] #['superpoint_inloc']
matcher_conf = match_features.confs['disk+lightglue'] #['superglue']

try:
    shutil.rmtree(output_perent)
except:
    pass
output_perent.mkdir(parents=True, exist_ok=True)
(output_perent / "images").mkdir(parents=True, exist_ok=True)
(output_perent / "depth").mkdir(parents=True, exist_ok=True)




unifeid_transform = {'frames': []}

initial_frame = True
last_frame = None
carry_transform = np.eye(4)

section_counter = 1

last_position, last_image, last_depth = None, None, None

for sec in tqdm(sections_dirs):

    transform_info = OmegaConf.load(base_dir / sec / "transforms.json")

    if initial_frame:

        unifeid_transform['cx'] = transform_info['cx']
        unifeid_transform['cy'] = transform_info['cy']
        unifeid_transform['fl_x'] = transform_info['fl_x']
        unifeid_transform['fl_y'] = transform_info['fl_y']
        unifeid_transform['w'] = transform_info['w']
        unifeid_transform['h'] = transform_info['h']
        unifeid_transform['camera_model'] = transform_info['camera_model']

        camera_params = {
                'model': transform_info['camera_model'],
                'width': transform_info['w'],
                'height': transform_info['h'],
                'params': np.array([transform_info['fl_x'], 
                                    transform_info['fl_y'], 
                                    transform_info['cx'], 
                                    transform_info['cy']])}
        
        # carry_transform[:3, 3] = np.asarray(transform_info.frames[0]['transform_matrix'])[:3, 3] * -1
        # carry_transform[:3, :3] = np.linalg.inv(np.asarray(transform_info.frames[0]['transform_matrix'])[:3, :3])
        carry_transform = np.eye(4)
        initial_frame = False

    else:

        shutil.copy(base_dir / sec / transform_info.frames[0].file_path, images_between / "target.png")
        shutil.copy(base_dir / sec / transform_info.frames[0]['depth_file_path'].replace("depth_neural_plus_meter","depth"), depth_between / "target.npy")
        references = [str(p.relative_to(images_between)) for p in (images_between).iterdir()]
        depths_path = [depth_between / "ref.npy", depth_between / "target.npy"]

        extract_features.main(feature_conf, images_between, image_list=references, feature_path=features)
        pairs_from_exhaustive.main(sfm_pairs, image_list=references)
        match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

        keypoints1, descriptors1 = load_features(features, references[0])
        keypoints2, descriptors2 = load_features(features, references[1])

        matches_pairs_idx = load_matches(matches, references[0], references[1])

        r, t, inliers = estimate_pose(keypoints1, keypoints2, matches_pairs_idx, camera_params)

        image_ref =  cv2.imread(str(images_between / references[0])) 
        depth_ref = load_depth(depths_path[0])
        image_target =  cv2.imread(str(images_between / references[1])) 
        depth_target = load_depth(depths_path[1])

        ref_traget_transform = scale_transition_between_sections(keypoints1, 
                                                                 keypoints2,
                                                                 image_ref,
                                                                 image_target,
                                                                 depth_ref, 
                                                                 depth_target,
                                                                 t, 
                                                                 r,
                                                                 inliers, 
                                                                 matches_pairs_idx, 
                                                                 camera_params,
                                                                 ref_pose=carry_transform)
        
        carry_transform[:3, -1] = carry_transform[:3, -1] + carry_transform[:3, :3] @ ref_traget_transform[:3, -1].T
        carry_transform[:3, :3] =  ref_traget_transform[:3, :3] @ carry_transform[:3, :3]
        
    for idx, frame in enumerate(tqdm(transform_info.frames)):

        new_image_name = f"images/{section_counter}_" + frame.file_path.split("/")[1]
        new_depth_name = f"depth/{section_counter}_" + frame.depth_file_path.split("/")[1]
        frame.depth_file_path = frame.depth_file_path.replace("depth_neural_plus_meter","depth")

        shutil.copy(base_dir / sec / frame.file_path, output_perent / new_image_name)
        shutil.copy(base_dir / sec / frame.depth_file_path, output_perent / new_depth_name)

        curr_image =  cv2.imread(str(output_perent / new_image_name))
        curr_depth = load_depth(output_perent / new_depth_name)


        relative_transform = np.asarray(frame.transform_matrix)
        if idx == 0:
            origin_reset_t = np.copy(relative_transform[:3, -1])
            origin_reset_r = np.linalg.inv(np.copy(relative_transform[:3, :3]))

        relative_transform[:3, 3] = relative_transform[:3, -1] - origin_reset_t
        relative_transform[:3, :3] =  relative_transform[:3, :3] @ origin_reset_r

        relative_transform[:3, 3] = carry_transform[:3, :3] @ relative_transform[:3, -1] + carry_transform[:3, -1]
        relative_transform[:3, :3] =  relative_transform[:3, :3] @ carry_transform[:3, :3]

        if not sec == sections_dirs[0] and idx != 0:
            relative_transform = refine_transform(relative_transform, 
                                             camera_params,
                                             last_position,
                                             curr_image,
                                             curr_depth,
                                             last_image,
                                             last_depth,
                                             max_iter=100)
            

        unifeid_transform['frames'].append({'camera_id':0,
                                            'depth_file_path': new_depth_name,
                                            'file_path': new_image_name,
                                            'transform_matrix': relative_transform.tolist()})
        
        last_position = relative_transform
        last_image = curr_image
        last_depth = curr_depth 
        

    carry_transform = np.copy(relative_transform)
    section_counter += 1
    
    try:
        shutil.rmtree(outputs)
    except:
        pass
    outputs.mkdir(parents=True, exist_ok=True)
    images_between.mkdir(parents=True, exist_ok=True)
    depth_between.mkdir(parents=True, exist_ok=True)
    shutil.copy(base_dir / sec / transform_info.frames[-1]['file_path'], images_between / "ref.png")
    shutil.copy(base_dir / sec / transform_info.frames[-1]['depth_file_path'].replace("depth_neural_plus_meter","depth"), depth_between / "ref.npy")



json.dump(unifeid_transform, (output_perent / "transforms.json").open("wt"), indent=2)
generate_pointcloud(json_path=f"{output_perent}/transforms.json",
                        image_dir=f"{output_perent}/images",
                        depth_dir=f"{output_perent}/depth",
                        output_path=Path(f"{output_perent}/pointcloud.pcd"),
                        scale=1,
                        max_distance=2)
show_pointcloud(output_perent / "pointcloud.pcd")



# coloros = ['red', 'blue', 'green']
# fig = go.Figure()

# fig.add_trace(go.Scatter3d(x=loc_array[:,0], y=loc_array[:, 1], z=loc_array[:, 2],     mode='markers',
#     marker=dict(
#         size=6,
#         color='red',                # set color to an array/list of desired values
#         opacity=0.8),
#         text=names
#         ))

# fig.update_layout(scene=dict(
#                     xaxis=dict(range=[min_value, max_value],  # Adjust range as needed
#                             autorange=False),
#                     yaxis=dict(range=[min_value, max_value],  # Adjust range as needed
#                             autorange=False),
#                     zaxis=dict(range=[min_value, max_value],  # Adjust range as needed
#                             autorange=False),
#                 ),
#                 )
# fig.show()




