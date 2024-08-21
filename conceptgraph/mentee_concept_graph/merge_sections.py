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

import json

import matplotlib
matplotlib.use('TkAgg')


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
    cam_pc.reshape(height, width, -1)
    if rot is not None and t is not None:
        cam_pc = np.matmul(cam_pc, rot.T) + t

    return cam_pc.reshape(height, width, -1)


def load_depth(path, clip_detstance=2.5):
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
    

    if debug_plot:
        pc2 = get_pc(camera_params, depth2)
        tmp_rotation = r.T @ rotation
        # translation = t - translation
        tmp_trans = t - translation

        pc1_clean = pc1.reshape(-1, 3)
        pc1_valid = ~np.isnan(pc1_clean).any(axis=1)
        pc1_clean = pc1_clean[pc1_valid, :]
        pc2_clean = pc2.reshape(-1, 3)
        pc2_valid = ~np.isnan(pc2_clean).any(axis=1)
        pc2_clean = np.matmul(pc2_clean[pc2_valid, :], tmp_rotation) + tmp_trans

        o3d_pc1 = open3d.geometry.PointCloud()
        o3d_pc1.points = open3d.utility.Vector3dVector(pc1_clean)
        o3d_pc1.colors = open3d.utility.Vector3dVector(image1.reshape(-1, 3)[pc1_valid, :] / 255)

        o3d_pc2 = open3d.geometry.PointCloud()
        o3d_pc2.points = open3d.utility.Vector3dVector(pc2_clean)
        o3d_pc2.colors = open3d.utility.Vector3dVector(image2.reshape(-1, 3)[pc2_valid, :] / 255)

        visualizer = open3d.visualization.Visualizer()
        visualizer.create_window()

        visualizer.add_geometry(o3d_pc1)
        visualizer.add_geometry(o3d_pc2)

        visualizer.run()
        visualizer.destroy_window()

        fig, axs = plt.subplots(2,2)
        axs[0,0].imshow(depth1, cmap='viridis')
        axs[0,1].imshow(image1)
        axs[1,0].imshow(depth2, cmap='viridis')
        axs[1,1].imshow(image2)
        fig.show()


    transform = np.eye(4)
    transform[:3, :3] = r.T @ rotation
    transform[:3, 3] = t - translation

    return transform
    



output_perent = Path("/home/liora/Lior/Datasets/svo/global/reconstruction")
merge_path = Path("/home/liora/Lior/Datasets/svo/global/merge")
base_dir = Path("/home/liora/Lior/Datasets/svo/global/records")
sections_dirs = ["global_1_skip_8", 
                "global_2_skip_8",]
                #"global_3_skip_8",
                # "global_4_skip_8", 
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



unifeid_transform = {'frames': []}

initial_frame = True
last_frame = None
carry_transform = np.eye(4)

section_counter = 1


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
        
        carry_transform[:3, 3] = np.asarray(transform_info.frames[0]['transform_matrix'])[:3, 3] * -1
        carry_transform[:3, :3] = np.linalg.inv(np.asarray(transform_info.frames[0]['transform_matrix'])[:3, :3])

        
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
                                                                 camera_params)
        
        carry_transform[:3, :3] = carry_transform[:3, :3] @ ref_traget_transform[:3, :3]
        carry_transform[:3, 3] = carry_transform[:3, 3] + ref_traget_transform[:3, 3]
        
    for idx, frame in enumerate(transform_info.frames):
        new_image_name = f"images/{section_counter}_" + frame.file_path.split("/")[1]
        new_depth_name = f"depth/{section_counter}_" + frame.depth_file_path.split("/")[1]
        frame.depth_file_path = frame.depth_file_path.replace("depth_neural_plus_meter","depth")

        shutil.copy(base_dir / sec / frame.file_path, output_perent / new_image_name)
        shutil.copy(base_dir / sec / frame.depth_file_path, output_perent / new_depth_name)

        relative_transform = np.asarray(frame.transform_matrix)
        relative_transform[:3, :3] = relative_transform[:3, :3] @ carry_transform[:3, :3].T
        relative_transform[:3, 3] = relative_transform[:3, 3] + carry_transform[:3, 3]

        unifeid_transform['frames'].append({'camera_id':0,
                                            'depth_file_path': new_depth_name,
                                            'file_path': new_image_name,
                                            'transform_matrix': relative_transform.tolist()})
        

    carry_transform = relative_transform
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




