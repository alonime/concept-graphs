import plotly.graph_objects as go
from omegaconf import OmegaConf
import shutil
import numpy as np
from PIL import Image
import cv2
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


# import matplotlib
# matplotlib.use('TkAgg')

def get_pc(camera_params, depth):
    intrinsic = np.asarray([[camera_params['params'][0], 0, camera_params['params'][2]], [0, camera_params['params'][1], camera_params['params'][3]], [0, 0, 1]])
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


def scale_transition_between_sections(kp1, kp2, depth1, depth2, t, inliers, matches_pairmatchess_idx, camera_param):
    ref_idx = np.where(matches_pairmatchess_idx != -1)[0]
    matched_keypoints1 = keypoints1[ref_idx, :]
    matched_keypoints2 = keypoints2[matches_pairmatchess_idx[ref_idx]]
    matched_keypoints1 = matched_keypoints1[inliers].astype(np.int64)
    matched_keypoints2 = matched_keypoints2[inliers].astype(np.int64)
    kp1_depth = depth1[matched_keypoints1[:, 1], matched_keypoints1[:, 0]]
    kp2_depth = depth2[matched_keypoints2[:, 1], matched_keypoints2[:, 0]]

    valid_points = np.all(np.vstack([kp1_depth != np.nan, kp2_depth != np.nan]), axis=0)




    pass
    



output_perent = Path("/home/liora/Lior/Datasets/svo/global/reconstruction")
merge_path = Path("/home/liora/Lior/Datasets/svo/global/merge")
base_dir = Path("/home/liora/Lior/Datasets/svo/global/records")
sections_dirs = ["global_1_skip_8", 
                "global_2_skip_8",
                "global_3_skip_8",
                "global_4_skip_8", 
                "global_5_skip_8",
                "global_6_skip_8",
                "global_7_skip_8", 
                "global_8_skip_8",
                "global_9_skip_8",
                "global_10_skip_8",
                "global_11_skip_8"]

merge_dirs = ["1_2", "2_3"]

coloros = ['red', 'blue', 'green']

unifeid_transform = {'frames': []}

fig = go.Figure()
sec = sections_dirs[0]
transform_info = OmegaConf.load(base_dir / sec / "transforms.json")

unifeid_transform['cx'] = transform_info['cx']
unifeid_transform['cy'] = transform_info['cy']
unifeid_transform['fl_x'] = transform_info['fl_x']
unifeid_transform['fl_y'] = transform_info['fl_y']
unifeid_transform['w'] = transform_info['w']
unifeid_transform['h'] = transform_info['h']

loc = []
ref_frame = transform_info.frames[0]
images_path = base_dir / sec 

section_counter = 1


for frame in transform_info.frames:
    new_image_name = f"images/{section_counter}_" + frame.file_path.split("/")[1]
    new_depth_name = f"depth/{section_counter}_" + frame.depth_file_path.split("/")[1]
    frame.depth_file_path = frame.depth_file_path.replace("depth_neural_plus_meter","depth")

    shutil.copy(base_dir / sec / frame.file_path, output_perent / new_image_name)
    shutil.copy(base_dir / sec / frame.depth_file_path, output_perent / new_depth_name)

    unifeid_transform['frames'].append({'camera_id':0,
                                        'depth_file_path': new_depth_name,
                                        'file_path': new_image_name,
                                        'transform_matrix': frame.transform_matrix})

    loc.append((frame.file_path, [frame.transform_matrix[0][3], frame.transform_matrix[1][3], frame.transform_matrix[2][3]]))

loc.sort(key= lambda x: x[0])

loc_array = np.array([lo[1] for lo in loc])
loc_array= loc_array[loc_array[:,1] < 100] 
names = [lo[0] for lo in loc]

min_value, max_value = loc_array.min(), loc_array.max()

merge_union_transform = OmegaConf.load(merge_path / merge_dirs[0] / "transforms.json")
second_section =  OmegaConf.load(base_dir / sections_dirs[1] / "transforms.json")

feature_conf = extract_features.confs['disk']
matcher_conf = match_features.confs['disk+lightglue']


# feature_conf = extract_features.confs['superpoint_inloc']
# matcher_conf = match_features.confs['superglue']

outputs = output_perent / "tmp"
images = outputs / "images"
depth =  outputs / "depth"

shutil.rmtree(outputs)

outputs.mkdir(parents=True, exist_ok=True)
images.mkdir(parents=True, exist_ok=True)
depth.mkdir(parents=True, exist_ok=True)


shutil.copy(base_dir / sec / transform_info.frames[-1]['file_path'], images / "ref.png")
shutil.copy(base_dir / sec / transform_info.frames[-1]['depth_file_path'].replace("depth_neural_plus_meter","depth"), depth / "ref.npy")
shutil.copy(base_dir / sections_dirs[1] / second_section.frames[0].file_path,images / "target.png")
shutil.copy(base_dir / sec / second_section.frames[0]['depth_file_path'].replace("depth_neural_plus_meter","depth"), depth / "target.npy")


sfm_pairs = outputs / 'pairs-sfm.txt'
loc_pairs = outputs / 'pairs-loc.txt'
sfm_dir = outputs / 'sfm'
features = outputs / 'features.h5'
matches = outputs / 'matches.h5'

references = [str(p.relative_to(images)) for p in (images).iterdir()]
depths_path = [depth / "ref.npy", depth / "target.npy"]

extract_features.main(feature_conf, images, image_list=references, feature_path=features)
pairs_from_exhaustive.main(sfm_pairs, image_list=references)
match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

# feature_path = extract_features.main(feature_conf, images, outputs)
# match_path = match_features.main(
#     matcher_conf, sfm_pairs, feature_conf["output"], outputs
# )


# Load features
keypoints1, descriptors1 = load_features(features, references[0])
keypoints2, descriptors2 = load_features(features, references[1])

# Load matches
matches_pairs_idx = load_matches(matches, references[0], references[1])

# Estimate pose
camera_params = {
        'model': second_section.camera_model,
        'width': second_section.w,
        'height': second_section.h,
        'params': np.array([second_section.fl_x, second_section.fl_y, second_section.cx, second_section.cy])  # Focal lengths and principal point
    }
r, t, inliers = estimate_pose(keypoints1, keypoints2, matches_pairs_idx, camera_params)


depth_ref = load_depth(depths_path[0])
depth_target = load_depth(depths_path[1])

scale_transition_between_sections(keypoints1, 
                                  keypoints2, 
                                  depth_ref, 
                                  depth_target, 
                                  t, 
                                  inliers, 
                                  matches_pairs_idx, 
                                  camera_params)



fig.add_trace(go.Scatter3d(x=loc_array[:,0], y=loc_array[:, 1], z=loc_array[:, 2],     mode='markers',
    marker=dict(
        size=6,
        color='red',                # set color to an array/list of desired values
        opacity=0.8),
        text=names
        ))

fig.update_layout(scene=dict(
                    xaxis=dict(range=[min_value, max_value],  # Adjust range as needed
                            autorange=False),
                    yaxis=dict(range=[min_value, max_value],  # Adjust range as needed
                            autorange=False),
                    zaxis=dict(range=[min_value, max_value],  # Adjust range as needed
                            autorange=False),
                ),
                )
fig.show()




for sec in sections_dirs:
    
    transform_info = OmegaConf.load(base_dir / sec / "transforms.json")

    loc = []
    ref_frame = transform_info.frames[0]
    images_path = base_dir / sec 

    for frame in transform_info.frames[1:]:
        loc.append((frame.file_path, [frame.transform_matrix[0][3], frame.transform_matrix[1][3], frame.transform_matrix[2][3]]))

    loc.sort(key= lambda x: x[0])

    loc_array = np.array([lo[1] for lo in loc])
    loc_array= loc_array[loc_array[:,1] < 100] 
    names = [lo[0] for lo in loc]

    min_value, max_value = loc_array.min(), loc_array.max()


    fig.add_trace(go.Scatter3d(x=loc_array[:,0], y=loc_array[:, 1], z=loc_array[:, 2],     mode='markers',
        marker=dict(
            size=6,
            color=np.linspace(0,1,len(loc)),                # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            opacity=0.8),
            text=names
            ))

    fig.update_layout(scene=dict(
                        xaxis=dict(range=[min_value, max_value],  # Adjust range as needed
                                autorange=False),
                        yaxis=dict(range=[min_value, max_value],  # Adjust range as needed
                                autorange=False),
                        zaxis=dict(range=[min_value, max_value],  # Adjust range as needed
                                autorange=False),
                    ),
                    )
    fig.show()


