import plotly.graph_objects as go
from omegaconf import OmegaConf
import shutil
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

from hloc import (
    extract_features,
    match_features,
    pairs_from_exhaustive,
    reconstruction,
    visualization,
    pairs_from_retrieval,
)
from hloc.visualization import plot_images, read_image


import matplotlib
matplotlib.use('TkAgg')

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


new_image_name = f"images/{section_counter}_" + frame.file_path.split("/")[1]
new_depth_name = f"depth/{section_counter}_" + frame.depth_file_path.split("/")[1]
frame.depth_file_path = frame.depth_file_path.replace("depth_neural_plus_meter","depth")

shutil.copy(base_dir / sec / loc[-1][0], output_perent / "tmp/ref.png")
shutil.copy(base_dir / sections_dirs[1] / second_section.frames[0].file_path, output_perent / "tmp/target.png")



images = output_perent / "tmp"
outputs = output_perent

sfm_pairs = outputs / 'pairs-sfm.txt'
loc_pairs = outputs / 'pairs-loc.txt'
sfm_dir = outputs / 'sfm'
features = outputs / 'features.h5'
matches = outputs / 'matches.h5'

references = [str(p.relative_to(images)) for p in (images).iterdir()]

extract_features.main(feature_conf, images, image_list=references, feature_path=features)
pairs_from_exhaustive.main(sfm_pairs, image_list=references)
match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

feature_path = extract_features.main(feature_conf, images, outputs)
match_path = match_features.main(
    matcher_conf, sfm_pairs, feature_conf["output"], outputs
)

model = reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path)


# features = extract_features.main(feature_conf, 
#                                   output_perent / "tmp",
#                                   output_perent)
# matches = match_features.main(matcher_conf, output_perent / "tmp",  feature_conf["output"], output_perent)

import pycolmap
import h5py


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
def estimate_pose(keypoints1, keypoints2, matches, camera_params):
    """Estimate the relative pose between two images."""
    # Select matched keypoints
    ref_idx = np.where(matches != -1)[0]
    matched_keypoints1 = keypoints1[ref_idx, :]
    matched_keypoints2 = keypoints2[matches[ref_idx]]


    import matplotlib.pyplot as plt
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
    estimator = pycolmap.essential_matrix_estimation(matched_keypoints1, matched_keypoints2, camera_params, camera_params)

    success, inliers, E, R, t = estimator.estimate(
        matched_keypoints1, matched_keypoints2)
    
    if success:
        print("Pose estimation successful")
        return R, t, inliers
    else:
        print("Pose estimation failed")
        return None, None, None


# Load features
keypoints1, descriptors1 = load_features(features, references[0])
keypoints2, descriptors2 = load_features(features, references[1])

# Load matches
matches_pairs_idx = load_matches(matches, references[1], references[2])

# Estimate pose
camera_params = {
        'model': second_section.camera_model,
        'width': second_section.w,
        'height': second_section.h,
        'params': np.array([second_section.fl_x, second_section.fl_y, second_section.cx, second_section.cy])  # Focal lengths and principal point
    }
R, t, inliers = estimate_pose(keypoints1, keypoints2, matches_pairs_idx, camera_params)



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


