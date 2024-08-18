import plotly.graph_objects as go
from omegaconf import OmegaConf
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


output_perent = Path("/home/liora/Lior/Datasets/svo/handed_unifined")
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

# transform_paths = "/home/liora/Lior/Datasets/svo/global_5_skip_8/transforms.json"

feature_conf = extract_features.confs['disk']
matcher_conf = match_features.confs['disk+lightglue']
# retrieval_conf = extract_features.confs["netvlad"]

sfm_pairs = output_perent / 'pairs-sfm.txt'
loc_pairs = output_perent / 'pairs-loc.txt'
sfm_dir = output_perent / 'sfm'
features = output_perent / 'features.h5'
matches = output_perent / 'matches.h5'

for sec in sections_dirs:
    transform_info = OmegaConf.load(base_dir / sec / "transforms.json")

    loc = []
    ref_frame = transform_info.frames[0]
    images_path = base_dir / sec 

    for frame in transform_info.frames[1:]:
        extract_features.main(feature_conf, images_path, image_list=[ref_frame.file_path, frame.file_path], feature_path=features)
        pairs_from_exhaustive.main(sfm_pairs, image_list=[ref_frame.file_path, frame.file_path])
        res  = match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)
        model = reconstruction.main(sfm_dir, images_path, sfm_pairs, features, matches, image_list=[ref_frame.file_path, frame.file_path])

        visualization.visualize_sfm_2d(model, images_path, color_by='visibility', n=2)
        pairs = [(0, 1)]  # assuming we are matching between two frames
        pairs = pairs_from_sequence.main(pairs)

        loc.append((frame.file_path, [frame.transform_matrix[0][3], frame.transform_matrix[1][3], frame.transform_matrix[2][3]]))

    loc.sort(key= lambda x: x[0])

    loc_array = np.array([lo[1] for lo in loc])
    loc_array= loc_array[loc_array[:,1] < 100] 
    names = [lo[0] for lo in loc]

    min_value, max_value = loc_array.min(), loc_array.max()

    fig = go.Figure()

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


