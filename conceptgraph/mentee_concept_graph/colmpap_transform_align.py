import plotly.graph_objects as go
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import cv2
from pathlib import Path


def get_pca(cov):
    eig_mag, eig_vecs = np.linalg.eig(ref_cov)
    order = np.argsort(-eig_mag)
    comps = np.eye(3)
    comps[:3, :3] = eig_vecs[:, order].T
    return comps





transform_path = "/home/liora/Lior/Datasets/svo/office_60fps_skip_8/transforms.json"
transform_hloc_path = "/home/liora/Lior/Datasets/svo/office_60fps_skip_8_hloc/transforms.json"

transform_info = OmegaConf.load(transform_path)
transform_hloc = OmegaConf.load(transform_hloc_path)



loc_ref = []
loc_hloc = []

for idx, frame in enumerate(transform_info.frames):
    hloc_frame = transform_hloc.frames[idx]
    loc_ref.append([frame.transform_matrix[0][3], frame.transform_matrix[1][3], frame.transform_matrix[2][3]])
    loc_hloc.append([hloc_frame.transform_matrix[0][3], hloc_frame.transform_matrix[1][3], hloc_frame.transform_matrix[2][3]])


loc_ref = np.array(loc_ref)
loc_hloc = np.array(loc_hloc)

ref_cov = np.cov(loc_ref.T)
hloc_cov = np.cov(loc_hloc.T)


floor_transform_ref = get_pca(ref_cov)
floor_transform_hloc = get_pca(hloc_cov)


loc_ref = loc_ref @ np.linalg.inv(floor_transform_ref)
loc_hloc = loc_hloc @ np.linalg.inv(floor_transform_hloc)


loc_ref = loc_ref - loc_ref[0, :]
loc_hloc = loc_hloc - loc_hloc[0, :]

# word cord to nerf
loc_ref = loc_ref * [-1, -1, 1]
loc_ref = loc_ref[:, [2, 1, 0]]




diff_loc_ref = np.diff(loc_ref, axis=0)
diff_loc_hloc = np.diff(loc_hloc, axis=0)

dist_loc_ref = np.linalg.norm(diff_loc_ref, axis=1)
dist_loc_hloc = np.linalg.norm(diff_loc_hloc, axis=1)

factor = np.mean(dist_loc_ref / dist_loc_hloc)

sign = np.mean(np.sign(diff_loc_ref) == np.sign(diff_loc_hloc), axis=0)



dim = 2
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.linspace(0, 1, len(diff_loc_ref)), 
                         y=diff_loc_ref[:, dim], 
                         mode='markers',))

fig.add_trace(go.Scatter(x=np.linspace(0, 1, len(diff_loc_ref)), 
                         y=diff_loc_hloc[:, dim], 
                         mode='markers',))
fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.linspace(0, 1, len(diff_loc_ref)), 
                         y=loc_ref[:, dim], 
                         mode='markers',))

fig.add_trace(go.Scatter(x=np.linspace(0, 1, len(diff_loc_ref)), 
                         y=loc_hloc[:, dim], 
                         mode='markers',))
fig.show()

min_value, max_value = loc_ref.min(), loc_ref.max()

fig = go.Figure()

fig.add_trace(go.Scatter3d(x=loc_ref[:,0], y=loc_ref[:, 1], z=loc_ref[:, 2],     
                           mode='markers',
    marker=dict(
        size=5,
        color=np.linspace(0,1,len(loc_ref)),                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8),
        ))

fig.add_trace(go.Scatter3d(x=loc_hloc[:,0] * factor, y=loc_hloc[:, 1]*factor, z=loc_hloc[:, 2]*factor,     
                           mode='markers',
    marker=dict(
        size=5,
        color=np.linspace(0,1,len(loc_hloc)),                # set color to an array/list of desired values
        colorscale='Inferno',   # choose a colorscale
        opacity=0.8),
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