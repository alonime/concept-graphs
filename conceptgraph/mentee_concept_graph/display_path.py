import plotly.graph_objects as go
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import cv2


def compute_blur_score(image_path):
    pass



transform_path = "/home/liora/Lior/Datasets/svo/global_5_skip_8/transforms.json"
transform_info = OmegaConf.load(transform_path)



loc = []

for frame in transform_info.frames:

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