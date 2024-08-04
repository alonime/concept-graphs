from pathlib import Path
import torch
import numpy as np
import plotly.graph_objects as go

from gradslam import RGBDImages
import torch.utils
from conceptgraph.mentee_concept_graph.svo_dataset import SVODataset

from omegaconf import OmegaConf
from tqdm import tqdm

# dataset_root = Path("/home/liora/Lior/Datasets/svo")
# scene_id = "CG_office_5"
# dataset_config = "transforms.json"

# dataset_config_path = dataset_root / scene_id / dataset_config
# dataset_config = OmegaConf.load(dataset_config_path)

# dataset = SVODataset(config_dict=dataset_config,
#                     start=0,
#                     end=-1,
#                     stride=1,
#                     basedir=dataset_root,
#                     sequence="CG_office_5",
#                     device="cpu",
#                     dtype=torch.float,
#                     image_suffix='png',
#                     depth_suffix='npy',
#                     scale_depth=True,
#                     batch_size=1,
#                     )



# # colors, depths, intrinsics, poses, *_ = next(iter(dataset))
# rgbdimages = RGBDImages(colors[None,None, ...], depths[None,None, ...], intrinsics[None,None, ...], poses[None,None, ...])

# rgbdimages.plotly(0).show()

# import necessary packages
import gradslam as gs
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from gradslam import Pointclouds, RGBDImages
from gradslam.datasets import ICL
from gradslam.slam import PointFusion
from torch.utils.data import DataLoader


def plotly_map_update_visualization(intermediate_pcs, poses, K, max_points_per_pc=50000, ms_per_frame=50):
    """
    Args:
        - intermediate_pcs (List[gradslam.Pointclouds]): list of gradslam.Pointclouds objects, each of batch size 1
        - poses (torch.Tensor): poses for drawing frustums
        - K (torch.Tensor): Intrinsics matrix
        - max_points_per_pc (int): maximum number of points to plot for each pointcloud
        - ms_per_frame (int): miliseconds per frame for the animation

    Shape:
        - poses: :math:`(L, 4, 4)`
        - K: :math:`(4, 4)`
    """
    def plotly_poses(poses, K):
        """
        Args:
            poses (np.ndarray):
            K (np.ndarray):

        Shapes:
            - poses: :math:`(L, 4, 4)`
            - K: :math:`(4, 4)`
        """
        fx = abs(K[0, 0])
        fy = abs(K[1, 1])
        f = (fx + fy) / 2
        cx = K[0, 2]
        cy = K[1, 2]

        cx = cx / f
        cy = cy / f
        f = 1.

        pos_0 = np.array([0., 0., 0.])
        fustum_0 = np.array(
            [
                [-cx, -cy, f],
                [cx, -cy, f],
                list(pos_0),
                [-cx, -cy, f],
                [-cx, cy, f],
                list(pos_0),
                [cx, cy, f],
                [-cx, cy, f],
                [cx, cy, f],
                [cx, -cy, f],
            ]
        )

        traj = []
        traj_frustums = []
        for pose in poses:
            rot = pose[:3, :3]
            tvec = pose[:3, 3]

            fustum_i = fustum_0 @ rot.T
            fustum_i = fustum_i + tvec
            pos_i = pos_0 + tvec

            pos_i = np.round(pos_i, decimals=2)
            fustum_i = np.round(fustum_i, decimals=2)

            traj.append(pos_i)
            traj_array = np.array(traj)
            traj_frustum = [
                go.Scatter3d(
                    x=fustum_i[:, 0], y=fustum_i[:, 1], z=fustum_i[:, 2],
                    marker=dict(
                        size=0.1,
                    ),
                    line=dict(
                        color='purple',
                        width=4,
                    )
                ),
                go.Scatter3d(
                    x=pos_i[None, 0], y=pos_i[None, 1], z=pos_i[None, 2],
                    marker=dict(
                        size=6.,
                        color='purple',
                    )
                ),
                go.Scatter3d(
                    x=traj_array[:, 0], y=traj_array[:, 1], z=traj_array[:, 2],
                    marker=dict(
                        size=0.1,
                    ),
                    line=dict(
                        color='purple',
                        width=2,
                    )
                ),
            ]
            traj_frustums.append(traj_frustum)
        return traj_frustums

    def frame_args(duration):
        return {
            "frame": {"duration": duration, "redraw": True},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    # visualization
    scatter3d_list = [pc.plotly(0, as_figure=False, max_num_points=max_points_per_pc) for pc in intermediate_pcs]
    traj_frustums = plotly_poses(poses.cpu().numpy(), K.cpu().numpy())
    data = [[*frustum, scatter3d] for frustum, scatter3d in zip(traj_frustums, scatter3d_list)]

    steps = [
        {"args": [[i], frame_args(0)], "label": i, "method": "animate"}
        for i in range(seq_len)
    ]
    sliders = [
        {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {"prefix": "Frame: "},
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": steps,
        }
    ]
    updatemenus = [
        {
            "buttons": [
                {
                    "args": [None, frame_args(ms_per_frame)],
                    "label": "&#9654;",
                    "method": "animate",
                },
                {
                    "args": [[None], frame_args(0)],
                    "label": "&#9724;",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 70},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top",
        }
    ]
/mnt/shared/Datasets/office_60fps_skip_1
    fig = go.Figure()
    frames = [{"data": frame, "name": i} for i, frame in enumerate(data)]
    fig.add_traces(frames[0]["data"])
    fig.update(frames=frames)
    fig.update_layout(
        updatemenus=updatemenus,
        sliders=sliders,
        showlegend=False,
        scene=dict(
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, visible=False,),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, visible=False,),
            zaxis=dict(showticklabels=False, showgrid=False, zeroline=False, visible=False,),
        )
    )
    fig.show()
    return fig


icl_path = '/home/liora/Lior/Datasets/gradslam/ICL/'

# load dataset
dataset = ICL(icl_path, seqlen=8, height=240, width=320)
loader = DataLoader(dataset=dataset, batch_size=1)

colors, depths, intrinsics, poses, *_ = next(iter(loader))

# # create rgbdimages object
# rgbdimages = RGBDImages(colors, depths, intrinsics, poses)
# # rgbdimages.plotly(0).update_layout(autosize=False, height=600, width=400).show()


# dataset_root = Path("/home/liora/Lior/Datasets/svo")
# scene_id = "CG_office_5"
# dataset_config = "transforms.json"

# dataset_config_path = dataset_root / scene_id / dataset_config
# dataset_config = OmegaConf.load(dataset_config_path)

# dataset = SVODataset(config_dict=dataset_config,
#                     start=0,
#                     end=-1,
#                     stride=1,
#                     basedir=dataset_root,
#                     sequence="CG_office_5",
#                     device="cpu",
#                     dtype=torch.float,
#                     image_suffix='png',
#                     depth_suffix='npy',
#                     scale_depth=True,
#                     batch_size=1,
#                     clip_depth=5,
#                     )



# colors, depths, intrinsics, poses, *_ = next(iter(dataset))
# rgbdimages = RGBDImages(colors[None,None, ...], depths[None,None, ...], intrinsics[None,None, ...], poses[None,None, ...])



# load dataset
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
loader = DataLoader(dataset=dataset, batch_size=1)
first_frame = True

initial_poses = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(1, 1, 1, 1)
slam = PointFusion(odom='gradicp', device=device)
pointclouds = Pointclouds(device=device)
prev_frame = None

for colors, depths, intrinsics, poses, *_ in tqdm(loader):

    # create rgbdimages object
    # rgbdimages = RGBDImages(colors[None, ...].to(device), depths[None, ...].to(device), intrinsics[None, ...].to(device))
    rgbdimages = RGBDImages(colors, depths, intrinsics)


    # step by step SLAM
    live_frame = rgbdimages[:, 0].to(device)
    if first_frame:
        live_frame.poses = initial_poses
        first_frame = False
    pointclouds, live_frame.poses = slam.step(pointclouds, live_frame, prev_frame)
    prev_frame = live_frame
pointclouds.plotly(0, max_num_points=20000).update_layout(autosize=False, width=600).show()
