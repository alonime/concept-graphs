import abc
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from natsort import natsorted
from scipy.spatial.transform import Rotation as R

from conceptgraph.dataset import conceptgraphs_datautils
from conceptgraph.utils.geometry import relative_transformation
from conceptgraph.dataset.conceptgraphs_rgbd_images import RGBDImages


from conceptgraph.utils.general_utils import measure_time

from conceptgraph.dataset.datasets_common import GradSLAMDataset


class SVODataset(GradSLAMDataset):
    """
    Dataset class to read in saved files from the structure created by our
    `save_record3d_stream.py` script
    """
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "poses")
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        # Attempt to find .jpg files in the directory
        color_paths = None
        jpg_paths = glob.glob(os.path.join(self.input_folder, "rgb", "*.jpg"))
        # If .jpg files are found, use them; otherwise, look for .png files
        if jpg_paths:
            color_paths = jpg_paths
        else:
            color_paths = glob.glob(os.path.join(self.input_folder, "rgb", "*.png"))
        color_paths = natsorted(color_paths)
        # check if "high_conf_depth" folder exists, if not, use "depth" folder
        if os.path.exists(os.path.join(self.input_folder, "high_conf_depth")):
            depth_folder = "high_conf_depth"
        else:
            depth_folder = "depth"
        depth_paths = natsorted(
            glob.glob(os.path.join(self.input_folder, depth_folder, "*.png"))
        )
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        posefiles = natsorted(glob.glob(os.path.join(self.pose_path, "*.npy")))
        poses = []
        P = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ]
        ).float()
        for posefile in posefiles:
            c2w = torch.from_numpy(np.load(posefile)).float()
            _R = c2w[:3, :3]
            _t = c2w[:3, 3]
            _pose = P @ c2w @ P.T
            poses.append(_pose)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)
