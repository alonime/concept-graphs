import glob
import json
import cv2
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import imageio
import numpy as np
import torch
from natsort import natsorted

from conceptgraph.dataset import conceptgraphs_datautils
from conceptgraph.utils.geometry import relative_transformation


class GradSLAMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config_dict,
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: int = 480,
        desired_width: int = 640,
        channels_first: bool = False,
        normalize_color: bool = False,
        device="cuda:0",
        dtype=torch.float,
        load_embeddings: bool = False,
        embedding_dir: str = "feat_lseg_240_320",
        embedding_dim: int = 512,
        relative_pose: bool = True, # If True, the pose is relative to the first frame
        scale_depth: bool = True,
        clip_depth: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.name = 'svo'
        self.device = device
        self.png_depth_scale = 1
        self.scale_depth = scale_depth
        self.clip_depth = clip_depth

        self.orig_height = config_dict["h"]
        self.orig_width = config_dict["w"]
        self.fx = config_dict["fl_x"]
        self.fy = config_dict["fl_y"]
        self.cx = config_dict["cx"]
        self.cy = config_dict["cy"]

        self.dtype = dtype

        self.desired_height = desired_height
        self.desired_width = desired_width
        self.height_downsample_ratio = float(self.desired_height) / self.orig_height
        self.width_downsample_ratio = float(self.desired_width) / self.orig_width
        self.channels_first = channels_first
        self.normalize_color = normalize_color

        self.load_embeddings = load_embeddings
        self.embedding_dir = embedding_dir
        self.embedding_dim = embedding_dim
        self.relative_pose = relative_pose

        self.start = start
        self.end = end
        if start < 0:
            raise ValueError("start must be positive. Got {0}.".format(stride))
        if not (end == -1 or end > start):
            raise ValueError(
                "end ({0}) must be -1 (use all images) or greater than start ({1})".format(end, start)
            )

        self.distortion = np.array(config_dict["distortion"]) if "distortion" in config_dict else None
        self.crop_size = config_dict["crop_size"] if "crop_size" in config_dict else None 

        self.crop_edge = None
        if "crop_edge" in config_dict.keys():
            self.crop_edge = config_dict["crop_edge"]

        self.color_paths, self.depth_paths, self.embedding_paths = self.get_filepaths()
        if len(self.color_paths) != len(self.depth_paths):
            raise ValueError("Number of color and depth images must be the same.")
        if self.load_embeddings:
            if len(self.color_paths) != len(self.embedding_paths):
                raise ValueError(
                    "Mismatch between number of color images and number of embedding files."
                )
            
        self.num_imgs = len(self.color_paths)
        
        if self.end == -1:
            self.end = self.num_imgs
            
        self.poses_path, self.poses = self.load_poses()

        self.color_paths = self.color_paths[self.start : self.end : stride]
        self.depth_paths = self.depth_paths[self.start : self.end : stride]
        if self.load_embeddings:
            self.embedding_paths = self.embedding_paths[self.start : self.end : stride]
        self.poses = self.poses[self.start : self.end : stride]
        self.poses_path = self.poses_path[self.start : self.end : stride]

        # Tensor of retained indices (indices of frames and poses that were retained)
        self.retained_inds = torch.arange(self.num_imgs)[self.start : self.end : stride]

        self.num_imgs = len(self.color_paths)
        
        if self.end == -1:
            self.end = self.num_imgs


        # self.transformed_poses = datautils.poses_to_transforms(self.poses)
        self.poses = torch.stack(self.poses)
        if self.relative_pose:
            self.transformed_poses = self._preprocess_poses(self.poses)
        else:
            self.transformed_poses = self.poses

    def __len__(self):
        return self.num_imgs

    def get_filepaths(self):
        """Return paths to color images, depth images. Implement in subclass."""
        raise NotImplementedError

    def load_poses(self):
        """Load camera poses. Implement in subclass."""
        raise NotImplementedError

    def _preprocess_color(self, color: np.ndarray):
        r"""Preprocesses the color image by resizing to :math:`(H, W, C)`, (optionally) normalizing values to
        :math:`[0, 1]`, and (optionally) using channels first :math:`(C, H, W)` representation.

        Args:
            color (np.ndarray): Raw input rgb image

        Retruns:
            np.ndarray: Preprocessed rgb image

        Shape:
            - Input: :math:`(H_\text{old}, W_\text{old}, C)`
            - Output: :math:`(H, W, C)` if `self.channels_first == False`, else :math:`(C, H, W)`.
        """
        color = cv2.resize(
            color,
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_LINEAR,
        )
        if self.normalize_color:
            color = conceptgraphs_datautils.normalize_image(color)
        if self.channels_first:
            color = conceptgraphs_datautils.channels_first(color)
        return color

    def _preprocess_depth(self, depth: np.ndarray, scale=True):
        r"""Preprocesses the depth image by resizing, adding channel dimension, and scaling values to meters. Optionally
        converts depth from channels last :math:`(H, W, 1)` to channels first :math:`(1, H, W)` representation.

        Args:
            depth (np.ndarray): Raw depth image

        Returns:
            np.ndarray: Preprocessed depth

        Shape:
            - depth: :math:`(H_\text{old}, W_\text{old})`
            - Output: :math:`(H, W, 1)` if `self.channels_first == False`, else :math:`(1, H, W)`.
        """
        if scale:
            depth /= 1000.0 
        depth[depth < 0] = np.nan

        if not self.clip_depth == 0:
            depth[depth > self.clip_depth] = np.nan

        depth = cv2.resize(
            depth.astype(float),
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_NEAREST,
        )


        depth = np.expand_dims(depth, -1)
        if self.channels_first:
            depth = conceptgraphs_datautils.channels_first(depth)
        return depth / self.png_depth_scale
    
    def _preprocess_poses(self, poses: torch.Tensor):
        r"""Preprocesses the poses by setting first pose in a sequence to identity and computing the relative
        homogenous transformation for all other poses.

        Args:
            poses (torch.Tensor): Pose matrices to be preprocessed

        Returns:
            Output (torch.Tensor): Preprocessed poses

        Shape:
            - poses: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
            - Output: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
        """
        return relative_transformation(
            poses[0].unsqueeze(0).repeat(poses.shape[0], 1, 1),
            poses,
            orthogonal_rotations=False,
        )
        
    def get_cam_K(self):
        '''
        Return camera intrinsics matrix K
        
        Returns:
            K (torch.Tensor): Camera intrinsics matrix, of shape (3, 3)
        '''
        K = np.array([self.fx, 0, self.cx,
                      0, self.fy,  self.cy,
                      0, 0, 1]).reshape(3,3)
        
        K = torch.from_numpy(K)
        return K
    
    def read_embedding_from_file(self, embedding_path: str):
        '''
        Read embedding from file and process it. To be implemented in subclass for each dataset separately.
        '''
        raise NotImplementedError

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        color = np.asarray(imageio.imread(color_path), dtype=float)
        color = self._preprocess_color(color)
        color = torch.from_numpy(color)
        if ".png" in depth_path:
            # depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth = np.asarray(imageio.imread(depth_path), dtype=np.int64)
        elif ".npy" in depth_path:
            depth = np.load(depth_path)
        else:
            raise NotImplementedError

        K = self.get_cam_K()
        if self.distortion is not None:
            # undistortion is only applied on color image, not depth!
            color = cv2.undistort(color, K, self.distortion)

        depth = self._preprocess_depth(depth, scale=self.scale_depth)
        depth = torch.from_numpy(depth)
        K = conceptgraphs_datautils.scale_intrinsics(
            K, self.height_downsample_ratio, self.width_downsample_ratio
        )
        intrinsics = torch.eye(4).to(K)
        intrinsics[:3, :3] = K

        pose = self.transformed_poses[index]
        pose_path = self.poses_path[index]
        assert Path(pose_path).stem == Path(color_path).stem or Path(depth_path).stem == Path(color_path).stem, "Not equal path to image/depth/pose"

        if self.load_embeddings:
            embedding = self.read_embedding_from_file(self.embedding_paths[index])
            return (
                color.to(self.device).type(self.dtype),
                depth.to(self.device).type(self.dtype),
                intrinsics.to(self.device).type(self.dtype),
                pose.to(self.device).type(self.dtype),
                embedding.to(self.device),  # Allow embedding to be another dtype
                # self.retained_inds[index].item(),
            )

        return (
            color.to(self.device).type(self.dtype),
            depth.to(self.device).type(self.dtype),
            intrinsics.to(self.device).type(self.dtype),
            pose.to(self.device).type(self.dtype),
            # self.retained_inds[index].item(),
        )




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
        desired_height: Optional[int] = None,
        desired_width: Optional[int] = None,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        depth_floder_name: str = "depth_neural_plus_meter",
        image_suffix:str = 'png',
        depth_suffix:str = 'npy',
        scale_depth: bool = True,
        clip_depth: int = 0,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.depth_floder_name = depth_floder_name
        self.config_dict = config_dict
        self.image_suffix = image_suffix
        self.depth_suffix = depth_suffix

        if desired_height is None:
            desired_height = config_dict['h']
        
        if desired_width is None:
            desired_width = config_dict['w']
        
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
            relative_pose=False,
            scale_depth=scale_depth,
            clip_depth = clip_depth,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = glob.glob(os.path.join(self.input_folder, "images", f"*.{self.image_suffix}"))
        color_paths = natsorted(color_paths)

        depth_paths = natsorted(glob.glob(os.path.join(self.input_folder, self.depth_floder_name, f"*.{self.depth_suffix}")))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses_sort = [(frame['file_path'], frame['transform_matrix']) for frame in self.config_dict['frames']]
        poses_sort.sort(key=lambda x: x[0])

        pose_path = [frame[0] for frame in poses_sort]

        poses_data = np.array([frame[1] for frame in poses_sort])

        poses = []
        # P = torch.tensor([[1, 0, 0, 0],
        #                   [0, -1, 0, 0],
        #                   [0, 0, -1, 0],
        #                   [0, 0, 0, 1]]).double()
        P = torch.tensor([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]]).double()
        for pose in poses_data:
            _pose = P @ pose @ P.T
            poses.append(_pose)
        return pose_path, poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)
