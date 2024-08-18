import gzip
import inspect
import json
import os
from glob import glob

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import open3d
import torch

from rich.progress import track

# from perception.keypoints.superpoint import SuperPointConfig
# from perception.utils.path import validate_path


ALT_EXTS = {
    ".png": ".jpg",
    ".npy": ".npy.gz",
}


def generate_pointcloud(
    json_path: Path,
    image_dir: Path = None,
    depth_dir: Path = None,
    output_path: Path = None,
    downsample: int = 1,
    skip: int = 1,
    max_distance: float = 2.0,
    voxel_size: float = 0.025,
    nb_neighbors: int = 25,
    std_ratio: float = 2.0,
    features: Literal["superpoint"] = None,
    show: bool = False,
    scale: float = 1,
):
    # Save arguments
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    arg_dict = {arg: values[arg] for arg in args}

    # Validate paths
    json_path = Path(json_path)
    assert json_path.exists(), f"{json_path} doesn't exist"
    image_dir = Path(image_dir)
    assert image_dir is None or image_dir.exists(), f"{image_dir} doesn't exist"
    depth_dir = Path(depth_dir)
    assert depth_dir is None or depth_dir.exists(), f"{depth_dir} doesn't exist"
    base_dir = json_path.parent
    if output_path is None:
        suffix = ".pointcloud.pcd" if features is None else f".pointcloud.{features}.pcd"
        output_path = json_path.with_suffix(suffix)
    else:
        assert output_path.suffix == ".pcd", f"Expected '.pcd' output suffix got '{output_path.suffix}'"
    output_info_path = output_path.with_suffix(".info.json")
    assert output_info_path != json_path, "sanity"

    # Optional: create feature extractor
    feature_extractor = None
    if features == "superpoint":
        feature_extractor = SuperPointConfig().setup(cuda=True)

    # Parse json file
    data = json.load(json_path.open())

    # Read intrinsics
    grid = _get_pixel_grid(data, downsample)

    # Collect point cloud data
    all_xyz = []
    all_rgb = []
    all_desc = [] if feature_extractor is not None else None

    depth_files = sorted(glob(f"{depth_dir}/*.npy"))

    idx = 0
    for frame in track(data["frames"][::skip]):
        # Read extrinsics
        extrinsics = _get_extrinsics(frame["transform_matrix"])
        extrinsics[0:3,3] *= scale

        # Read image data
        image = _get_image(frame["file_path"], image_dir, base_dir, downsample)

        # Read depth data
        depth = _get_depth(depth_files[idx], depth_dir, base_dir, downsample)
        # depth = _get_depth(frame["depth_file_path"], depth_dir, base_dir, downsample)

        # Get pointcloud
        xyz, rgb, desc = _get_pointcloud(image, depth, extrinsics, grid, max_distance, feature_extractor)
        if desc is None:
            xyz, rgb = _clean_pointcloud(xyz, rgb, voxel_size, nb_neighbors, std_ratio)
        else:
            assert (
                voxel_size == 0 and nb_neighbors == 0
            ), "Point cloud cleaning is not supported with feature extraction"

        # Append
        all_xyz.append(xyz)
        all_rgb.append(rgb)
        if desc is not None:
            all_desc.append(desc)
        
        idx += 1

    # Concatenate
    xyz = np.concatenate(all_xyz, 0)
    rgb = np.concatenate(all_rgb, 0)
    if all_desc is not None:
        desc = np.concatenate(all_desc, 0)
    else:
        desc = None

    # Final cleanup
    if desc is None:
        xyz, rgb = _clean_pointcloud(xyz, rgb, voxel_size, nb_neighbors, std_ratio)
    else:
        assert voxel_size == 0 and nb_neighbors == 0, "Point cloud cleaning is not supported with feature extraction"

    # Create point cloud
    pc = open3d.geometry.PointCloud()
    pc.points = open3d.utility.Vector3dVector(xyz)
    pc.colors = open3d.utility.Vector3dVector(rgb)
    print(f"Created pointcloud with {len(pc.points)} points")

    # Save point cloud
    open3d.io.write_point_cloud(str(output_path), pc)
    print(f"Pointcloud saved to: {output_path}")
    # with open(str(output_info_path), 'wt') as file:
    #     json.dump(arg_dict, file, indent=2)

    if desc is not None:
        features_output_path = output_path.with_suffix(".npy")
        print(f"Features saved to: {features_output_path}")
        np.save(str(features_output_path), desc)

    # Show point cloud
    if show:
        open3d.visualization.draw_geometries([pc])


def _get_pixel_grid(data, downsample):
    intrinsics = np.asarray([data["fl_x"], 0, data["cx"], 0, data["fl_y"], data["cy"], 0, 0, 1]).reshape(3, 3)
    intrinsics[:2] /= downsample
    intrinsics_inv = np.linalg.inv(intrinsics).astype(np.float32)
    width, height = data["w"], data["h"]
    grid = np.stack(np.meshgrid(np.arange(width // downsample), np.arange(height // downsample)), -1)
    grid = grid.astype(np.float32) @ intrinsics_inv[:, :2].T + intrinsics_inv[:, 2]
    return grid


def _get_extrinsics(transform_matrix):
    extrinsics = np.asarray(transform_matrix).astype(np.float32)  # x: right, y: up, z: backwards
    return extrinsics


def _get_image(image_path, image_dir, base_dir, downsample):
    if image_dir is None:
        image_path = base_dir / image_path
    else:
        image_path = image_dir / os.path.basename(image_path)
    if not image_path.exists():
        image_path = image_path.with_suffix(ALT_EXTS[image_path.suffix])
    assert image_path.exists(), f"{image_path} doesn't exist"
    image = cv2.imread(str(image_path))[::downsample, ::downsample, ::-1] / 255.0
    return image


def _get_depth(depth_path, depth_dir, base_dir, downsample):
    if depth_dir is None:
        depth_path = base_dir / depth_path
    else:
        depth_path = depth_dir / os.path.basename(depth_path)
    if not depth_path.exists():
        depth_path = depth_path.with_suffix(ALT_EXTS[depth_path.suffix])
    assert depth_path.exists(), f"{depth_path} doesn't exist"
    if depth_path.suffix == ".gz":
        with gzip.open(depth_path, "rb") as f:
            depth = np.load(f)
    else:
        depth = np.load(depth_path.open("rb"))
        depth /= 1000.0  # depth generate from nerf is in meters, but GT is in centimeters
    depth = depth[::downsample, ::downsample]
    return depth


def _get_pointcloud(image, depth, extrinsics, grid, max_distance, feature_extractor):
    xyz_cam = grid * depth[..., None]  # x: right, y: down, z: forward
    xyz_cam = xyz_cam * [1, -1, -1]  # x: right, y: up, z: backward
    xyz = xyz_cam @ extrinsics[:3, :3].T + extrinsics[:3, -1]
    rgb = image

    desc = None
    if feature_extractor is not None:
        pts, desc, _ = feature_extractor.run(torch.from_numpy(image.astype(np.float32)).mean(-1))
        desc = desc.T
        xs, ys = pts[:2].astype(int)  # pts is [x,y,confidence]
        depth = depth[ys, xs]
        xyz = xyz[ys, xs]
        rgb = rgb[ys, xs]

    v = ~(np.isnan(depth) | np.isinf(depth))
    v &= (depth >= 0) & (depth <= max_distance)

    xyz = xyz[v]
    rgb = rgb[v]
    if desc is not None:
        desc = desc[v]

    return xyz, rgb, desc


def _clean_pointcloud(xyz, rgb, voxel_size, nb_neighbors, std_ratio):
    pc = open3d.geometry.PointCloud()
    pc.points = open3d.utility.Vector3dVector(xyz)
    pc.colors = open3d.utility.Vector3dVector(rgb)

    if voxel_size > 0:
        pc = pc.voxel_down_sample(voxel_size=voxel_size)

    if nb_neighbors > 0:
        pc, _ = pc.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    xyz = np.asarray(pc.points)
    rgb = np.asarray(pc.colors)
    return xyz, rgb



if __name__ == "__main__":
    generate_pointcloud(json_path="/home/liora/Lior/Datasets/svo/global_6_skip_8_hloc/transforms.json",
                        image_dir="/home/liora/Lior/Datasets/svo/global_6_skip_8_hloc/images",
                        depth_dir="/home/liora/Lior/Datasets/svo/global_6_skip_8_hloc/depth",
                 output_path=Path("/home/liora/Lior/Datasets/svo/global_6_skip_8_hloc/pointcloud.pcd"),
                        scale=1,
                        max_distance=2)
