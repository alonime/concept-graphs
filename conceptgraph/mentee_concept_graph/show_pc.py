from pathlib import Path
from typing import List

import os

import numpy as np
import open3d

import os
import pickle
import sys
import matplotlib.pyplot as plt


from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Union
from urllib.parse import urlparse

import boto3
import requests

from botocore.exceptions import ClientError
from tqdm import tqdm

COLORS = np.asarray(plt.get_cmap("tab20").colors)


def validate_path(path):
    if path is None:
        return None

    return Path(get_local_path(path))


def download_from_s3(local_filename, url):
    p = url.replace("s3://", "").split("/")
    bucket_name = p[0]
    object_key = "/".join(p[1:])

    s3 = boto3.client("s3")
    meta_data = s3.head_object(Bucket=bucket_name, Key=object_key)
    total_length = int(meta_data.get("ContentLength", 0))

    prog_bar = tqdm(
        total=total_length,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        file=sys.stdout,
        desc=f"Downloading from s3://{bucket_name}/{object_key}",
    )
    os.makedirs(os.path.dirname(local_filename), exist_ok=True)
    with open(local_filename, "wb") as f:
        s3.download_fileobj(bucket_name, object_key, f, Callback=prog_bar.update)


def download_from_http(local_filename, url):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte

    os.makedirs(os.path.dirname(local_filename), exist_ok=True)
    progress_bar = tqdm(
        total=total_size_in_bytes, unit="iB", unit_scale=True, file=sys.stdout, desc=f"Downloading from {url}"
    )
    with open(local_filename, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")


def resolve(p: Union[Path, List[Path]]) -> List[Path]:
    if isinstance(p, Path):
        pass
    elif isinstance(p, list) and all(map(lambda pp: isinstance(pp, Path), p)):
        return sum(map(resolve, p), [])
    else:
        raise ValueError("input must be a path or list of paths")

    if "*" not in str(p):
        return [p]

    parts = p.parts
    idx = min(i for i, part in enumerate(parts) if "*" in part)
    prefix = os.path.join(*parts[:idx])
    suffix = os.path.join(*parts[idx:])
    return list(Path(prefix).glob(suffix))


def create_new_folder_in_s3(bucket_name: str, folder_name: str) -> bool:
    try:
        # Create an S3 client
        s3 = boto3.client("s3")

        # Create an empty object in the S3 bucket to represent the folder
        s3.put_object(Bucket=bucket_name, Key=f"{folder_name}/")
        print(f"Folder '{folder_name}' created successfully in S3 bucket '{bucket_name}'.")
        return True
    except ClientError as e:
        print(f"Error creating folder '{folder_name}' in S3 bucket '{bucket_name}': {e}")
        return False


def dump_file_to_s3(filepath: str, obj_to_save, delete_if_exists: bool = False) -> bool:
    try:
        # Create an S3 client
        s3 = boto3.client("s3")

        # Extract the bucket name and folder path from the S3 filepath
        parsed_url = urlparse(str(filepath))
        bucket_name = parsed_url.path.split("/")[1]
        folder_name = parsed_url.path.split("/")[-2]
        file_path_without_bucket = "/".join(parsed_url.path.split("/")[-2:])

        # Create the directory if it doesn't exist
        try:
            resp = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)
            if "Contents" not in resp:
                create_new_folder_in_s3(bucket_name, folder_name)
        except ClientError as e:
            print(f"An error occurred while checking if folder '{folder_name}' exists in bucket '{bucket_name}': {e}")
            return False

        # Delete file if exists
        if delete_if_exists:
            try:
                s3.delete_object(Bucket=bucket_name, Key=file_path_without_bucket)
            except Exception as e:
                print(f"Error deleting file '{file_path_without_bucket}' from S3 bucket '{bucket_name}': {e}")

        # Upload the file to S3
        # Convert the NumPy array to bytes using pickle
        bytes_buffer = pickle.dumps(obj_to_save)

        # Upload the bytes buffer to S3
        s3.upload_fileobj(BytesIO(bytes_buffer), bucket_name, file_path_without_bucket)

        print(f"File '{filepath}' uploaded successfully to S3 bucket '{bucket_name}'.")
        return True
    except ClientError as e:
        print(f"Error uploading file '{filepath}' to S3 bucket '{bucket_name}': {e}")
        return False


def get_local_path(url: Union[str, Tuple[str], List[str]]):
    if url is None:
        return None

    if isinstance(url, (tuple, list)):
        return list(map(get_local_path, url))

    url = str(url)

    for prefix in ["s3", "http", "https"]:
        if url.startswith(prefix + ":/") and not url.startswith(prefix + "://"):
            url = url.replace(prefix + ":/", prefix + "://")
            break

    if os.path.exists(url):
        return url

    trg = os.path.join(os.path.expanduser("~"), ".cache", "perception", url.replace("://", "/"))
    if os.path.exists(trg):
        return trg

    if url.startswith("http"):
        download_from_http(trg, url)
    elif url.startswith("s3"):
        if check_s3_path_type(url) == "file":
            download_from_s3(trg, url)
        elif check_s3_path_type(url) == "folder":
            files = get_all_aws_files_from_path(url)
            for file in files:
                file_name = file.split("/")[-1]
                if file_name != "":
                    local_path = os.path.join(trg, file_name)
                    s3_path = os.path.join(url, file_name)
                    download_from_s3(local_path, s3_path)
        else:
            raise RuntimeError("S3 path: " + url + ", not found!")
    else:
        return url

    return trg


def check_s3_path_type(url):
    tmp = url.split("/", 1)[-1][1:]
    bucket_name = tmp.split("/")[0]
    prefix = tmp.split("/", 1)[-1]
    s3_client = boto3.client("s3")
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if "Contents" in response:
        if len(response["Contents"]) > 1:
            return "folder"
        elif len(response["Contents"]) == 1:
            if response["Contents"][0]["Key"] == prefix:
                return "file"
    else:
        return "not_found"


def get_all_aws_files_from_path(url):
    # Extract bucket name and prefix from the provided AWS path
    tmp = url.split("/", 1)[-1][1:]
    bucket_name = tmp.split("/")[0]
    prefix = tmp.split("/", 1)[-1]
    # Use Boto3 to list objects in the specified bucket and prefix
    s3_client = boto3.client("s3")
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    # Iterate through the response to collect object URLs
    object_urls = []
    for obj in response["Contents"]:
        object_url = f"https://{bucket_name}.s3.amazonaws.com/{obj['Key']}"
        object_urls.append(object_url)
    # Update the member variable with the collected object URLs
    return object_urls



def get_grid_line_set(min_bound, max_bound, voxel_size=1.0, color=(0, 0, 0)):
    min_bound = np.floor(np.asarray(min_bound) / voxel_size) * voxel_size
    max_bound = np.ceil(np.asarray(max_bound) / voxel_size) * voxel_size
    xx = np.arange(min_bound[0], max_bound[0] + voxel_size, voxel_size)
    yy = np.arange(min_bound[1], max_bound[1] + voxel_size, voxel_size)
    zz = np.arange(min_bound[2], max_bound[2] + voxel_size, voxel_size)
    points = {}
    lines = []
    for x in xx:
        for y in yy:
            p0 = (x, y, zz[0])
            p1 = (x, y, zz[-1])
            idx0 = points.setdefault(p0, len(points))
            idx1 = points.setdefault(p1, len(points))
            lines.append((idx0, idx1))
    for x in xx:
        for z in zz:
            p0 = (x, yy[0], z)
            p1 = (x, yy[-1], z)
            idx0 = points.setdefault(p0, len(points))
            idx1 = points.setdefault(p1, len(points))
            lines.append((idx0, idx1))
    for y in yy:
        for z in zz:
            p0 = (xx[0], y, z)
            p1 = (xx[-1], y, z)
            idx0 = points.setdefault(p0, len(points))
            idx1 = points.setdefault(p1, len(points))
            lines.append((idx0, idx1))
    points = sorted([(v, k) for k, v in points.items()])
    points = [k for v, k in points]

    line_set = open3d.geometry.LineSet()
    line_set.points = open3d.utility.Vector3dVector(points)
    line_set.lines = open3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color(color)
    return line_set



def show_pointcloud(
    pointcloud_paths: List[Path], /, side_by_side: bool = False, color_by_source: bool = False, draw_grid: bool = False
):
    """Visualize a pointcloud file

    Args:
        pointcloud_path: path to pointcloud file
    """
    pointcloud_paths_ = pointcloud_paths
    pointcloud_paths = resolve(pointcloud_paths)
    if len(pointcloud_paths) == 0:
        raise ValueError(f"No paths found for '{pointcloud_paths_}'")

    visualizer = open3d.visualization.Visualizer()
    visualizer.create_window()

    total = 0
    shift = 0
    min_bound = (np.inf, np.inf, np.inf)
    max_bound = (-np.inf, -np.inf, -np.inf)
    for idx, pointcloud_path in enumerate(pointcloud_paths):
        pointcloud_path = get_local_path(pointcloud_path)
        pointcloud = open3d.io.read_point_cloud(str(pointcloud_path))

        if color_by_source:
            pointcloud.colors = open3d.utility.Vector3dVector(
                np.tile(COLORS[idx % len(COLORS)], (len(pointcloud.points), 1))
            )

        if side_by_side:
            points = np.asarray(pointcloud.points)
            points[:, 0] += shift - np.min(points[:, 0])
            pointcloud.points = open3d.utility.Vector3dVector(points)
            shift = np.max(points[:, 0]) + 2

        visualizer.add_geometry(pointcloud)
        print(f"Loaded {len(pointcloud.points)} points from {pointcloud_path}")
        total += len(pointcloud.points)

        bounding_box = pointcloud.get_axis_aligned_bounding_box()
        min_bound = np.minimum(min_bound, bounding_box.min_bound)
        max_bound = np.maximum(max_bound, bounding_box.max_bound)

    if draw_grid:
        line_set = get_grid_line_set(min_bound, max_bound)
        visualizer.add_geometry(line_set)

    print(f"Total #points: {total}")
    visualizer.run()
    visualizer.destroy_window()
