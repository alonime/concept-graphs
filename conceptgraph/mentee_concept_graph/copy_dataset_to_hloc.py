import plotly.graph_objects as go
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import shutil




data_path = Path("/home/liora/Lior/Datasets/svo/office_60fps_skip_8")
transform_info = OmegaConf.load(data_path / "transforms.json")

output_path = Path("/home/liora/Lior/Datasets/svo/office_60fps_skip_8_tmp")
output_images_path = output_path / "images"
output_depth_path =  output_path / "depth"

output_path.mkdir(exist_ok=True)
output_depth_path.mkdir(exist_ok=True)
output_images_path.mkdir(exist_ok=True)



loc = []

for frame in transform_info.frames:
    loc.append((frame.file_path, frame.depth_file_path))

loc.sort(key= lambda x: x[0])

for idx, frame in enumerate(loc):
    file_name = str(idx + 1).zfill(6)
    image_name = file_name + ".png"
    depth_name = file_name + ".npy"
    shutil.copy(data_path / frame[0], output_images_path / image_name)
    shutil.copy(data_path / frame[1], output_depth_path / depth_name)



