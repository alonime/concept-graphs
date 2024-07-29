
import cv2 
import glob
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
from pathlib import Path
import shutil

import plotly.graph_objects as go

def detect_blur_images(data_path):
    color_paths = glob.glob(os.path.join(data_path, "images", "*.png"))
    vars = []
    for img_path in tqdm(color_paths):
        image = cv2.imread(str(img_path)) 
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(image_gray, cv2.CV_64F).var()
        vars.append(lap_var)
    
    fig = go.Figure(data=[go.Histogram(x=vars)])
    fig.show()

    thershold = 100
    vars_np = np.array(vars)
    vaild_images = [Path(color_paths[idx]) for idx in np.where(vars_np > thershold)[0]]

    new_dir = Path("/home/liora/Lior/Datasets/svo/robot_walking" + "_clean")
    new_dir.mkdir(exist_ok=True, parents=True)
    image_dir = new_dir / "images"
    depth_dir =  new_dir / "depth_neural_plus_meter"
    image_dir.mkdir(exist_ok=True, parents=True)
    depth_dir.mkdir(exist_ok=True, parents=True)



    for path_img in vaild_images:
        image_name = f"{path_img.stem}{path_img.suffix}"
        depth_name = f"{path_img.stem}.npy"
        depth_path = path_img.parent.parent / "depth_neural_plus_meter" / depth_name

        shutil.copy(path_img, image_dir / image_name)
        shutil.copy(depth_path, depth_dir / depth_name)





if __name__ == "__main__":
    data_path = "/home/liora/Lior/Datasets/svo/robot_walking"
    detect_blur_images(data_path)