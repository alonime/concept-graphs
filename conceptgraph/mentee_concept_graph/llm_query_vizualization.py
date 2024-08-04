import cv2
import os
# import PyQt5

# # Set the QT_QPA_PLATFORM_PLUGIN_PATH environment variable
# pyqt_plugin_path = os.path.join(os.path.dirname(PyQt5.__file__), "Qt", "plugins", "platforms")
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = pyqt_plugin_path

import supervision as sv
from supervision.draw.color import Color, ColorPalette

import copy
import json
import os
import pickle
import gzip
import argparse
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import open_clip
from openai import OpenAI
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import distinctipy
import open3d

# from conceptgraph.utils.pointclouds import Pointclouds
from conceptgraph.utils.pointclouds import Pointclouds

from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.utils.vis import LineMesh
from conceptgraph.slam.utils import filter_objects, merge_objects
from conceptgraph.mentee_concept_graph.llm_query import llm_retrive_object

def create_ball_mesh(center, radius, color=(0, 1, 0)):
    """
    Create a colored mesh sphere.
    
    Args:
    - center (tuple): (x, y, z) coordinates for the center of the sphere.
    - radius (float): Radius of the sphere.
    - color (tuple): RGB values in the range [0, 1] for the color of the sphere.
    
    Returns:
    - o3d.geometry.TriangleMesh: Colored mesh sphere.
    """
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh_sphere.translate(center)
    mesh_sphere.paint_uniform_color(color)
    return mesh_sphere

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--rgb_pcd_path", type=str, default=None)
    parser.add_argument("--edge_file", type=str, default=None)
    
    parser.add_argument("--no_clip", action="store_true", 
                        help="If set, the CLIP model will not init for fast debugging.")
    
    # To inspect the results of merge_overlap_objects
    # This is mainly to quickly try out different thresholds
    parser.add_argument("--merge_overlap_thresh", type=float, default=-1)
    parser.add_argument("--merge_visual_sim_thresh", type=float, default=-1)
    parser.add_argument("--merge_text_sim_thresh", type=float, default=-1)
    parser.add_argument("--obj_min_points", type=int, default=0)
    parser.add_argument("--obj_min_detections", type=int, default=0)
    
    return parser

def load_result(result_path):
    # check if theres a potential symlink for result_path and resolve it
    potential_path = os.path.realpath(result_path)
    if potential_path != result_path:
        print(f"Resolved symlink for result_path: {result_path} -> \n{potential_path}")
        result_path = potential_path
    with gzip.open(result_path, "rb") as f:
        results = pickle.load(f)

    if not isinstance(results, dict):
        raise ValueError("Results should be a dictionary! other types are not supported!")
    
    objects = MapObjectList()
    objects.load_serializable(results["objects"])
    bg_objects = MapObjectList()
    bg_objects.extend(obj for obj in objects if obj['is_background'])
    if len(bg_objects) == 0:
        bg_objects = None
    class_colors = results['class_colors']
        
    
        
    return objects, bg_objects, class_colors

def main(args):
    result_path = args.result_path
    rgb_pcd_path = args.rgb_pcd_path
    openai_client =  OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    
    assert not (result_path is None and rgb_pcd_path is None), \
        "Either result_path or rgb_pcd_path must be provided."

    if rgb_pcd_path is not None:        
        # pointclouds = Pointclouds.load_pointcloud_from_h5(rgb_pcd_path)
        # global_pcd = pointclouds.open3d(0, include_colors=True)
        global_pcd = open3d.io.read_point_cloud(rgb_pcd_path)

        if result_path is None:
            print("Only visualizing the pointcloud...")
            o3d.visualization.draw_geometries([global_pcd])
            exit()
        
    objects, bg_objects, class_colors = load_result(result_path)
    
    if args.edge_file is not None:
        # Load edge files and create meshes for the scene graph
        scene_graph_geometries = []
        with open(args.edge_file, "r") as f:
            edges = json.load(f)
        
        classes = objects.get_most_common_class()
        colors = [class_colors[str(c)] for c in classes]
        obj_centers = []
        for obj, c in zip(objects, colors):
            pcd = obj['pcd']
            bbox = obj['bbox']
            points = np.asarray(pcd.points)
            center = np.mean(points, axis=0)
            extent = bbox.get_max_bound()
            extent = np.linalg.norm(extent)
            # radius = extent ** 0.5 / 25
            radius = 0.10
            obj_centers.append(center)

            # remove the nodes on the ceiling, for better visualization
            ball = create_ball_mesh(center, radius, c)
            scene_graph_geometries.append(ball)
            
        for edge in edges:
            if edge['object_relation'] == "none of these":
                continue
            id1 = edge["object1"]['id']
            id2 = edge["object2"]['id']

            line_mesh = LineMesh(
                points = np.array([obj_centers[id1], obj_centers[id2]]),
                lines = np.array([[0, 1]]),
                colors = [1, 0, 0],
                radius=0.02
            )

            scene_graph_geometries.extend(line_mesh.cylinder_segments)
    
    if not args.no_clip:
        print("Initializing CLIP model...")
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
        clip_model = clip_model.to("cuda")
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
        print("Done initializing CLIP model.")

    print("Building json captioning")
    objects_captions = {}
    # for idx in range(len(objects)):
    #     objects_captions[idx] = objects[idx]['caption']
    
    # objects_captions_json = json.dumps(objects_captions)

    cmap = matplotlib.colormaps.get_cmap("turbo")
    
    if bg_objects is not None:
        indices_bg = []
        for obj_idx, obj in enumerate(objects):
            if obj['is_background']:
                indices_bg.append(obj_idx)
        # indices_bg = np.arange(len(objects), len(objects) + len(bg_objects))
        # objects.extend(bg_objects)
        
    # Sub-sample the point cloud for better interactive experience
    for i in range(len(objects)):
        pcd = objects[i]['pcd']
        # pcd = pcd.voxel_down_sample(0.05)
        objects[i]['pcd'] = pcd
    
    pcds = copy.deepcopy(objects.get_values("pcd"))
    bboxes = copy.deepcopy(objects.get_values("bbox"))
    
    # Get the color for each object when colored by their class
    object_classes = []
    for i in range(len(objects)):
        obj = objects[i]
        pcd = pcds[i]
        obj_classes = np.asarray(obj['class_id'])
        # Get the most common class for this object as the class
        values, counts = np.unique(obj_classes, return_counts=True)
        obj_class = values[np.argmax(counts)]
        object_classes.append(obj_class)
    
    # Set the title of the window
    vis = o3d.visualization.VisualizerWithKeyCallback()

    if result_path is not None:
        vis.create_window(window_name=f'Open3D - {os.path.basename(result_path)}', width=1280, height=720)
    else:
        vis.create_window(window_name=f'Open3D', width=1280, height=720)

    # Add geometry to the scene
    for geometry in pcds + bboxes:
        vis.add_geometry(geometry)

    print('Done loading!')
        
    main.show_bg_pcd = True
    def toggle_bg_pcd(vis):
        if bg_objects is None:
            print("No background objects found.")
            return
        
        for idx in indices_bg:
            if main.show_bg_pcd:
                vis.remove_geometry(pcds[idx], reset_bounding_box=False)
                vis.remove_geometry(bboxes[idx], reset_bounding_box=False)
            else:
                vis.add_geometry(pcds[idx], reset_bounding_box=False)
                vis.add_geometry(bboxes[idx], reset_bounding_box=False)
        
        main.show_bg_pcd = not main.show_bg_pcd
        
    main.show_global_pcd = False
    def toggle_global_pcd(vis):
        if args.rgb_pcd_path is None:
            print("No RGB pcd path provided.")
            return
        
        if main.show_global_pcd:
            vis.remove_geometry(global_pcd, reset_bounding_box=False)
        else:
            vis.add_geometry(global_pcd, reset_bounding_box=False)
        
        main.show_global_pcd = not main.show_global_pcd
        
    main.show_scene_graph = False
    def toggle_scene_graph(vis):
        if args.edge_file is None:
            print("No edge file provided.")
            return
        
        if main.show_scene_graph:
            for geometry in scene_graph_geometries:
                vis.remove_geometry(geometry, reset_bounding_box=False)
        else:
            for geometry in scene_graph_geometries:
                vis.add_geometry(geometry, reset_bounding_box=False)
        
        main.show_scene_graph = not main.show_scene_graph
        
    def color_by_class(vis):
        for i in range(len(objects)):
            pcd = pcds[i]
            obj_class = object_classes[i]
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile(
                    class_colors[str(obj_class)],
                    (len(pcd.points), 1)
                )
            )

        for pcd in pcds:
            vis.update_geometry(pcd)
            
    def color_by_rgb(vis):
        for i in range(len(pcds)):
            pcd = pcds[i]
            pcd.colors = objects[i]['pcd'].colors
        
        for pcd in pcds:
            vis.update_geometry(pcd)
            
    def color_by_instance(vis):
        instance_colors = cmap(np.linspace(0, 1, len(pcds)))
        for i in range(len(pcds)):
            pcd = pcds[i]
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile(
                    instance_colors[i, :3],
                    (len(pcd.points), 1)
                )
            )
            
        for pcd in pcds:
            vis.update_geometry(pcd)
        
    def color_by_clip_sim(vis):
        if args.no_clip:
            print("CLIP model is not initialized.")
            return

        text_query = input("Enter your query: ")
        text_queries = [text_query]
        
        text_queries_tokenized = clip_tokenizer(text_queries).to("cuda")
        text_query_ft = clip_model.encode_text(text_queries_tokenized)
        text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)
        text_query_ft = text_query_ft.squeeze()
        
        # similarities = objects.compute_similarities(text_query_ft)
        objects_clip_fts = objects.get_stacked_values_torch("clip_ft")
        objects_clip_fts = objects_clip_fts.to("cuda")
        similarities = F.cosine_similarity(
            text_query_ft.unsqueeze(0), objects_clip_fts, dim=-1
        )
        max_value = similarities.max()
        min_value = similarities.min()
        normalized_similarities = (similarities - min_value) / (max_value - min_value)
        probs = F.softmax(similarities, dim=0)
        max_prob_idx = torch.argmax(probs)
        similarity_colors = cmap(normalized_similarities.detach().cpu().numpy())[..., :3]

        max_prob_object = objects[max_prob_idx]
        print(f"Most probable object is at index {max_prob_idx} with class name '{max_prob_object['class_name']}'")
        print(f"location xyz: {max_prob_object['bbox'].center}")
        
        for i in range(len(objects)):
            pcd = pcds[i]
            map_colors = np.asarray(pcd.colors)
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile(
                    [
                        similarity_colors[i, 0].item(),
                        similarity_colors[i, 1].item(),
                        similarity_colors[i, 2].item()
                    ], 
                    (len(pcd.points), 1)
                )
            )

        for pcd in pcds:
            vis.update_geometry(pcd)

    def color_by_llm_query(vis):

        text_query = input("Enter your query: ")

        llm_response = llm_retrive_object(openai_client, text_query, objects_captions_json)
     
        if llm_response['query_achievable'] and len(llm_response['relevant_objects']):
            print("Found a fits object")
            print(f"LLM inferred task: {llm_response['inferred_query']}")
            best_fits = llm_response['final_relevant_objects'][0]
            print(f"Most probable object is with class name {objects[best_fits]['class_name']}")
            print(f"The first Caption is {objects[best_fits]['caption'][0]}")
            print(f"The LLM reasoning is: {llm_response['explanation']}")

            print(f"location xyz: {objects[best_fits]['bbox'].center}")


            objects_colors = np.zeros(len(objects))
            objects_colors[best_fits] = 1
            similarity_colors = cmap(objects_colors)[..., :3]
            img = Image.open(objects[best_fits]['color_path'][0])


            for i in range(len(objects)):
                pcd = pcds[i]
                map_colors = np.asarray(pcd.colors)
                pcd.colors = o3d.utility.Vector3dVector(
                    np.tile(
                        [
                            similarity_colors[i, 0].item(),
                            similarity_colors[i, 1].item(),
                            similarity_colors[i, 2].item()
                        ], 
                        (len(pcd.points), 1)
                    )
                )

            for pcd in pcds:
                vis.update_geometry(pcd)

            box_annotator = sv.BoxAnnotator(color = ColorPalette.default(),
                                            text_scale=0.3,
                                            text_thickness=1,
                                            text_padding=2,)
            
            curr_det = sv.Detections(xyxy=objects[best_fits]['xyxy'][0][None, ...])
            
            annotated_image = box_annotator.annotate(scene=img, detections=curr_det)
            fig = px.imshow(annotated_image)
            fig.show()



        else:
            print("Didn't find suitable object")


    def color_by_visual_query(vis):

        ref_img_query_path = "/home/liora/Lior/Datasets/svo/office_8_mono/chair.jpg"

        image = Image.open(ref_img_query_path)
        preprocessed_image = clip_preprocess(image).unsqueeze(0)
        image_features = clip_model.encode_image(preprocessed_image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        objects_clip_fts = objects.get_stacked_values_torch("clip_ft")
        objects_clip_fts = objects_clip_fts.to("cuda")
        similarities = F.cosine_similarity(image_features.unsqueeze(0), objects_clip_fts, dim=-1)
        max_value = similarities.max()
        min_value = similarities.min()
        normalized_similarities = (similarities - min_value) / (max_value - min_value)
        probs = F.softmax(similarities, dim=0)
        max_prob_idx = torch.argmax(probs)
        similarity_colors = cmap(normalized_similarities.detach().cpu().numpy())[..., :3]

        max_prob_object = objects[max_prob_idx]
        print(f"Most probable object is at index {max_prob_idx} with class name '{max_prob_object['class_name']}'")
        print(f"location xyz: {max_prob_object['bbox'].center}")
        
        for i in range(len(objects)):
            pcd = pcds[i]
            map_colors = np.asarray(pcd.colors)
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile(
                    [
                        similarity_colors[i, 0].item(),
                        similarity_colors[i, 1].item(),
                        similarity_colors[i, 2].item()
                    ], 
                    (len(pcd.points), 1)
                )
            )
        
        for pcd in pcds:
            vis.update_geometry(pcd)
      

        box_annotator = sv.BoxAnnotator(color = ColorPalette.default(),
                                        text_scale=0.3,
                                        text_thickness=1,
                                        text_padding=2,)
        
        curr_det = sv.Detections(xyxy=objects[max_prob_object]['xyxy'][0][None, ...])
        
        img = Image.open(objects[max_prob_object]['color_path'][0])

        annotated_image = box_annotator.annotate(scene=img, detections=curr_det)

        fig = make_subplots(rows=1, cols=2, subplot_titles=['refernce', 'best fit'])
        fig.add_trace(go.Image(z=annotated_image), row=1, col=1)
        fig.show()
   

            
    def save_view_params(vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters("temp.json", param)
        
    vis.register_key_callback(ord("B"), toggle_bg_pcd)
    vis.register_key_callback(ord("S"), toggle_global_pcd)
    vis.register_key_callback(ord("C"), color_by_class)
    vis.register_key_callback(ord("R"), color_by_rgb)
    vis.register_key_callback(ord("F"), color_by_clip_sim)
    vis.register_key_callback(ord("I"), color_by_instance)
    vis.register_key_callback(ord("V"), save_view_params)
    vis.register_key_callback(ord("G"), toggle_scene_graph)
    vis.register_key_callback(ord("L"), color_by_llm_query)
    vis.register_key_callback(ord("V"), color_by_visual_query)


    
    # Render the scene
    vis.run()
    
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
