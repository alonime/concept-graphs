import os
from pathlib import Path
import pickle
import gzip
import dataclasses
from typing import Literal
import logging
import datetime
import json
import signal

# Third-party imports
import cv2
import numpy as np
import torch
from tqdm import trange
from open3d.io import read_pinhole_camera_parameters
import open_clip
from ultralytics import YOLO, SAM
import supervision as sv
from omegaconf import OmegaConf


from openai import OpenAI

# Local application/library specific imports
from conceptgraph.utils.optional_rerun_wrapper import (
    OptionalReRun, 
    orr_log_annotated_image, 
    orr_log_camera, 
    orr_log_depth_image, 
    orr_log_edges, 
    orr_log_objs_pcd_and_bbox, 
    orr_log_rgb_image, 
    orr_log_vlm_image
)
from conceptgraph.utils.optional_wandb_wrapper import OptionalWandB
from conceptgraph.utils.geometry import rotation_matrix_to_quaternion
from conceptgraph.utils.logging_metrics import DenoisingTracker, MappingTracker
from conceptgraph.utils.vlm import get_obj_rel_from_image_gpt4v
from conceptgraph.utils.ious import mask_subtract_contained
from conceptgraph.utils.general_utils import (
    load_saved_detections, 
    load_saved_hydra_json_config, 
    make_vlm_edges, 
    save_detection_results,
    save_objects_for_frame, 
    save_pointcloud, 
    should_exit_early, 
    vis_render_image
)

from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils.vis import (
    OnlineObjectRenderer, 
    save_video_from_frames, 
    vis_result_fast_on_depth, 
    vis_result_for_vlm, 
    vis_result_fast, 
    save_video_detections
)
from conceptgraph.slam.slam_classes import MapEdgeMapping, MapObjectList
from conceptgraph.slam.utils import (
    filter_gobs,
    filter_objects,
    get_bounding_box,
    init_process_pcd,
    make_detection_list_from_pcd_and_gobs,
    denoise_objects,
    merge_objects, 
    detections_to_obj_pcd_and_bbox,
    prepare_objects_save_vis,
    process_edges,
    process_pcd,
    processing_needed,
    resize_gobs
)
from conceptgraph.slam.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    match_detections_to_objects,
    merge_obj_matches
)
from conceptgraph.utils.model_utils import compute_clip_features_batched

from conceptgraph.mentee_concept_graph.mentee_vlm import image_captioning
from conceptgraph.mentee_concept_graph.svo_dataset import SVODataset


# Disable torch gradient computation
torch.set_grad_enabled(False)

@dataclasses.dataclass
class SceneGraphConfigs:

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_stride: int = 8

    wandb: bool = False
    "Log in wand indecator"
    build_visualization: Literal['rerun', 'open3d', 'none'] = 'rerun'
    dataset_root = Path("/home/liora/Lior/Datasets/svo")
    scene_id= "robot_walking"
    dataset_config: Literal["transforms.json", "dataconfig.yaml"] = "transforms.json"

    classes_file = "conceptgraph/scannet200_classes.txt"
    bg_classes = ["wall", "floor", "ceiling"]
    skip_bg: bool = False

    detectoin_model: str = "yolov8l-world.pt"
    detection_threshold: float = 0.3
    segmentaion_model: str = "mobile_sam.pt"  # UltraLytics SAM
    clip_name: str = "ViT-H-14"
    clip_pretrained: str = "laion2b_s32b_b79k"

    save_detections = True
    load_detections = True
    filter_persons = True

    build_spetial_edges: bool = False
    mask_area_threshold: float = 25   # mask with pixel area less than this will be skipped
    mask_conf_threshold: float = 0.2   # mask with lower confidence score will be skipped
    max_bbox_area_ratio: float = 1.0  # boxes with larger areas than this will be skipped
    min_points_threshold: int = 16    # projected and sampled pcd with less points will be skipped
    spatial_sim_type: Literal["iou", "giou", "overlap"] = "overlap" 
    obj_pcd_max_points: int = 5000 
    """ Determines the maximum point count for object point clouds; exceeding this triggers downsampling to approx max points. Set to -1 to disable """

    # point cloud processing
    downsample_voxel_size: float = 0.025
    dbscan_remove_noise: bool = True
    dbscan_eps: float = 0.05
    dbscan_min_points: float = 10

    phys_bias: float = 0.0
    match_method: Literal["sep_thresh", "sim_sum"] =  "sim_sum"
    # Only when match_method=="sep_thresh"
    semantic_threshold: float = 0.5
    physical_threshold: float = 0.5
    # Only when match_method=="sim_sum"
    sim_threshold: float = 1.2

    denoise_interval: int = 5           # Run DBSCAN every k frame. This operation is heavy
    filter_interval: int = 5            # Filter objects that have too few associations or are too small
    merge_interval: int = 5             # Merge objects based on geometric and semantic similarity
    run_denoise_final_frame: bool = True
    run_filter_final_frame: bool = True
    run_merge_final_frame: bool = True

    # For merge_overlap_objects() function
    merge_overlap_thresh: float = 0.7      # -1 means do not perform the merge_overlap_objects()
    merge_visual_sim_thresh: float = 0.7   # Merge only if the visual similarity is larger
    merge_text_sim_thresh: float = 0.7     # Merge only if the text cosine sim is larger

    # Selection criteria of the fused object point cloud
    obj_min_points: int = 0
    obj_min_detections: int = 3



class SceneGraph:

    def __init__(self) -> None:
        self.configs = SceneGraphConfigs()

        self.tracker = MappingTracker()
        self.viz = None
        if self.configs.build_visualization == 'rerun':
            self.viz = OptionalReRun()
            self.viz.set_use_rerun(True)
            self.viz.init("realtime_mapping")
            self.viz.spawn()

        if self.configs.wandb:
            self.owandb = OptionalWandB()
            self.owandb.set_use_wandb(self.configs.wandb)
            self.owandb.init(project="concept-graphs", config=dataclasses.asdict(self.configs),)


        self.obj_classes =  ObjectClasses(classes_file_path=self.configs.classes_file, 
                                          bg_classes=self.configs.bg_classes, 
                                          skip_bg=self.configs.skip_bg)

        self.detection_model, self.segmentaion_model = None, None
        self.clip_model, self.clip_preprocess, self.clip_tokenizer = None, None, None
        self.dataset = None

        self.dataset_config_path = self.configs.dataset_root / self.configs.scene_id / self.configs.dataset_config
        self.dataset_config = OmegaConf.load(self.dataset_config_path)

        self.objects, self.map_edges= None



    def init_dataset(self,):
         # Initialize the dataset
        self.dataset = SVODataset(config_dict=self.dataset_config,
                                  start=0,
                                  end=-1,
                                  stride=self.configs.dataset_stride,
                                  basedir=self.configs.dataset_root,
                                  sequence=self.configs.scene_id,
                                  device="cpu",
                                  dtype=torch.float)
        # self.dataset = get_dataset(dataconfig=self.configs.dataset_root / self.configs.scene_id / self.configs.dataset_config, #TODO: @lior, refactor to dataset class
        #                            start=0,
        #                            end=-1,
        #                            stride=self.configs.dataset_stride,
        #                            basedir=self.configs.dataset_root,
        #                            sequence=self.configs.scene_id,
        #                            desired_height=self.dataset_config.camera_params.image_height,
        #                            desired_width=self.dataset_config.camera_params.image_width,
        #                            device="cpu",
        #                            dtype=torch.float)
        
    def init_models(self,):
        logging.info("Iinitazlize models:")
        logging.info(f"Detectoin Model: {self.configs.detectoin_model}")
        self.detection_model = YOLO(self.configs.detectoin_model)
        self.detection_model.set_classes(self.obj_classes.get_classes_arr())

        logging.info(f"Segmentaion Model: {self.configs.detectoin_model}")
        # segmentaion_model = SAM('sam_l.pt') # SAM('mobile_sam.pt') # UltraLytics SAM
        self.segmentaion_model = SAM('mobile_sam.pt') # UltraLytics SAM
        # segmentaion_model = measure_time(get_segmentaion_model)(cfg) # Normal SAM
        
        logging.info(f"CLIP Model: {self.configs.clip_name} | Pretrained Weights: {self.configs.clip_pretrained}")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(self.configs.clip_name, 
                                                                               self.configs.clip_pretrained)
        self.clip_model = self.clip_model.to(self.configs.device)
        self.clip_tokenizer = open_clip.get_tokenizer(self.configs.clip_name)
        
        assert os.getenv('OPENAI_API_KEY'), "OpenAI API key is required is the environment variables, and it don't exsit"
        logging.info("OpenAI clinet: GPT-4 LLM")
        self.openai_client =  OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


    def build_scene(self):

        assert self.dataset is not None, "Dataset isn't initalize"

        self.results_dir =  Path(self.configs.dataset_root) / self.configs.scene_id / "exps" / f"reconstruction_stride_{self.configs.dataset_stride}"
        self.results_dir_detections = self.results_dir / "detections"
        self.results_dir_vis = self.results_dir / "vis"
        self.resutls_dir_objects = self.results_dir / "saved_obj_all_frames"
        
        self.results_dir_detections.mkdir(exist_ok=True, parents=True)
        self.results_dir_vis.mkdir(exist_ok=True, parents=True)
        self.resutls_dir_objects.mkdir(exist_ok=True, parents=True)


        logging.info(f"Save results in {self.results_dir}")

        self.objects = MapObjectList(device=self.configs.device)
        self.map_edges = MapEdgeMapping(self.objects)

        
        prev_adjusted_pose = None

        counter = 0
        for frame_idx in trange(len(self.dataset)):
        # for frame_idx in trange(10):
            self.tracker.curr_frame_idx = frame_idx
            counter += 1
            self.viz.set_time_sequence("frame", frame_idx)


            rgb_path = Path(self.dataset.color_paths[frame_idx])
            rgb_tensor, depth_tensor, intrinsics, *_ = self.dataset[frame_idx]

            # Covert to numpy and do some sanity checks
            depth_tensor = depth_tensor[..., 0]
            depth_array = depth_tensor.cpu().numpy()
            color_np = rgb_tensor.cpu().numpy() # (H, W, 3)
            image_rgb = (color_np).astype(np.uint8) # (H, W, 3)
            assert image_rgb.max() > 1, "Image is not in range [0, 255]"

            # Load image detections for the current frame
            raw_grounded_obs = None
            grounded_obs = None # stands for grounded observations
                        
            
            results = None
            image = cv2.imread(str(rgb_path)) 
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Do initial object detection
            results = self.detection_model.predict(rgb_path, conf=self.configs.detection_threshold, verbose=False)
            confidences = results[0].boxes.conf.cpu().numpy()
            detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            detection_class_labels = [f"{self.obj_classes.get_classes_arr()[class_id]} {class_idx}" for class_idx, class_id in enumerate(detection_class_ids)]
            xyxy_tensor = results[0].boxes.xyxy
            xyxy_np = xyxy_tensor.cpu().numpy()

            # if there are detections, use segmentaion model
            if xyxy_tensor.numel() != 0:
                sam_out = self.segmentaion_model.predict(rgb_path, bboxes=xyxy_tensor, verbose=False)
                masks_tensor = sam_out[0].masks.data

                masks_np = masks_tensor.cpu().numpy()
            else:
                masks_np = np.empty((0, *rgb_tensor.shape[:2]), dtype=np.float64)

            # Create a detections object that we will save later
            curr_det = sv.Detections(
                xyxy=xyxy_np,
                confidence=confidences,
                class_id=detection_class_ids,
                mask=masks_np,
            )
            

            # Make the edges
            labels, edges, edge_image = make_vlm_edges(image, 
                                                                 curr_det, 
                                                                 self.obj_classes, 
                                                                 detection_class_labels, 
                                                                 self.results_dir_vis, 
                                                                 rgb_path, 
                                                                 self.configs.build_spetial_edges, 
                                                                 self.openai_client)

            image_crops, image_feats, text_feats = compute_clip_features_batched(image_rgb, 
                                                                                 curr_det, 
                                                                                 self.clip_model, 
                                                                                 self.clip_preprocess, 
                                                                                 self.clip_tokenizer, 
                                                                                 self.obj_classes.get_classes_arr(), 
                                                                                 self.configs.device)
            


            # increment total object detections
            self.tracker.increment_total_detections(len(curr_det.xyxy))

            # Save results
            # Convert the detections to a dict. The elements are in np.array
            results = {
                # add new uuid for each detection 
                "xyxy": curr_det.xyxy,
                "confidence": curr_det.confidence,
                "class_id": curr_det.class_id,
                "mask": curr_det.mask,
                "classes": self.obj_classes.get_classes_arr(),
                "image_crops": image_crops,
                "image_feats": image_feats,
                "text_feats": text_feats,
                "detection_class_labels": detection_class_labels,
                "labels": labels,
                "edges": edges,
            }

            # save the detections if needed
            if self.configs.save_detections:

                vis_save_path = (self.results_dir_vis / rgb_path.name).with_suffix(".jpg")
                # Visualize and save the annotated image
                annotated_image, labels = vis_result_fast(image, curr_det, self.obj_classes.get_classes_arr())
                cv2.imwrite(str(vis_save_path), annotated_image)

                depth_image_rgb = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
                depth_image_rgb = depth_image_rgb.astype(np.uint8)
                depth_image_rgb = cv2.cvtColor(depth_image_rgb, cv2.COLOR_GRAY2BGR)
                annotated_depth_image, labels = vis_result_fast_on_depth(depth_image_rgb, curr_det, self.obj_classes.get_classes_arr())
                cv2.imwrite(str(vis_save_path).replace(".jpg", "_depth.jpg"), annotated_depth_image)
                cv2.imwrite(str(vis_save_path).replace(".jpg", "_depth_only.jpg"), depth_image_rgb)
                save_detection_results(self.results_dir_detections / vis_save_path.stem, results)


            raw_grounded_obs = results
        
            # get pose, this is the untrasformed pose.
            unt_pose = self.dataset.poses[frame_idx]
            unt_pose = unt_pose.cpu().numpy()

            # Don't apply any transformation otherwise
            adjusted_pose = unt_pose
            
            if self.viz is not None:
                prev_adjusted_pose = orr_log_camera(intrinsics,
                                                    adjusted_pose,
                                                    prev_adjusted_pose,
                                                    self.dataset_config.w,
                                                    self.dataset_config.h,
                                                    frame_idx)
                orr_log_rgb_image(rgb_path)
                orr_log_annotated_image(rgb_path, vis_save_path)
                orr_log_depth_image(depth_tensor)
                orr_log_vlm_image((self.results_dir_vis / rgb_path.name).with_suffix(".jpg"))
                orr_log_vlm_image((self.results_dir_vis / rgb_path.name).with_suffix(".jpg"), label="w_edges")


            # resize the observation if needed
            resized_grounded_obs = resize_gobs(raw_grounded_obs, image_rgb)
            # filter the observations
            filtered_grounded_obs = filter_gobs(resized_grounded_obs, 
                                                image_rgb, 
                                                skip_bg=self.configs.skip_bg,
                                                BG_CLASSES=self.obj_classes.get_bg_classes_arr(),
                                                mask_area_threshold=self.configs.mask_area_threshold,
                                                max_bbox_area_ratio=self.configs.max_bbox_area_ratio,
                                                mask_conf_threshold=self.configs.mask_conf_threshold,)

            grounded_obs = filtered_grounded_obs

            if len(grounded_obs['mask']) == 0: # no detections in this frame
                continue

            # this helps make sure things like pillows on couches are separate objects
            grounded_obs['mask'] = mask_subtract_contained(grounded_obs['xyxy'], grounded_obs['mask'])

            obj_pcds_and_bboxes = detections_to_obj_pcd_and_bbox(depth_array=depth_array,
                                                                 masks=grounded_obs['mask'],
                                                                 cam_K=intrinsics.cpu().numpy()[:3, :3],  # Camera intrinsics
                                                                 image_rgb=image_rgb,
                                                                 trans_pose=adjusted_pose,
                                                                 min_points_threshold=self.configs.min_points_threshold,
                                                                 spatial_sim_type=self.configs.spatial_sim_type,
                                                                 obj_pcd_max_points=self.configs.obj_pcd_max_points,
                                                                 device=self.configs.device)

            for obj in obj_pcds_and_bboxes:
                if obj:
                    obj["pcd"] = init_process_pcd(pcd=obj["pcd"],
                                                  downsample_voxel_size=self.configs.downsample_voxel_size,
                                                  dbscan_remove_noise=self.configs.dbscan_remove_noise,
                                                  dbscan_eps=self.configs.dbscan_eps,
                                                  dbscan_min_points=self.configs.dbscan_min_points,)
                    
                    obj["bbox"] = get_bounding_box(spatial_sim_type=self.configs.spatial_sim_type, 
                                                   pcd=obj["pcd"])
                    
            
            captions_crops, detections_captions = image_captioning(image_rgb, self.openai_client, grounded_obs, obj_pcds_and_bboxes)
            grounded_obs['caption'] = detections_captions

            detection_list = make_detection_list_from_pcd_and_gobs(obj_pcds_and_bboxes, 
                                                                   grounded_obs, 
                                                                   rgb_path, 
                                                                   self.obj_classes, 
                                                                   frame_idx)

            if len(detection_list) == 0: # no detections, skip
                continue

            # if no objects yet in the map,
            # just add all the objects from the current frame
            # then continue, no need to match or merge
            if len(self.objects) == 0:
                self.objects.extend(detection_list)
                self.tracker.increment_total_objects(len(detection_list))
                if self.configs.wandb:
                    self.owandb.log({"total_objects_so_far": self.tracker.get_total_objects(),
                                     "objects_this_frame": len(detection_list)})
                continue 

            ### compute similarities and then merge
            spatial_sim = compute_spatial_similarities(
                spatial_sim_type=self.configs.spatial_sim_type, 
                detection_list=detection_list, 
                objects=self.objects,
                downsample_voxel_size=self.configs.downsample_voxel_size
            )

            visual_sim = compute_visual_similarities(detection_list, self.objects)

            agg_sim = aggregate_similarities(match_method=self.configs.match_method, 
                                             phys_bias=self.configs.phys_bias, 
                                             spatial_sim=spatial_sim, 
                                             visual_sim=visual_sim)
            

            # Perform matching of detections to existing objects
            match_indices = match_detections_to_objects(agg_sim=agg_sim, 
                                                        detection_threshold=self.configs.sim_threshold)

            # Now merge the detected objects into the existing objects based on the match indices
            self.objects = merge_obj_matches(detection_list=detection_list, 
                                             objects=self.objects,
                                             match_indices=match_indices,
                                             downsample_voxel_size=self.configs.downsample_voxel_size, 
                                             dbscan_remove_noise=self.configs.dbscan_remove_noise, 
                                             dbscan_eps=self.configs.dbscan_eps, 
                                             dbscan_min_points=self.configs.dbscan_min_points, 
                                             spatial_sim_type=self.configs.spatial_sim_type, 
                                             device=self.configs.device
                                             # Note: Removed 'match_method' and 'phys_bias' as they do not appear in the provided merge function
                                             )
            self.map_edges = process_edges(match_indices, grounded_obs, len(self.objects), self.objects, self.map_edges)

            is_final_frame = frame_idx == len(self.dataset) - 1
            if is_final_frame:
                print("Final frame detected. Performing final post-processing...")

            ### Perform post-processing periodically if told so

            # Denoising
            if processing_needed(self.configs.denoise_interval,
                                 self.configs.run_denoise_final_frame,
                                 frame_idx,  
                                 is_final_frame):
                
                self.objects = denoise_objects(downsample_voxel_size=self.configs.downsample_voxel_size, 
                                          dbscan_remove_noise=self.configs.dbscan_remove_noise, 
                                          dbscan_eps=self.configs.dbscan_eps, 
                                          dbscan_min_points=self.configs.dbscan_min_points, 
                                          spatial_sim_type=self.configs.spatial_sim_type, 
                                          device=self.configs.device, 
                                          objects=self.objects)

            # Filtering
            if processing_needed(self.configs.filter_interval,
                                 self.configs.run_filter_final_frame,
                                 frame_idx,
                                 is_final_frame):
                
                self.objects = filter_objects(obj_min_points=self.configs.obj_min_points, 
                                         obj_min_detections=self.configs.obj_min_detections, 
                                         objects=self.objects,
                                         map_edges=self.map_edges)

            # Merging
            if processing_needed(self.configs.merge_interval,
                                 self.configs.run_merge_final_frame,
                                 frame_idx,
                                 is_final_frame):
                
                self.objects, self.map_edges = merge_objects(merge_overlap_thresh=self.configs.merge_overlap_thresh,
                                                   merge_visual_sim_thresh=self.configs.merge_visual_sim_thresh,
                                                   merge_text_sim_thresh=self.configs.merge_text_sim_thresh,
                                                   objects=self.objects,
                                                   downsample_voxel_size=self.configs.downsample_voxel_size,
                                                   dbscan_remove_noise=self.configs.dbscan_remove_noise,
                                                   dbscan_eps=self.configs.dbscan_eps,
                                                   dbscan_min_points=self.configs.dbscan_min_points,
                                                   spatial_sim_type=self.configs.spatial_sim_type,
                                                   device=self.configs.device,
                                                   do_edges=True,
                                                   map_edges=self.map_edges)
                
            orr_log_objs_pcd_and_bbox(self.objects, self.obj_classes)
            orr_log_edges(self.objects, self.map_edges, self.obj_classes)

            if self.configs.save_detections:
                save_objects_for_frame(self.resutls_dir_objects,
                                       frame_idx,
                                       self.objects,
                                       self.configs.obj_min_detections,
                                       adjusted_pose,
                                       rgb_path)
            
            if self.configs.wandb:
                self.owandb.log({"frame_idx": frame_idx,
                                 "counter": counter,
                                 "is_final_frame": is_final_frame,})

            self.tracker.increment_total_objects(len(self.objects))
            self.tracker.increment_total_detections(len(detection_list))
            if self.configs.wandb:
                self.owandb.log({"total_objects": self.tracker.get_total_objects(),
                                 "objects_this_frame": len(self.objects),
                                 "total_detections": self.tracker.get_total_detections(),
                                 "detections_this_frame": len(detection_list),
                                 "frame_idx": frame_idx,
                                 "counter": counter,
                                 "is_final_frame": is_final_frame,})
                
        # LOOP OVER -----------------------------------------------------

        # Save the pointcloud

    def save_pcd(self,):

        save_pointcloud(exp_suffix="final_results",
                        exp_out_path=self.results_dir,
                        cfg=dataclasses.asdict(self.configs),
                        objects=self.objects,
                        obj_classes=self.obj_classes,
                        latest_pcd_filepath="latest_pcd_save.pcd",
                        create_symlink=True,
                        edges=self.map_edges)

        # Save metadata if all frames are saved
        if self.configs.save_detections:
            save_meta_path = self.resutls_dir_objects / f"meta.pkl.gz"
            with gzip.open(save_meta_path, "wb") as f:
                pickle.dump({'cfg': dataclasses.asdict(self.configs),
                             'class_names': self.obj_classes.get_classes_arr(),
                             'class_colors': self.obj_classes.get_class_color_dict_by_index()}, f)

        if self.configs.wandb:
            self.owandb.finish()


        
if __name__ == "__main__":
    scene_graph = SceneGraph()

    signal.signal()
    scene_graph.init_dataset()
    scene_graph.init_models()
    scene_graph.build_scene()
