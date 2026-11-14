#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import numpy as np


class FilterCam:
    """Minimal camera for GaussianModel.compute_3D_filter (CoMe)."""
    def __init__(self, cam):
        self.image_name = cam.image_name
        self.image_width = cam.image_width
        self.image_height = cam.image_height
        self.world_view_transform = cam.world_view_transform
        self.focal_x = cam.image_width / (2.0 * np.tan(cam.FoVx / 2.0))
        self.focal_y = cam.image_height / (2.0 * np.tan(cam.FoVy / 2.0))


def come_filter_cameras(cameras, llffhold=8):
    """CoMe computes filter_3D from its --eval training split: all images sorted by name, idx % llffhold != 0."""
    cams = sorted(cameras, key=lambda c: c.image_name)
    return [FilterCam(c) for idx, c in enumerate(cams) if idx % llffhold != 0]

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
       
        if args.train_distill:
            self.vanilla_3dgs_path = args.vanilla_3dgs_path
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if args.train_distill and load_iteration == -1:
                print(self.vanilla_3dgs_path)
                self.loaded_iter = searchForMaxIteration(os.path.join(self.vanilla_3dgs_path, "point_cloud"))
            elif load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))  
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        if "inpaint360" in args.source_path:
            self.inpaint_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args, args.images, args.eval, args.object_path, n_views=args.n_views, random_init=args.random_init, train_split=args.train_split)  #  COLMAP
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"


        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            if "inpaint360" in args.source_path:
                if scene_info.inpaint_cameras:
                    camlist.extend(scene_info.inpaint_cameras)

            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling    training dataset
            random.shuffle(scene_info.test_cameras)   # Multi-res consistent random shuffling
            if "inpaint360" in args.source_path:
                random.shuffle(scene_info.inpaint_cameras) 

        self.cameras_extent = scene_info.nerf_normalization["radius"]  

        for resolution_scale in resolution_scales:
            # print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            # print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

            if "inpaint360" in args.source_path:
                # print("Loading Inpainting Cameras")
                self.inpaint_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.inpaint_cameras, resolution_scale, args)

        if self.loaded_iter: 
            if args.train_distill:
                print("\nWe load vanilla_3dgs_path for distillation.")
                
                self.gaussians.load_ply(os.path.join(self.vanilla_3dgs_path,    
                                                        "point_cloud",
                                                        "iteration_" + str(self.loaded_iter),
                                                        "point_cloud.ply"))
            elif ".ply" in str(self.loaded_iter):
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                        "point_cloud"+self.loaded_iter))

            elif isinstance(self.loaded_iter,str):                                 
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud"+self.loaded_iter,
                                                            "point_cloud.ply"))    
            else:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "point_cloud.ply"))   
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        # CoMe: filter_3D from the CoMe training cameras (as CoMe render.py after loading); the cameras stay attached
        # so the renderer can recompute it whenever the number of points changes
        self.gaussians.filter_cameras = come_filter_cameras(self.train_cameras[resolution_scales[0]] +
                                                            self.test_cameras[resolution_scales[0]])
        self.gaussians.compute_3D_filter()
        print("3D filter computed from {} cameras".format(len(self.gaussians.filter_cameras)))

    def save(self, iteration):
      
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getInpaintCameras(self, scale=1.0):
        return self.inpaint_cameras[scale]
