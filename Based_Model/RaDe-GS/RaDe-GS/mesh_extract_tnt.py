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
# refer to https://github.com/zju3dv/PGSR/blob/main/scripts/render_tnt.py
import os, sys
import math
from pathlib import Path

dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
print(f"dir_path {dir_path}")
sys.path.append(dir_path.__str__())

import json
import torch
from scene import Scene
from tqdm import tqdm
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import open3d as o3d
import trimesh, copy
from utils.graphics_utils import depth_to_normal, get_points_from_depth


def post_process_mesh(mesh, cluster_to_keep=1):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy

    print(
        "post processing the mesh to have {} clusterscluster_to_kep".format(
            cluster_to_keep
        )
    )
    mesh_0 = mesh
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh_0.cluster_connected_triangles()
        )

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50)  # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0


def clean_mesh(mesh, min_len=1000):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh.cluster_connected_triangles()
        )
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_len
    mesh_0 = copy.deepcopy(mesh)
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    return mesh_0


def render_set(
    views,
    gaussians,
    pipeline,
    background,
    max_depth=5.0,
    volume=None,
    use_depth_filter=False,
    bounds=None,
    depth_name = "expected_depth"
):

    for view in tqdm(views, desc="Rendering progress"):
        # print(view.FoVx, view.FoVy)
        render_pkg = render(view, gaussians, pipeline, background, 0.0)
        rendering = render_pkg["render"]
        _, H, W = rendering.shape

        depth_tsdf = render_pkg[depth_name].squeeze()

        if use_depth_filter:
            view_dir = torch.nn.functional.normalize(view.get_rays(), p=2, dim=-1)
            depth_normal = depth_to_normal(view, depth_tsdf).permute(1, 2, 0)
            depth_normal = torch.nn.functional.normalize(depth_normal, p=2, dim=-1)
            dot = torch.sum(view_dir * depth_normal, dim=-1).abs()
            angle = torch.acos(dot)
            mask = angle > (80.0 / 180 * 3.14159)
            depth_tsdf[mask] = 0

        if bounds is not None:
            pts = get_points_from_depth(view, depth_tsdf)
            unvalid_mask = (pts[...,0] < bounds[0,0]) | (pts[...,0] > bounds[0,1]) |\
                            (pts[...,1] < bounds[1,0]) | (pts[...,1] > bounds[1,1]) |\
                            (pts[...,2] < bounds[2,0]) | (pts[...,2] > bounds[2,1])
            unvalid_mask = unvalid_mask.reshape(H,W)
            depth_tsdf[unvalid_mask] = 0
            
        ref_depth = depth_tsdf.squeeze().cpu()

        color = (
            render_pkg["render"].cpu().numpy().transpose(1, 2, 0).clip(0, 1) * 255
        ).astype(np.uint8)
        color = np.ascontiguousarray(color)
        pose = np.identity(4)
        pose[:3, :3] = view.R.cpu().numpy().transpose(-1, -2)
        pose[:3, 3] = view.T.cpu().numpy()
        color = o3d.geometry.Image(color)
        ref_depth = ref_depth.detach().cpu().numpy()
        depth = o3d.geometry.Image(ref_depth[..., None])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_trunc=max_depth,
            convert_rgb_to_intensity=False,
            depth_scale=1.0,
        )
        volume.integrate(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(W, H, view.Fx, view.Fy, view.Cx, view.Cy),
            pose,
        )


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    max_depth: float,
    voxel_size: float,
    num_cluster: int,
    use_depth_filter: bool,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        depth_name = "expected_depth" if dataset.depth_ratio < 0.5 else "median_depth"

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        bounds = None
        js_file = f"{dataset.source_path}/transforms.json"
        if os.path.exists(js_file):
            with open(js_file) as file:
                meta = json.load(file)
                if "aabb_range" in meta:
                    bounds = np.array(meta["aabb_range"])
        if bounds is not None:
            max_dis = np.max(bounds[:, 1] - bounds[:, 0])
            voxel_size = max_dis / 2048.0
        print(f"TSDF voxel_size {voxel_size}")
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=4.0 * voxel_size,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        render_set(
            scene.getTrainCameras(),
            gaussians,
            pipeline,
            background,
            max_depth=max_depth,
            volume=volume,
            use_depth_filter=use_depth_filter,
            bounds=bounds,
            depth_name = depth_name
        )
        print(f"extract_triangle_mesh")
        mesh = volume.extract_triangle_mesh()

        o3d.io.write_triangle_mesh(
            os.path.join(dataset.model_path, "recon.ply"),
            mesh,
            write_triangle_uvs=True,
            write_vertex_colors=True,
            write_vertex_normals=True,
        )

        mesh = post_process_mesh(mesh, num_cluster)
        o3d.io.write_triangle_mesh(
            os.path.join(dataset.model_path, "recon_post.ply"),
            mesh,
            write_triangle_uvs=True,
            write_vertex_colors=True,
            write_vertex_normals=True,
        )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max_depth", default=20.0, type=float)
    parser.add_argument("--voxel_size", default=0.002, type=float)
    parser.add_argument("--num_cluster", default=1, type=int)
    parser.add_argument("--use_depth_filter", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    print(f"multi_view_num {model.multi_view_num}")
    with torch.no_grad():
        render_sets(
            model.extract(args),
            args.iteration,
            pipeline.extract(args),
            args.max_depth,
            args.voxel_size,
            args.num_cluster,
            args.use_depth_filter,
        )
