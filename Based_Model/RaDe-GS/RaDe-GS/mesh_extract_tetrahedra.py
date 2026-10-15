# adopted from https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/extract_mesh.py
import torch
from scene import Scene
import os
from gaussian_renderer import integrate
import random
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import trimesh
from tetranerf.utils.extension import cpp
from utils.tetmesh import marching_tetrahedra


@torch.no_grad()
def evaluage_alpha(points, views, gaussians, pipeline, background, kernel_size):
    final_weight = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
    for view in tqdm(views, desc="Rendering progress"):
        ret = integrate(points, view, gaussians, pipeline, background, kernel_size)
        final_weight = torch.where(ret["inside"], torch.min(ret["alpha_integrated"], final_weight), final_weight)
    final_sdf = 0.5 - final_weight
    return final_sdf


@torch.no_grad()
def marching_tetrahedra_with_binary_search(model_path, views, gaussians: GaussianModel, pipeline, background, kernel_size, move_cpu:bool):
    # generate tetra points here
    points, points_scale = gaussians.get_tetra_points()
    cells = cpp.triangulate(points)
    torch.save(cells, os.path.join(model_path, "cells.pt"))
    # if os.path.exists(os.path.join(model_path, "cells.pt")):
    #     print("load existing cells")
    #     cells = torch.load(os.path.join(model_path, "cells.pt"))
    # else:
    #     # create cell and save cells
    #     print("create cells and save")
    #     cells = cpp.triangulate(points)
    #     # we should filter the cell if it is larger than the gaussians
    #     torch.save(cells, os.path.join(model_path, "cells.pt"))

    sdf = evaluage_alpha(points, views, gaussians, pipeline, background, kernel_size)

    torch.cuda.empty_cache()
    # the function marching_tetrahedra costs much memory, so we move it to cpu.
    if move_cpu:
        verts_list, scale_list, faces_list, _ = marching_tetrahedra(points.cpu()[None], cells.cpu().long(), sdf[None].cpu(), points_scale[None].cpu())
    else:
        verts_list, scale_list, faces_list, _ = marching_tetrahedra(points[None], cells.long(), sdf[None], points_scale[None])
    del points
    del points_scale
    del cells
    end_points, end_sdf = verts_list[0]
    end_scales = scale_list[0]
    end_points, end_sdf, end_scales = end_points.cuda(), end_sdf.cuda(), end_scales.cuda()

    faces = faces_list[0].cpu().numpy()
    points = (end_points[:, 0, :] + end_points[:, 1, :]) / 2.0

    left_points = end_points[:, 0, :]
    right_points = end_points[:, 1, :]
    left_sdf = end_sdf[:, 0, :]
    right_sdf = end_sdf[:, 1, :]
    left_scale = end_scales[:, 0, 0]
    right_scale = end_scales[:, 1, 0]
    distance = torch.norm(left_points - right_points, dim=-1)
    scale = left_scale + right_scale

    n_binary_steps = 8
    for step in range(n_binary_steps):
        print("binary search in step {}".format(step))
        mid_points = (left_points + right_points) / 2
        mid_sdf = evaluage_alpha(mid_points, views, gaussians, pipeline, background, kernel_size)
        mid_sdf = mid_sdf.unsqueeze(-1)
        ind_low = ((mid_sdf < 0) & (left_sdf < 0)) | ((mid_sdf > 0) & (left_sdf > 0))

        left_sdf[ind_low] = mid_sdf[ind_low]
        right_sdf[~ind_low] = mid_sdf[~ind_low]
        left_points[ind_low.flatten()] = mid_points[ind_low.flatten()]
        right_points[~ind_low.flatten()] = mid_points[~ind_low.flatten()]
        points = (left_points + right_points) / 2

    mesh = trimesh.Trimesh(vertices=points.cpu().numpy(), faces=faces, process=False)
    # filter
    vertice_mask = (distance <= scale).cpu().numpy()
    face_mask = vertice_mask[faces].all(axis=1)
    mesh.update_vertices(vertice_mask)
    mesh.update_faces(face_mask)

    mesh.export(os.path.join(model_path, "recon.ply"))


def extract_mesh(dataset: ModelParams, iteration: int, pipeline: PipelineParams, move_cpu:bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = dataset.kernel_size

        cams = scene.getTrainCameras()
        marching_tetrahedra_with_binary_search(dataset.model_path, cams, gaussians, pipeline, background, kernel_size, move_cpu)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--move_cpu", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

    extract_mesh(model.extract(args), args.iteration, pipeline.extract(args), args.move_cpu)