"""Diagnose "CUDA error: invalid configuration argument" of CoMe extract_mesh_tets.py on an object-only model: same
setup as extract_mesh() (Scene, 3D filter from the training cameras, tetra points), then integrate() view by view with a
synchronize after each, printing per view the tetra points / Gaussians in front of the camera and inside the image.
Run from the CoMe code folder (it imports CoMe modules):

  cd /workspace/CoMe && python <this file> -m <model> -s <data> --iteration 1
"""
import os
import sys

sys.path.insert(0, os.getcwd())     # the CoMe code folder (the script itself may live elsewhere)
import torch  # noqa: E402
from argparse import ArgumentParser

from arguments import SplattingSettings, ModelParams, PipelineParams, get_combined_args, MeshingSettings, BoundingSetting
from gaussian_renderer import integrate, GaussianModel
from scene import Scene


def in_view(view, X):
    Xh = torch.cat([X, torch.ones_like(X[:, :1])], 1)
    p = Xh @ view.full_proj_transform
    w = p[:, 3:4]
    ndc = p[:, :3] / w
    z = (Xh @ view.world_view_transform)[:, 2]
    return ((z > 0.2) & (ndc[:, 0].abs() < 1) & (ndc[:, 1].abs() < 1)).sum().item()


def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    ss = SplattingSettings(parser, render=True)
    parser.add_argument("--iteration", default=30000, type=int)
    args = get_combined_args(parser)
    dataset, pipe = model.extract(args), pipeline.extract(args)
    splat_args = ss.get_settings(args)
    ms = MeshingSettings(0.02, 1e6, False, BoundingSetting.STP, "dbg", near_far_culling=True, opacity_cutoff_tetra=0.0039,
                         load_cells=False)
    dataset.init_type = "sfm"
    with torch.no_grad():
        g = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, g, load_iteration=args.iteration, shuffle=False)
        cams = scene.getTrainCameras()
        g.compute_3D_filter(cameras=cams.copy())
        points, _ = g.get_tetra_points(cams, ms)
        print(f"gaussians {g.get_xyz.shape[0]}, tetra points {points.shape[0]}", flush=True)
        bg = torch.zeros(3, device="cuda")
        # exactly as marching_tetrahedra_with_binary_search -> evaluate_alpha(exact_alpha_eval=True)
        from gaussian_renderer import GlobalSortOrder
        splat_args.sort_settings.sort_order = GlobalSortOrder.MIN_Z_BOUNDING
        splat_args.meshing_settings.alpha_early_stop = True
        alpha = torch.ones(points.shape[0], device="cuda")
        final_alpha = torch.ones(points.shape[0], device="cuda")
        P = points.cuda()
        for i, view in enumerate(cams):
            live = alpha != 0
            msg = (f"view {i} {view.image_name}: points in view {in_view(view, P)}, live (alpha != 0) points in view "
                   f"{in_view(view, P[live])}, gaussians in view {in_view(view, g.get_xyz)}")
            try:
                integrate(points, alpha, view, g, pipe, bg, kernel_size=0.003, splat_args=splat_args)
                torch.cuda.synchronize()
                final_alpha = torch.min(final_alpha, alpha)
                alpha = torch.ones(points.shape[0], device="cuda") * ~(alpha == 0)
                print(msg + " ok", flush=True)
            except RuntimeError as e:
                print(msg + f" FAILED: {str(e).splitlines()[0]}", flush=True)
                sys.exit(1)


if __name__ == "__main__":
    main()
