import argparse

import numpy as np

from .args import get_args
from src.geometry import CameraParameters
from src.reconstruction import ImageFrame
from src.reconstruction.colmap import ColmapTriangulationReconstructor, ColmapIncrementalReconstructor
from src.meshing import AlphaShapeMeshReconstructor, PoissonMeshReconstructor
from src.grasping import MujocoGraspEstimator
from src.pipeline import GraspingPipeline
from src.synthetic_rendering import Open3DRenderer, BlenderRenderer


def run_grasp_generation_simulation(
    obj_file: str,
    verbose: bool
) -> None:
    fx, fy = 2871.63, 2871.63
    cx, cy = 1161.31, 1161.31
    intrinsic: np.array = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1],
    ], dtype=float)
    camera_parameters: CameraParameters = CameraParameters(matrix=intrinsic)

    renderer: Open3DRenderer = Open3DRenderer(
        camera_parameters=camera_parameters
    )
    # renderer: BlenderRenderer = BlenderRenderer(
    #     camera_parameters=camera_parameters
    # )
    image_frames: list[ImageFrame] = renderer.run(
        obj_file=obj_file,
        verbose=False,
        frame_count=100,
        output_dir="data/synthetic"
    )

    grasping_pipeline: GraspingPipeline = GraspingPipeline(
        reconstructor=ColmapIncrementalReconstructor(),
        mesh_reconstructor=AlphaShapeMeshReconstructor(),
        grasp_estimator=MujocoGraspEstimator()
    )

    grasping_pipeline.run(
        frames=image_frames,
        camera_parameters=camera_parameters,
        verbose=verbose
    )


if __name__ == "__main__":
    args: argparse.Namespace = get_args()
    run_grasp_generation_simulation(
        obj_file="data/ycb/strawberry/012_strawberry/google_512k/textured.obj",
        verbose=args.verbose
    )
