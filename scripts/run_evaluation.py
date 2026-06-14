import argparse
import copy
import json
import pathlib

import numpy as np
import open3d
from pydantic import BaseModel

from .args import get_args
from src.geometry import CameraParameters
from src.reconstruction import ImageFrame
from src.reconstruction.colmap import ColmapTriangulationReconstructor, ColmapIncrementalReconstructor
from src.meshing import AlphaShapeMeshReconstructor, PoissonMeshReconstructor
from src.grasping import IGraspEstimator, FrankaEmikaPandaGrasper, PhysicalCoefficients, GraspTrajectory
from src.pipeline import GraspingPipeline, IntermediateResults
from src.synthetic_rendering import Open3DRenderer, BlenderRenderer
from src.utils import load_mesh


class ReconstructionEvaluationMetrics(BaseModel):
    accuracy_mean: float
    accuracy_std: float
    completeness_mean: float
    completeness_std: float
    fscore_1mm: float
    fscore_5mm: float


def fscore(accuracy: np.ndarray, completeness: np.ndarray, threshold: float) -> float:
    precision: float = np.mean(np.array(accuracy) < threshold)
    recall: float = np.mean(np.array(completeness) < threshold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_reconstruction(
    reconstructed: open3d.geometry.PointCloud,
    ground_truth_mesh: open3d.geometry.TriangleMesh,
    n_samples: int = 100000,
    icp: bool = False,
) -> ReconstructionEvaluationMetrics:
    ground_truth: open3d.geometry.PointCloud = ground_truth_mesh.sample_points_poisson_disk(n_samples)

    if icp:
        reconstructed = copy.deepcopy(reconstructed)
        result = open3d.pipelines.registration.registration_icp(
            reconstructed,
            ground_truth,
            max_correspondence_distance=0.05,
            estimation_method=open3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        reconstructed = reconstructed.transform(result.transformation)

    accuracy: np.ndarray = np.array(reconstructed.compute_point_cloud_distance(ground_truth))
    completeness: np.ndarray = np.array(ground_truth.compute_point_cloud_distance(reconstructed))

    return ReconstructionEvaluationMetrics(
        accuracy_mean=np.mean(accuracy),
        accuracy_std=np.std(accuracy),
        completeness_mean=np.mean(completeness),
        completeness_std=np.std(completeness),
        fscore_1mm=fscore(accuracy, completeness, threshold=0.001),
        fscore_5mm=fscore(accuracy, completeness, threshold=0.005),
    )


def add_gaussian_noise(
    coefficients: PhysicalCoefficients,
    std: float = 0.01,
    seed: int = 42,
) -> PhysicalCoefficients:
    rng = np.random.default_rng(seed)
    return PhysicalCoefficients(
        mass=max(coefficients.mass - std, coefficients.mass + rng.normal(0, std)),
        friction_sliding=max(coefficients.friction_sliding - std, coefficients.friction_sliding + rng.normal(0, std)),
        friction_torsional=max(coefficients.friction_torsional - std, coefficients.friction_torsional + rng.normal(0, std)),
        friction_rolling=max(coefficients.friction_rolling - std, coefficients.friction_rolling + rng.normal(0, std)),
    )


def run_evaluation(
    obj_files: list[str],
    physical_coeffs_list: list[PhysicalCoefficients],
    model_path: str,
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
    renderer: BlenderRenderer = BlenderRenderer(
        camera_parameters=camera_parameters
    )

    reconstruction_metrics: list[ReconstructionEvaluationMetrics] = []
    reconstruction_metrics_with_icp: list[ReconstructionEvaluationMetrics] = []
    grasp_accuracies: list[float] = []
    for obj_file, physical_coeffs in zip(obj_files, physical_coeffs_list):
        image_frames: list[ImageFrame] = renderer.run(
            obj_file=obj_file,
            verbose=False,
            frame_count=100,
            output_dir="data/synthetic"
        )

        grasp_estimator: IGraspEstimator = FrankaEmikaPandaGrasper(model_path=model_path)
        grasping_pipeline: GraspingPipeline = GraspingPipeline(
            reconstructor=ColmapIncrementalReconstructor(),
            mesh_reconstructor=AlphaShapeMeshReconstructor(),
            grasp_estimator=grasp_estimator
        )

        grasps: list[GraspTrajectory | None] = []
        grasp_trajectory: GraspTrajectory | None = grasping_pipeline.run(
            frames=image_frames,
            camera_parameters=camera_parameters,
            physical_coeffs=physical_coeffs,
            verbose=verbose
        )
        grasps.append(grasp_trajectory)

        ground_truth_mesh: open3d.geometry.TriangleMesh = load_mesh(obj_file)
        intermediate_results: IntermediateResults = grasping_pipeline.get_intermediate_results()

        reconstruction_evaluation: ReconstructionEvaluationMetrics = evaluate_reconstruction(
            reconstructed=intermediate_results.point_cloud,
            ground_truth_mesh=ground_truth_mesh,
            icp=False
        )
        reconstruction_metrics.append(reconstruction_evaluation)

        reconstruction_evaluation_icp: ReconstructionEvaluationMetrics = evaluate_reconstruction(
            reconstructed=intermediate_results.point_cloud,
            ground_truth_mesh=ground_truth_mesh,
            icp=True
        )
        reconstruction_metrics_with_icp.append(reconstruction_evaluation_icp)

        print(reconstruction_evaluation)
        print(reconstruction_metrics_with_icp)

        n_grasps: int = 10
        for _ in range(n_grasps - 1):
            grasp_trajectory: GraspTrajectory | None = grasp_estimator.run(
                mesh=intermediate_results.mesh,
                physical_coeffs=physical_coeffs,
                verbose=verbose
            )
            grasps.append(grasp_trajectory)

        grasp_accuracy: float = 0.0
        for grasp in grasps:
            if grasp:
                grasped: bool = grasp_estimator.evaluate_grasp(
                    mesh=ground_truth_mesh,
                    physical_coeffs=add_gaussian_noise(physical_coeffs),
                    grasp_trajectory=grasp,
                    verbose=True
                )
                grasp_accuracy += int(grasped)
        grasp_accuracy /= n_grasps
        grasp_accuracies.append(grasp_accuracy)

    evaluation_results: dict = {
        "reconstruction": {
            field: round(np.mean([getattr(m, field) for m in reconstruction_metrics]), 2)
            for field in ReconstructionEvaluationMetrics.model_fields
        },
        "reconstruction_icp": {
            field: round(np.mean([getattr(m, field) for m in reconstruction_metrics_with_icp]), 2)
            for field in ReconstructionEvaluationMetrics.model_fields
        },
        "grasping": {
            "accuracy_mean": np.mean(grasp_accuracies),
        }
    }

    output_path: pathlib.Path = pathlib.Path("evaluation_results.json")
    output_path.write_text(json.dumps(evaluation_results, indent=2))


if __name__ == "__main__":
    args: argparse.Namespace = get_args()
    run_evaluation(
        model_path="third_party/mujoco_menagerie/franka_emika_panda/panda.xml",
        obj_files=[
            "data/ycb/012_strawberry/google_512k/textured.obj",
            #"data/ycb/017_orange/google_512k/textured.obj",
            # "data/ycb/011_banana/google_512k/textured.obj",
            #"data/ycb/007_tuna_fish_can/google_512k/textured.obj"
        ],
        physical_coeffs_list=[
            PhysicalCoefficients(
                mass=0.02,
                friction_sliding=0.6,
                friction_torsional=0.05,
                friction_rolling=0.01
            )
        ],
        verbose=args.verbose
    )
