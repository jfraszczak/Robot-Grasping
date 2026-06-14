import argparse
import cv2

from .args import get_args
from src.geometry.camera_calibration import calibrate_camera
from src.aruco.markers_detection import get_full_circle_aruco_board
from src.aruco.estimate_poses_and_mask import estimate_poses_and_mask
from src.geometry import CameraParameters
from src.reconstruction import ImageFrame
from src.reconstruction.colmap import ColmapTriangulationReconstructor, ColmapIncrementalReconstructor
from src.reconstruction.depth_anything import DepthAnythingReconstructor
from src.meshing import AlphaShapeMeshReconstructor
from src.grasping import FrankaEmikaPandaGrasper, PhysicalCoefficients
from src.pipeline import GraspingPipeline
from scripts.config import Config


def run_colmap_grasp_generation(config: Config, verbose: bool) -> None:
    camera_parameters: CameraParameters = calibrate_camera(
        images_path=config.camera_calibration.path,
        chessboard_size=(config.camera_calibration.chessboard_rows, config.camera_calibration.chessboard_columns),
        square_size=config.camera_calibration.square_size,
        verbose=False
    )
    
    aruco_dictionary: int = cv2.aruco.DICT_4X4_250
    aruco_board: cv2.aruco.Board = get_full_circle_aruco_board(
        marker_size=config.aruco_board.marker_size,
        radius=config.aruco_board.radius,
        aruco_dictionary=aruco_dictionary
    )

    image_frames: list[ImageFrame] = estimate_poses_and_mask(
        images_path=config.image_source.path,
        camera_parameters=camera_parameters,
        aruco_board=aruco_board,
        verbose=verbose
    )

    grasping_pipeline: GraspingPipeline = GraspingPipeline(
        reconstructor=DepthAnythingReconstructor(),
        mesh_reconstructor=AlphaShapeMeshReconstructor(),
        grasp_estimator=FrankaEmikaPandaGrasper(model_path="third_party/mujoco_menagerie/franka_emika_panda/panda.xml")
    )

    grasping_pipeline.run(
        frames=image_frames,
        camera_parameters=camera_parameters,
        physical_coeffs=PhysicalCoefficients(
            mass=0.02,
            friction_sliding=2.0,
            friction_torsional=0.05,
            friction_rolling=0.01
        ),
        verbose=True
    )


if __name__ == "__main__":
    args: argparse.Namespace = get_args()
    config: Config = Config.from_yaml("scripts/config.yaml")
    run_colmap_grasp_generation(
        config=config,
        verbose=args.verbose
    )
