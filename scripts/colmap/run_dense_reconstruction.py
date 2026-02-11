import argparse

import opencv as cv2
import numpy as np
import yaml

from .args import get_args
from src.geometry.camera_calibration import calibrate_camera
from src.aruco.markers_detection import get_full_circle_aruco_board
from src.reconstruction.colmap.estimate_poses_and_mask import estimate_poses_and_mask
from src.reconstruction.colmap.dense_reconstruction import (
    dense_reconstruction_based_on_images,
    dense_reconstruction_based_on_poses
)


def run_dense_reconstruction(
    images_path: str,
    calibration_path: str,
    use_aruco_markers: bool,
    verbose: bool
) -> None:
    with open("scripts/config.yaml", "r") as file:
        config: dict = yaml.safe_load(file)

    if use_aruco_markers:
        camera_matrix, distortion_coeffs = calibrate_camera(
            images_path=calibration_path,
            chessboard_size=(15, 6),
            square_size=1.6,
            verbose=False
        )
        aruco_dictionary: int = cv2.aruco.DICT_4X4_250
        aruco_board: cv2.aruco.Board = get_full_circle_aruco_board(
            marker_size=config["aruco_board"]["marker_size"],
            radius=config["aruco_board"]["radius"],
            aruco_dictionary=aruco_dictionary
        )
        poses: dict[str, np.ndarray] = estimate_poses_and_mask(
            images_path=images_path,
            camera_matrix=camera_matrix,
            distortion_coeffs=distortion_coeffs,
            aruco_board=aruco_board,
            verbose=verbose
        )

        dense_reconstruction_based_on_poses(
            images_path=images_path,
            camera_matrix=camera_matrix,
            distortion_coeffs=distortion_coeffs,
            poses=poses,
            verbose=verbose
        )
    else:
        dense_reconstruction_based_on_images(
            images_path=images_path,
            verbose=verbose
        )


if __name__ == "__main__":
    args: argparse.Namespace = get_args()
    run_dense_reconstruction(
        images_path=args.images_path,
        calibration_path=args.calibration_path,
        use_aruco_markers=args.aruco_markers,
        verbose=args.verbose
    )
