import os
import argparse

import yaml
import cv2
import numpy as np
from src.geometry.camera_calibration import calibrate_camera
from src.aruco.markers_detection import get_full_circle_aruco_board
from src.reconstruction.colmap.sparse_reconstruction import (
    sparse_reconstruction_based_on_poses,
    sparse_reconstruction_based_on_images
)
from src.reconstruction.colmap.estimate_poses_and_mask import estimate_poses_and_mask


def get_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_path",
        type=str,
        help="Path to input images directory"
    )
    parser.add_argument(
        "--aruco_markers",
        action="store_true",
        help="Use AruCo markers for pose estimation"
    )
    parser.add_argument(
        "--verbose",
        action="store_true"
    )
    return parser.parse_args()


def run_sparse_reconstruction(
    images_path: str,
    use_aruco_markers: bool,
    verbose: bool
) -> None:
    with open("scripts/config.yaml", "r") as file:
        config: dict = yaml.safe_load(file)

    if use_aruco_markers:
        camera_matrix, distortion_coeffs = calibrate_camera(
            images_path=config["camera_calibration"]["path"],
            chessboard_size=(config["camera_calibration"]["chessboard_rows"], config["camera_calibration"]["chessboard_columns"]),
            square_size=config["camera_calibration"]["square_size"],
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

        sparse_reconstruction_based_on_poses(
            images_path=os.path.dirname(list(poses.keys())[0]),
            camera_matrix=camera_matrix,
            distortion_coeffs=distortion_coeffs,
            poses=poses,
            verbose=verbose
        )

    else:
        sparse_reconstruction_based_on_images(
            images_path=images_path,
            verbose=verbose
        )


if __name__ == "__main__":
    args: argparse.Namespace = get_args()
    run_sparse_reconstruction(
        images_path=args.images_path,
        use_aruco_markers=args.aruco_markers,
        verbose=args.verbose
    )
