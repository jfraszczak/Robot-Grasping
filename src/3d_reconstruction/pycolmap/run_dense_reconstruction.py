import os
import argparse

import numpy as np
import pycolmap

from ..camera_calibration import calibrate_camera
from ..utils import measure_time
from .run_sparse_reconstruction import (
    sparse_reconstruction_based_on_poses,
    sparse_reconstruction_based_on_images,
    estimate_poses,
    get_args
)
from .visualization import visualize_keypoints, visualize_reconstruction


@measure_time
def dense_reconstruction_based_on_images(
    images_path: str,
    verbose: bool = False
) -> dict[str, np.ndarray]:
    sparse_reconstruction_based_on_images(images_path)

    sparse_path: str = "sparse"
    dense_path: str = "dense"
    model_path: str = os.path.join(sparse_path, "0")

    pycolmap.undistort_images(
        image_path=images_path,
        output_path=dense_path,
        input_path=model_path
    )

    pycolmap.patch_match_stereo(
        workspace_path=dense_path,
        workspace_format="COLMAP"
    )

    reconstruction = pycolmap.stereo_fusion(
        workspace_path=dense_path,
        workspace_format="COLMAP",
        output_path=dense_path
    )

    if verbose:
        print("Cameras:", reconstruction.cameras)
        print(len(reconstruction.images), "registered images")
        print(len(reconstruction.points3D), "3D points")
        visualize_reconstruction(reconstruction)


@measure_time
def dense_reconstruction_based_on_poses(
    images_path: str,
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    poses: dict[str, np.ndarray],
    verbose: bool = False
) -> dict[str, np.ndarray]:
    sparse_reconstruction_based_on_poses(
        images_path=images_path,
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs,
        poses=poses
    )

    sparse_path: str = "sparse"
    dense_path: str = "dense"
    model_path: str = os.path.join(sparse_path, "0")

    pycolmap.undistort_images(
        image_path=images_path,
        output_path=dense_path,
        input_path=model_path
    )

    pycolmap.patch_match_stereo(
        workspace_path=dense_path,
        workspace_format="COLMAP"
    )

    reconstruction = pycolmap.stereo_fusion(
        workspace_path=dense_path,
        workspace_format="COLMAP",
        output_path=dense_path
    )

    if verbose:
        print("Cameras:", reconstruction.cameras)
        print(len(reconstruction.images), "registered images")
        print(len(reconstruction.points3D), "3D points")
        visualize_reconstruction(reconstruction)


def run_dense_reconstruction(
    images_path: str,
    calibration_path: str,
    use_aruco_markers: bool,
    verbose: bool
) -> None:
    if use_aruco_markers:
        camera_matrix, distortion_coeffs = calibrate_camera(
            images_path=calibration_path,
            chessboard_size=(15, 6),
            square_size=1.6,
            verbose=False
        )
        poses: dict[str, np.ndarray] = estimate_poses(
            images_path=images_path,
            camera_matrix=camera_matrix,
            distortion_coeffs=distortion_coeffs,
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
