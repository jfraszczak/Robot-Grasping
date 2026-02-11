import os

import numpy as np
import pycolmap

from src.utils import measure_time
from .sparse_reconstruction import (
    sparse_reconstruction_based_on_poses,
    sparse_reconstruction_based_on_images,
)
from .visualization import visualize_reconstruction, save_reconstruction


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
    save_reconstruction(reconstruction)

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

    pycolmap.undistort_images(
        image_path=images_path,
        output_path=dense_path,
        input_path=sparse_path
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
    save_reconstruction(reconstruction)

    if verbose:
        print("Cameras:", reconstruction.cameras)
        print(len(reconstruction.images), "registered images")
        print(len(reconstruction.points3D), "3D points")
        visualize_reconstruction(reconstruction)
