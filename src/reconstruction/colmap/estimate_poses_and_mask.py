import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from src.aruco.markers_detection import (
    detect_aruco_markers,
    mask_board
)
from src.aruco.visualization import draw_detected_markers
from src.geometry.pose_estimation import estimate_pose


def estimate_poses_and_mask(
    images_path: str,
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    aruco_board: cv2.aruco.Board,
    verbose: bool = False
) -> dict[str, np.ndarray]:
    images_masked_path: str = os.path.join(os.path.dirname(images_path), "masked")
    path: Path = Path(images_masked_path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir()
    
    poses: dict[str, np.ndarray] = {}
    for img_name in os.listdir(images_path):
        img_path: str = os.path.join(images_path, img_name)   
        img: np.ndarray = cv2.imread(img_path)
        marker_corners, marker_ids = detect_aruco_markers(
            img=img,
            aruco_board=aruco_board,
            camera_matrix=camera_matrix,
            distortion_coeffs=distortion_coeffs
        )

        if verbose:
            draw_detected_markers(
                img=img,
                marker_corners=marker_corners,
                marker_ids=marker_ids
            )

        # Estimate pose
        obj_points, img_points = aruco_board.matchImagePoints(marker_corners, marker_ids)
        t_camera_object: np.ndarray = estimate_pose(
            obj_points=obj_points,
            img_points=img_points,
            camera_matrix=camera_matrix,
            distortion_coeffs=distortion_coeffs
        )

        img_masked_path: str = os.path.join(images_masked_path, img_name)
        poses[img_masked_path] = t_camera_object

        # Mask boards
        img_masked: np.ndarray = mask_board(
            img=img,
            obj_points=obj_points,
            img_points=img_points,
            marker_ids=marker_ids,
            board_radius=6,
            verbose=False
        )
        cv2.imwrite(img_masked_path, img_masked)
            
    return poses
