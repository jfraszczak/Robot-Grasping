import os
import argparse

import cv2
import numpy as np
import pycolmap
import open3d as o3d

from ..camera_calibration import calibrate_camera
from ..aruco_markers_detection import (
    detect_aruco_markers,
    get_full_circle_aruco_board,
    draw_detected_markers
)
from ..pose_estimation import estimate_pose
from ..utils import measure_time
from .visualization import (
    visualize_keypoints,
    visualize_reconstruction,
    save_reconstruction
)


def estimate_poses(
    images_path: str,
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    verbose: bool = False
) -> dict[str, np.ndarray]:
    poses: dict[str, np.ndarray] = {}
    for img_name in os.listdir(images_path):
        img_path: str = os.path.join(images_path, img_name)   
        img: np.ndarray = cv2.imread(img_path)

        # Detect AruCo markers
        marker_size: float = 1.5
        radius: float = 4.5
        aruco_dictionary: int = cv2.aruco.DICT_4X4_250
        aruco_board: cv2.aruco.Board = get_full_circle_aruco_board(
            marker_size=marker_size,
            radius=radius,
            aruco_dictionary=aruco_dictionary
        )
        markers_corners, markers_ids = detect_aruco_markers(
            img=img,
            aruco_board=aruco_board,
            camera_matrix=camera_matrix,
            distortion_coeffs=distortion_coeffs
        )

        if verbose:
            draw_detected_markers(
                img=img,
                markers_corners=markers_corners,
                markers_ids=markers_ids
            )

        # Estimate pose
        obj_points, img_points = aruco_board.matchImagePoints(markers_corners, markers_ids)
        t_camera_object: np.ndarray = estimate_pose(
            obj_points=obj_points,
            img_points=img_points,
            camera_matrix=camera_matrix,
            distortion_coeffs=distortion_coeffs
        )
        poses[img_path] = t_camera_object
    
    return poses


def construct_reconstruction(
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    poses: dict[str, np.ndarray]
) -> None:
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    k1, k2, p1, p2 = distortion_coeffs.flatten()[:4]

    sample_img = cv2.imread(list(poses.keys())[0])
    height, width = sample_img.shape[:2]

    reconstruction: pycolmap.Reconstruction = pycolmap.Reconstruction()
    camera_id: int = 1
    camera = pycolmap.Camera({
        "camera_id": camera_id,
        "model": pycolmap.CameraModelId.OPENCV,
        "width": width,
        "height": height,
        "params": [fx, fy, cx, cy, k1, k2, p1, p2],
    })
    reconstruction.add_camera(camera)

    rig = pycolmap.Rig()
    rig.rig_id = 1
    sensor_t = pycolmap.sensor_t()
    sensor_t.type = pycolmap.SensorType.CAMERA
    sensor_t.id = camera.camera_id
    rig.add_ref_sensor(sensor_t)
    reconstruction.add_rig(rig)

    image_id: int = 1
    frame_id: int = 1

    for img_name, pose in poses.items():
        T_cw = np.linalg.inv(pose)
        R_cw = T_cw[:3, :3]
        t_cw = T_cw[:3, 3]

        frame = pycolmap.Frame()
        frame.frame_id = frame_id
        frame.rig_id = rig.rig_id
        rotation = pycolmap.Rotation3d(R_cw)
        translation = t_cw
        rigid3d = pycolmap.Rigid3d(rotation, translation)

        image = pycolmap.Image(
            image_id=image_id, name=os.path.basename(img_name), camera_id=camera.camera_id, frame_id=frame.frame_id
        )
        frame.add_data_id(image.data_id)
        reconstruction.add_frame(frame)
        reconstruction.frame(frame.frame_id).set_cam_from_world(camera_id=camera.camera_id, cam_from_world=rigid3d)
        reconstruction.register_frame(frame_id)

        reconstruction.add_image(image)

        image_id += 1
        frame_id += 1

    return reconstruction


@measure_time
def sparse_reconstruction_based_on_poses(
    images_path: str,
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    poses: dict[str, np.ndarray],
    verbose: bool = False
) -> dict[str, np.ndarray]:
    database_path: str = "database.db"
    sparse_path: str = "sparse"

    if os.path.exists(database_path):
        os.remove(database_path)

    pycolmap.extract_features(
        database_path=database_path,
        image_path=images_path,
        camera_mode=pycolmap.CameraMode.AUTO,
        camera_model="OPENCV"
    )

    pycolmap.match_exhaustive(database_path)

    if verbose:
        visualize_keypoints(
            database_path=database_path,
            images_path=images_path
        )

    reconstruction: pycolmap.Reconstruction = construct_reconstruction(
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs,
        poses=poses
    )
    reconstruction = pycolmap.triangulate_points(
        reconstruction=reconstruction,
        database_path=database_path,
        image_path=images_path,
        output_path=sparse_path,
        clear_points=True,
        refine_intrinsics=False
    )
    save_reconstruction(reconstruction)

    if verbose:
        print("Cameras:", reconstruction.cameras)
        print(len(reconstruction.images), "registered images")
        print(len(reconstruction.points3D), "3D points")
        visualize_reconstruction(reconstruction)


@measure_time
def sparse_reconstruction_based_on_images(
    images_path: str,
    verbose: bool = False
) -> dict[str, np.ndarray]:
    database_path: str = "database.db"
    sparse_path: str = "sparse"

    if os.path.exists(database_path):
        os.remove(database_path)

    pycolmap.extract_features(
        database_path=database_path,
        image_path=images_path,
        camera_mode=pycolmap.CameraMode.SINGLE,
        camera_model="OPENCV"
    )
    pycolmap.match_exhaustive(database_path)

    if verbose:
        visualize_keypoints(
            database_path=database_path,
            images_path=images_path
        )

    reconstruction: pycolmap.Reconstruction = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=images_path,
        output_path=sparse_path
    )[0]

    if verbose:
        print("Cameras:", reconstruction.cameras)
        print(len(reconstruction.images), "registered images")
        print(len(reconstruction.points3D), "3D points")
        visualize_reconstruction(reconstruction)


def get_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_path",
        type=str,
        help="Path to input images directory"
    )
    parser.add_argument(
        "--calibration_path",
        type=str,
        default="",
        help="Path to calibration images"
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
        sparse_reconstruction_based_on_poses(
            images_path=images_path,
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
        calibration_path=args.calibration_path,
        use_aruco_markers=args.aruco_markers,
        verbose=args.verbose
    )
