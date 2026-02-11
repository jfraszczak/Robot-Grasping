import cv2
import numpy as np

from ...geometry.camera_calibration import calibrate_camera
from ..aruco_markers_detection import (
    detect_aruco_markers,
    get_full_circle_aruco_board,
    draw_detected_markers
)
from ..pose_estimation import estimate_pose, draw_object_frame
from ..epipolar_geometry import (
    get_corresponding_points,
    find_fundamental_matrix,
    draw_epilines,
    estimate_disparity_uncalibrated,
    fundamental_matrix_from_pose,
    estimate_disparity_calibrated
)


def compute_disparity(
    img_path1: str,
    img_path2: str,
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    verbose: bool = False
) -> None:
    # Calibrate camera
    camera_matrix, distortion_coeffs = calibrate_camera(
        images_path="data/calibration/*.jpg",
        chessboard_size=(15, 6),
        square_size=1.6,
        verbose=False
    )

    # Read stereo images
    img1: np.ndarray = cv2.imread(img_path1)
    img2: np.ndarray = cv2.imread(img_path2)

    # Detect AruCo markers
    marker_size: float = 1.5
    radius: float = 4.75
    aruco_dictionary: int = cv2.aruco.DICT_4X4_250
    aruco_board: cv2.aruco.Board = get_full_circle_aruco_board(
        marker_size=marker_size,
        radius=radius,
        aruco_dictionary=aruco_dictionary
    )
    markers_corners1, markers_ids1 = detect_aruco_markers(
        img=img1,
        aruco_board=aruco_board,
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs
    )
    markers_corners2, markers_ids2 = detect_aruco_markers(
        img=img2,
        aruco_board=aruco_board,
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs
    )

    if verbose:
        draw_detected_markers(
            img=img1,
            markers_corners=markers_corners1,
            markers_ids=markers_ids1
        )

        draw_detected_markers(
            img=img2,
            markers_corners=markers_corners2,
            markers_ids=markers_ids2
        )


    # Estimate poses
    obj_points1, img_points1 = aruco_board.matchImagePoints(markers_corners1, markers_ids1)
    obj_points2, img_points2 = aruco_board.matchImagePoints(markers_corners2, markers_ids2)

    t_camera_object1: np.ndarray = estimate_pose(
        obj_points=obj_points1,
        img_points=img_points1,
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs
    )

    t_camera_object2: np.ndarray = estimate_pose(
        obj_points=obj_points2,
        img_points=img_points2,
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs
    )

    if verbose:
        draw_object_frame(
            img=img1,
            t_object_camera=np.linalg.inv(t_camera_object1),
            camera_matrix=camera_matrix,
            distortion_coeffs=distortion_coeffs
        )

        draw_object_frame(
            img=img2,
            t_object_camera=np.linalg.inv(t_camera_object2),
            camera_matrix=camera_matrix,
            distortion_coeffs=distortion_coeffs
        )

    img_points1, img_points2 = get_corresponding_points(
        img_points1=img_points1,
        img_points2=img_points2,
        markers_ids1=markers_ids1,
        markers_ids2=markers_ids2
    )

    fundamental_matrix: np.ndarray = find_fundamental_matrix(img_points1, img_points2)

    if verbose:
        draw_epilines(
            img1=img1,
            img2=img2,
            img_points1=img_points1,
            img_points2=img_points2,
            fundamental_matrix=fundamental_matrix
        )

    estimate_disparity_uncalibrated(
        img1=img1,
        img2=img2,
        img_points1=img_points1,
        img_points2=img_points2,
        fundamental_matrix=fundamental_matrix,
        verbose=verbose
    )

    t_cam1_cam2: np.ndarray = np.linalg.inv(t_camera_object2) @ t_camera_object1
    rotation: np.ndarray = t_cam1_cam2[:3, :3]
    translation: np.ndarray = t_cam1_cam2[:3, 3]

    fundamental_matrix: np.ndarray = fundamental_matrix_from_pose(
        rotation=rotation,
        translation=translation,
        camera_matrix=camera_matrix
    )
    if verbose:
        draw_epilines(
            img1=img1,
            img2=img2,
            img_points1=img_points1,
            img_points2=img_points2,
            fundamental_matrix=fundamental_matrix
        )

    estimate_disparity_calibrated(
        img1=img1,
        img2=img2,
        t_camera_object1=t_camera_object1,
        t_camera_object2=t_camera_object2,
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs,
        verbose=verbose
    )


if __name__ == "__main__":
    compute_disparity(
        img_path1="data/green-rasp-1/IMG_20251210_101023320.jpg",
        img_path2="data/green-rasp-1/IMG_20251210_101029776.jpg",
        verbose=True
    )

    # compute_disparity(
    #     img_path1="data/stereo/IMG_20251210_101829904.jpg",
    #     img_path2="data/stereo/IMG_20251210_101835457.jpg",
    #     verbose=True
    # )
