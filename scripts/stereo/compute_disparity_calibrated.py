import cv2
import numpy as np
import yaml

from src.aruco.markers_detection import detect_aruco_markers, get_full_circle_aruco_board, mask_board
from src.aruco.visualization import draw_detected_markers
from src.geometry.camera_calibration import calibrate_camera, undistort
from src.geometry.pose_estimation import estimate_pose, draw_object_frame
from src.geometry.epipolar_geometry import get_corresponding_points, draw_epilines, fundamental_matrix_from_pose
from src.reconstruction.stereo.disparity_calibrated import estimate_disparity_calibrated


def compute_disparity(
    img_path1: str,
    img_path2: str,
    verbose: bool = False
) -> None:
    # Read config
    with open("scripts/config.yaml", "r") as file:
        config: dict = yaml.safe_load(file)

    # Calibrate camera
    camera_matrix, distortion_coeffs = calibrate_camera(
        images_path=config["camera_calibration"]["path"],
        chessboard_size=(config["camera_calibration"]["chessboard_rows"], config["camera_calibration"]["chessboard_columns"]),
        square_size=config["camera_calibration"]["square_size"],
        verbose=False
    )

    # Read images
    img1: np.ndarray = cv2.imread(img_path1)
    img2: np.ndarray = cv2.imread(img_path2)

    # Undistort
    img1, camera_matrix = undistort(
        img=img1,
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs
    )
    img2, camera_matrix = undistort(
        img=img2,
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs
    )
    distortion_coeffs = np.zeros((1, 5))

    # Detect AruCo markers
    aruco_dictionary: int = cv2.aruco.DICT_4X4_250
    aruco_board: cv2.aruco.Board = get_full_circle_aruco_board(
        marker_size=config["aruco_board"]["marker_size"],
        radius=config["aruco_board"]["radius"],
        aruco_dictionary=aruco_dictionary
    )

    marker_corners1, marker_ids1 = detect_aruco_markers(
        img=img1,
        aruco_board=aruco_board,
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs
    )
    marker_corners2, marker_ids2 = detect_aruco_markers(
        img=img2,
        aruco_board=aruco_board,
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs
    )

    if verbose:
        draw_detected_markers(
            img=img1,
            marker_corners=marker_corners1,
            marker_ids=marker_ids1
        )

        draw_detected_markers(
            img=img2,
            marker_corners=marker_corners2,
            marker_ids=marker_ids2
        )

    # Estimate poses
    obj_points1, img_points1 = aruco_board.matchImagePoints(marker_corners1, marker_ids1)
    obj_points2, img_points2 = aruco_board.matchImagePoints(marker_corners2, marker_ids2)

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

    # Mask boards
    img1 = mask_board(
        img=img1,
        obj_points=obj_points1,
        img_points=img_points1,
        marker_ids=marker_ids1,
        board_radius=config["aruco_board"]["radius"] - config["camera_calibration"]["square_size"] / 2,
        verbose=True
    )
    img2 = mask_board(
        img=img2,
        obj_points=obj_points2,
        img_points=img_points2,
        marker_ids=marker_ids2,
        board_radius=config["aruco_board"]["radius"] - config["camera_calibration"]["square_size"] / 2,
        verbose=True
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
        markers_ids1=marker_ids1,
        markers_ids2=marker_ids2
    )

    if verbose:
        fundamental_matrix, _ = cv2.findFundamentalMat(img_points1, img_points2)
        
        t_cam1_cam2: np.ndarray = np.linalg.inv(t_camera_object2) @ t_camera_object1
        rotation: np.ndarray = t_cam1_cam2[:3, :3]
        translation: np.ndarray = t_cam1_cam2[:3, 3]
        fundamental_matrix = fundamental_matrix_from_pose(
            rotation=rotation,
            translation=translation,
            camera_matrix=camera_matrix
        )

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
    # compute_disparity(
    #     img_path1="data/green-rasp-1/IMG_20251210_101023320.jpg",
    #     img_path2="data/green-rasp-1/IMG_20251210_101029776.jpg",
    #     verbose=True
    # )

    compute_disparity(
        img_path1="data/stereo_car/IMG_6014.jpeg",
        img_path2="data/stereo_car/IMG_6015.jpeg",
        verbose=True
    )
