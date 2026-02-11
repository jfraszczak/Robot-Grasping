import cv2
import numpy as np
import yaml

from src.aruco.markers_detection import detect_aruco_markers, get_full_circle_aruco_board, mask_board
from src.aruco.visualization import draw_aruco_board, draw_detected_markers
from src.geometry.camera_calibration import calibrate_camera
from src.utils import measure_time


@measure_time
def run_markers_detection(img_path: str) -> None:
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

    # Detect AruCo markers
    aruco_dictionary: int = cv2.aruco.DICT_4X4_250
    aruco_board: cv2.aruco.Board = get_full_circle_aruco_board(
        marker_size=config["aruco_board"]["marker_size"],
        radius=config["aruco_board"]["radius"],
        aruco_dictionary=aruco_dictionary
    )

    img: np.ndarray = cv2.imread(img_path)
    marker_corners, marker_ids = detect_aruco_markers(
        img=img,
        aruco_board=aruco_board,
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs
    )

    # Visualize result
    draw_aruco_board(aruco_board)
    draw_detected_markers(
        img=img,
        marker_corners=marker_corners,
        marker_ids=marker_ids
    )

    # Mask board
    obj_points, img_points = aruco_board.matchImagePoints(marker_corners, marker_ids)
    mask_board(
        img=img,
        obj_points=obj_points,
        img_points=img_points,
        marker_ids=marker_ids,
        board_radius=config["aruco_board"]["radius"] - config["camera_calibration"]["square_size"] / 2,
        verbose=True
    )


if __name__ == "__main__":
    run_markers_detection("data/green-rasp-1/IMG_20251210_101023320.jpg")
