
import cv2
import numpy as np


def detect_aruco_markers(
    img: np.ndarray,
    aruco_board: cv2.aruco.Board | None = None,
    camera_matrix: np.ndarray | None = None,
    distortion_coeffs: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    gray: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector: cv2.aruco.ArucoDetector = cv2.aruco.ArucoDetector(
        aruco_board.getDictionary(),
        cv2.aruco.DetectorParameters()
    )
    markers_corners, markers_ids, rejected = detector.detectMarkers(gray)
    if aruco_board is not None and (camera_matrix is not None or distortion_coeffs is not None):
        markers_corners, markers_ids, _, _ = detector.refineDetectedMarkers(
            img,
            aruco_board,
            markers_corners,
            markers_ids,
            rejected,
            camera_matrix,
            distortion_coeffs
        )

    return markers_corners, markers_ids


def get_full_circle_aruco_board(marker_size: float, radius: float, aruco_dictionary: int) -> cv2.aruco.Board:
    marker_corners: np.ndarray  = np.zeros((3, 4))
    marker_corners[:, 0] = np.array([-0.5, -0.5, 1.0])
    marker_corners[:, 1] = np.array([ 0.5, -0.5, 1.0])
    marker_corners[:, 2] = np.array([ 0.5, 0.5, 1.0])
    marker_corners[:, 3] = np.array([-0.5, 0.5, 1.0])
    marker_corners[:2, :] *= marker_size

    markers_number: int = 8
    markers_corners: np.ndarray = np.zeros((markers_number, 4, 3))
    for i in range(markers_number):
        angle: float = i * (2 * np.pi / markers_number)
        tx: float = radius * np.cos(angle)
        ty: float = radius * np.sin(angle)

        t: np.ndarray = np.array([
            [np.cos(angle), -np.sin(angle), tx],
            [np.sin(angle), np.cos(angle), ty],
            [0.0, 0.0, 1.0]
        ])
        markers_corners[i, :, 0:2] = (t @ marker_corners).T[:, 0:2]

    markers_ids: np.ndarray = np.array([10, 17, 16, 15, 14, 13, 12, 11])

    return cv2.aruco.Board(
        markers_corners.astype(np.float32),
        cv2.aruco.getPredefinedDictionary(aruco_dictionary),
        markers_ids
    )
