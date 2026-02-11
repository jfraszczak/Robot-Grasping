
import cv2
import numpy as np
from src.geometry.transformations import transformation_matrix_2d


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

        t: np.ndarray = transformation_matrix_2d(
            angle=angle,
            tx=tx,
            ty=ty
        )
        markers_corners[i, :, 0:2] = (t @ marker_corners).T[:, 0:2]

    markers_ids: np.ndarray = np.array([10, 17, 16, 15, 14, 13, 12, 11])

    return cv2.aruco.Board(
        markers_corners.astype(np.float32),
        cv2.aruco.getPredefinedDictionary(aruco_dictionary),
        markers_ids
    )


def mask_board(
    img: np.ndarray,
    obj_points: np.ndarray,
    img_points: np.ndarray,
    marker_ids: np.ndarray,
    board_radius: int,
    verbose: bool = False
) -> np.ndarray:
    w, h = (np.shape(img)[1], np.shape(img)[0])
    mask: np.ndarray = np.zeros((h, w), dtype=np.uint8)
    center: tuple[int, int] = (w // 2, h // 2)
    radius_scale: int = 100
    radius: int = int(board_radius * radius_scale)

    obj_points = obj_points[:, :, :2]
    obj_points = obj_points * radius_scale + np.array([[w // 2, h // 2]])

    cv2.circle(mask, center, radius, 1, -1)
    if verbose:
        cv2.imshow("Mask", mask * 255)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    homography, _ = cv2.findHomography(obj_points, img_points)
    dst_points: np.ndarray = cv2.perspectiveTransform(obj_points, homography)

    mask = cv2.warpPerspective(
        mask,
        homography,
        dsize=(w, h),
        flags=cv2.INTER_NEAREST
    )

    if verbose:
        cv2.imshow("Warped Mask", mask * 255)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if verbose:
        img_copy: np.ndarray = img.copy()
        for i in range(len(dst_points)):
            color = tuple(np.random.randint(0,255,3).tolist())
            point: tuple[int, int] = (int(dst_points[i][0][0]), int(dst_points[i][0][1]))
            cv2.circle(img_copy, point, 10, color, -1)
            cv2.putText(img_copy, f"{marker_ids[int(i / 4)]}.{i % 4}", point, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        
        if verbose:
            cv2.imshow("Warped Markers", img_copy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    img_masked: np.ndarray = img.copy()
    img_masked[mask == 0] = np.array([0, 0, 0])
    
    if verbose:
        cv2.imshow("Masked Image", img_masked)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img_masked
