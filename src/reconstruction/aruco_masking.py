import cv2
import numpy as np

from ..geometry.camera_calibration import calibrate_camera
from .aruco_markers_detection import (
    detect_aruco_markers,
    get_full_circle_aruco_board,
    draw_detected_markers
)


def mask_img(
    img_path: str,
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    verbose: bool = False
) -> None:
    img: np.ndarray = cv2.imread(img_path)

    # Detect AruCo markers
    marker_size: float = 1.5
    radius: float = 4.75
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

    obj_points, img_points = aruco_board.matchImagePoints(markers_corners, markers_ids)
    obj_points = obj_points[:, :, :2]


    print(np.shape(obj_points), np.shape(img_points))

    homography, _ = cv2.findHomography(obj_points, img_points)
    dst: np.ndarray = cv2.perspectiveTransform(obj_points, homography)

    img_board: np.ndarray = np.zeros((1500, 1500, 3), dtype=np.uint8)

    img_copy = img[:, :, :]
    obj_points = obj_points * 100 + 750
    print(img.dtype, img_board.dtype)
    for i in range(len(dst)):
        color = tuple(np.random.randint(0,255,3).tolist())
        
        cv2.circle(img, (int(dst[i][0][0]), int(dst[i][0][1])), 20, color, -1)
        cv2.circle(img, (int(img_points[i][0][0]), int(img_points[i][0][1])), 10, color, -1)
        cv2.putText(img, f"{markers_ids[int(i / 4)]}.{i % 4}", (int(img_points[i][0][0]), int(img_points[i][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        
        pts = (int(obj_points[i][0][0]), int(obj_points[i][0][1]))
        print(obj_points[i], i % 4)
        cv2.circle(img_board, pts, 10, color, -1)
        cv2.putText(img_board, f"{aruco_board.getIds()[int(i / 4)]}.{i % 4}", pts, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)


    if img_board.shape[0] != img.shape[0]:
        img_board = cv2.resize(
            img_board,
            (img_board.shape[1], img.shape[0])
        )

    combined = np.hstack([img_board, img])

    cv2.imshow("Mask | Image", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
    # cv2.imshow("Mask", img_board)
    # cv2.waitKey(0)

    # cv2.imshow("Mask", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    mask = np.zeros((1500, 1500), dtype=np.uint8)

    # Base circle
    center = (750, 750)
    radius = int(3.5 * 100)
    cv2.circle(mask, center, radius, 1, -1)

    cv2.imshow("Mask", mask * 255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    homography, _ = cv2.findHomography(obj_points, img_points)
    dst: np.ndarray = cv2.perspectiveTransform(obj_points, homography)

    warped_mask = cv2.warpPerspective(
        mask,
        homography,
        dsize=(np.shape(img)[1], np.shape(img)[0]),
        flags=cv2.INTER_NEAREST
    )

    cv2.imshow("Mask", warped_mask * 255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for i in range(len(dst)):
        color = tuple(np.random.randint(0,255,3).tolist())

        cv2.circle(img_copy, (int(dst[i][0][0]), int(dst[i][0][1])), 20, color, -1)
        cv2.circle(img_copy, (int(img_points[i][0][0]), int(img_points[i][0][1])), 10, color, -1)
        cv2.putText(img_copy, f"{markers_ids[int(i / 4)]}.{i % 4}", (int(img_points[i][0][0]), int(img_points[i][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    
    cv2.imshow("Mask", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img_copy[warped_mask == 0] = np.array([0, 0, 0])
    
    cv2.imshow("Mask", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Calibrate camera
    camera_matrix, distortion_coeffs = calibrate_camera(
        images_path="data/calibration/*.jpg",
        chessboard_size=(15, 6),
        square_size=1.6,
        verbose=False
    )

    mask_img(
        img_path="data/green-rasp-1/IMG_20251210_101023320.jpg",
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs
    )
