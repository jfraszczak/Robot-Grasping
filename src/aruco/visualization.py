import cv2
import numpy as np


def draw_detected_markers(
    img: np.ndarray,
    marker_corners: np.ndarray,
    marker_ids: np.ndarray
) -> None:
    img_copy: np.ndarray = img.copy()
    cv2.aruco.drawDetectedMarkers(img_copy, marker_corners, marker_ids)
    cv2.imshow("Detected ArUco Markers", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_aruco_board(aruco_board: cv2.aruco.Board) -> None:
    img: np.ndarray = np.zeros((1500, 1500, 3), dtype=np.uint8)

    obj_points: np.ndarray = np.array(aruco_board.getObjPoints())
    obj_points = obj_points[:, :, :2]
    obj_points = obj_points.reshape(-1, 1, 2)
    obj_points = obj_points / np.max(np.absolute(obj_points)) * 500 + 750

    for i in range(np.shape(obj_points)[0]):
        color = tuple(np.random.randint(0,255,3).tolist())
        point: tuple[int, int] = (int(obj_points[i][0][0]), int(obj_points[i][0][1]))

        cv2.circle(img, point, 5, color, -1)
        cv2.putText(img, f"{aruco_board.getIds()[int(i / 4)]}.{i % 4}", point, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    
    cv2.imshow("ArUco Board", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
