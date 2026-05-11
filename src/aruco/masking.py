import cv2
import numpy as np


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
