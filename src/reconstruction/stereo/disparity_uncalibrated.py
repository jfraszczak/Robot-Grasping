import cv2
import numpy as np


def estimate_disparity_uncalibrated(
    img1: np.ndarray,
    img2: np.ndarray,
    img_points1: np.ndarray,
    img_points2: np.ndarray,
    fundamental_matrix: np.ndarray,
    verbose: bool = False
) -> None:
    h, w = img1.shape[:2]
    img_size: tuple[int, int] = (w, h)

    success, homography1, homography2 = cv2.stereoRectifyUncalibrated(
        img_points1,
        img_points2,
        fundamental_matrix,
        img_size
    )

    if not success:
        raise RuntimeError("Rectification failed")

    rect1 = cv2.warpPerspective(img1, homography1, img_size)
    rect2 = cv2.warpPerspective(img2, homography2, img_size)

    gray1 = cv2.cvtColor(rect1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(rect2, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoSGBM_create(
        numDisparities=32,
        blockSize=9
    )
    disparity: np.ndarray = stereo.compute(gray1, gray2)

    if verbose:
        cv2.imshow("Rectified 1", rect1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow("Rectified 2", rect2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disp_vis = np.uint8(disp_vis)
        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        cv2.imshow("Disparity Color", disp_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return disparity