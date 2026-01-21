import cv2
import numpy as np


def draw_epilines(
    img1: np.ndarray,
    img2: np.ndarray,
    img_points1: np.ndarray,
    img_points2: np.ndarray,
    fundamental_matrix: np.ndarray
) -> None:
    img1_copy: np.ndarray = img1.copy()
    img2_copy: np.ndarray = img2.copy()
    combined_img: np.ndarray = np.hstack((img1_copy, img2_copy))
    w1: int = img1.shape[1]

    # Compute epilines in image 2 for points in image 1
    lines1_to_2 = cv2.computeCorrespondEpilines(img_points1.reshape(-1,1,2), 1, fundamental_matrix)
    lines1_to_2 = lines1_to_2.reshape(-1,3)
    
    # Compute epilines in image 1 for points in image 2
    lines2_to_1 = cv2.computeCorrespondEpilines(img_points2.reshape(-1,1,2), 2, fundamental_matrix)
    lines2_to_1 = lines2_to_1.reshape(-1,3)
    
    for i in range(len(img_points1)):
        color = tuple(np.random.randint(0,255,3).tolist())
        
        r = lines1_to_2[i]
        x0, y0 = 0, int(-r[2]/r[1])
        x1, y1 = img2.shape[1], int(-(r[2] + r[0]*img2.shape[1])/r[1])
        cv2.line(combined_img, (x0 + w1, y0), (x1 + w1, y1), color, 5)
        cv2.circle(combined_img, tuple(img_points1[i].astype(int)), 20, color, -1)
        
        r = lines2_to_1[i]
        x0, y0 = 0, int(-r[2]/r[1])
        x1, y1 = img1.shape[1], int(-(r[2] + r[0]*img1.shape[1])/r[1])
        cv2.line(combined_img, (x0, y0), (x1, y1), color, 5)
        cv2.circle(combined_img, tuple((img_points2[i] + np.array([w1,0])).astype(int)), 20, color, -1)
    
    cv2.imshow("Epilines", combined_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_corresponding_points(
    img_points1: np.ndarray,
    img_points2: np.ndarray,
    markers_ids1: np.ndarray,
    markers_ids2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    markers_ids1 = markers_ids1.flatten()
    markers_ids2 = markers_ids2.flatten()

    order1 = np.argsort(markers_ids1)
    order2 = np.argsort(markers_ids2)

    markers_ids1 = markers_ids1[order1]
    markers_ids2 = markers_ids2[order2]

    corners_per_marker: int = 4
    sorted_img_points1 = np.vstack([img_points1[i * corners_per_marker:(i + 1) * corners_per_marker] for i in order1])
    sorted_img_points2 = np.vstack([img_points2[i * corners_per_marker:(i + 1) * corners_per_marker] for i in order2])

    img_points1 = sorted_img_points1
    img_points2 = sorted_img_points2

    common_ids = set(list(markers_ids1)).intersection(set(list(markers_ids2)))
    mask1 = np.array([marker_id in common_ids for marker_id in markers_ids1])
    mask2 = np.array([marker_id in common_ids for marker_id in markers_ids2])

    mask1 = np.repeat(mask1, corners_per_marker)
    mask2 = np.repeat(mask2, corners_per_marker)

    img_points1 = img_points1[mask1]
    img_points2 = img_points2[mask2]

    img_points1 = img_points1.reshape(-1, 2).astype(np.float32)
    img_points2 = img_points2.reshape(-1, 2).astype(np.float32)

    return img_points1, img_points2


def find_fundamental_matrix(
    img_points1: np.ndarray,
    img_points2: np.ndarray,
) -> np.ndarray:
    fundamental_matrix, mask = cv2.findFundamentalMat(img_points1, img_points2)
    img_points1 = img_points1[mask.flatten() == 1]
    img_points2 = img_points2[mask.flatten() == 1]
    return fundamental_matrix


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
        minDisparity=0,
        numDisparities=16,
        blockSize=15
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
    

def fundamental_matrix_from_pose(
    rotation: np.ndarray,
    translation: np.ndarray,
    camera_matrix: np.ndarray
) -> np.ndarray:
    translation = translation.reshape(3)
    tx = np.array([
        [0,     -translation[2],  translation[1]],
        [translation[2],   0,    -translation[0]],
        [-translation[1],  translation[0],  0]
    ])

    essential_matrix: np.ndarray = tx @ rotation
    fundamental_matrix: np.ndarray = np.linalg.inv(camera_matrix).T @ essential_matrix @ np.linalg.inv(camera_matrix)
    return fundamental_matrix


def rotation_matrix_to_yaw_pitch_roll(rotation_matrix: np.ndarray) -> tuple[float, float, float]:
    pitch: float = np.arctan2(-rotation_matrix[2, 1], rotation_matrix[2, 2])
    yaw: float   = np.arctan2(rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] **2 ))
    roll: float  = np.arctan2(-rotation_matrix[1, 0], rotation_matrix[0, 0])
    yaw, pitch, roll = np.degrees([yaw, pitch, roll])
    return yaw, pitch, roll


def estimate_disparity_calibrated(
    img1: np.ndarray,
    img2: np.ndarray,
    t_camera_object1: np.ndarray,
    t_camera_object2: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    verbose: bool = False
) -> np.ndarray:
    t_cam1_cam2: np.ndarray = np.linalg.inv(t_camera_object2) @ t_camera_object1
    rotation: np.ndarray = t_cam1_cam2[:3, :3]
    translation: np.ndarray = t_cam1_cam2[:3, 3]

    h, w = img1.shape[:2]

    R1r, R2r, P1, P2, Q, _, _ = cv2.stereoRectify(
        camera_matrix, distortion_coeffs,
        camera_matrix, distortion_coeffs,
        (w, h),
        rotation,
        translation,
        alpha=1
    )

    map1x, map1y = cv2.initUndistortRectifyMap(
        camera_matrix, distortion_coeffs, R1r, P1, (w, h), cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        camera_matrix, distortion_coeffs, R2r, P2, (w, h), cv2.CV_32FC1
    )

    rect1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
    rect2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)

    gray1 = cv2.cvtColor(rect1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(rect2, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=160,
        blockSize=5
    )
    disparity = stereo.compute(gray1, gray2).astype(np.float32)

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
