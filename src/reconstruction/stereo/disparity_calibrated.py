import cv2
import numpy as np
import open3d


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
        alpha=-1
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
        numDisparities=128,
        blockSize=9
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

    points3d: np.ndarray = cv2.reprojectImageTo3D(disparity, Q)
    print(np.shape(points3d))

    mask: np.ndarray = disparity > 0
    points3d = points3d[mask]
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points3d.astype(np.float64))

    colors = rect1[mask][:, ::-1] / 255.0
    pcd.colors = open3d.utility.Vector3dVector(colors.astype(np.float64))
    open3d.visualization.draw_geometries([pcd])

    return disparity
