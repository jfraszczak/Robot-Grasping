import cv2
import numpy as np
from .models import CameraParameters


def estimate_pose(
    obj_points: np.ndarray,
    img_points: np.ndarray,
    camera_parameters: CameraParameters
) -> tuple[np.ndarray]:
    """
    Estimates the camera pose in object coordinates from 3D-2D correspondences.
    
    Returns:
        np.ndarray: 4 x 4 homogeneous camera to object frame transformation
    """
    success, rotation_vector, translation_vector = cv2.solvePnP(
        obj_points,
        img_points,
        camera_parameters.matrix,
        camera_parameters.distortion_coeffs
    )

    if not success:
        raise Exception("Failed to estimate pose")
    
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    t_object_camera: np.ndarray = np.eye(4)
    t_object_camera[:3, :3] = rotation_matrix
    t_object_camera[:3, 3] = translation_vector.flatten()
    t_camera_object: np.ndarray = np.linalg.inv(t_object_camera)

    return t_camera_object


def draw_object_frame(
    img: np.ndarray,
    t_object_camera: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray
) -> None:
    img_copy: np.ndarray = img.copy()
    cv2.drawFrameAxes(img_copy, camera_matrix, distortion_coeffs, t_object_camera[:3, :3], t_object_camera[:3, 3], 1.0)
    cv2.imshow("axes", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
