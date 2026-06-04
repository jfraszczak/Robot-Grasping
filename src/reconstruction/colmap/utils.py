import os

import cv2
import numpy as np
import pycolmap
import open3d
from scipy import stats

from src.reconstruction import ImageFrame
from src.geometry import Transformation3D


def visualize_keypoints(database_path: str, images_path: str) -> None:
    db: pycolmap.Database = pycolmap.Database.open(database_path)
    for image in db.read_all_images():
        img_path: str = os.path.join(images_path, image.name)
        img: np.ndarray = cv2.imread(img_path)
        keypoints: np.ndarray = db.read_keypoints(image.image_id)
        pts: np.ndarray = keypoints[:, :2]
        for x, y in pts:
            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)

        cv2.imshow(f"Keypoints: {image.name}", img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    db.close()


def build_point_cloud(reconstruction: pycolmap.Reconstruction) -> open3d.geometry.PointCloud:
    points: list[float] = []
    colors: list[float]  = []

    for point in reconstruction.points3D.values():
        points.append(point.xyz)
        colors.append(point.color / 255.0)

    pcd: open3d.geometry.PointCloud = open3d.geometry.PointCloud()
    if len(reconstruction.points3D) == 0:
        return pcd
    
    pcd.points = open3d.utility.Vector3dVector(np.asarray(points))
    pcd.colors = open3d.utility.Vector3dVector(np.asarray(colors))
    pcd, _ = pcd.remove_statistical_outlier(5, 0.05)
    return pcd


def get_camera_transformations(reconstruction: pycolmap.Reconstruction) -> dict[str, Transformation3D]:
    camera_transformations: dict[str, Transformation3D] = {}
    for image in reconstruction.images.values():
        cam_from_world: pycolmap.Rigid3d = image.cam_from_world()
        rigid3d_inverse = cam_from_world.inverse()
        cam_to_world: np.ndarray = np.eye(4)
        cam_to_world[:3, 3] = rigid3d_inverse.translation
        cam_to_world[:3, :3] = rigid3d_inverse.rotation.matrix()
        camera_transformations[image.name] = Transformation3D(matrix=cam_to_world)
    return camera_transformations


def get_scaling_factor(
    frames: list[ImageFrame],
    colmap_camera_transformations: dict[str, Transformation3D]
) -> float:
    x: list[float] = []
    y: list[float] = []
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            x.append(
                np.linalg.norm(frames[i].t_cam_to_world.matrix[:3, 3] - frames[j].t_cam_to_world.matrix[:3, 3])
            )
            y.append(
                np.linalg.norm(colmap_camera_transformations[os.path.basename(frames[i].path)].matrix[:3, 3] - colmap_camera_transformations[os.path.basename(frames[j].path)].matrix[:3, 3])
            )
    
    x_numpy: np.ndarray = np.array(x)
    y_numpy: np.ndarray = np.array(y)

    return 1 / stats.linregress(x_numpy, y_numpy).slope


def transform_from_colmap_to_original_frame(reconstruction: pycolmap.Reconstruction, frames: list[ImageFrame]) -> open3d.geometry.PointCloud:
    point_cloud: open3d.geometry.PointCloud = build_point_cloud(reconstruction)
    camera_transformations: dict[str, Transformation3D] = get_camera_transformations(reconstruction)

    scale_factor: float = get_scaling_factor(frames, camera_transformations)
    point_cloud_scaled: open3d.geometry.PointCloud = point_cloud.scale(scale=scale_factor, center=point_cloud.get_center())

    for frame in frames:
        if os.path.basename(frame.path) in camera_transformations:
            t_colmap: np.ndarray = camera_transformations[os.path.basename(frame.path)].inverse()
            t_colmap.matrix[:3, 3] *= scale_factor

            t_colmap_to_world: Transformation3D = frame.t_cam_to_world @ t_colmap
            point_cloud_scaled.transform(t_colmap_to_world.matrix)
            return point_cloud_scaled
    raise RuntimeError("Failed to transform object to original coordinate frame")
