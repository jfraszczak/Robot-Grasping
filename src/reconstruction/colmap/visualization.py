import os
import uuid

import cv2
import numpy as np
import pycolmap
import open3d as o3d


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


def _build_pcd(reconstruction: pycolmap.Reconstruction) -> o3d.geometry.PointCloud:
    points: list[float] = []
    colors: list[float]  = []

    for point in reconstruction.points3D.values():
        points.append(point.xyz)
        colors.append(point.color / 255.0)

    pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors))
    pcd = pcd.voxel_down_sample(0.01)
    pcd, _ = pcd.remove_statistical_outlier(20, 2.0)
    return pcd


def visualize_reconstruction(reconstruction: pycolmap.Reconstruction) -> None:
    pcd: o3d.geometry.PointCloud = _build_pcd(reconstruction)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0, 0, 0]
    )
    o3d.visualization.draw_geometries([pcd, frame])


def save_reconstruction(reconstruction: pycolmap.Reconstruction) -> None:
    pcd: o3d.geometry.PointCloud = _build_pcd(reconstruction)
    o3d.io.write_point_cloud(f"reconstruction.ply", pcd)
