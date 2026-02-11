import os

import cv2
import numpy as np
import pycolmap

from src.utils import measure_time
from src.reconstruction.colmap.visualization import (
    visualize_keypoints,
    visualize_reconstruction,
    save_reconstruction
)


def construct_reconstruction(
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    poses: dict[str, np.ndarray]
) -> None:
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    k1, k2, p1, p2 = distortion_coeffs.flatten()[:4]

    sample_img = cv2.imread(list(poses.keys())[0])
    height, width = sample_img.shape[:2]

    reconstruction: pycolmap.Reconstruction = pycolmap.Reconstruction()
    camera_id: int = 1
    camera = pycolmap.Camera({
        "camera_id": camera_id,
        "model": pycolmap.CameraModelId.OPENCV,
        "width": width,
        "height": height,
        "params": [fx, fy, cx, cy, k1, k2, p1, p2],
    })
    reconstruction.add_camera(camera)

    rig = pycolmap.Rig()
    rig.rig_id = 1
    sensor_t = pycolmap.sensor_t()
    sensor_t.type = pycolmap.SensorType.CAMERA
    sensor_t.id = camera.camera_id
    rig.add_ref_sensor(sensor_t)
    reconstruction.add_rig(rig)

    image_id: int = 1
    frame_id: int = 1

    for img_name, pose in poses.items():
        frame = pycolmap.Frame()
        frame.frame_id = frame_id
        frame.rig_id = rig.rig_id
        rigid3d = pycolmap.Rigid3d(np.linalg.inv(pose)[:3, :])
        print(np.linalg.inv(pose))
        image = pycolmap.Image(
            image_id=image_id, name=os.path.basename(img_name), camera_id=camera.camera_id, frame_id=frame.frame_id
        )
        frame.add_data_id(image.data_id)
        reconstruction.add_frame(frame)
        reconstruction.frame(frame.frame_id).set_cam_from_world(camera_id=camera.camera_id, cam_from_world=rigid3d)
        reconstruction.register_frame(frame_id)
        reconstruction.add_image(image)

        image_id += 1
        frame_id += 1

    return reconstruction


@measure_time
def sparse_reconstruction_based_on_poses(
    images_path: str,
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    poses: dict[str, np.ndarray],
    verbose: bool = False
) -> dict[str, np.ndarray]:
    database_path: str = "database.db"
    sparse_path: str = "sparse"

    if os.path.exists(database_path):
        os.remove(database_path)

    reconstruction: pycolmap.Reconstruction = construct_reconstruction(
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs,
        poses=poses
    )
    #reconstruction.write_text(sparse_path)

    if verbose:
        print("Cameras:", reconstruction.cameras)
        print(len(reconstruction.images), "registered images")
        print(reconstruction.frames)

    db: pycolmap.Database = pycolmap.Database.open(database_path)

    db.write_camera(next(iter(reconstruction.cameras.values())))
    db.write_rig(next(iter(reconstruction.rigs.values())))

    for image_id in sorted(reconstruction.images.keys()):
        db.write_image(reconstruction.images[image_id])

    for frame_id in sorted(reconstruction.frames.keys()):
        db.write_frame(reconstruction.frames[frame_id])

    pycolmap.extract_features(
        database_path=database_path,
        image_path=images_path,
        camera_mode=pycolmap.CameraMode.AUTO,
        camera_model="OPENCV"
    )
    pycolmap.match_exhaustive(database_path)

    if verbose:
        visualize_keypoints(
            database_path=database_path,
            images_path=images_path
        )

    reconstruction = pycolmap.triangulate_points(
        reconstruction=reconstruction,
        database_path=database_path,
        image_path=images_path,
        output_path=sparse_path,
        clear_points=True,
        refine_intrinsics=True
    )
    save_reconstruction(reconstruction)

    if verbose:
        print(len(reconstruction.points3D), "3D points")
        visualize_reconstruction(reconstruction)


@measure_time
def sparse_reconstruction_based_on_images(
    images_path: str,
    verbose: bool = False
) -> dict[str, np.ndarray]:
    database_path: str = "database.db"
    sparse_path: str = "sparse"

    if os.path.exists(database_path):
        os.remove(database_path)

    pycolmap.extract_features(
        database_path=database_path,
        image_path=images_path,
        camera_mode=pycolmap.CameraMode.SINGLE,
        camera_model="OPENCV"
    )
    pycolmap.match_exhaustive(database_path)

    if verbose:
        visualize_keypoints(
            database_path=database_path,
            images_path=images_path
        )

    reconstruction: pycolmap.Reconstruction = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=images_path,
        output_path=sparse_path
    )[0]
    save_reconstruction(reconstruction)

    if verbose:
        print("Cameras:", reconstruction.cameras)
        print(len(reconstruction.images), "registered images")
        print(reconstruction.frames)
        print(len(reconstruction.points3D), "3D points")
        visualize_reconstruction(reconstruction)
