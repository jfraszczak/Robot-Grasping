import os
from contextlib import contextmanager
from pathlib import Path
import tempfile
from typing import Generator
import pathlib

import numpy as np
import pycolmap
import open3d
import cv2

from src.reconstruction import IReconstructor, ImageFrame
from src.geometry import CameraParameters, Transformation3D
from src.utils import measure_time
from .utils import transform_from_colmap_to_original_frame
from src.utils import create_dir


def construct_reconstruction(
    image_frames: list[ImageFrame],
    camera_parameters: CameraParameters
) -> pycolmap.Reconstruction:
    height, width = image_frames[0].img.shape[:2]

    reconstruction: pycolmap.Reconstruction = pycolmap.Reconstruction()
    camera_id: int = 1
    camera = pycolmap.Camera({
        "camera_id": camera_id,
        "model": pycolmap.CameraModelId.OPENCV,
        "width": width,
        "height": height,
        "params": (
            [camera_parameters.fx, camera_parameters.fy, camera_parameters.cx, camera_parameters.cy, *camera_parameters.distortion_coeffs.flatten()[:4]]
        ),
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

    for image_frame in image_frames:
        frame = pycolmap.Frame()
        frame.frame_id = frame_id
        frame.rig_id = rig.rig_id
        rigid3d = pycolmap.Rigid3d(np.linalg.inv(image_frame.t_cam_to_world.matrix)[:3, :])
        image = pycolmap.Image(
            image_id=image_id, name=os.path.basename(image_frame.path), camera_id=camera.camera_id, frame_id=frame.frame_id
        )
        frame.add_data_id(image.data_id)
        reconstruction.add_frame(frame)
        reconstruction.frame(frame.frame_id).set_cam_from_world(camera_id=camera.camera_id, cam_from_world=rigid3d)
        reconstruction.register_frame(frame_id)
        reconstruction.add_image(image)

        image_id += 1
        frame_id += 1

    return reconstruction


@contextmanager
def tmp_images_dir(image_frames: list[ImageFrame],) -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        for image_frame in image_frames:
            cv2.imwrite(f"{tmp_dir}/{os.path.basename(image_frame.path)}", image_frame.img)
        yield Path(tmp_dir)


class ColmapTriangulationReconstructor(IReconstructor):

    def __init__(self) -> None:
        super().__init__()
        self.reconstruction: pycolmap.Reconstruction | None = None
    
    @measure_time
    def run(
        self,
        frames: list[ImageFrame],
        camera_parameters: CameraParameters,
        verbose: bool = False
    ) -> open3d.geometry.PointCloud:
        output_path: pathlib.Path = pathlib.Path("output_colmap")
        database_path: pathlib.Path = output_path / "database.db"
        create_dir(output_path)

        reconstruction: pycolmap.Reconstruction = construct_reconstruction(
            image_frames=frames,
            camera_parameters=camera_parameters
        )
        reconstruction.write(output_path)

        db: pycolmap.Database = pycolmap.Database.open(database_path)

        db.write_camera(next(iter(reconstruction.cameras.values())))
        db.write_rig(next(iter(reconstruction.rigs.values())))

        for frame_id in sorted(reconstruction.frames.keys()):
            db.write_frame(reconstruction.frames[frame_id])

        for image_id in sorted(reconstruction.images.keys()):
            db.write_image(reconstruction.images[image_id])

        with tmp_images_dir(frames) as images_path:
            pycolmap.extract_features(
                database_path=database_path,
                image_path=images_path,
                camera_mode=pycolmap.CameraMode.AUTO,
                reader_options=pycolmap.ImageReaderOptions(
                    camera_model="OPENCV"
                )
            )
            pycolmap.match_exhaustive(database_path=database_path)
            reconstruction = pycolmap.triangulate_points(
                reconstruction=reconstruction,
                database_path=database_path,
                image_path=images_path,
                output_path=output_path
            )
            self.reconstruction = reconstruction

        point_cloud: open3d.geometry.PointCloud = transform_from_colmap_to_original_frame(
            reconstruction=reconstruction,
            frames=frames
        )
        if verbose:
            coordinate_frame: open3d.geometry.TriangleMesh = open3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.05,
                origin=[0, 0, 0]
            )
            open3d.visualization.draw_geometries([point_cloud, coordinate_frame])

        return point_cloud


    def get_reconstruction(self) -> pycolmap.Reconstruction | None:
        return self.reconstruction
    