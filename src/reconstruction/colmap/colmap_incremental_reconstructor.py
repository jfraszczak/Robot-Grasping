import pathlib

import pycolmap
import open3d

from src.reconstruction import IReconstructor, ImageFrame
from src.geometry import CameraParameters
from src.utils import measure_time
from .utils import transform_from_colmap_to_original_frame
from src.utils import create_dir


class ColmapIncrementalReconstructor(IReconstructor):

    def __init__(self) -> None:
        super().__init__()
        self.reconstruction: pycolmap.Reconstruction | None = None

    @measure_time
    def run(
        self,
        frames: list[ImageFrame],
        camera_parameters: CameraParameters,
        dense: bool = False,
        verbose: bool = False
    ) -> open3d.geometry.PointCloud:
        output_path: pathlib.Path = pathlib.Path("output_colmap")
        images_path: pathlib.Path = pathlib.Path(frames[0].path).parent
        database_path: pathlib.Path = output_path / "database.db"
        create_dir(output_path)

        pycolmap.extract_features(
            database_path=database_path,
            image_path=images_path,
            reader_options=pycolmap.ImageReaderOptions(
                camera_model="PINHOLE",
                camera_params=f"{camera_parameters.fx},{camera_parameters.fy},{camera_parameters.cx},{camera_parameters.cy}"
            )
        )
        pycolmap.match_exhaustive(database_path=database_path)
        reconstruction: pycolmap.Reconstruction = pycolmap.incremental_mapping(
            database_path=database_path,
            image_path=images_path,
            output_path=output_path,
            options=pycolmap.IncrementalPipelineOptions(random_seed=42)
        )[0]
        reconstruction.write(output_path)
        
        if dense:
            mvs_path = output_path / "mvs"
            pycolmap.undistort_images(mvs_path, output_path, images_path)
            pycolmap.patch_match_stereo(mvs_path)
            reconstruction: pycolmap.Reconstruction = pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)

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
