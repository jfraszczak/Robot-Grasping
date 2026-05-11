import open3d

from src.geometry import CameraParameters, Transformation3D
from src.reconstruction import Frame, IReconstructor
from src.meshing import IMeshReconstructor
from src.grasping import IGraspEstimator


class GraspingPipeline:

    def __init__(
        self,
        reconstructor: IReconstructor,
        mesh_reconstructor: IMeshReconstructor,
        grasp_estimator: IGraspEstimator
    ) -> None:
        self.reconstructor: IReconstructor = reconstructor
        self.mesh_reconstructor: IMeshReconstructor = mesh_reconstructor
        self.grasp_estimator: IGraspEstimator = grasp_estimator

    def run(
        self, 
        frames: list[Frame],
        camera_parameters: CameraParameters,
        verbose: bool = False
    ) -> Transformation3D:
        point_cloud: open3d.geometry.PointCloud = self.reconstructor.run(
            frames=frames,
            camera_parameters=camera_parameters,
            verbose=verbose
        )
        mesh: open3d.geometry.TriangleMesh = self.mesh_reconstructor.run(
            point_cloud=point_cloud,
            verbose=verbose
        )
        grasp: Transformation3D = self.grasp_estimator.run(
            mesh=mesh,
            verbose=verbose
        )
        return grasp
