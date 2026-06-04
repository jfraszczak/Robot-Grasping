import open3d

from src.geometry import CameraParameters
from src.reconstruction import Frame, IReconstructor
from src.meshing import IMeshReconstructor
from src.grasping import IGraspEstimator, PhysicalCoefficients, GraspTrajectory


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
        physical_coeffs: PhysicalCoefficients,
        verbose: bool = False
    ) -> GraspTrajectory | None:
        point_cloud: open3d.geometry.PointCloud = self.reconstructor.run(
            frames=frames,
            camera_parameters=camera_parameters,
            verbose=verbose
        )
        mesh: open3d.geometry.TriangleMesh = self.mesh_reconstructor.run(
            point_cloud=point_cloud,
            verbose=verbose
        )
        grasp_trajectory: GraspTrajectory | None = self.grasp_estimator.run(
            mesh=mesh,
            physical_coeffs=physical_coeffs,
            verbose=verbose
        )
        return grasp_trajectory
