from abc import ABC, abstractmethod
import open3d
from .models import PhysicalCoefficients, GraspTrajectory


class IGraspEstimator(ABC):

    @abstractmethod
    def run(
        self,
        mesh: open3d.geometry.TriangleMesh,
        physical_coeffs: PhysicalCoefficients,
        verbose: bool = False
    ) -> GraspTrajectory | None:
        pass

    @abstractmethod
    def evaluate_grasp(
        self,
        mesh: open3d.geometry.TriangleMesh,
        physical_coeffs: PhysicalCoefficients,
        grasp_trajectory: GraspTrajectory,
        verbose: bool = False
    ) -> bool:
        pass
