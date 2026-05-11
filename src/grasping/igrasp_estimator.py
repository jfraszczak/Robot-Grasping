from abc import ABC, abstractmethod
import open3d
from src.geometry import Transformation3D


class IGraspEstimator(ABC):

    @abstractmethod
    def run(
        self,
        mesh: open3d.geometry.TriangleMesh,
        verbose: bool = False
    ) -> Transformation3D:
        pass
