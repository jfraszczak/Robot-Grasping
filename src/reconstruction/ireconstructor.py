from abc import ABC, abstractmethod
import open3d
from src.geometry.models import CameraParameters
from .models import Frame


class IReconstructor(ABC):

    @abstractmethod
    def run(
        frames: list[Frame],
        camera_parameters: CameraParameters,
        verbose: bool = False
    ) -> open3d.geometry.PointCloud:
        pass
