from abc import ABC, abstractmethod
from src.geometry import CameraParameters
from src.reconstruction import ImageFrame, StereoFrame


class IRenderer(ABC):

    def __init__(
        self,
        camera_parameters: CameraParameters
    ) -> None:
        self.camera_parameters: CameraParameters = camera_parameters
    
    @abstractmethod
    def run(
        self,
        obj_file: str,
        stereo: bool = False,
        frame_count: int = 8,
        output_dir: str = "render",
        verbose: bool = False
    ) -> list[ImageFrame | StereoFrame]:
        pass
