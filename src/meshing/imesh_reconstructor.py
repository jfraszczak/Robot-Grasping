from abc import ABC, abstractmethod
import open3d


class IMeshReconstructor(ABC):

    @abstractmethod
    def run(
        self,
        point_cloud: open3d.geometry.PointCloud,
        max_retries: int = 10,
        verbose: bool = False
    ) -> open3d.geometry.TriangleMesh:
        pass
