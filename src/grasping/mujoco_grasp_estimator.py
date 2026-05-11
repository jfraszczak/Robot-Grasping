import open3d
import numpy as np

from src.geometry import Transformation3D
from .igrasp_estimator import IGraspEstimator


class MujocoGraspEstimator(IGraspEstimator):

    def run(
        self,
        mesh: open3d.geometry.TriangleMesh,
        verbose: bool = False
    ) -> Transformation3D:
        return Transformation3D(
            matrix=np.eye(4)
        )
