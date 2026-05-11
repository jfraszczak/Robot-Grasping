from __future__ import annotations
from pydantic import BaseModel, ConfigDict, model_validator
import numpy as np


class CameraParameters(BaseModel):
    matrix: np.ndarray
    distortion_coeffs: np.ndarray = np.zeros((1, 4))
    resolution_x: int = 2000
    resolution_y: int = 2000

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    @property
    def fx(self) -> float:
        return self.matrix[0, 0]
    
    @property
    def fy(self) -> float:
        return self.matrix[1, 1]
    
    @property
    def cx(self) -> float:
        return self.matrix[0, 2]
    
    @property
    def cy(self) -> float:
        return self.matrix[1, 2]


class Transformation3D(BaseModel):
    matrix: np.ndarray
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_matrix(self) -> Transformation3D:
        if self.matrix.shape != (4, 4):
            raise ValueError(f"Expected 4x4 matrix, got {self.matrix.shape}")
        return self
    
    def inverse(self) -> Transformation3D:
        return Transformation3D(matrix=np.linalg.inv(self.matrix))

    @property
    def x(self) -> float:
        return self.matrix[0, 3] / self.matrix[3, 3]
    
    @property
    def y(self) -> float:
        return self.matrix[1, 3] / self.matrix[3, 3]
    
    @property
    def z(self) -> float:
        return self.matrix[2, 3] / self.matrix[3, 3]
    
    @property
    def yaw(self) -> float:
        return float(np.arctan2(self.matrix[1, 0], self.matrix[0, 0]))

    @property
    def pitch(self) -> float:
        return float(np.arctan2(-self.matrix[2, 0], np.sqrt(self.matrix[2, 1] ** 2 + self.matrix[2, 2] ** 2)))

    @property
    def roll(self) -> float:
        return float(np.arctan2(self.matrix[2, 1], self.matrix[2, 2]))

    @classmethod
    def from_yaw_pitch_roll(
        cls,
        yaw: float = 0.0,
        pitch: float = 0.0,
        roll: float = 0.0,
        t: np.ndarray | None = None,
    ) -> Transformation3D:
        if t is None:
            t = np.array([0.0, 0.0, 0.0])

        if np.shape(t) != (3,):
            raise ValueError("Translation shall be of shape (3,)")

        cy, sy = np.cos(yaw),   np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll),  np.sin(roll)

        rotation_z: np.ndarray = np.array([
            [ cy, -sy, 0],
            [ sy,  cy, 0],
            [  0,   0, 1],
        ])
        rotation_y: np.ndarray = np.array([ 
            [ cp,  0, sp],
            [  0,  1,  0],
            [-sp,  0, cp],
        ])
        rotation_x: np.ndarray = np.array([
            [1,   0,   0],
            [0,  cr, -sr],
            [0,  sr,  cr],
        ])

        matrix: np.ndarray = np.eye(4)
        matrix[:3, :3] = rotation_z @ rotation_y @ rotation_x
        matrix[:3,  3] = t

        return cls(matrix=matrix)
    
    def __matmul__(self, other: Transformation3D) -> Transformation3D:
        return Transformation3D(matrix=self.matrix @ other.matrix)
