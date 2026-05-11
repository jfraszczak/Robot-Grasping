from __future__ import annotations
from pydantic import BaseModel
import yaml


class ImageSourceConfig(BaseModel):
    path: str


class CameraCalibrationConfig(BaseModel):
    path: str
    chessboard_rows: int
    chessboard_columns: int
    square_size: float


class ArucoBoardConfig(BaseModel):
    marker_size: float
    radius: float


class Config(BaseModel):
    image_source: ImageSourceConfig
    camera_calibration: CameraCalibrationConfig
    aruco_board: ArucoBoardConfig | None = None

    @classmethod
    def from_yaml(cls, path: str) -> Config:
        with open(path) as file:
            data: dict = yaml.safe_load(file)
        return cls(**data)
