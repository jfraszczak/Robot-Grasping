from __future__ import annotations
from abc import ABC

from pydantic import BaseModel, model_validator, ConfigDict
import numpy as np
import cv2

from src.geometry import Transformation3D


class Frame(ABC, BaseModel):
    pass
    

class ImageFrame(Frame):
    path: str
    img: np.ndarray | None = None
    t_cam_to_world: Transformation3D

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def load_image(self) -> ImageFrame:
        if self.img is not None:
            raise ValueError("img must not be set manually, it is loaded from path")
        img = cv2.imread(str(self.path))
        if img is None:
            raise ValueError(f"Failed to load image: {self.path}")
        self.img = img
        return self


class StereoFrame(Frame):
    img_left: np.ndarray
    img_right: np.ndarray
    t_left_to_right: Transformation3D
    t_world_to_left: Transformation3D

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)
