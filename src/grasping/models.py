from pydantic import BaseModel, Field, ConfigDict
import numpy as np


class PhysicalCoefficients(BaseModel):
    mass: float = Field(..., ge=0.0)
    friction_sliding: float = Field(..., ge=0.0)
    friction_torsional: float = Field(..., ge=0.0)
    friction_rolling: float = Field(..., ge=0.0)


class GrasperState(BaseModel):
    qpos: np.ndarray
    grasping: bool
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)


class GraspTrajectory(BaseModel):
    trajectory: list[GrasperState]
