import numpy as np
from src.geometry import Transformation3D


def get_fibonacci_hemisphere_trajectory(
    steps: int = 24,
    radius: float = 3.0
) -> list[Transformation3D]:
    idx: np.ndarray = np.arange(0, steps)
    golden_ratio: float = (1 + 5 ** 0.5) / 2

    theta: np.ndarray = 2 * np.pi * idx / golden_ratio
    phi: np.ndarray = np.arccos(1 - 2 * (idx + 0.5) / steps)

    x: np.ndarray  = radius * np.cos(theta) * np.sin(phi)
    y: np.ndarray  = radius * np.sin(theta) * np.sin(phi)
    z: np.ndarray  = radius * np.cos(phi)

    trajectory: list[Transformation3D] = []
    for step in range(steps):
        trajectory.append(
            Transformation3D.from_yaw_pitch_roll(
                yaw=np.pi / 2 + theta[step],
                pitch=phi[step],
                t=np.array([
                    x[step],
                    y[step],
                    z[step]
                ])
            )
        )

    return trajectory


def get_fibonacci_hemisphere_trajectory_opencv(
    steps: int = 24,
    radius: float = 3.0
) -> list[Transformation3D]:
    target = np.zeros(3)
    up = np.array([0.0, 1.0, 0.0])

    idx          = np.arange(steps, dtype=float)
    golden_ratio = (1.0 + 5.0 ** 0.5) / 2.0
    theta        = 2.0 * np.pi * idx / golden_ratio
    phi          = np.arccos(1.0 - 2.0 * (idx + 0.5) / steps)

    x = radius * np.cos(theta) * np.sin(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(phi)

    trajectory: list[Transformation3D] = []
    for i in range(steps):
        eye = np.array([x[i], y[i], z[i]])

        z_axis = target - eye
        z_axis /= np.linalg.norm(z_axis)

        x_axis = np.cross(z_axis, up)
        norm = np.linalg.norm(x_axis)
        if norm < 1e-6:
            x_axis = np.cross(z_axis, np.array([0.0, 0.0, 1.0]))
            norm = np.linalg.norm(x_axis)
        x_axis /= norm

        y_axis = np.cross(z_axis, x_axis)

        R = np.stack([x_axis, y_axis, z_axis], axis=0)
        t = R @ (-eye)

        matrix = np.eye(4)
        matrix[:3, :3] = R
        matrix[:3, 3] = t
        trajectory.append(Transformation3D(matrix=matrix))

    return trajectory
