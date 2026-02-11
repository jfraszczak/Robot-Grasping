import numpy as np


def transformation_matrix_2d(angle: float = 0.0, tx: float = 0.0, ty: float = 0.0) -> np.ndarray:
    return np.array([
        [np.cos(angle), -np.sin(angle), tx],
        [np.sin(angle), np.cos(angle), ty],
        [0.0, 0.0, 1.0]
    ])
