import os

from transformers import pipeline
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import open3d
import cv2
import torch

from src.aruco import detect_aruco_markers, get_full_circle_aruco_board
from src.aruco.visualization import draw_detected_markers
from src.geometry.camera_calibration import undistort
from src.geometry.pose_estimation import estimate_pose
from src.utils import measure_time

from src.reconstruction import IReconstructor, ImageFrame
from src.geometry import CameraParameters
from src.utils import measure_time
from scripts.config import Config


def run_relative_depth_estimation(img: Image) -> np.ndarray:
    if torch.backends.mps.is_available():
        print("Metal GPU available!")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUPA with GPU.")
    else:
        device = torch.device("cpu")
        print("Metal GPU not available. Using CPU.")
    print(device)
    
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)
    depth_pil: Image = pipe(img)["depth"]
    depth_np: np.ndarray = np.array(depth_pil)

    plt.imshow(depth_np, cmap="inferno")
    plt.colorbar(label="Depth")
    plt.axis("off")
    plt.show()

    return depth_np


def compute_depth(
    img: np.ndarray,
    relative_depth: np.ndarray,
    camera_matrix: np.ndarray,
    aruco_board: cv2.aruco.Board,
    verbose: bool = False
) -> np.ndarray:
    marker_corners, marker_ids = detect_aruco_markers(
        img=img,
        aruco_board=aruco_board,
        camera_matrix=camera_matrix
    )
    draw_detected_markers(
        img=img,
        marker_corners=marker_corners,
        marker_ids=marker_ids
    )

    # Estimate poses
    obj_points, img_points = aruco_board.matchImagePoints(marker_corners, marker_ids)
    t_camera_object: np.ndarray = estimate_pose(
        obj_points=obj_points,
        img_points=img_points,
        camera_parameters=CameraParameters(matrix=camera_matrix)
    )

    obj_points = obj_points[:, 0, :]
    obj_points = np.concatenate([obj_points, np.ones((np.shape(obj_points)[0], 1))], axis=1)
    obj_points_camera_coords = (np.linalg.inv(t_camera_object) @ obj_points.T).T
    camera_depth_points: np.ndarray = obj_points_camera_coords[:, 2]
    relative_depth_points = relative_depth[img_points[:, 0, 1].astype(int), img_points[:, 0, 0].astype(int)].flatten()

    if verbose:
        plt.scatter(relative_depth_points, camera_depth_points)
        plt.xlabel("Relative Depth")
        plt.ylabel("Camera Depth")
        plt.show()

    coefs = np.polyfit(relative_depth_points, camera_depth_points, 1)
    depth: np.ndarray = relative_depth * coefs[0] + coefs[1]

    return depth


@measure_time
def depth_to_3d(depth: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    fx: float = camera_matrix[0][0]
    fy: float = camera_matrix[1][1]
    cx: float = camera_matrix[0][2]
    cy: float = camera_matrix[1][2]

    x: np.ndarray = np.linspace(0, np.shape(depth)[1] - 1, np.shape(depth)[1])
    y: np.ndarray = np.linspace(0, np.shape(depth)[0] - 1, np.shape(depth)[0])
    x, y = np.meshgrid(x, y)

    x = (x - cx) / fx * depth
    y = (y - cy) / fy * depth

    x = x[:, :, np.newaxis]
    y = y[:, :, np.newaxis]
    depth = depth[:, :, np.newaxis]

    points3d = np.concatenate((x, y, depth), axis=2)
    points3d = points3d.reshape(-1, 3)

    return points3d


class DepthAnythingReconstructor(IReconstructor):

    @measure_time
    def run(
        self,
        frames: list[ImageFrame],
        camera_parameters: CameraParameters,
        verbose: bool = False
    ) -> open3d.geometry.PointCloud:

        for frame in frames:
            # Undistort Image
            img, camera_matrix = undistort(
                img=frame.img,
                camera_matrix=camera_parameters.matrix,
                distortion_coeffs=camera_parameters.distortion_coeffs
            )

            # Relative Depth Estimation
            img_pil: Image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            relative_depth: np.ndarray = run_relative_depth_estimation(img_pil)

            # Compute Depth
            config: Config = Config.from_yaml("scripts/config.yaml")
            aruco_dictionary: int = cv2.aruco.DICT_4X4_250
            aruco_board: cv2.aruco.Board = get_full_circle_aruco_board(
                marker_size=config.aruco_board.marker_size,
                radius=config.aruco_board.radius,
                aruco_dictionary=aruco_dictionary
            )

            depth: np.ndarray = compute_depth(
                img=img,
                relative_depth=relative_depth,
                camera_matrix=camera_matrix,
                aruco_board=aruco_board,
                verbose=verbose
            )

            # Compute 3d reconstruction
            points3d: np.ndarray = depth_to_3d(depth=depth, camera_matrix=camera_matrix)

            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(points3d.astype(np.float64))

            colors = img[:, :, ::-1] / 255.0
            colors = colors.reshape(-1, 3)
            pcd.colors = open3d.utility.Vector3dVector(colors.astype(np.float64))

            if verbose:
                frame = open3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.5, origin=[0, 0, 0]
                )
                open3d.visualization.draw_geometries([pcd, frame])

            return pcd
