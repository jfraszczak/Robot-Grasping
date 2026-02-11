from transformers import pipeline
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import open3d
import cv2
import yaml

from src.geometry.camera_calibration import calibrate_camera, undistort
from src.aruco.markers_detection import detect_aruco_markers, get_full_circle_aruco_board
from src.aruco.visualization import draw_detected_markers
from src.geometry.camera_calibration import calibrate_camera, undistort
from src.geometry.pose_estimation import estimate_pose
from src.utils import measure_time


def run_relative_depth_estimation(img: Image) -> np.ndarray:
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Base-hf")
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
    aruco_board: cv2.aruco.Board
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
        camera_matrix=camera_matrix
    )

    obj_points = obj_points[:, 0, :]
    obj_points = np.concatenate([obj_points, np.ones((np.shape(obj_points)[0], 1))], axis=1)
    obj_points_camera_coords = (np.linalg.inv(t_camera_object) @ obj_points.T).T
    camera_depth_points: np.ndarray = obj_points_camera_coords[:, 2]
    relative_depth_points = relative_depth[img_points[:, 0, 1].astype(int), img_points[:, 0, 0].astype(int)].flatten()

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


def run_3d_reconstruction(img_path: str) -> None:
    with open("scripts/config.yaml", "r") as file:
        config: dict = yaml.safe_load(file)

    # Calibrate camera
    camera_matrix, distortion_coeffs = calibrate_camera(
        images_path=config["camera_calibration"]["path"],
        chessboard_size=(config["camera_calibration"]["chessboard_rows"], config["camera_calibration"]["chessboard_columns"]),
        square_size=config["camera_calibration"]["square_size"],
        verbose=False
    )

    # Read image
    img: np.ndarray = cv2.imread(img_path)
    img, camera_matrix = undistort(
        img=img,
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs
    )

    # Relative Depth Estimation
    img_pil: Image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    relative_depth: np.ndarray = run_relative_depth_estimation(img_pil)

    # Compute Depth
    aruco_dictionary: int = cv2.aruco.DICT_4X4_250
    aruco_board: cv2.aruco.Board = get_full_circle_aruco_board(
        marker_size=config["aruco_board"]["marker_size"],
        radius=config["aruco_board"]["radius"],
        aruco_dictionary=aruco_dictionary
    )

    depth: np.ndarray = compute_depth(
        img=img,
        relative_depth=relative_depth,
        camera_matrix=camera_matrix,
        aruco_board=aruco_board
    )

    # Compute 3d reconstruction
    points3d: np.ndarray = depth_to_3d(depth=depth, camera_matrix=camera_matrix)

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points3d.astype(np.float64))

    colors = img[:, :, ::-1] / 255.0
    colors = colors.reshape(-1, 3)
    pcd.colors = open3d.utility.Vector3dVector(colors.astype(np.float64))

    frame = open3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0, 0, 0]
    )

    open3d.visualization.draw_geometries([pcd, frame])


if __name__ == "__main__":
    #run_3d_reconstruction("data/green-rasp-1/IMG_20251210_101023320.jpg")
    run_3d_reconstruction("data/object-4/IMG_20251210_101421714.jpg")
