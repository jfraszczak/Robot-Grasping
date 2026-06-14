import pathlib
import copy

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2
import open3d
import pycolmap

from src.aruco import detect_aruco_markers, get_full_circle_aruco_board
from src.aruco.visualization import draw_detected_markers
from src.geometry.camera_calibration import undistort
from src.geometry.pose_estimation import estimate_pose
from src.utils import measure_time

from src.reconstruction import IReconstructor, ImageFrame
from src.reconstruction.colmap import ColmapIncrementalReconstructor
from src.reconstruction.colmap.utils import get_camera_transformations, get_scaling_factor
from src.geometry import CameraParameters
from src.utils import measure_time
from scripts.config import Config
from src.utils import create_dir

import sys
import subprocess
import pickle
import tempfile


@measure_time
def run_relative_depth_estimation(img: Image) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as img_file:
        img.save(img_file.name)
        img_path = img_file.name

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as out_file:
        out_path = out_file.name

    subprocess.run(
        [sys.executable, "src/reconstruction/depth_anything/run.py", img_path, out_path],
        check=True
    )

    with open(out_path, "rb") as f:
        return pickle.load(f)


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


def reconstruct_with_colmap(frames: list[ImageFrame], camera_parameters: CameraParameters) -> pycolmap.Reconstruction:
    reconstructor: ColmapIncrementalReconstructor = ColmapIncrementalReconstructor()
    reconstructor.run(frames=frames, camera_parameters=camera_parameters)
    return reconstructor.get_reconstruction() 


def compute_depth_colmap(
    frame: ImageFrame,
    relative_depth: np.ndarray,
    reconstruction: pycolmap.Reconstruction,
    verbose: bool = False
) -> np.ndarray:
    relative_depth_points: list[float] = []
    camera_depth_points: list[float] = []

    frame_name: pathlib.Path = pathlib.Path(frame.path).name
    colmap_image = next(
        (img for img in reconstruction.images.values() if img.name == frame_name),
        None
    )
    if colmap_image is None:
        raise RuntimeError(f"Frame '{frame_name}' not found in COLMAP reconstruction")

    for p2d in colmap_image.points2D:
        if not p2d.has_point3D():
            continue

        px, py = int(round(p2d.xy[0])), int(round(p2d.xy[1]))
        if not (0 <= py < relative_depth.shape[0] and 0 <= px < relative_depth.shape[1]):
            continue

        p3d_world: np.ndarray = reconstruction.points3D[p2d.point3D_id].xyz
        cam_from_world = colmap_image.cam_from_world()
        p3d_cam: np.ndarray = cam_from_world * p3d_world
        metric_depth: float = float(p3d_cam[2])

        if metric_depth <= 0:
            continue

        relative_depth_points.append(relative_depth[py, px])
        camera_depth_points.append(metric_depth)

    if len(camera_depth_points) < 2:
        raise RuntimeError(
            f"Too few COLMAP reference points ({len(camera_depth_points)}) to scale depth"
        )

    coefs: np.ndarray = np.polyfit(relative_depth_points, camera_depth_points, 1)
    depth: np.ndarray = relative_depth * coefs[0] + coefs[1]

    if verbose:
        plt.scatter(relative_depth_points, camera_depth_points)
        plt.xlabel("Relative Depth")
        plt.ylabel("Camera Depth")
        plt.show()

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


def get_mask(
    img: np.ndarray,
    relative_depth: np.ndarray,
    threshold: float = 0.2,
    verbose: bool = False
) -> np.ndarray:
    colors: np.ndarray = img[:, :, ::-1] / 255.0
    mask: np.ndarray = ~np.all(colors == 0, axis=2)

    relative_depth_uint8: np.ndarray = cv2.normalize(
        relative_depth, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    _, otsu_mask = cv2.threshold(
        relative_depth_uint8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    mask = mask & otsu_mask.astype(bool)

    sobel_x: np.ndarray = cv2.Sobel(relative_depth, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y: np.ndarray = cv2.Sobel(relative_depth, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude: np.ndarray = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    mask = mask & (gradient_magnitude < threshold)

    if verbose:
        plt.imshow(mask, cmap="gray")
        plt.axis("off")
        plt.show()

    return mask


def preprocess(pcd: open3d.geometry.PointCloud, voxel_size: float) -> open3d.geometry.PointCloud:
    # Downsample to speed up ICP
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # Normals required for point-to-plane ICP
    pcd_down.estimate_normals(
        search_param=open3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2,
            max_nn=30
        )
    )
    return pcd_down


def apply_icp(
    pcds: list[open3d.geometry.PointCloud],
    voxel_size: float = 0.05,
    max_correspondence_distance: float = 0.1,
) -> open3d.geometry.PointCloud:

    merged = pcds[0]

    for i in range(1, len(pcds)):
        source = pcds[i]
        target = merged

        # Preprocess both
        source_down = preprocess(source, voxel_size)
        target_down = preprocess(target, voxel_size)

        # Point-to-plane ICP
        result = open3d.pipelines.registration.registration_icp(
            source=source_down,
            target=target_down,
            max_correspondence_distance=max_correspondence_distance,
            init=np.eye(4),
            estimation_method=open3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=open3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6,
                relative_rmse=1e-6,
                max_iteration=100
            )
        )

        print(f"ICP [{i-1} → {i}] fitness={result.fitness:.3f}  rmse={result.inlier_rmse:.4f}")

        # Apply transformation to the ORIGINAL (not downsampled) cloud
        source.transform(result.transformation)
        merged = merged + source

        # Downsample merged to prevent it growing too large
        merged = merged.voxel_down_sample(voxel_size)

    return merged


def merge_point_clouds(
    pcds: list[open3d.geometry.PointCloud],
    t_cam_to_world_matrices: list[np.ndarray]
) -> open3d.geometry.PointCloud:
    
    merged = open3d.geometry.PointCloud()
    for pcd, t_cam_to_world in zip(pcds, t_cam_to_world_matrices):
        # Transform point cloud from camera space into world space
        pcd_world = copy.deepcopy(pcd)
        pcd_world.transform(t_cam_to_world)  # (4, 4) homogeneous matrix
        merged += pcd_world
    
    return merged


class DepthAnythingReconstructor(IReconstructor):

    @measure_time
    def run(
        self,
        frames: list[ImageFrame],
        camera_parameters: CameraParameters,
        verbose: bool = False
    ) -> open3d.geometry.PointCloud:
        reconstruction: pycolmap.Reconstruction = reconstruct_with_colmap(
            frames=frames,
            camera_parameters=camera_parameters
        )
        pcds: list[open3d.geometry.PointCloud] = []
        t_cam_to_world_matrices: list[np.ndarray] = []
        import random
        random.shuffle(frames)
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
            
            # config: Config = Config.from_yaml("scripts/config.yaml")
            # aruco_dictionary: int = cv2.aruco.DICT_4X4_250
            # aruco_board: cv2.aruco.Board = get_full_circle_aruco_board(
            #     marker_size=config.aruco_board.marker_size,
            #     radius=config.aruco_board.radius,
            #     aruco_dictionary=aruco_dictionary
            # )

            # depth: np.ndarray = compute_depth(
            #     img=img,
            #     relative_depth=relative_depth,
            #     camera_matrix=camera_matrix,
            #     aruco_board=aruco_board,
            #     verbose=verbose
            # )

            try:
                depth: np.ndarray = compute_depth_colmap(
                    frame=frame,
                    relative_depth=relative_depth,
                    reconstruction=reconstruction,
                    verbose=verbose
                )
                scale_factor: float = get_scaling_factor(
                    frames=frames,
                    colmap_camera_transformations=get_camera_transformations(reconstruction)
                )
                depth *= scale_factor
            except (ValueError, RuntimeError):
                continue

            # Compute 3d reconstruction
            points3d: np.ndarray = depth_to_3d(depth=depth, camera_matrix=camera_matrix)

            # Clear depth map
            mask: np.ndarray = get_mask(img=img, relative_depth=relative_depth, verbose=verbose)
            points3d = points3d[mask.flatten()]
            colors: np.ndarray = img[:, :, ::-1] / 255.0
            colors: np.ndarray = colors[mask]

            pcd: open3d.geometry.PointCloud = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(points3d.astype(np.float64))
            pcd.colors = open3d.utility.Vector3dVector(colors.astype(np.float64))

            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=20,
                std_ratio=1.0
            )
            pcd, _ = pcd.remove_radius_outlier(
                nb_points=16,
                radius=0.05
            )

            if verbose:
                frame = open3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.5, origin=[0, 0, 0]
                )
                open3d.visualization.draw_geometries([pcd, frame])

            pcds.append(pcd)
            t_cam_to_world_matrices.append(frame.t_cam_to_world.matrix)
            frame.t_cam_to_world
            if len(pcds) == 5:
                break
        
        pcd_merged: open3d.geometry.PointCloud = merge_point_clouds(
            pcds=pcds,
            t_cam_to_world_matrices=t_cam_to_world_matrices
        )
        if verbose:
            frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
            open3d.visualization.draw_geometries([pcd_merged, frame])

        return pcd_merged
