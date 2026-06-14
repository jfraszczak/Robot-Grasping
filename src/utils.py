import time
from typing import Callable, Any
import shutil
import pathlib

import trimesh
import open3d
import numpy as np


def measure_time(func: Callable) -> None:
    def wrapper(*args, **kwargs) -> Any:
        start: float = time.perf_counter()
        result: Any = func(*args, **kwargs)
        end: float = time.perf_counter()
        print(f"{func.__name__} took {end - start:.4f}s")
        return result
    return wrapper


def create_dir(path: str | pathlib.Path) -> None:
    path: pathlib.Path = pathlib.Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def load_mesh(obj_file: str) -> open3d.geometry.TriangleMesh:
    mesh_trimesh = trimesh.load_mesh(obj_file)
    vertex_colors = mesh_trimesh.visual.to_color().vertex_colors[:, :3] / 255.0

    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(np.array(mesh_trimesh.vertices))
    mesh.triangles = open3d.utility.Vector3iVector(np.array(mesh_trimesh.faces))
    mesh.vertex_colors = open3d.utility.Vector3dVector(vertex_colors)

    verts = np.asarray(mesh.vertices)
    verts -= verts.mean(axis=0)
    mesh.vertices = open3d.utility.Vector3dVector(verts)

    mesh.compute_vertex_normals()

    return mesh
