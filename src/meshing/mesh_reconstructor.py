import open3d

from .imesh_reconstructor import IMeshReconstructor
from src.utils import measure_time


def show_mesh(mesh: open3d.geometry.TriangleMesh) -> None:
    frame: open3d.geometry.TriangleMesh = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=(0, 0, 0))
    open3d.visualization.draw_geometries([mesh, frame])


class AlphaShapeMeshReconstructor(IMeshReconstructor):

    @measure_time
    def run(
        self,
        point_cloud: open3d.geometry.PointCloud,
        max_retries: int = 10,
        verbose: bool = False
    ) -> open3d.geometry.TriangleMesh:
        is_watertight: bool = False
        retries: int = 0
        while not is_watertight and retries <= max_retries:
            mesh: open3d.geometry.TriangleMesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                pcd=point_cloud,
                alpha=0.1 + retries * 0.05
            )
            is_watertight = mesh.is_watertight()
            retries += 1

        if verbose:
            show_mesh(mesh)

        if not mesh.is_watertight():
            raise RuntimeError("Obtained mesh is not watertight")

        return mesh


class PoissonMeshReconstructor(IMeshReconstructor):

    @measure_time
    def run(
        self,
        point_cloud: open3d.geometry.PointCloud,
        max_retries: int = 10,
        verbose: bool = False
    ) -> open3d.geometry.TriangleMesh:
        depth: int = 12
        if max_retries >= depth:
            raise ValueError(f"max_rerties cannot exceed {depth - 1}")

        point_cloud.estimate_normals(
            search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        point_cloud.orient_normals_consistent_tangent_plane(100)

        is_watertight: bool = False
        retries: int = 0
        mesh: open3d.geometry.TriangleMesh | None = None
        while not is_watertight and retries <= max_retries:
            try:
                mesh, _ = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    point_cloud,
                    depth=depth - retries
                )
            except:
                retries += 1
                continue
            is_watertight = mesh.is_watertight()
            retries += 1

        if mesh is None:
            raise RuntimeError("Failed to contruct mesh")

        if verbose:
            show_mesh(mesh)

        if not mesh.is_watertight():
            raise RuntimeError("Obtained mesh is not watertight")

        return mesh
