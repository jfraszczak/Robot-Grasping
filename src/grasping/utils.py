import open3d
import numpy as np


def get_bounding_sphere(
    mesh: open3d.geometry.TriangleMesh,
    verbose: bool = False
) -> tuple[float, float]:
    verts: np.ndarray = np.asarray(mesh.vertices)
    center: float = verts.mean(axis=0)
    radius: float = np.max(np.linalg.norm(verts - center, axis=1))

    if verbose:
        print(f"Center: {center}")
        print(f"Radius: {radius:.4f}m")

        # Create sphere mesh for visualization
        sphere = open3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(center)
        sphere.paint_uniform_color([1, 0, 0])  # red

        # Make sphere wireframe by converting to lineset
        sphere_lines = open3d.geometry.LineSet.create_from_triangle_mesh(sphere)
        sphere_lines.paint_uniform_color([1, 0, 0])

        # Visualize
        open3d.visualization.draw_geometries(
            [mesh, sphere_lines],
            window_name="Mesh + Bounding Sphere",
            width=1024,
            height=768
        )
    
    return center, radius


if __name__ == "__main__":
    mesh: open3d.geometry.TriangleMesh = open3d.io.read_triangle_mesh(
        "data/ycb/strawberry/012_strawberry/google_512k/textured.obj"
    )
    mesh.compute_vertex_normals()
    get_bounding_sphere(mesh, verbose=True)
