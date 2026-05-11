import open3d
import numpy as np


frame = open3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.5, origin=[0, 0, 0]
)
box = open3d.geometry.TriangleMesh.create_box()

box.compute_triangle_normals()
print(np.asarray(box.triangle_normals))

vertices: np.ndarray = np.asarray(box.vertices)
triangles: np.ndarray = np.asarray(box.triangles)

print(vertices)
print(triangles)

contact_points: np.ndarray = np.empty((0, vertices.shape[1]))
for i in range(np.shape(triangles)[0]):
    vertex1: int = vertices[triangles[i][0]]
    vertex2: int = vertices[triangles[i][1]]
    vertex3: int = vertices[triangles[i][2]]

    contact_points = np.vstack((
        contact_points,
        np.array([
            (vertex1 + vertex2) / 2,
            (vertex1 + vertex3) / 2,
            (vertex2 + vertex3) / 2
        ])
    ))

print(contact_points)

candidates: set[tuple[np.ndarray, np.ndarray]] = set()
for i in range(np.shape(contact_points)[0] - 1):
    for j in range(i + 1, np.shape(contact_points)[0]):
        if np.all(box.triangle_normals[i // 3] + box.triangle_normals[j // 3] == 0):
            torque1: np.ndarray = np.cross(contact_points[i], box.triangle_normals[i // 3])
            torque2: np.ndarray = np.cross(contact_points[j], box.triangle_normals[j // 3])
            if np.all(torque1 + torque2 == 0):
                candidates.add((str(contact_points[i]), str(contact_points[j])))

print(candidates, len(candidates))

open3d.visualization.draw_geometries([box, frame], mesh_show_wireframe=True)
