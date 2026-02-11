import open3d as o3d

frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.5, origin=[0, 0, 0]
)

pcd = o3d.io.read_point_cloud(f"reconstructions/raspberry.ply")
o3d.visualization.draw_geometries([pcd, frame])

pcd = o3d.io.read_point_cloud(f"reconstructions/cashew.ply")
o3d.visualization.draw_geometries([pcd, frame])

pcd = o3d.io.read_point_cloud(f"reconstructions/mario.ply")
o3d.visualization.draw_geometries([pcd, frame])

pcd = o3d.io.read_point_cloud(f"reconstructions/object-1.ply")
o3d.visualization.draw_geometries([pcd, frame])
