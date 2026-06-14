import open3d
import numpy as np
import cv2

from src.geometry import Transformation3D, CameraParameters
from src.reconstruction import ImageFrame, StereoFrame
from .irenderer import IRenderer
from .trajectory import get_fibonacci_hemisphere_trajectory
from src.utils import create_dir, load_mesh


def render(
    mesh: open3d.geometry.TriangleMesh,
    extrinsic: Transformation3D,
    camera_parameters: CameraParameters
) -> np.ndarray:
    vis = open3d.visualization.Visualizer()
    vis.create_window(visible=False, width=camera_parameters.resolution_x, height=camera_parameters.resolution_y)
    vis.add_geometry(mesh)

    # Render options
    opt = vis.get_render_option()
    opt.background_color = np.array([0.0, 0.0, 0.0])
    opt.mesh_show_back_face = True
    opt.light_on = True

    # Set camera from intrinsic + extrinsic
    ctr = vis.get_view_control()
    cam = open3d.camera.PinholeCameraParameters()

    cam.intrinsic = open3d.camera.PinholeCameraIntrinsic(
        width=camera_parameters.resolution_x,
        height=camera_parameters.resolution_y,
        fx=camera_parameters.fx,
        fy=camera_parameters.fy,
        cx=camera_parameters.cx,
        cy=camera_parameters.cy
    )

    cam.extrinsic = extrinsic.matrix
    ctr.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)

    vis.poll_events()
    vis.update_renderer()

    # Capture RGB
    rgb = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    rgb = (rgb * 255).astype(np.uint8)

    vis.destroy_window()

    return rgb


class Open3DRenderer(IRenderer):
    
    def run(
        self,
        obj_file: str,
        stereo: bool = False,
        frame_count: int = 8,
        output_dir: str = "render",
        verbose: bool = False
    ) -> list[ImageFrame]:
        mesh: open3d.geometry.TriangleMesh = load_mesh(obj_file)

        if verbose:
            frame = open3d.geometry.TriangleMesh.create_coordinate_frame(
                size=1.0, origin=[0, 0, 0]
            )
            open3d.visualization.draw_geometries([mesh, frame])

        create_dir(output_dir)

        trajectory: list[Transformation3D] = get_fibonacci_hemisphere_trajectory(
            steps=frame_count,
            radius=0.2
        )
        image_frames: list[ImageFrame] = []
        for step, camera_orientation in enumerate(trajectory):
            if verbose:
                world_frame  = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
                camera_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
                camera_frame.transform(camera_orientation.inverse().matrix)

                camera_marker = open3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                camera_marker.translate([camera_orientation.inverse().x, camera_orientation.inverse().y, camera_orientation.inverse().z])
                camera_marker.paint_uniform_color([1.0, 0.0, 0.0])

                open3d.visualization.draw_geometries([mesh, world_frame, camera_frame, camera_marker])

            img_rendered: np.ndarray = render(
                mesh=mesh,
                extrinsic=camera_orientation,
                camera_parameters=self.camera_parameters
            )

            img_path: str = f"{output_dir}/frame_{step:04d}.png"
            cv2.imwrite(img_path, cv2.cvtColor(img_rendered, cv2.COLOR_RGB2BGR))

            image_frames.append(
                ImageFrame(
                    path=img_path,
                    t_cam_to_world=camera_orientation.inverse()
                )
            )

        return image_frames
