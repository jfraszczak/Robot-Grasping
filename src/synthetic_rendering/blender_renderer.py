import os

import bpy
from pydantic import BaseModel
import numpy as np

from src.geometry import Transformation3D, CameraParameters
from src.reconstruction import ImageFrame
from .irenderer import IRenderer
from .trajectory import get_fibonacci_hemisphere_trajectory
from src.utils import create_dir


class Coordinates(BaseModel):
    x: float
    y: float
    z: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


def clear_scene() -> None:
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
    bpy.ops.object.delete()


def import_object(filepath: str, center: bool = True, normalize: bool = False) -> bpy.types.Object:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"OBJ file not found: {filepath}")

    bpy.ops.wm.obj_import(filepath=filepath)

    if not bpy.context.selected_objects:
        raise RuntimeError("OBJ import produced no objects.")

    obj: bpy.types.Object = bpy.context.active_object

    if center:
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        obj.location = (0.0, 0.0, 0.0)

    if normalize:
        max_dim = max(obj.dimensions)
        if max_dim > 0:
            scale_factor = 1.0 / max_dim
            obj.scale = (scale_factor, scale_factor, scale_factor)
            bpy.ops.object.transform_apply(scale=True)

    return obj


def add_camera(location: Coordinates) -> bpy.types.Object:
    if bpy.context.scene.camera:
        return bpy.context.scene.camera

    bpy.ops.object.camera_add(location=location.as_tuple())
    camera: bpy.types.Object = bpy.context.active_object
    bpy.context.scene.camera = camera
    return camera


def set_camera_parameters(
    camera: bpy.types.Object,
    camera_parameters: CameraParameters
) -> bpy.types.Object:
    if abs(camera_parameters.fx - camera_parameters.fy) > 1e-3:
        raise ValueError("Blender does not support fx != fy")

    camera.data.lens = camera_parameters.fx * camera.data.sensor_width / camera_parameters.resolution_x
    camera.data.shift_x = (camera_parameters.cx - camera_parameters.resolution_x / 2) / camera_parameters.resolution_x
    camera.data.shift_y = -(camera_parameters.cy - camera_parameters.resolution_y / 2) / camera_parameters.resolution_y

    return camera


def add_light(location: Coordinates, light_type: str = 'POINT') -> None:
    bpy.ops.object.light_add(type=light_type, location=location.as_tuple())
    light_obj = bpy.context.active_object
    light_obj.data.energy = 200.0


def add_ambient_light(strength: float = 1.0) -> None:
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = (1, 1, 1, 1)
        bg_node.inputs["Strength"].default_value = strength


def rotate_and_render(
    obj_file: str,
    output_dir: str,
    camera_parameters: CameraParameters,
    steps: int = 24,
    radius: float = 3.0,
    verbose: bool = False
) -> list[ImageFrame]:
    clear_scene()
    import_object(filepath=obj_file, normalize=False)
    camera: bpy.types.Object = add_camera(location=Coordinates(x=0.0, y=-radius, z=0.0))
    camera = set_camera_parameters(
        camera=camera,
        camera_parameters=camera_parameters
    )
    add_ambient_light()
    create_dir(output_dir)
    
    trajectory: list[Transformation3D] = get_fibonacci_hemisphere_trajectory(
        steps=steps,
        radius=radius
    )

    scene = bpy.context.scene
    scene.render.resolution_x = camera_parameters.resolution_x
    scene.render.resolution_y = camera_parameters.resolution_y
    scene.render.image_settings.file_format = 'JPEG'
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.film_transparent = True

    original_rotation = camera.rotation_euler.copy()
    original_location = camera.location.copy()

    image_frames: list[ImageFrame] = []
    for step, camera_orientation in enumerate(trajectory):
        camera_orientation_blender: Transformation3D = camera_orientation.inverse()
        camera_orientation_blender = camera_orientation_blender @ Transformation3D(matrix=np.diag([1, -1, -1, 1]))

        camera.location = (camera_orientation_blender.x, camera_orientation_blender.y, camera_orientation_blender.z)
        camera.rotation_euler[0] = camera_orientation_blender.roll
        camera.rotation_euler[1] = camera_orientation_blender.pitch
        camera.rotation_euler[2] = camera_orientation_blender.yaw

        print(f"[step {step:03d}] loc={camera.location[:]} rot={list(camera.rotation_euler)}")

        scene.render.filepath = os.path.join(output_dir, f"frame_{step:04d}")
        bpy.ops.render.render(write_still=True)

        image_frames.append(
            ImageFrame(
                path=os.path.join(output_dir, f"frame_{step:04d}.jpg"),
                t_cam_to_world=camera_orientation.inverse()
            )
        )

    camera.rotation_euler = original_rotation
    camera.location = original_location

    return image_frames


class BlenderRenderer(IRenderer):

    def run(
        self,
        obj_file: str,
        frame_count: int = 8,
        output_dir: str = "render",
        verbose: bool = False
    ) -> list[ImageFrame]:
        return rotate_and_render(
            obj_file=obj_file,
            output_dir=output_dir,
            camera_parameters=self.camera_parameters,
            steps=frame_count,
            radius=0.2,
            verbose=verbose
        )
