import tempfile
from typing import Generator, Callable, Iterator
from functools import wraps
import copy

import open3d
import numpy as np
import mujoco
import cv2

from .igrasp_estimator import IGraspEstimator
from .models import PhysicalCoefficients, GrasperState, GraspTrajectory
from .utils import get_bounding_sphere


class Constants:
    object_to_grasp_name: str = "object_to_grasp"
    object_world_pose: np.ndarray = np.array([0.4, 0.0, 0.0])
    q_init: np.ndarray = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
    fingers_offset: float = 0.04
    lift: np.ndarray = np.array([0, 0, 0.10])


def is_grasped(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    object_id: int = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, Constants.object_to_grasp_name)
    left_id: int = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_id: int = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

    left_contact: bool = False
    right_contact: bool = False

    for i in range(data.ncon):
        con = data.contact[i]
        b1 = model.geom_bodyid[con.geom1]
        b2 = model.geom_bodyid[con.geom2]
        bodies = {b1, b2}

        if object_id in bodies:
            if left_id  in bodies: left_contact  = True
            if right_id in bodies: right_contact = True

    return left_contact and right_contact


def get_fingertips(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float]:
    left_id: int = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_id: int = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

    left_tips, right_tips = [], []
    for i in range(model.ngeom):
        body_id: int = model.geom_bodyid[i]
        if body_id == left_id and model.geom_size[i][0] < 0.01:
            left_tips.append(data.geom_xpos[i].copy())
        elif body_id == right_id and model.geom_size[i][0] < 0.01:
            right_tips.append(data.geom_xpos[i].copy())

    left_center: float = np.mean(left_tips, axis=0)
    right_center: float = np.mean(right_tips, axis=0)
    return left_center, right_center


def get_fingertip_midpoint(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    left, right = get_fingertips(model, data)
    return (left + right) / 2


def compute_inverse_kinematics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target: np.ndarray,
    n_steps: int = 500
) -> np.ndarray:
    hand_id: int = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    jac_pos: np.ndarray = np.zeros((3, model.nv))
    jac_rot: np.ndarray = np.zeros((3, model.nv))

    for _ in range(n_steps):
        mujoco.mj_forward(model, data)
        
        tip_mid: float = get_fingertip_midpoint(model, data)
        error: float = target - tip_mid

        if np.linalg.norm(error) < 0.001:
            break

        mujoco.mj_jacBody(model, data, jac_pos, jac_rot, hand_id)
        
        r: np.ndarray = tip_mid - data.xpos[hand_id]
        jacobian: np.ndarray = (jac_pos + np.cross(jac_rot.T, r).T)[:, :7]
        
        lam: float = 0.01
        dq: np.ndarray = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + lam * np.eye(3), error)
        data.qpos[:7] += 0.1 * dq

        for i in range(7):
            data.qpos[i] = np.clip(data.qpos[i], model.jnt_range[i, 0], model.jnt_range[i, 1])

    mujoco.mj_forward(model, data)
    return data.qpos[:7].copy()


def compute_inverse_kinematics_fingers(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_left: float,
    target_right: float,
    n_steps: int = 500
) -> np.ndarray:
    left_id: int  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_id: int = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

    jac_pos_l: np.ndarray = np.zeros((3, model.nv))
    jac_rot_l: np.ndarray = np.zeros((3, model.nv))
    jac_pos_r: np.ndarray = np.zeros((3, model.nv))
    jac_rot_r: np.ndarray = np.zeros((3, model.nv))

    for _ in range(n_steps):
        mujoco.mj_forward(model, data)
        left_pos, right_pos = get_fingertips(model=model, data=data)

        err_left  = target_left  - left_pos
        err_right = target_right - right_pos
        error = np.concatenate([err_left, err_right])

        if np.linalg.norm(error) < 0.001:
            break

        mujoco.mj_jacBody(model, data, jac_pos_l, jac_rot_l, left_id)
        mujoco.mj_jacBody(model, data, jac_pos_r, jac_rot_r, right_id)

        r_l = left_pos  - data.xpos[left_id]
        r_r = right_pos - data.xpos[right_id]
        jac_l = (jac_pos_l + np.cross(jac_rot_l.T, r_l).T)[:, :7]  # (3, 7)
        jac_r = (jac_pos_r + np.cross(jac_rot_r.T, r_r).T)[:, :7]  # (3, 7)

        jacobian = np.vstack([jac_l, jac_r])  # (6, 7)

        lam = 0.01
        dq = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + lam * np.eye(6), error)
        data.qpos[:7] += 0.5 * dq

        for i in range(7):
            data.qpos[i] = np.clip(data.qpos[i], model.jnt_range[i, 0], model.jnt_range[i, 1])

    mujoco.mj_forward(model, data)
    return data.qpos[:7].copy()


def build_environment(model_path: str, mesh_path: str, physical_coeffs: PhysicalCoefficients) -> tuple[mujoco.MjModel, mujoco.MjData]:
    spec: mujoco.MjSpec = mujoco.MjSpec.from_file(model_path)

    vis = spec.visual
    vis.global_.offwidth = 1280
    vis.global_.offheight = 720

    mesh_asset = spec.add_mesh()
    mesh_asset.name = Constants.object_to_grasp_name
    mesh_asset.file = mesh_path

    floor = spec.worldbody.add_geom()
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [2, 2, 0.1]
    floor.pos = [0, 0, 0]
    floor.contype = 1
    floor.conaffinity = 1

    object_to_grasp = spec.worldbody.add_body()
    object_to_grasp.name = Constants.object_to_grasp_name
    object_to_grasp.pos = list(Constants.object_world_pose)
    object_to_grasp.add_freejoint()
    object_to_grasp.mass = physical_coeffs.mass

    visual = object_to_grasp.add_geom()
    visual.type = mujoco.mjtGeom.mjGEOM_MESH
    visual.meshname = Constants.object_to_grasp_name
    visual.contype = 1
    visual.conaffinity = 1
    visual.friction = [
        physical_coeffs.friction_sliding,
        physical_coeffs.friction_torsional,
        physical_coeffs.friction_rolling
    ]

    model: mujoco.MjModel = spec.compile()
    data: mujoco.MjData = mujoco.MjData(model)

    return model, data


def get_finger_targets(strawberry_center, approach_axis=np.array([0, 1, 0]), offset=0.04):
    if abs(1.0 - np.linalg.norm(approach_axis)) > 1e-3:
        raise ValueError("approach_axis can't have length differing from 1.0")

    target_left  = strawberry_center - approach_axis * offset
    target_right = strawberry_center + approach_axis * offset
    return target_left, target_right


def run_simulation(model: mujoco.MjModel, data: mujoco.MjData, trajectory: GraspTrajectory) -> Generator[mujoco.MjData, None, None]:
    steps: int = 500
    for state in trajectory.trajectory:
        for _ in range(steps):
            data.ctrl[:7]  = state.qpos
            data.ctrl[7:9] = (1 - state.grasping) * 255
            mujoco.mj_step(model, data)
            yield data


def render(fn: Callable[..., Iterator[mujoco.MjData]]) -> Callable[..., None]:
    @wraps(fn)
    def with_rendering(model: mujoco.MjModel, *args, **kwargs) -> None:
        fps: int = 50

        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.4, 0.0, 0.2]
        camera.distance  = 1.5
        camera.azimuth   = 135
        camera.elevation = -20

        opt = mujoco.MjvOption()
        render_dt: float  = 1.0 / fps
        render_every: int = max(1, int(render_dt / model.opt.timestep))

        cv2.namedWindow("MuJoCo Grasp", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("MuJoCo Grasp", 1280, 720)

        renderer: mujoco.Renderer = mujoco.Renderer(model, 720, 1280)
        for step, data in enumerate(fn(model, *args, **kwargs)):
            if step % render_every == 0:
                renderer.update_scene(data, camera, opt)
                rgb_buf = renderer.render()
                frame = cv2.cvtColor(rgb_buf, cv2.COLOR_RGB2BGR)
                cv2.imshow("MuJoCo Grasp", frame)
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    return
        cv2.destroyAllWindows()

    return with_rendering


def sample_grasps(mesh_path: str, shuffle: bool = False) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    o3d_mesh: open3d.geometry.TriangleMesh = open3d.io.read_triangle_mesh(mesh_path)
    o3d_mesh.compute_vertex_normals()
    center, radius = get_bounding_sphere(o3d_mesh)
    object_center_world: np.ndarray = Constants.object_world_pose + center

    n_samples: int = 10
    phi_list: np.ndarray = np.linspace(0, np.pi, n_samples, endpoint=False)
    theta_list: np.ndarray = np.linspace(-np.pi / 3, np.pi / 3, n_samples)
    z_list: np.ndarray = np.linspace(-radius * 0.5, radius * 0.5, 10)

    if shuffle:
        rng: Generator = np.random.default_rng(seed=42)
        rng.shuffle(phi_list)
        rng.shuffle(theta_list)
        rng.shuffle(z_list)

    for z_offset in z_list:
        grasp_center = object_center_world + np.array([0, 0, z_offset])
        for phi in phi_list:
            for theta in theta_list:
                approach_axis: np.ndarray = np.array([
                    np.cos(phi) * np.cos(theta),
                    np.sin(phi) * np.cos(theta),
                    np.sin(theta)
                ])

                left, right = get_finger_targets(
                    grasp_center,
                    approach_axis=approach_axis,
                    offset=Constants.fingers_offset
                )

                if left[2] < 0 or right[2] < 0:
                    continue

                yield left, right


def reset(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    data.qpos[:7] = Constants.q_init
    data.qpos[7:9] = 0

    body_id: int = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, Constants.object_to_grasp_name)
    jnt_id: int = model.body_jntadr[body_id]
    qpos_addr: int = model.jnt_qposadr[jnt_id]
    data.qpos[qpos_addr:qpos_addr + 3] = list(Constants.object_world_pose)
    data.qpos[qpos_addr + 3:qpos_addr + 7] = [1, 0, 0, 0]

    data.qvel[:] = 0
    data.qacc[:] = 0 

    mujoco.mj_forward(model, data)


def evaluate_grasp(
    model_path: str,
    mesh_path: str,
    physical_coeffs: PhysicalCoefficients,
    grasp_trajectory: GraspTrajectory,
    verbose: bool = False
) -> bool:
    model, data = build_environment(
        model_path=model_path,
        mesh_path=mesh_path,
        physical_coeffs=physical_coeffs
    )
    reset(model=model, data=data)
    if verbose:
        render(run_simulation)(model=model, data=data, trajectory=grasp_trajectory)
    else:
        for _ in run_simulation(model=model, data=data, trajectory=grasp_trajectory): pass
    
    return is_grasped(model=model, data=data)
    

def estimate_grasp(
    model_path: str,
    mesh_path: str,
    physical_coeffs: PhysicalCoefficients,
    verbose: bool = False
) -> GraspTrajectory | None:
    model, data = build_environment(
        model_path=model_path,
        mesh_path=mesh_path,
        physical_coeffs=physical_coeffs
    )

    for target_left, target_right in sample_grasps(mesh_path=mesh_path, shuffle=True):
        reset(model=model, data=data)
        lift_qpos: np.ndarray = compute_inverse_kinematics_fingers(
            model=model,
            data=data,
            target_left=target_left + Constants.lift,
            target_right=target_right + Constants.lift
        )

        data.qpos[:7] = lift_qpos
        data.qvel[:]  = 0
        ik_qpos: np.ndarray = compute_inverse_kinematics_fingers(
            model=model,
            data=data,
            target_left=target_left,
            target_right=target_right
        )

        trajectory: GraspTrajectory = GraspTrajectory(
            trajectory=[
                GrasperState(qpos=Constants.q_init, grasping=False),
                GrasperState(qpos=lift_qpos, grasping=False),
                GrasperState(qpos=ik_qpos, grasping=False),
                GrasperState(qpos=ik_qpos, grasping=True),
                GrasperState(qpos=lift_qpos, grasping=True),
            ]
        )

        reset(model=model, data=data)
        if verbose:
            render(run_simulation)(model=model, data=data, trajectory=trajectory)
        else:
            for _ in run_simulation(model=model, data=data, trajectory=trajectory): pass
        
        is_grasp_successful: bool = is_grasped(model=model, data=data)
        if is_grasp_successful:
            return trajectory

    return None


def offset_along_z_axis(mesh: open3d.geometry.TriangleMesh) -> open3d.geometry.TriangleMesh:
    vertices: np.asarray = np.asarray(mesh.vertices)
    min_z: float = vertices[:, 2].min()
    mesh = copy.deepcopy(mesh)
    vertices = np.asarray(mesh.vertices)
    vertices[:, 2] -= min_z
    mesh.vertices = open3d.utility.Vector3dVector(vertices)
    return mesh


class FrankaEmikaPandaGrasper(IGraspEstimator):

    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.model_path: str = model_path

    def run(
        self,
        mesh: open3d.geometry.TriangleMesh,
        physical_coeffs: PhysicalCoefficients,
        verbose: bool = False
    ) -> GraspTrajectory | None:
        mesh = offset_along_z_axis(mesh)
        tmp = tempfile.NamedTemporaryFile(suffix=".obj")
        open3d.io.write_triangle_mesh(tmp.name, mesh)
        grasp_trajectory: GraspTrajectory | None = estimate_grasp(
            model_path=self.model_path,
            mesh_path=tmp.name,
            physical_coeffs=physical_coeffs,
            verbose=verbose
        )
        tmp.close()
        return grasp_trajectory
    
    def evaluate_grasp(
        self,
        mesh: open3d.geometry.TriangleMesh,
        physical_coeffs: PhysicalCoefficients,
        grasp_trajectory: GraspTrajectory,
        verbose: bool = False
    ) -> bool:
        mesh = offset_along_z_axis(mesh)
        tmp = tempfile.NamedTemporaryFile(suffix=".obj")
        open3d.io.write_triangle_mesh(tmp.name, mesh)
        is_grasp_successful: bool = evaluate_grasp(
            model_path=self.model_path,
            mesh_path=tmp.name,
            physical_coeffs=physical_coeffs,
            grasp_trajectory=grasp_trajectory,
            verbose=verbose
        )
        tmp.close()
        return is_grasp_successful


if __name__ == "__main__":
    grasper: FrankaEmikaPandaGrasper = FrankaEmikaPandaGrasper(
        model_path="third_party/mujoco_menagerie/franka_emika_panda/panda.xml"
    )
    mesh: open3d.geometry.TriangleMesh = open3d.io.read_triangle_mesh(
        "data/ycb/012_strawberry/google_512k/textured.obj",
        enable_post_processing=True
    )
    mesh.compute_vertex_normals()
    grasper.run(
        mesh=mesh,
        physical_coeffs=PhysicalCoefficients(
            mass=0.02,
            friction_sliding=2.0,
            friction_torsional=0.05,
            friction_rolling=0.01
        ),
        verbose=True
    )
