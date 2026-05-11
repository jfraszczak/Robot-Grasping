import mujoco
import cv2
import numpy as np
from .environment import get_environment_xml


# ── Force constants ───────────────────────────────────────────────────────────
# Squeeze needs to overcome friction; lift needs to overcome gravity on palm+box.
SQUEEZE_SCALE = 10.0    # × mg
LIFT_SCALE    = 5.0   # × mg

# Simulation timing
N_FRAMES      = 300
FRAME_RATE    = 60
SQUEEZE_END   = 1.0    # seconds of pure squeeze before lifting begins


# ── Actuator helpers ──────────────────────────────────────────────────────────

def _actuator_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def apply_squeeze(force: float, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for name in ("f1_squeeze", "f2_squeeze", "f3_squeeze"):
        data.ctrl[_actuator_id(model, name)] = force


def apply_lift(force: float, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    data.ctrl[_actuator_id(model, "palm_lift")] = force


# ── Contact query ─────────────────────────────────────────────────────────────

def is_in_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body1_name: str,
    body2_name: str,
) -> bool:
    b1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body1_name)
    b2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body2_name)
    for c in data.contact[:data.ncon]:
        g1b = model.geom_bodyid[c.geom1]
        g2b = model.geom_bodyid[c.geom2]
        if (g1b == b1 and g2b == b2) or (g1b == b2 and g2b == b1):
            return True
    return False


def all_fingers_gripping(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    return all(
        is_in_contact(model, data, f"finger{i}", "box")
        for i in range(1, 4)
    )


# ── Simulation loop ───────────────────────────────────────────────────────────

def grasp_and_lift_simulation(
    f_squeeze: float,
    f_lift: float,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> list[np.ndarray]:
    options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(options)
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT]  = True

    frames: list[np.ndarray] = []

    with mujoco.Renderer(model, 960, 1280) as renderer:
        for frame_idx in range(N_FRAMES):
            target_time = frame_idx / FRAME_RATE
            while data.time < target_time:

                if data.time >= 1.0:
                    apply_squeeze(f_squeeze, model, data)
                if data.time >= 2.0:
                    apply_lift(f_lift, model, data)
                mujoco.mj_step(model, data)

            renderer.update_scene(data, "track", options)
            frames.append(renderer.render())

    return frames


def show_simulation(frames: list[np.ndarray]) -> None:
    delay_ms = max(1, 1000 // FRAME_RATE)
    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Grab and Lift", bgr)
        if cv2.waitKey(delay_ms) == 27:   # ESC to quit early
            break
    cv2.destroyAllWindows()               # called once, outside the loop


# ── Entry point ───────────────────────────────────────────────────────────────

def run_simulation(
    gripper_height: float,
    gripper_angle: float,
    show: bool = True,
) -> bool:
    xml = get_environment_xml(
        mesh_path="reconstructions/raspberry_mesh.obj",
        gripper_height=gripper_height,
        gripper_angle=gripper_angle,
        friction=1.0,
    )

    model = mujoco.MjModel.from_xml_string(xml)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    mass   = model.body_mass[box_id]
    g      = abs(model.opt.gravity[2])

    f_squeeze = mass * g * SQUEEZE_SCALE
    f_lift    = mass * g * LIFT_SCALE

    frames = grasp_and_lift_simulation(f_squeeze, f_lift, model, data)

    stable = all_fingers_gripping(model, data)

    #if show:
    show_simulation(frames)

    return stable


if __name__ == "__main__":
    # Palm should be at ~box top face (z=0.20). Fingers drop FINGER_Z=-0.10
    # below palm to reach box centre. Sweep a small range around that.
    heights = np.arange(0.2, 0.28, 0.02).tolist()
    angles  = np.arange(0, 360, 45).tolist()

    results = []
    for height in heights:
        for angle in angles:
            print(f"height={height:.2f}  angle={angle:>3}°  ... ", end="", flush=True)
            stable = run_simulation(gripper_height=height, gripper_angle=angle, show=False)
            results.append({"height": height, "angle": angle, "stable": stable})
            print("✓ stable" if stable else "✗ failed")

    print("\n--- Results ---")
    print(f"{'Height':>8} {'Angle':>8} {'Stable':>8}")
    for r in results:
        mark = "yes" if r["stable"] else "no"
        print(f"{r['height']:>8.2f} {r['angle']:>8}° {mark:>8}")

    successful = [r for r in results if r["stable"]]
    print(f"\n{len(successful)}/{len(results)} configurations succeeded")
