import numpy as np

# ── Geometry constants ────────────────────────────────────────────────────────
RADIUS     = 0.23   # distance from palm centre to finger body origin
CAPSULE_R  = 0.02   # capsule cross-section radius
BOX_HALF   = 0.10   # box half-size (box geom size=".1 .1 .1")
HALF_LEN   = 0.09   # half the finger capsule length (total = 0.18 m)

# Finger travel: start at RADIUS, stop when capsule surface touches box surface
MAX_TRAVEL = 0.15 # 0.23 - 0.12 = 0.11 m

# Finger Z offset below the palm so horizontal fingers sit at box mid-height.
# Palm at z=0.20 (top of box) + FINGER_Z=-0.10 → fingers at z=0.10 (box centre).
FINGER_Z = -0.10

# Contact solver — stiff enough to prevent capsule tunnelling into the box
SOLIMP = "0.99 0.999 0.0001"
SOLREF = "0.004 1"

# Elastic finger spring
STIFFNESS = 0   # N/m  — restores finger to rest position when released
DAMPING   =  0.2   # N·s/m


def gripper_geometry(height: float, angle: float = 0.0) -> tuple[list, np.ndarray]:
    finger_angles: np.ndarray = np.radians(np.array([0, 120, 240]) + angle)
    finger_positions = [
        (RADIUS * np.cos(a), RADIUS * np.sin(a), FINGER_Z)
        for a in finger_angles
    ]
    return finger_positions, finger_angles


def get_environment_xml(
    gripper_height: float,
    mesh_path: str,
    gripper_angle: float = 0.0,
    friction: float = 0.5,
) -> str:
    finger_positions, finger_angles = gripper_geometry(
        height=gripper_height,
        angle=gripper_angle,
    )

    def fromto(i: int) -> str:
        # Capsule oriented along the tangent to the radial direction so the
        # flat side (not the hemispherical tip) faces the box.
        # Radial outward: (cos a, sin a)  →  tangent: (-sin a, cos a)
        tx = -np.sin(finger_angles[i])
        ty =  np.cos(finger_angles[i])
        x1, y1 = -tx * HALF_LEN, -ty * HALF_LEN
        x2, y2 =  tx * HALF_LEN,  ty * HALF_LEN
        return f"{x1:.4f} {y1:.4f} 0   {x2:.4f} {y2:.4f} 0"

    def inward_axis(i: int) -> str:
        # Squeeze joint slides inward → negate the outward radial direction
        return f"{-np.cos(finger_angles[i]):.4f} {-np.sin(finger_angles[i]):.4f} 0"

    def finger_xml(idx: int) -> str:
        i   = idx - 1
        px  = finger_positions[i][0]
        py  = finger_positions[i][1]
        pz  = finger_positions[i][2]
        return f"""\
            <body name="finger{idx}" pos="{px:.4f} {py:.4f} {pz:.4f}">
                <joint name="f{idx}_squeeze" type="slide" limited="true"
                       range="0 {MAX_TRAVEL:.4f}"
                       axis="{inward_axis(i)}"
                       stiffness="{STIFFNESS}" damping="{DAMPING}"/>
                <geom name="finger{idx}_geom" type="capsule" size="{CAPSULE_R}"
                      fromto="{fromto(i)}"
                      rgba=".9 .7 .5 1"
                      solimp="{SOLIMP}" solref="{SOLREF}"
                      friction="{friction} {friction} {friction}"/>
            </body>"""

    return f"""
<mujoco>
    <option gravity="0 0 -9.81" timestep="0.002"/>
    <visual>
        <global offwidth="1280" offheight="960"/>
    </visual>
    <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
                 rgb2=".2 .3 .4" width="300" height="300" mark="edge" markrgb=".2 .3 .4"/>
        <material name="grid" texture="grid" texrepeat="2 2" texuniform="true"
                  reflectance=".2"/>
    </asset>
    <asset>
        <!-- existing texture/material ... -->
        <mesh name="raspberry" file="{mesh_path}" scale="1 1 1"/>
    </asset>
    <worldbody>
        <light pos="0 0 1.5" mode="trackcom"/>
        <geom name="ground" type="plane" pos="0 0 0" size="2 2 .1" material="grid"
              solimp="{SOLIMP}" solref="{SOLREF}"/>

        <!-- Box: half-size=0.1 → centre at z=0.1, top face at z=0.2 -->
        <body name="box" pos="0 0 0">
            <freejoint/>
            <geom name="raspberry_geom" type="mesh" mesh="raspberry"
                rgba="1 0 0 1"
                solimp="{SOLIMP}" solref="{SOLREF}"
                friction="{friction} {friction} {friction}"/>
            <camera name="track" pos="0 -1.5 0.8" xyaxes="1 0 0 0 1 2" mode="track"/>
        </body>

        <!-- Palm at gripper_height (~0.20). Fingers offset FINGER_Z=-0.10
             below palm so they sit at box mid-height z=0.10. -->
        <body name="palm" pos="0 0 {gripper_height:.4f}">
            <joint name="palm_lift" type="slide" axis="0 0 1"/>
            <geom name="palm_geom" type="cylinder" size="0.05 0.01" rgba=".5 .5 .5 1"
                  solimp="{SOLIMP}" solref="{SOLREF}"/>
{finger_xml(1)}
{finger_xml(2)}
{finger_xml(3)}
        </body>
    </worldbody>

    <actuator>
        <motor name="palm_lift"  joint="palm_lift"  gear="1" ctrlrange="-200 200"/>
        <motor name="f1_squeeze" joint="f1_squeeze" gear="1" ctrlrange="0 100"/>
        <motor name="f2_squeeze" joint="f2_squeeze" gear="1" ctrlrange="0 100"/>
        <motor name="f3_squeeze" joint="f3_squeeze" gear="1" ctrlrange="0 100"/>
    </actuator>
</mujoco>
"""