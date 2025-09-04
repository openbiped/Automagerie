#!/usr/bin/env python3

import mujoco
import numpy as np
import pickle
import math
import time
import argparse
import os, sys
from pathlib import Path
import mujoco.viewer
from mujoco_scenes import mjcf
from poly_reference_motion_numpy import PolyReferenceMotion

import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import warnings
# Ignore pygame warning
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*pkg_resources is deprecated as an API.*"
)

SCRIPT_DIR = Path(__file__).resolve().parent
AUTOMAGERIE_DIR = SCRIPT_DIR.parent
ROBOTS_DIR = AUTOMAGERIE_DIR / "robots"
SCENES_DIR = mjcf.get_template_dir()

available_scenes = []
if os.path.isdir(SCENES_DIR):
    for name in os.listdir(SCENES_DIR):
        if (name.endswith(".xml")
            and os.path.isfile(os.path.join(SCENES_DIR, name))
        ):
            scene_name = Path(name).stem
            available_scenes.append(scene_name)
if len(available_scenes) == 0:
    print(f"No scenes found in: {SCENES_DIR}")
    sys.exit(1)

# Parse command-line arguments.
parser = argparse.ArgumentParser(description="Reference Motion Viewer")
parser.add_argument(
    "--reference_data",
    type=str,
    required=True,
    help="Path to the polynomial reference motion file.",
)
parser.add_argument(
    "-joystick", action="store_true", default=False, help="Use joystick control"
)
# Command parameters: dx, dy, dtheta
parser.add_argument(
    "--command",
    nargs=3,
    type=float,
    default=[1.0, 0, 0],
    help="Reference motion command as three floats: dx, dy, dtheta.",
)
parser.add_argument(
    "--scene",
    type=str,
    choices=available_scenes,
    default=None,
)
parser.add_argument(
    "--camera-sweep",
    action="store_true",
    default=False,
    help="Enable continuous camera sweep (360° rotation)"
)
parser.add_argument('--show-gui', action='store_true', help='Show the viewer gui')
args = parser.parse_args()

# -------------------------------------------------------------------
# Load Reference Motion & Set Default Pose
# -------------------------------------------------------------------
PRM = PolyReferenceMotion(args.reference_data)

model_path = f"{ROBOTS_DIR}/{PRM.robot}/scene.xml"
command = args.command

# -------------------------------------------------------------------
# Optional Joystick Initialization
# -------------------------------------------------------------------
joystick1 = None
joystick2 = None
if args.joystick:
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() > 0:
        joystick1 = pygame.joystick.Joystick(0)
        joystick1.init()
        command = [0.0, 0.0, 0.0]
        print("Joystick initialized:", joystick1.get_name())
        if pygame.joystick.get_count() > 1:
            joystick2 = pygame.joystick.Joystick(1)
            joystick2.init()
            print("Joystick 2 (theta) initialized:", joystick2.get_name())
        else:
            print("Only one joystick detected; theta via second joystick will be disabled.")
    else:
        print("No joystick found!")

decimation = 10  # update reference motion every 10 iterations
print(PRM.dx_range)
print(PRM.dy_range)
print(PRM.dtheta_range)

COMMANDS_RANGE_X = PRM.dx_range
COMMANDS_RANGE_Y = PRM.dy_range
COMMANDS_RANGE_THETA = PRM.dtheta_range

# -------------------------------------------------------------------
# Load Model & Initialize State
# -------------------------------------------------------------------

base_model = model = mujoco.MjModel.from_xml_path(str(model_path))
if args.scene is not None:
    model = mjcf.load_mjmodel(f"{ROBOTS_DIR}/{PRM.robot}/{PRM.robot}.xml", args.scene)

data = mujoco.MjData(model)
model.opt.timestep = 1.0 / PRM.fps / decimation

# Step the simulation once to initialize.
mujoco.mj_step(model, data)

# Build mapping from reference joints -> model qpos indices
joint_mapping = {}
for idx, joint_name in enumerate(PRM.joints):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid != -1:
        adr = model.jnt_qposadr[jid]
        joint_mapping[idx] = adr
    else:
        print(f"Warning: Reference joint '{joint_name}' not found in model; ignoring.")
left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")

# -------------------------------------------------------------------
# Helper: Euler to Quaternion Conversion
# -------------------------------------------------------------------
def euler_to_quat(roll, pitch, yaw):
    """Convert Euler angles (roll, pitch, yaw) to quaternion [w, x, y, z]."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z])

# Get the "home" keyframe to use as a default pose.
home_frame = base_model.keyframe("home")
default_qpos = np.array(home_frame.qpos)
default_ctrl = np.array(home_frame.ctrl)
#default_qpos[2] += 0.2  # Increase the base height by 0.2 meters

# Set initial state.
data.qpos[:] = default_qpos.copy()
data.ctrl[:] = default_ctrl.copy()

cur_lin_vel_x = 0

# -------------------------------------------------------------------
# Input Callback Functions
# -------------------------------------------------------------------
def key_callback(keycode):
    global cur_lin_vel_x
    if joystick1 is not None:
        return

    lin_vel_x = 0
    lin_vel_y = 0
    ang_vel = 0
    if keycode == 265:  # arrow up
        lin_vel_x = COMMANDS_RANGE_X[1]
    if keycode == 264:  # arrow down
        lin_vel_x = COMMANDS_RANGE_X[0]
    if keycode == 263:  # arrow left
        lin_vel_y = COMMANDS_RANGE_Y[1]
    if keycode == 262:  # arrow right
        lin_vel_y = COMMANDS_RANGE_Y[0]
    if keycode == 81:  # q
        ang_vel = COMMANDS_RANGE_THETA[1]
    if keycode == 69:  # e
        ang_vel = COMMANDS_RANGE_THETA[0]
    if keycode == 331:  # e
        if cur_lin_vel_x > COMMANDS_RANGE_X[0]:
            cur_lin_vel_x -= 0.1
        lin_vel_x = cur_lin_vel_x
    if keycode == 332:  # e
        if cur_lin_vel_x < COMMANDS_RANGE_X[1]:
            cur_lin_vel_x += 0.1
        lin_vel_x = cur_lin_vel_x
    if keycode == 61:
        model.opt.timestep = model.opt.timestep / 2
    elif keycode == 45:
        model.opt.timestep = model.opt.timestep * 2
    else:
        command[0] = lin_vel_x
        command[1] = lin_vel_y
        command[2] = ang_vel


def handle_joystick():
    if joystick1 is None:
        return

    pygame.event.pump()
    joy_y = joystick1.get_axis(1)
    joy_x = joystick1.get_axis(0)
    joy_z = joystick2.get_axis(0) if joystick2 is not None else 0
    lin_vel_x = (-joy_y) * (COMMANDS_RANGE_X[1] if joy_y < 0 else abs(COMMANDS_RANGE_X[0]))
    lin_vel_y = -joy_x * COMMANDS_RANGE_Y[1]
    lin_vel_z = -joy_z * COMMANDS_RANGE_THETA[1]
    command[0] = lin_vel_x
    command[1] = lin_vel_y
    command[2] = lin_vel_z
    print(f"command: {command}")

CAMERA_ROTATION_SPEED = 10.0  # degrees per second

# -------------------------------------------------------------------
# Main Simulation & Viewer Loop
# -------------------------------------------------------------------
with mujoco.viewer.launch_passive(
        model, data,
        show_left_ui=args.show_gui,
        show_right_ui=args.show_gui,
        key_callback=key_callback) as viewer:    
    step = 0
    dt = model.opt.timestep
    counter = 0
    new_qpos = default_qpos.copy()

    # One–time pitch alignment flag and storage.
    pitch_aligned = False
    pitch_quat = None

    viewer.cam.distance = 1.3
    viewer.cam.elevation = -20

    while viewer.is_running():
        step_start = time.time()
        handle_joystick()
        counter += 1

        # Reset base pose for each iteration.
        new_qpos[:7] = default_qpos[:7].copy()

        # Update reference motion at decimation frequency.
        if counter % decimation == 0:
            new_qpos = default_qpos.copy()
            if not all(val == 0.0 for val in command):
                imitation_i = step % PRM.nb_steps_in_period
                ref_motion = np.array(
                    PRM.get_reference_motion(command[0], command[1], command[2], imitation_i)
                )
                # first slice is root+quat, skip those positions
                ref_joint_pos = ref_motion[: PRM.nb_joints]
                # apply only matching joints
                for ref_idx, qpos_idx in joint_mapping.items():
                    if ref_idx < len(ref_joint_pos):
                        new_qpos[qpos_idx] = ref_joint_pos[ref_idx]
                    else:
                        print(f"Warning: ref index {ref_idx} out of bounds.")
                step += 1
            else:
                step = 0

            # Apply updated qpos and update kinematics.
            data.qpos[:] = new_qpos
            mujoco.mj_forward(model, data)

            # One–time pitch alignment: capture the foot’s rotation matrix
            if not pitch_aligned:
                foot_mat = data.xmat[left_foot_id].reshape(3, 3)
                # For a foot with forward along +X and up along +Z:
                pitch_angle = math.atan2(foot_mat[0, 2], foot_mat[2, 2])
                final_pitch = -pitch_angle  # cancel the pitch
                pitch_quat = euler_to_quat(0.0, final_pitch, 0.0)
                pitch_aligned = True
                # Uncomment to print alignment details:
                # print(f"Pitch alignment: foot_pitch={pitch_angle:.3f}, final_pitch={final_pitch:.3f}")

        # Apply the one–time computed pitch quaternion.
        if pitch_aligned and pitch_quat is not None:
            new_qpos[3:7] = pitch_quat

        data.qpos[:] = new_qpos

        if args.camera_sweep:
            viewer.cam.azimuth += CAMERA_ROTATION_SPEED * dt

        # Step simulation and update viewer.
        mujoco.mj_step(model, data)
        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

