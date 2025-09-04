import argparse
import json
import os
import sys
import time
import webbrowser
import threading
import numpy as np
import warnings
from pathlib import Path
import xml.etree.ElementTree as ET
from placo_utils.visualization import footsteps_viz, robot_frame_viz, robot_viz
from scipy.spatial.transform import Rotation as R

from placo_walk_engine import PlacoWalkEngine
import placo

def open_browser():
    try:
        webbrowser.open_new("http://127.0.0.1:7000/static/")
    except:
        print("Failed to open the default browser. Trying Google Chrome.")
        try:
            webbrowser.get("google-chrome").open_new("http://127.0.0.1:7000/static/")
        except:
            print(
                "Failed to open Google Chrome. Make sure it's installed and accessible."
            )


class RoundingFloat(float):
    __repr__ = staticmethod(lambda x: format(x, ".5f"))

def strip_visuals_from_urdf(urdf_path: str, suffix: str = "_min") -> str:
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for parent in root.iter():
        for child in list(parent):
            tag = child.tag
            if tag == 'visual' or tag.endswith('}visual'):
                parent.remove(child)

    base, ext = os.path.splitext(urdf_path)
    out_path = f"{base}{suffix}{ext}"

    tree.write(out_path, encoding='utf-8', xml_declaration=True)
    return out_path

SCRIPT_DIR = Path(__file__).resolve().parent
AUTOMAGERIE_DIR = SCRIPT_DIR.parent
ROBOTS_DIR = AUTOMAGERIE_DIR / "robots"

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", type=str, required=True)
parser.add_argument(
    "-o", "--output_dir", type=str, default=os.path.join(SCRIPT_DIR, "recordings")
)
parser.add_argument("--dx", type=float, default=None)
parser.add_argument("--dy", type=float, default=None)
parser.add_argument("--dtheta", type=float, default=None)
parser.add_argument("--double_support_ratio", type=float, default=None)
parser.add_argument("--startend_double_support_ratio", type=float, default=None)
parser.add_argument("--planned_timesteps", type=float, default=None)
parser.add_argument("--walk_com_height", type=float, default=None)
parser.add_argument("--walk_foot_height", type=float, default=None)
parser.add_argument("--walk_trunk_pitch", type=float, default=None)
parser.add_argument("--walk_foot_rise_ratio", type=float, default=None)
parser.add_argument("--single_support_duration", type=float, default=None)
parser.add_argument("--single_support_timesteps", type=float, default=None)
parser.add_argument("--foot_length", type=float, default=None)
parser.add_argument("--feet_spacing", type=float, default=None)
parser.add_argument("--zmp_margin", type=float, default=None)
parser.add_argument("--foot_zmp_target_x", type=float, default=None)
parser.add_argument("--foot_zmp_target_y", type=float, default=None)
parser.add_argument("--walk_max_dtheta", type=float, default=None)
parser.add_argument("--walk_max_dy", type=float, default=None)
parser.add_argument("--walk_max_dx_forward", type=float, default=None)
parser.add_argument("--walk_max_dx_backward", type=float, default=None)
parser.add_argument("-l", "--length", type=int, default=10)
parser.add_argument("-m", "--meshcat_viz", action="store_true", default=False)
# new, preferred argument
type_group = parser.add_mutually_exclusive_group(required=True)
type_group.add_argument(
    "--robot",
    help="Robot type",
)
# deprecated alias: write into the same dest, but suppress it from the help
type_group.add_argument(
    "--duck",
    help=argparse.SUPPRESS,
    dest="robot",
)
parser.add_argument("--min_urdf", action="store_true", default=False)
parser.add_argument("--fps", type=float, default=50)
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--preset", type=str, default="")
parser.add_argument(
    "-s",
    "--skip_warmup",
    action="store_true",
    default=False,
    help="don't record warmup motion",
)
parser.add_argument(
    "--stand",
    action="store_true",
    default=False,
    help="hack to record a standing pose",
)
parser.add_argument(
    "--index_by_vx",
    action="store_true",
    default=None,
    help="Index by v instead of dx",
)

args = parser.parse_args()
args.hardware = True

FPS = args.fps  # 50 for mujoco playground, 30 for AWD
MESHCAT_FPS = 20
DISPLAY_MESHCAT = args.meshcat_viz

# For IsaacGymEnvs (OUTDATED)
# [root position, root orientation, joint poses (e.g. rotations)]
# [x, y, z, qw, qx, qy, qz, j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12, j13, j14]

# For AWD and amp for hardware
# [root position, root orientation, joint poses (e.g. rotations), target toe positions, linear velocity, angular velocity, joint velocities, target toe velocities]
# [x, y, z, qw, qx, qy, qz, j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12, j13, j14, l_toe_x, l_toe_y, l_toe_z, r_toe_x, r_toe_y, r_toe_z, lin_vel_x, lin_vel_y, lin_vel_z, ang_vel_x, ang_vel_y, ang_vel_z, j1_vel, j2_vel, j3_vel, j4_vel, j5_vel, j6_vel, j7_vel, j8_vel, j9_vel, j10_vel, j11_vel, j12_vel, j13_vel, j14_vel, l_toe_vel_x, l_toe_vel_y, l_toe_vel_z, r_toe_vel_x, r_toe_vel_y, r_toe_vel_z]

robot = args.robot
episode = {
    "LoopMode": "Wrap",
    "FPS": FPS,
    "FrameDuration": np.around(1 / FPS, 4),
    "EnableCycleOffsetPosition": True,
    "EnableCycleOffsetRotation": False,
    "Robot": robot,
    "Joints": [],
    "Home": [],
    "Vel_x": [],
    "Vel_y": [],
    "Yaw": [],
    "Placo": [],
    "Frame_offset": [],
    "Frame_size": [],
    "Frames": [],
    "MotionWeight": 1,
}
if args.debug:
    episode["Debug_info"] = []

asset_path = os.path.join(ROBOTS_DIR, robot)
robot_urdf = f"{robot}.urdf"
robot_min_urdf = f"{robot}_min.urdf"
if args.min_urdf:
    if not os.path.exists(os.path.join(asset_path, robot_min_urdf)):
        strip_visuals_from_urdf(os.path.join(asset_path, robot_urdf))
    robot_urdf = robot_min_urdf

default_file = Path(asset_path) / "placo_defaults.json"
preset_file = Path(args.preset) if args.preset else None
with default_file.open("r") as f:
    defaults = json.load(f)

presets = {}
if preset_file:
    if not preset_file.exists():
        alt_preset_file = Path(gait.asset_path) / "placo_presets" / f"{preset_file.stem}.json"
        if alt_preset_file.exists():
            preset_file = alt_preset_file
    if preset_file.exists():
        with preset_file.open("r") as f:
            presets = json.load(f)
    else:
        warnings.warn(
            f"Preset file not found ({preset_file}); using defaults only",
            stacklevel=2
        )

gait_parameters = {**defaults, **presets}

if args.index_by_vx == None:
    args.index_by_vx = gait_parameters.get('use_vx', False)
if args.dx == None:
    args.dx = gait_parameters["dx"]
if args.dy == None:
    args.dy = gait_parameters["dy"]
if args.dtheta == None:
    args.dtheta = gait_parameters["dtheta"]
if args.double_support_ratio is not None:
    gait_parameters['double_support_ratio'] = args.double_support_ratio
if args.startend_double_support_ratio is not None:
    gait_parameters['startend_double_support_ratio'] = args.startend_double_support_ratio
if args.planned_timesteps is not None:
    gait_parameters['planned_timesteps'] = args.planned_timesteps
if args.walk_com_height is not None:
    gait_parameters['walk_com_height'] = args.walk_com_height
if args.walk_foot_height is not None:
    gait_parameters['walk_foot_height'] = args.walk_foot_height
if args.walk_trunk_pitch is not None:
    gait_parameters['walk_trunk_pitch'] = args.walk_trunk_pitch
if args.walk_foot_rise_ratio is not None:
    gait_parameters['walk_foot_rise_ratio'] = args.walk_foot_rise_ratio
if args.single_support_duration is not None:
    gait_parameters['single_support_duration'] = args.single_support_duration
if args.single_support_timesteps is not None:
    gait_parameters['single_support_timesteps'] = args.single_support_timesteps
if args.foot_length is not None:
    gait_parameters['foot_length'] = args.foot_length
if args.feet_spacing is not None:
    gait_parameters['feet_spacing'] = args.feet_spacing
if args.zmp_margin is not None:
    gait_parameters['zmp_margin'] = args.zmp_margin
if args.foot_zmp_target_x is not None:
    gait_parameters['foot_zmp_target_x'] = args.foot_zmp_target_x
if args.foot_zmp_target_y is not None:
    gait_parameters['foot_zmp_target_y'] = args.foot_zmp_target_y
if args.walk_max_dtheta is not None:
    gait_parameters['walk_max_dtheta'] = args.walk_max_dtheta
if args.walk_max_dy is not None:
    gait_parameters['walk_max_dy'] = args.walk_max_dy
if args.walk_max_dx_forward is not None:
    gait_parameters['walk_max_dx_forward'] = args.walk_max_dx_forward
if args.walk_max_dx_backward is not None:
    gait_parameters['walk_max_dx_backward'] = args.walk_max_dx_backward

pwe = PlacoWalkEngine(asset_path, robot_urdf, gait_parameters)

first_joints_positions = list(pwe.get_angles().values())
first_T_world_fbase = pwe.robot.get_T_world_fbase()
first_T_world_leftFoot = pwe.robot.get_T_world_left()
first_T_world_rightFoot = pwe.robot.get_T_world_right()

pwe.set_traj(args.dx, args.dy, args.dtheta)
if DISPLAY_MESHCAT:
    viz = robot_viz(pwe.robot)
    threading.Timer(1, open_browser).start()
DT = 0.001
start = time.time()

last_record = 0
last_meshcat_display = 0
prev_root_position = [0, 0, 0]
prev_root_orientation_quat = None
prev_root_orientation_euler = [0, 0, 0]
prev_left_toe_pos = [0, 0, 0]
prev_right_toe_pos = [0, 0, 0]
prev_joints_positions = None
i = 0
prev_initialized = False
avg_x_lin_vel = []
avg_y_lin_vel = []
avg_yaw_vel = []
added_frame_info = False
# center_y_pos = None
# center_y_pos = -(pwe.parameters.feet_spacing / 2)
# print(f"center_y_pos: {center_y_pos}")


def compute_angular_velocity(quat, prev_quat, dt):
    # Convert quaternions to scipy Rotation objects
    if prev_quat is None:
        prev_quat = quat
    r1 = R.from_quat(quat)  # Current quaternion
    r0 = R.from_quat(prev_quat)  # Previous quaternion

    # Compute relative rotation: r_rel = r0^(-1) * r1
    r_rel = r0.inv() * r1

    # Convert relative rotation to axis-angle
    axis, angle = r_rel.as_rotvec(), np.linalg.norm(r_rel.as_rotvec())

    # Angular velocity (in radians per second)
    angular_velocity = axis * (angle / dt)

    return list(angular_velocity)


# # convert to linear and angular velocity
def steps_to_vel(step_size, period):
    return (step_size * 2) / period


while True:
    pwe.tick(DT)
    if pwe.t <= 0 + args.skip_warmup * 1:
        start = pwe.t
        last_record = pwe.t + 1 / FPS
        last_meshcat_display = pwe.t + 1 / MESHCAT_FPS
        continue

    # print(np.around(pwe.robot.get_T_world_fbase()[:3, 3], 3))

    if pwe.t - last_record >= 1 / FPS:
        if args.stand:
            T_world_fbase = first_T_world_fbase
        else:
            T_world_fbase = pwe.robot.get_T_world_fbase()
        root_position = list(T_world_fbase[:3, 3])
        # if center_y_pos is None:
        #    center_y_pos = root_position[1]

        # TODO decomment this line for big duck ?
        # root_position[1] = root_position[1] - center_y_pos
        if round(root_position[2], 5) < 0:
            print(f"BAD root_position: {root_position[2]:.5f}")
        root_orientation_quat = list(R.from_matrix(T_world_fbase[:3, :3]).as_quat())
        joints_positions = list(pwe.get_angles().values())

        if args.stand:
            joints_positions = first_joints_positions
            T_world_leftFoot = first_T_world_leftFoot
            T_world_rightFoot = first_T_world_rightFoot
        else:
            joints_positions = list(pwe.get_angles().values())
            T_world_leftFoot = pwe.robot.get_T_world_left()
            T_world_rightFoot = pwe.robot.get_T_world_right()

        # T_body_leftFoot = (
        #     T_world_leftFoot  # np.linalg.inv(T_world_fbase) @ T_world_leftFoot
        # )
        # T_body_rightFoot = (
        #     T_world_rightFoot  # np.linalg.inv(T_world_fbase) @ T_world_rightFoot
        # )

        T_body_leftFoot = np.linalg.inv(T_world_fbase) @ T_world_leftFoot
        T_body_rightFoot = np.linalg.inv(T_world_fbase) @ T_world_rightFoot

        left_toe_pos = list(T_body_leftFoot[:3, 3])
        right_toe_pos = list(T_body_rightFoot[:3, 3])

        if not prev_initialized:
            prev_root_position = root_position.copy()
            prev_root_orientation_euler = (
                R.from_quat(root_orientation_quat).as_euler("xyz").copy()
            )
            prev_left_toe_pos = left_toe_pos.copy()
            prev_right_toe_pos = right_toe_pos.copy()
            prev_joints_positions = joints_positions.copy()
            prev_initialized = True

        world_linear_vel = list(
            (np.array(root_position) - np.array(prev_root_position)) / (1 / FPS)
        )
        avg_x_lin_vel.append(world_linear_vel[0])
        avg_y_lin_vel.append(world_linear_vel[1])
        body_rot_mat = T_world_fbase[:3, :3]
        body_linear_vel = list(body_rot_mat.T @ world_linear_vel)
        # print("world linear vel", world_linear_vel)
        # print("body linear vel", body_linear_vel)

        world_angular_vel = compute_angular_velocity(
            root_orientation_quat, prev_root_orientation_quat, (1 / FPS)
        )

        # world_angular_vel = list(
        #     (
        #         R.from_quat(root_orientation_quat).as_euler("xyz")
        #         - prev_root_orientation_euler
        #     )
        #     / (1 / FPS)
        # )
        avg_yaw_vel.append(world_angular_vel[2])
        body_angular_vel = list(body_rot_mat.T @ world_angular_vel)
        # print("world angular vel", world_angular_vel)
        # print("body angular vel", body_angular_vel)

        if prev_joints_positions == None:
            prev_joints_positions = [0] * len(joints_positions)
        joints_vel = list(
            (np.array(joints_positions) - np.array(prev_joints_positions)) / (1 / FPS)
        )
        left_toe_vel = list(
            (np.array(left_toe_pos) - np.array(prev_left_toe_pos)) / (1 / FPS)
        )
        right_toe_vel = list(
            (np.array(right_toe_pos) - np.array(prev_right_toe_pos)) / (1 / FPS)
        )

        foot_contacts = pwe.get_current_support_phase()

        if prev_initialized:
            if args.hardware:
                episode["Frames"].append(
                    root_position
                    + root_orientation_quat
                    + joints_positions
                    + left_toe_pos
                    + right_toe_pos
                    + world_linear_vel
                    + world_angular_vel
                    + joints_vel
                    + left_toe_vel
                    + right_toe_vel
                    + foot_contacts
                )
            else:
                episode["Frames"].append(
                    root_position + root_orientation_quat + joints_positions
                )
            if args.debug:
                episode["Debug_info"].append(
                    {
                        "left_foot_pose": list(T_world_leftFoot.flatten()),
                        "right_foot_pose": list(T_world_rightFoot.flatten()),
                    }
                )
            if not added_frame_info:
                added_frame_info = True
                offset = 0
                offset_root_pos = offset
                offset = offset + len(root_position)
                offset_root_quat = offset
                offset = offset + len(root_orientation_quat)
                offset_joints_pos = offset
                offset = offset + len(joints_positions)
                offset_left_toe_pos = offset
                offset = offset + len(left_toe_pos)
                offset_right_toe_pos = offset
                offset = offset + len(right_toe_pos)
                offset_world_linear_vel = offset
                offset = offset + len(world_linear_vel)
                offset_world_angular_vel = offset
                offset = offset + len(world_angular_vel)
                offset_joints_vel = offset
                offset = offset + len(joints_vel)
                offset_left_toe_vel = offset
                offset = offset + len(left_toe_vel)
                offset_right_toe_vel = offset
                offset = offset + len(right_toe_vel)
                offset_foot_contacts = offset
                offset = offset + len(foot_contacts)

                episode["Joints"] = list(pwe.get_angles().keys())
                episode["Frame_offset"].append(
                    {
                        "root_pos": offset_root_pos,
                        "root_quat": offset_root_quat,
                        "joints_pos": offset_joints_pos,
                        "left_toe_pos": offset_left_toe_pos,
                        "right_toe_pos": offset_right_toe_pos,
                        "world_linear_vel": offset_world_linear_vel,
                        "world_angular_vel": offset_world_angular_vel,
                        "joints_vel": offset_joints_vel,
                        "left_toe_vel": offset_left_toe_vel,
                        "right_toe_vel": offset_right_toe_vel,
                        "foot_contacts": offset_foot_contacts,
                    }
                )
                episode["Frame_size"].append(
                    {
                        "root_pos": len(root_position),
                        "root_quat": len(root_orientation_quat),
                        "joints_pos": len(joints_positions),
                        "left_toe_pos": len(left_toe_pos),
                        "right_toe_pos": len(right_toe_pos),
                        "world_linear_vel": len(world_linear_vel),
                        "world_angular_vel": len(world_angular_vel),
                        "joints_vel": len(joints_vel),
                        "left_toe_vel": len(left_toe_vel),
                        "right_toe_vel": len(right_toe_vel),
                        "foot_contacts": len(foot_contacts),
                    }
                )
                episode["Home"] = pwe.home

        prev_root_position = root_position.copy()
        prev_root_orientation_quat = root_orientation_quat.copy()
        prev_root_orientation_euler = (
            R.from_quat(root_orientation_quat).as_euler("xyz").copy()
        )
        prev_left_toe_pos = left_toe_pos.copy()
        prev_right_toe_pos = right_toe_pos.copy()
        prev_joints_positions = joints_positions.copy()
        prev_initialized = True

        last_record = pwe.t

    if DISPLAY_MESHCAT and pwe.t - last_meshcat_display >= 1 / MESHCAT_FPS:
        last_meshcat_display = pwe.t
        viz.display(pwe.robot.state.q)
        footsteps_viz(pwe.trajectory.get_supports())
        robot_frame_viz(pwe.robot, "trunk")
        robot_frame_viz(pwe.robot, "left_foot")
        robot_frame_viz(pwe.robot, "right_foot")

    # if pwe.t - start > args.length:
    #    break
    if len(episode["Frames"]) == args.length * FPS:
        break

    i += 1

# skip first 2 seconds to get better average speed
mean_avg_x_lin_vel = np.around(np.mean(avg_x_lin_vel[240:]), 4)
mean_avg_y_lin_vel = np.around(np.mean(avg_y_lin_vel[240:]), 4)
mean_yaw_vel = np.around(np.mean(avg_yaw_vel[240:]), 4)

print("recorded", len(episode["Frames"]), "frames")
print(f"avg lin_vel_x {mean_avg_x_lin_vel} (world frame)")
print(f"avg lin_vel_y {mean_avg_y_lin_vel} (world frame)")
print(f"avg yaw {mean_yaw_vel} (world frame)")

x_vel = np.around(steps_to_vel(args.dx, pwe.period), 3)
y_vel = np.around(steps_to_vel(args.dy, pwe.period), 3)
theta_vel = np.around(steps_to_vel(args.dtheta, pwe.period), 3)

print(f"computed xvel: {x_vel}, mean xvel: {mean_avg_x_lin_vel}")
print(f"computed yvel: {y_vel}, mean yvel: {mean_avg_y_lin_vel}")
print(f"computed thetavel: {theta_vel}, mean thetavel: {mean_yaw_vel}")

episode["Vel_x"] = mean_avg_x_lin_vel
episode["Vel_y"] = mean_avg_y_lin_vel
episode["Yaw"] = mean_yaw_vel
episode["Placo"] = {
    "dx": args.dx,
    "dy": args.dy,
    "dtheta": args.dtheta,
    "duration": args.length,
    "hardware": args.hardware,
    "double_support_ratio": pwe.parameters.double_support_ratio,
    "startend_double_support_ratio": pwe.parameters.startend_double_support_ratio,
    "planned_timesteps": pwe.parameters.planned_timesteps,
    "walk_com_height": pwe.parameters.walk_com_height,
    "walk_foot_height": pwe.parameters.walk_foot_height,
    "walk_trunk_pitch": np.rad2deg(pwe.parameters.walk_trunk_pitch),
    "walk_foot_rise_ratio": pwe.parameters.walk_foot_rise_ratio,
    "single_support_duration": pwe.parameters.single_support_duration,
    "single_support_timesteps": pwe.parameters.single_support_timesteps,
    "foot_length": pwe.parameters.foot_length,
    "foot_width": pwe.parameters.foot_width,
    "feet_spacing": pwe.parameters.feet_spacing,
    "zmp_margin": pwe.parameters.zmp_margin,
    "foot_zmp_target_x": pwe.parameters.foot_zmp_target_x,
    "foot_zmp_target_y": pwe.parameters.foot_zmp_target_y,
    "walk_max_dtheta": pwe.parameters.walk_max_dtheta,
    "walk_max_dy": pwe.parameters.walk_max_dy,
    "walk_max_dx_forward": pwe.parameters.walk_max_dx_forward,
    "walk_max_dx_backward": pwe.parameters.walk_max_dx_backward,
    "avg_x_lin_vel": mean_avg_x_lin_vel,
    "avg_y_lin_vel": mean_avg_y_lin_vel,
    "avg_yaw_vel": mean_yaw_vel,
    "preset_name": args.preset.split("/")[-1].split(".")[0],
    "period": pwe.period,
}

# Only available in placo 0.9+
if hasattr(pwe.parameters, 'conic_overlap_clip'):
    episode["Placo"].update({
        "left_target": list(pwe.parameters.conic_overlap_clip(placo.HumanoidRobot_Side.left, np.array([args.dx, args.dy, args.dtheta]))),
        "right_target": list(pwe.parameters.conic_overlap_clip(placo.HumanoidRobot_Side.right, np.array([args.dx, args.dy, args.dtheta]))),
    })

if args.index_by_vx:
    x_vel = np.around(steps_to_vel(args.dx, pwe.period), 3)
    y_vel = np.around(steps_to_vel(args.dy, pwe.period), 3)
    theta_vel = np.around(steps_to_vel(args.dtheta, pwe.period), 3)

    # print(f"computed xvel: {x_vel}, mean xvel: {mean_avg_x_lin_vel}")
    # print(f"computed yvel: {y_vel}, mean yvel: {mean_avg_y_lin_vel}")
    # print(f"computed thetavel: {theta_vel}, mean thetavel: {mean_yaw_vel}")
    name = f"{args.name}_{x_vel}_{y_vel}_{theta_vel}"
else: # Default
    name = f"{args.name}_{args.dx}_{args.dy}_{args.dtheta}"


file_name = name + str(".json")
file_path = os.path.join(args.output_dir, file_name)
os.makedirs(args.output_dir, exist_ok=True)
print("DONE, saving", file_name)
with open(file_path, "w") as f:
    json.encoder.c_make_encoder = None
    json.encoder.float = RoundingFloat
    json.dump(episode, f, indent=4)
