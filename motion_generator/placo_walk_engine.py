import time
import warnings
import json
import math

import numpy as np
import placo
import os

from placo_utils.tf import tf
from scipy.spatial.transform import Rotation as R

warnings.filterwarnings("ignore")

DT = 0.01
REFINE = 10

import time

class PlacoWalkEngine:
    def __init__(
        self,
        asset_path: str = "",
        model_filename: str = "go_bdx.urdf",
        init_params: dict = {},
        ignore_feet_contact: bool = False,
        knee_limits: list = None,
    ) -> None:
        model_filename = os.path.join(asset_path, model_filename)
        self.asset_path = asset_path.removesuffix("/")
        self.model_filename = model_filename
        self.ignore_feet_contact = ignore_feet_contact

        robot_type = asset_path.split("/")[-1]

        # Loading the robot
        start_time = time.time()
        self.robot = placo.HumanoidRobot(model_filename)
        end_time = time.time()

        print("Execution time:", end_time - start_time, "seconds")

        self.parameters = placo.HumanoidParameters()
        if init_params is not None:
            self.load_parameters(init_params)
        else:
            defaults_filename = os.path.join(asset_path, "placo_defaults.json")
            self.load_defaults(defaults_filename)

        knee_limits = self.parameters.knee_limits
        # Creating the kinematics solver
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.enable_velocity_limits(True)
        self.robot.set_velocity_limits(self.parameters.velocity_limits)
        self.solver.enable_joint_limits(self.parameters.joint_limits)
        self.solver.dt = DT / REFINE
        if knee_limits is not None and len(knee_limits) > 0:
            self.robot.set_joint_limits("left_knee", *knee_limits)
            self.robot.set_joint_limits("right_knee", *knee_limits)

        # Creating the walk QP tasks
        self.tasks = placo.WalkTasks()
        self.tasks.trunk_mode = self.parameters.trunk_mode
        self.tasks.com_x = 0.0
        self.tasks.initialize_tasks(self.solver, self.robot)
        for axis, frame in self.parameters.left_foot_axises.items():
            self.tasks.left_foot_task.orientation().mask.set_axises(axis, frame)
        for axis, frame in self.parameters.right_foot_axises.items():
            self.tasks.right_foot_task.orientation().mask.set_axises(axis, frame)

        # self.tasks.trunk_orientation_task.configure("trunk_orientation", "soft", 1e-4)
        # tasks.left_foot_task.orientation().configure("left_foot_orientation", "soft", 1e-6)
        # tasks.right_foot_task.orientation().configure("right_foot_orientation", "soft", 1e-6)

        # # Creating a joint task to assign DoF values for upper body
        self.joints = self.parameters.joints
        joint_degrees = self.parameters.joint_angles
        joint_radians = {joint: np.deg2rad(degrees) for joint, degrees in joint_degrees.items()}
        self.joints_task = self.solver.add_joints_task()
        self.joints_task.set_joints(joint_radians)
        self.joints_task.configure("joints", "soft", 1.0)

        need_update = False
        for joint, rad in self.parameters.initial_pos.items():
            print(f"set_joint('{joint}', rad)")
            self.robot.set_joint(joint, rad)
            need_update = True
        if need_update:
            self.robot.update_kinematics()

        # Placing the robot in the initial position
        print("Placing the robot in the initial position...")
        print(f"self.parameters.walk_com_height: {self.parameters.walk_com_height}")
        self.tasks.reach_initial_pose(
            np.eye(4),
            self.parameters.feet_spacing,
            self.parameters.walk_com_height,
            self.parameters.walk_trunk_pitch,
        )
        print("Initial position reached")

        angles = self.get_angles()
        print(f"raw: {angles}")

        self.enable_bob = self.parameters.enable_bob
        if self.enable_bob:
            self.bob_joints = self.parameters.bob_joints

        joint_pairs = {}
        for full_name, angle in angles.items():
            if full_name.startswith(('left_', 'right_')):
                side, joint = full_name.split('_', 1)
                joint_pairs.setdefault(joint, {})[side] = angle

        mean_abs = {}
        for joint, sides in joint_pairs.items():
            if 'left' in sides and 'right' in sides:
                mean_abs[joint] = (abs(sides['left']) + abs(sides['right'])) / 2

        for name in angles:
            if name.startswith(('left_', 'right_')):
                _, joint = name.split('_', 1)
                if joint in mean_abs:
                    # apply the averaged magnitude with the original sign
                    angles[name] = math.copysign(mean_abs[joint], angles[name])
        self.print_keyframe("home", angles)
        self.home = self.get_keyframe(angles)
        # exit()

        # Creating the FootstepsPlanner
        self.repetitive_footsteps_planner = placo.FootstepsPlannerRepetitive(
            self.parameters
        )
        self.d_x = 0.0
        self.d_y = 0.0
        self.d_theta = 0.0
        self.nb_steps = 5
        self.repetitive_footsteps_planner.configure(
            self.d_x, self.d_y, self.d_theta, self.nb_steps
        )

        # Planning footsteps
        self.T_world_left = placo.flatten_on_floor(self.robot.get_T_world_left())
        self.T_world_right = placo.flatten_on_floor(self.robot.get_T_world_right())
        self.footsteps = self.repetitive_footsteps_planner.plan(
            placo.HumanoidRobot_Side.left, self.T_world_left, self.T_world_right
        )

        if hasattr(self.parameters, 'replan_timesteps'):
            # Old-style
            self.supports = placo.FootstepsPlanner.make_supports(
                self.footsteps, True, self.parameters.has_double_support(), True
            )
        else:
            self.supports = placo.FootstepsPlanner.make_supports(
                self.footsteps, 0.0, True, self.parameters.has_double_support(), True
            )

        # Creating the pattern generator and making an initial plan
        self.walk = placo.WalkPatternGenerator(self.robot, self.parameters)
        self.trajectory = self.walk.plan(self.supports, self.robot.com_world(), 0.0)

        self.time_since_last_right_contact = 0.0
        self.time_since_last_left_contact = 0.0
        self.start = None
        self.initial_delay = -1.0
        # self.initial_delay = 0
        self.t = self.initial_delay
        self.last_replan = 0

        # TODO remove startend_double_support_duration() when starting and ending ?
        self.period = (
            2 * self.parameters.single_support_duration
            + 2 * self.parameters.double_support_duration()
        )
        print("## period:", self.period)

    def get_keyframe(self, angles):
        """
        Build and return a dict with:
          - 'qpos': [x, y, z, w, x, y, z, joint_1, joint_2, …]
          - 'qval': [joint_1, joint_2, …]
        where qpos is the free-floating position + orientation (wxyz) 
        followed by all joint angles, and qval is just the joint angles.
        """
        # extract joint values as floats
        vals = list(angles.values())

        # 1) base transform
        T = self.robot.get_T_world_fbase()
        root_position = T[:3, 3]                        # array([x, y, z])
        quat_xyzw = R.from_matrix(T[:3, :3]).as_quat()  # [x, y, z, w]
        w, x, y, z = quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]

        # 2) assemble qpos: [pos(3), quat(wxyz), joint angles…]
        qpos = [
            root_position[0], root_position[1], root_position[2],
            w, x, y, z,
            *vals
        ]

        # 3) qval is just the joint angles
        qval = vals.copy()

        return {
            'qpos': qpos,
            'qval': qval
        }

    def print_keyframe(self, name, angles):
        vals = list(angles.values())
        chunks = [vals[i:i + 5] for i in range(0, len(vals), 5)]
        formatted_chunks = [' '.join(f"{v:.5f}" for v in chunk) for chunk in chunks]

        # 2) extract the base transform
        T = self.robot.get_T_world_fbase()
        root_position = T[:3, 3]                         # [x, y, z]
        quat_xyzw = R.from_matrix(T[:3, :3]).as_quat()   # returns [x, y, z, w]
        # reorder to [w, x, y, z]
        w, x, y, z = quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]

        # 3) format them to 5-decimal strings
        pos_str  = ' '.join(f"{p:.5f}" for p in root_position)
        quat_str = ' '.join(f"{q:.5f}" for q in (w, x, y, z))

        # 4) print out your XML
        print(f'<key name="{name}" ')
        print('  qpos="')
        print(f'    {pos_str}')
        print(f'    {quat_str}')
        print('')  # blank line before the joint rows
        for row in formatted_chunks:
            print(f'    {row}')
        print('  "')
        print('  ctrl="')
        for row in formatted_chunks:
            print(f'    {row}')
        print('  "/>')

    def load_defaults(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        params = self.parameters
        self.load_parameters(data)

    def load_parameters(self, data):
        params = self.parameters
        params.double_support_ratio = data.get('double_support_ratio', params.double_support_ratio)
        params.startend_double_support_ratio = data.get('startend_double_support_ratio', params.startend_double_support_ratio)
        params.planned_timesteps = data.get('planned_timesteps', params.planned_timesteps)
        params.walk_com_height = data.get('walk_com_height', params.walk_com_height)
        params.walk_foot_height = data.get('walk_foot_height', params.walk_foot_height)
        params.walk_trunk_pitch = np.deg2rad(data.get('walk_trunk_pitch', np.rad2deg(params.walk_trunk_pitch)))
        params.walk_foot_rise_ratio = data.get('walk_foot_rise_ratio', params.walk_foot_rise_ratio)
        params.single_support_duration = data.get('single_support_duration', params.single_support_duration)
        params.single_support_timesteps = data.get('single_support_timesteps', params.single_support_timesteps)
        params.foot_length = data.get('foot_length', params.foot_length)
        params.feet_spacing = data.get('feet_spacing', params.feet_spacing)
        params.zmp_margin = data.get('zmp_margin', params.zmp_margin)
        params.foot_zmp_target_x = data.get('foot_zmp_target_x', params.foot_zmp_target_x)
        params.foot_zmp_target_y = data.get('foot_zmp_target_y', params.foot_zmp_target_y)
        params.walk_max_dtheta = data.get('walk_max_dtheta', params.walk_max_dtheta)
        params.walk_max_dy = data.get('walk_max_dy', params.walk_max_dy)
        params.walk_max_dx_forward = data.get('walk_max_dx_forward', params.walk_max_dx_forward)
        params.walk_max_dx_backward = data.get('walk_max_dx_backward', params.walk_max_dx_backward)
        params.joints = data.get('joints', [])
        params.joint_angles = data.get('joint_angles', [])
        params.trunk_mode = data.get('trunk_mode', False)
        params.velocity_limits = data.get('velocity_limits', 12.0)
        params.knee_limits = data.get('knee_limits', None)
        params.joint_limits = data.get('joint_limits', False)
        params.left_foot_axises = data.get('left_foot_axises', {})
        params.right_foot_axises = data.get('right_foot_axises', {})
        params.enable_bob = data.get('enable_bob', False)
        params.initial_pos = data.get('initial_pos', {})
        if params.enable_bob:
            params.bob_toe_axis = data.get('bob_toe_axis', 0)
            params.bob_joints = data.get('bob_joints', {})

    def get_angles(self):
        angles = {joint: self.robot.get_joint(joint) for joint in self.joints}
        return angles

    def reset(self):
        self.t = 0
        self.start = None
        self.last_replan = 0
        self.time_since_last_right_contact = 0.0
        self.time_since_last_left_contact = 0.0

        self.tasks.reach_initial_pose(
            np.eye(4),
            self.parameters.feet_spacing,
            self.parameters.walk_com_height,
            self.parameters.walk_trunk_pitch,
        )

        # Planning footsteps
        self.T_world_left = placo.flatten_on_floor(self.robot.get_T_world_left())
        self.T_world_right = placo.flatten_on_floor(self.robot.get_T_world_right())
        self.footsteps = self.repetitive_footsteps_planner.plan(
            placo.HumanoidRobot_Side.left, self.T_world_left, self.T_world_right
        )

        if hasattr(self.parameters, 'replan_timesteps'):
            # Old-style
            self.supports = placo.FootstepsPlanner.make_supports(
                self.footsteps, True, self.parameters.has_double_support(), True
            )
        else:
            self.supports = placo.FootstepsPlanner.make_supports(
                self.footsteps, 0.0, True, self.parameters.has_double_support(), True
            )
        self.trajectory = self.walk.plan(self.supports, self.robot.com_world(), 0.0)

    def set_traj(self, d_x, d_y, d_theta):
        self.d_x = d_x
        self.d_y = d_y
        self.d_theta = d_theta
        self.repetitive_footsteps_planner.configure(
            self.d_x, self.d_y, self.d_theta, self.nb_steps
        )

    def get_footsteps_in_world(self):
        footsteps = self.trajectory.get_supports()
        footsteps_in_world = []
        for footstep in footsteps:
            if not footstep.is_both():
                footsteps_in_world.append(footstep.frame())

        for i in range(len(footsteps_in_world)):
            footsteps_in_world[i][:3, 3][1] += self.parameters.feet_spacing / 2

        return footsteps_in_world

    def get_footsteps_in_robot_frame(self):
        T_world_fbase = self.robot.get_T_world_fbase()

        footsteps = self.trajectory.get_supports()
        footsteps_in_robot_frame = []
        for footstep in footsteps:
            if not footstep.is_both():
                T_world_footstepFrame = footstep.frame().copy()
                T_fbase_footstepFrame = (
                    np.linalg.inv(T_world_fbase) @ T_world_footstepFrame
                )
                T_fbase_footstepFrame = placo.flatten_on_floor(T_fbase_footstepFrame)
                T_fbase_footstepFrame[:3, 3][2] = -T_world_fbase[:3, 3][2]

                footsteps_in_robot_frame.append(T_fbase_footstepFrame)

        return footsteps_in_robot_frame

    def get_current_support_phase(self):
        if self.trajectory.support_is_both(self.t):
            return [1, 1]
        elif str(self.trajectory.support_side(self.t)) == "left":
            return [1, 0]
        elif str(self.trajectory.support_side(self.t)) == "right":
            return [0, 1]
        else:
            raise AssertionError(f"Invalid phase: {self.trajectory.support_side(self.t)}")

    def tick(self, dt, left_contact=True, right_contact=True):
        if self.start is None:
            self.start = time.time()

        if not self.ignore_feet_contact:
            if left_contact:
                self.time_since_last_left_contact = 0.0
            if right_contact:
                self.time_since_last_right_contact = 0.0

        falling = not self.ignore_feet_contact and (
            self.time_since_last_left_contact > self.parameters.single_support_duration
            or self.time_since_last_right_contact
            > self.parameters.single_support_duration
        )

        if self.enable_bob:
            T_world_fbase = self.robot.get_T_world_fbase()
            T_world_leftFoot = self.robot.get_T_world_left()
            T_world_rightFoot = self.robot.get_T_world_right()
            T_body_leftFoot = (
                T_world_leftFoot  # np.linalg.inv(T_world_fbase) @ T_world_leftFoot
            )
            T_body_rightFoot = (
                T_world_rightFoot  # np.linalg.inv(T_world_fbase) @ T_world_rightFoot
            )
            T_body_leftFoot = np.linalg.inv(T_world_fbase) @ T_world_leftFoot
            T_body_rightFoot = np.linalg.inv(T_world_fbase) @ T_world_rightFoot

            left_toe_pos = list(T_body_leftFoot[:3, 3])
            right_toe_pos = list(T_body_rightFoot[:3, 3])

            toe_axis = self.parameters.bob_toe_axis
            toe_distance = np.abs(left_toe_pos[toe_axis] - right_toe_pos[toe_axis])
            joint_cmds = {
                name: param["center"] - param["range"] * toe_distance
                for name, param in self.parameters.bob_joints.items()
            }
            self.joints_task.set_joints(joint_cmds)

        for k in range(REFINE):
            # Updating the QP tasks from planned trajectory
            if not falling:
                self.tasks.update_tasks_from_trajectory(
                    self.trajectory, self.t - dt + k * dt / REFINE
                )

            # Update trunk pitch
            # print(f"pitch: {0.01 * np.sin(self.t - dt)}")
            # self.tasks.trunk_orientation_task.R_world_frame = \
            #     self.tasks.trunk_orientation_task.R_world_frame @ tf.rotation_matrix(0.2 * np.sin(self.t - dt), [0., 1., 0.])[:3, :3]
            self.robot.update_kinematics()
            _ = self.solver.solve(True)

        # If enough time elapsed and we can replan, do the replanning

        replan_timesteps = getattr(self.parameters, 'replan_timesteps', 10)
        if (
            self.t - self.last_replan
            > replan_timesteps * self.parameters.dt()
            and self.walk.can_replan_supports(self.trajectory, self.t)
        ):
            # Replanning footsteps from current trajectory
            if hasattr(self.parameters, 'replan_timesteps'):
                # Old-style
                self.supports = self.walk.replan_supports(
                    self.repetitive_footsteps_planner, self.trajectory, self.t
                )
            else:
                self.supports = self.walk.replan_supports(
                    self.repetitive_footsteps_planner, self.trajectory, self.t, self.last_replan
                )

            self.last_replan = self.t

            # Replanning CoM trajectory, yielding a new trajectory we can switch to
            self.trajectory = self.walk.replan(self.supports, self.trajectory, self.t)

        self.time_since_last_left_contact += dt
        self.time_since_last_right_contact += dt
        self.t += dt

        # while time.time() < self.start_t + self.t:
        #     time.sleep(1e-3)
