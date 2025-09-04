import os
import re
import sys
import json
import time
import psutil
import argparse
from pathlib import Path
import subprocess
from fit_poly import fit_ref_motions
from glob import glob
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
AUTOMAGERIE_DIR = SCRIPT_DIR.parent
ROBOTS_DIR = AUTOMAGERIE_DIR / "robots"

def run_command_with_logging(cmd_log_tuple):
    cmd, log_file = cmd_log_tuple
    if log_file is None:
        print(f"cmd: {cmd}")
        result = subprocess.run(cmd)
    else:
        with open(log_file, "w") as outfile:
            result = subprocess.run(cmd, stdout=outfile, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        if log_file is not None:
            print(f"Error: {cmd}")
            with open(log_file, "r") as infile:
                print(infile.read())
        elif result.stdout:
            print("Error:\n", result.stdout)
        return False
    return True


def numeric_prefix_sort_key(item):
    total_speed, preset_name = item
    match = re.match(r"(\d+)(.*)", preset_name)
    if match:
        number_part = int(match.group(1))
        rest_part = match.group(2)
        return (number_part, rest_part)
    return (float("inf"), preset_name)

def categorize_speed(speed: float, limits: list[float], names: list[str]) -> str:
    for limit, name in zip(limits, names):
        if abs(speed) <= limit:
            return name
    return names[-1]

def main(args):
    """
    Generates random preset data, creates gait motions, filters recordings,
    and (optionally) plots anim/sim if --plot is given.
    """

    start_time = time.time()

    # ---------------------------------------------------------------
    # 1. Load parameters from auto_gait.json
    # ---------------------------------------------------------------
    # This JSON should contain keys like: slow, medium, fast, dx_max, dy_max, dtheta_max
    props_path = f"{ROBOTS_DIR}/{args.robot}/auto_gait.json"
    if not os.path.isfile(props_path):
        raise FileNotFoundError(f"Could not find props file at: {props_path}")

    with open(props_path, "r") as f:
        gait_props = json.load(f)

    required_sweep_keys = [
        "min_sweep_x", "max_sweep_x",
        "min_sweep_y", "max_sweep_y",
        "min_sweep_theta", "max_sweep_theta",
        #"sweep_xy_granularity", "sweep_theta_granularity"
    ]
    if args.sweep:
        missing_keys = [key for key in required_sweep_keys if key not in gait_props]
        if missing_keys:
            sys.exit(
                "Error: The following sweep properties are required but missing: " +
                ", ".join(missing_keys)
            )

    # Extract needed values
    slow = gait_props["slow"]
    medium = gait_props["medium"]
    fast = gait_props["fast"]
    dx_max = gait_props["dx_max"]  # e.g. [0, 0.05]
    dy_max = gait_props["dy_max"]  # e.g. [0, 0.05]
    dtheta_max = gait_props["dtheta_max"]  # e.g. [0, 0.25]
    if args.sweep:
        min_sweep_x = gait_props["min_sweep_x"]
        max_sweep_x = gait_props["max_sweep_x"]
        min_sweep_y = gait_props["min_sweep_y"]
        max_sweep_y = gait_props["max_sweep_y"]
        min_sweep_theta = gait_props["min_sweep_theta"]
        max_sweep_theta = gait_props["max_sweep_theta"]
        if "sweep_xy_granularity" in gait_props:
            sweep_xy_granularity = gait_props["sweep_xy_granularity"]
        else:
            sweep_xy_granularity = (abs(min_sweep_x) + abs(max_sweep_x)) / 5
        if "sweep_theta_granularity" in gait_props:
            sweep_theta_granularity = gait_props["sweep_theta_granularity"]
        else:
            sweep_theta_granularity = (abs(min_sweep_theta) + abs(max_sweep_theta)) / 8

    # ---------------------------------------------------------------
    # 2. Paths and directories
    # ---------------------------------------------------------------
    presets_dir = f"{ROBOTS_DIR}/{args.robot}/placo_presets"

    recording_output_dir = args.output_dir
    os.makedirs(recording_output_dir, exist_ok=True)
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)

    # ---------------------------------------------------------------
    # 3. Generate random presets (n times)
    #    "medium" and "fast" as example base speeds
    # ---------------------------------------------------------------
    preset_speeds = []
    preset_speed_name = []
    if args.slow:
        preset_speed_name += ["slow"]
        preset_speeds += [slow]
    if args.medium:
        preset_speed_name += ["medium"]
        preset_speeds += [medium]
    if args.fast:
        preset_speed_name += ["fast"]
        preset_speeds += [fast]

    if args.sweep:
        dxs = np.arange(min_sweep_x, max_sweep_x + sweep_xy_granularity, sweep_xy_granularity)
        dys = np.arange(min_sweep_y, max_sweep_y + sweep_xy_granularity, sweep_xy_granularity)
        dthetas = np.arange(min_sweep_theta, max_sweep_theta + sweep_theta_granularity, sweep_theta_granularity)
        all_n = len(dxs) * len(dys) * len(dthetas)
    else:
        all_n = args.num

    nb_moves_message = f"=== GENERATING {all_n} MOVES ==="
    spacer = "=" * len(nb_moves_message)
    print(spacer)
    print(nb_moves_message)
    print(spacer)

    commands = []
    for i in range(all_n):
        if args.sweep:
            dx_idx = i % len(dxs)
            dy_idx = (i // len(dxs)) % len(dys)
            dtheta_idx = (i // (len(dxs) * len(dys))) % len(dthetas)

            dx = round(dxs[dx_idx], 2)
            dy = round(dys[dy_idx], 2)
            dtheta = round(dthetas[dtheta_idx], 2)
        else:
            # Randomize dx, dy, dtheta within the specified max ranges
            dx = round(
                np.random.uniform(dx_max[0], dx_max[1]) * np.random.choice([-1, 1]), 2
            )
            dy = round(
                np.random.uniform(dy_max[0], dy_max[1]) * np.random.choice([-1, 1]), 2
            )
            dtheta = round(
                np.random.uniform(dtheta_max[0], dtheta_max[1])
                * np.random.choice([-1, 1]),
                2,
            )

        # Select preset based on available speeds
        selected_speed = categorize_speed(dx, preset_speeds, preset_speed_name)
        #selected_speed = np.random.choice(preset_speeds)
        # Load the corresponding .json preset from placo_presets
        preset_file = selected_speed
        if args.preset is not None:
            preset_file = args.preset
        preset_file = os.path.join(presets_dir, f"{preset_file}.json")
        if not os.path.isfile(preset_file):
            raise FileNotFoundError(f"Preset file not found: {preset_file}")

        with open(preset_file, "r") as file:
            data = json.load(file)

        # Call gait_generator (no bdx-specific arguments, references removed)
        cmd = [
            "python",
            f"{AUTOMAGERIE_DIR}/motion_generator/gait_generator.py",
            "--robot",
            args.robot,
            "--preset",
            preset_file,
            "--fps",
            str(args.fps),
            "--name",
            str(i),
            "--dx",
            str(dx),
            "--dy",
            str(dy),
            "--dtheta",
            str(dtheta),
            "--output_dir",
            recording_output_dir,
        ]
        if args.min_urdf:
            cmd += ["--min_urdf"]
        if args.walk_com_height:
            cmd += ["--walk_com_height", str(args.walk_com_height)]
        if args.walk_trunk_pitch:
            cmd += ["--walk_trunk_pitch", str(args.walk_trunk_pitch)]
        if args.foot_zmp_target_x:
            cmd += ["--foot_zmp_target_x", str(args.foot_zmp_target_x)]

        log_file = None if args.verbose else os.path.join(log_dir, f"{i}.log")
        commands.append((cmd, log_file))

    if args.verbose:
        if args.jobs > 1:
            with ThreadPoolExecutor(max_workers=args.jobs) as executor:
                for ok in executor.map(run_command_with_logging, commands):
                    if not ok:
                        executor.shutdown(wait=False, cancel_futures=True)
                        print(f"Error running")
                        sys.exit(1)
        else:
            for cmd in commands:
                print(cmd)
                if not run_command_with_logging(cmd):
                    sys.exit(1)
    elif args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(run_command_with_logging, c)
                       for c in commands]
            for _ in tqdm(as_completed(futures),
                          total=len(commands),
                          desc="Generating motion",
                          dynamic_ncols=True):
                ok = futures.pop(0).result()
                if not ok:
                    executor.shutdown(wait=False, cancel_futures=True)
                    sys.exit(1)
    else:
        for cmd in tqdm(commands, desc="Generating motion"):
            if not run_command_with_logging(cmd):
                sys.exit(1)

    # ---------------------------------------------------------------
    # 5. Check the JSON outputs in ../recordings; remove if out of range
    # ---------------------------------------------------------------
    totals = []
    min_x_lin_vel=0
    max_x_lin_vel=0
    min_y_lin_vel=0
    max_y_lin_vel=0
    min_z_lin_vel=0
    max_z_lin_vel=0
    min_x_lin_vel_local=0
    min_y_lin_vel_local=0
    min_z_lin_vel_local=0
    max_x_lin_vel_local=0
    max_y_lin_vel_local=0
    max_z_lin_vel_local=0
    if os.path.isdir(recording_output_dir):
        for filename in os.listdir(recording_output_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(recording_output_dir, filename)
                with open(file_path, "r") as file:
                    data = json.load(file)

                mean_avg_x_vel_local = data.get("Vel_x", 0)
                mean_avg_y_vel_local = data.get("Vel_y", 0)
                mean_avg_z_vel_local = data.get("Yaw", 0)
                if mean_avg_x_vel_local < min_x_lin_vel_local:
                    min_x_lin_vel_local = mean_avg_x_vel_local
                if mean_avg_x_vel_local > max_x_lin_vel_local:
                    max_x_lin_vel_local = mean_avg_x_vel_local

                if mean_avg_y_vel_local < min_y_lin_vel_local:
                    min_y_lin_vel_local = mean_avg_y_vel_local
                if mean_avg_y_vel_local > max_y_lin_vel_local:
                    max_y_lin_vel_local = mean_avg_y_vel_local

                if mean_avg_z_vel_local < min_z_lin_vel_local:
                    min_z_lin_vel_local = mean_avg_z_vel_local
                if mean_avg_z_vel_local > max_z_lin_vel_local:
                    max_z_lin_vel_local = mean_avg_z_vel_local

                placo_data = data.get("Placo", {})
                avg_x_vel = placo_data.get("avg_x_lin_vel", 0)
                avg_y_vel = placo_data.get("avg_y_lin_vel", 0)
                avg_z_vel = placo_data.get("avg_yaw_vel", 0)

                if avg_x_vel < min_x_lin_vel:
                    min_x_lin_vel = avg_x_vel
                if avg_y_vel < min_y_lin_vel:
                    min_y_lin_vel = avg_y_vel
                if avg_x_vel > max_x_lin_vel:
                    max_x_lin_vel = avg_x_vel
                if avg_y_vel > max_y_lin_vel:
                    max_y_lin_vel = avg_y_vel
                if avg_z_vel > max_z_lin_vel:
                    max_z_lin_vel = avg_z_vel
                if avg_z_vel > max_z_lin_vel:
                    max_z_lin_vel = avg_z_vel
                preset_name = placo_data.get("preset_name", "unknown")

                total_speed = np.sqrt(avg_x_vel**2 + avg_y_vel**2)

                # If the speeds do not fit the indicated preset name, remove
                if (
                    (preset_name == "slow" and total_speed > slow)
                    or (
                        preset_name == "medium"
                        and (total_speed <= slow or total_speed > fast)
                    )
                    or (preset_name == "fast" and total_speed <= medium)
                ):
                    # os.remove(file_path)
                    if args.verbose:
                        print(f"Possibly not right {file_path}")
                else:
                    totals.append((total_speed, preset_name))
    elif args.verbose:
        print(f"No directory found at {recording_output_dir}; skipping file checks.")
    if args.verbose:
        totals = sorted(totals, key=numeric_prefix_sort_key)
        for speed, preset_name in totals:
            print(f"Preset: {preset_name}, Total Speed: {speed:.4f}")
    print(f"x_lin_vel: {min_x_lin_vel},{max_x_lin_vel} (world frame)")
    print(f"y_lin_vel: {min_y_lin_vel},{max_y_lin_vel} (world frame)")
    print(f"theta_lin_vel: {min_z_lin_vel},{max_z_lin_vel} (world frame)")

    print(f"x_lin_vel: {min_x_lin_vel_local},{max_x_lin_vel_local} (local frame)")
    print(f"y_lin_vel: {min_y_lin_vel_local},{max_y_lin_vel_local} (local frame)")
    print(f"z_lin_vel: {min_z_lin_vel_local},{max_z_lin_vel_local} (local frame)")

    # ---------------------------------------------------------------
    # 6. Optional plotting of anim.npy and sim.npy
    # ---------------------------------------------------------------
    if args.plot:
        import matplotlib.pyplot as plt

        anim_path = os.path.join(SCRIPT_DIR, "anim.npy")
        sim_path = os.path.join(SCRIPT_DIR, "sim.npy")

        if os.path.isfile(anim_path) and os.path.isfile(sim_path):
            anim = np.load(anim_path)
            sim = np.load(sim_path)
            print("anim shape:", anim.shape)
            print("sim shape:", sim.shape)

            plt.plot(anim[:, 0, 0, 2], label="anim z-pos")
            plt.plot(sim[:, 0, 0, 2], label="sim z-pos")
            plt.legend()
            plt.title("Comparison of anim & sim")
            plt.show()
        else:
            print("anim.npy or sim.npy not found; skipping plot.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")

    all_files = glob(f"{recording_output_dir}/*.json")
    fit_ref_motions(all_files, f"{recording_output_dir}/ref_motion.poly")

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    available_robots=[]
    if os.path.isdir(ROBOTS_DIR):
        for name in os.listdir(ROBOTS_DIR):
            if os.path.isdir(os.path.join(ROBOTS_DIR, name)):
                available_robots.append(name)
    if len(available_robots) == 0:
        print(f"No robots found in: {ROBOTS_DIR}")
    parser = argparse.ArgumentParser(description="Generate AMP walking animations")
    # new, preferred argument
    type_group = parser.add_mutually_exclusive_group(required=True)
    type_group.add_argument(
        "--robot",
        choices=available_robots,
        help="Robot type",
    )
    # deprecated alias: write into the same dest, but suppress it from the help
    type_group.add_argument(
        "--duck",
        choices=available_robots,
        help=argparse.SUPPRESS,
        dest="robot",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=100,
        help="Number of random motion files to generate.",
    )
    # Estimated resident memory per job in bytes (using ~550MB as an estimate)
    gb_per_job = 0.55
    available_memory = psutil.virtual_memory().available / (1024 ** 3)
    max_jobs_by_memory = int(available_memory // gb_per_job)
    default_jobs = max(1, min(os.cpu_count(), max_jobs_by_memory))
    parser.add_argument(
        "-j",
        "--jobs",
        nargs="?",  # Makes the argument optional
        type=int,
        const=default_jobs,  # Used when -j is provided without a number
        default=1,  # Default value when -j is not provided
        help=(
            "Number of parallel jobs. "
            f"If -j is provided without a number, uses {default_jobs} on this computer. "
        ),
    )
    parser.add_argument(
        "--sweep", action="store_true", help="Sweep through the dx, dy, dtheta values."
    )
    parser.add_argument(
        "--min_urdf", action="store_true", help="Use minimal URDF if available."
    )
    parser.add_argument(
        "--slow", action="store_true", help="Slow speed"
    )
    parser.add_argument(
        "--medium", action="store_true", help="Medium speed"
    )
    parser.add_argument(
        "--fast", action="store_true", help="High speed"
    )
    parser.add_argument(
        "--fps", type=float, default=50, help="Frames per second (50 default)."
    )
    parser.add_argument("--walk_com_height", type=float, default=None)
    parser.add_argument("--walk_trunk_pitch", type=float, default=None)
    parser.add_argument("--foot_zmp_target_x", type=float, default=None)
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Optionally plot anim.npy and sim.npy if they exist.",
    )
    parser.add_argument(
        "--preset",
        help="Specify preset file",
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        help="Output directory for the recordings",
        default=None,
    )
    parser.add_argument(
        "--log_dir",
        help="Logging output directory",
        default=None,
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(AUTOMAGERIE_DIR, "recordings")
    if args.log_dir is None:
        args.log_dir = os.path.join(args.output_dir, "log")
    args.output_dir = os.path.join(args.output_dir, args.robot)
    args.log_dir = os.path.join(args.log_dir, args.robot)
    if args.preset is not None:
        args.output_dir = os.path.join(args.output_dir, args.preset)
        args.log_dir = os.path.join(args.log_dir, args.preset)
    if args.walk_com_height is not None:
        args.output_dir = os.path.join(args.output_dir, f"height{args.walk_com_height}")
        args.log_dir = os.path.join(args.log_dir, f"height{args.walk_com_height}")
    if args.walk_trunk_pitch is not None:
        args.output_dir = os.path.join(args.output_dir, f"pitch{args.walk_trunk_pitch}")
        args.log_dir = os.path.join(args.log_dir, f"pitch{args.walk_trunk_pitch}")
    if not args.slow and not args.medium and not args.fast:
        args.medium = True
    main(args)
