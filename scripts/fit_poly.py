import numpy as np
import json
from glob import glob
import os
import argparse
import pickle
import warnings
import gzip

def fit_ref_motion(file):
    # TODO: maybe not ignore Polyfit warning
    try:
        RankWarning = np.RankWarning
    except AttributeError:
        from numpy.exceptions import RankWarning

    warnings.filterwarnings("ignore", category=RankWarning)
    data = json.load(open(file))
    Y_all = np.array(data["Frames"])
    period = data["Placo"]["period"]
    fps = data["FPS"]
    robot = data["Robot"]
    joints = data["Joints"]
    home = data["Home"]
    # root_pos = data["Joints"]
    # root_quat = data["Joints"]
    trunk_pitch = data["Placo"]["walk_trunk_pitch"]
    com_height = data["Placo"]["walk_com_height"]
    frame_offsets = data["Frame_offset"][0]
    frame_sizes = data["Frame_size"][0]
    startend_double_support_ratio = data["Placo"]["startend_double_support_ratio"]
    start_offset = int(startend_double_support_ratio * fps)
    nb_steps_in_period = int(period * fps)
    _Y = Y_all[start_offset : start_offset + int(nb_steps_in_period)]

    slices = {}
    slice_start = 0
    
    joints_pos_start = frame_offsets["joints_pos"]
    joints_pos_size = frame_sizes["joints_pos"]
    joints_pos_end = joints_pos_start + joints_pos_size
    joints_pos = _Y[:, joints_pos_start : joints_pos_end]
    slices["joints_pos"] = [ slice_start, slice_start + joints_pos_size ]
    slice_start += joints_pos_size
    
    joints_vel_start = frame_offsets["joints_vel"]
    joints_vel_size = frame_sizes["joints_vel"]
    joints_vel_end = joints_vel_start + joints_vel_size
    joints_vel = _Y[:, joints_vel_start : joints_vel_end]
    slices["joints_vel"] = [ slice_start, slice_start + joints_vel_size ]
    slice_start += joints_vel_size

    foot_contacts_start = frame_offsets["foot_contacts"]
    foot_contacts_size = frame_sizes["foot_contacts"]
    foot_contacts_end = foot_contacts_start + foot_contacts_size
    foot_contacts = _Y[:, foot_contacts_start : foot_contacts_end]
    slices["foot_contacts"] = [ slice_start, slice_start + foot_contacts_size ]
    slice_start += foot_contacts_size

    world_linear_vel_start = frame_offsets["world_linear_vel"]
    world_linear_vel_size = frame_sizes["world_linear_vel"]
    world_linear_vel_end = world_linear_vel_start + world_linear_vel_size
    base_linear_vel = _Y[:, world_linear_vel_start : world_linear_vel_end]
    slices["world_linear_vel"] = [ slice_start, slice_start + world_linear_vel_size ]
    slice_start += world_linear_vel_size

    world_angular_vel_start = frame_offsets["world_angular_vel"]
    world_angular_vel_size = frame_sizes["world_angular_vel"]
    world_angular_vel_end = world_angular_vel_start + world_angular_vel_size
    base_angular_vel = _Y[:, world_angular_vel_start : world_angular_vel_end]
    slices["world_angular_vel"] = [ slice_start, slice_start + world_angular_vel_size ]
    slice_start += world_angular_vel_size

    Y = np.concatenate(
        [joints_pos, joints_vel, foot_contacts, base_linear_vel, base_angular_vel],
        axis=1,
    ).astype(np.float32)

    # Generate time feature
    X = np.linspace(0, 1, Y.shape[0]).reshape(-1, 1).astype(np.float32)  # Time variable

    # Polynomial degree
    degree = 15

    # Store coefficients
    coefficients = {}

    # ====== Fit Polynomial Regression per Dimension ======
    for dim in range(Y.shape[1]):
        coeffs = np.polyfit(X.flatten(), Y[:, dim], degree)
        coefficients[f"dim_{dim}"] = list(np.flip(coeffs))

    ret_data = {
        "coefficients": coefficients,
        "period": period,
        "fps": fps,
        "robot": robot,
        "joints": joints,
        "home": home,
        "slices": slices,
        "trunk_pitch": trunk_pitch,
        "com_height": com_height,
        "nb_steps_in_period": nb_steps_in_period,
        "startend_double_support_ratio": startend_double_support_ratio,
        "Placo": data["Placo"],
        "feet_spacing": data["Placo"]["feet_spacing"],
    }
    if "left_target" in data["Placo"]:
        ret_data.update({
            "left_target": data["Placo"]["left_target"],
            "right_target": data["Placo"]["right_target"],
        })

    return ret_data

def fit_ref_motions(all_files, output_poly, output_pickle=None):
    all_coefficients = {}
    for file in all_files:
        name = os.path.basename(file).strip(".json")
        tmp = name.split("_")
        name = f"{tmp[1]}_{tmp[2]}_{tmp[3]}"

        all_coefficients[name] = fit_ref_motion(file)
    # Output compressed pickle (new-style)
    with gzip.open(output_poly, "wb") as f:
        pickle.dump(all_coefficients, f)
    # Output uncompressed pickle if requested (old-style)
    if output_pickle:
        pickle.dump(all_coefficients, open(output_pickle, "wb"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_motion", type=str, default="ref_motion")
    args = parser.parse_args()

    all_files = glob(f"{args.ref_motion}/*.json")
    output_poly_file = os.path.join(args.ref_motion, "ref_motion.poly")
    output_pickle_file = os.path.join(args.ref_motion, "polynomial_coefficients.pkl")
    fit_ref_motions(all_files, output_poly_file, output_pickle_file)
