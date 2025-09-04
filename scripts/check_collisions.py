import argparse
import mujoco
import numpy as np
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Run MuJoCo scene and print collisions.")
    parser.add_argument("scene", nargs="?", default="scene.xml",
                        help="Path to the XML scene file (default: scene.xml)")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Number of simulation steps to run (default: 1000)")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)

    home_frame = model.keyframe("home")
    default_qpos = np.array(home_frame.qpos)
    default_ctrl = np.array(home_frame.ctrl)
    data.qpos[:] = default_qpos.copy()
    data.ctrl[:] = default_ctrl.copy()

    collisions = defaultdict(int)

    for _ in range(args.steps):
        mujoco.mj_step(model, data)

        for i in range(data.ncon):
            contact = data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            body1 = model.body(model.geom_bodyid[geom1]).name
            body2 = model.body(model.geom_bodyid[geom2]).name
            key = tuple(sorted((body1, body2)))
            collisions[key] += 1

    if collisions:
        print("Collisions observed:")
        for (a, b), count in collisions.items():
            print(f"{a} <-> {b}: {count} time(s)")
    else:
        print("No collisions detected.")

if __name__ == "__main__":
    main()
