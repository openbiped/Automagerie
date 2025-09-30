import numpy as np
import pickle
import gzip


class PolyReferenceMotion:
    def __init__(self, polynomial_coefficients: str):
        with gzip.open(polynomial_coefficients, "rb") as f:
            data = pickle.load(f)
        self.dx_range = [0, 0]
        self.dy_range = [0, 0]
        self.dtheta_range = [0, 0]
        self.dxs = []
        self.dys = []
        self.dthetas = []
        self.data_array = []
        self.period = None
        self.fps = None
        self.frame_offsets = None
        self.startend_double_support_ratio = None
        self.start_offset = None
        self.nb_steps_in_period = None

        self.process(data)

    def process(self, data):
        print("[Poly ref data] Processing ...")
        _data = {}
        self.robot = None
        self.joints = []
        for name in data.keys():
            split = name.split("_")
            dx = round(float(split[0]), 3)
            dy = round(float(split[1]), 3)
            dtheta = round(float(split[2]), 3)

            if self.period is None:
                self.robot = data[name]["robot"]
                self.joints = data[name]["joints"]
                self.period = data[name]["period"]
                self.fps = data[name]["fps"]
                #self.frame_offsets = data[name]["frame_offsets"]
                self.startend_double_support_ratio = data[name][
                    "startend_double_support_ratio"
                ]
                self.start_offset = int(self.startend_double_support_ratio * self.fps)
                self.nb_steps_in_period = int(self.period * self.fps)

            if dx not in self.dxs:
                self.dxs.append(dx)

            if dy not in self.dys:
                self.dys.append(dy)

            if dtheta not in self.dthetas:
                self.dthetas.append(dtheta)

            self.dx_range = [min(dx, self.dx_range[0]), max(dx, self.dx_range[1])]
            self.dy_range = [min(dy, self.dy_range[0]), max(dy, self.dy_range[1])]
            self.dtheta_range = [
                min(dtheta, self.dtheta_range[0]),
                max(dtheta, self.dtheta_range[1]),
            ]

            if dx not in _data:
                _data[dx] = {}

            if dy not in _data[dx]:
                _data[dx][dy] = {}

            if dtheta not in _data[dx][dy]:
                _data[dx][dy][dtheta] = data[name]

            _coeffs = data[name]["coefficients"]

            coeffs = []
            for k, v in _coeffs.items():
                coeffs.append(v)
            _data[dx][dy][dtheta] = coeffs

        self.dxs = sorted(self.dxs)
        self.dys = sorted(self.dys)
        self.dthetas = sorted(self.dthetas)

        nb_dx = len(self.dxs)
        nb_dy = len(self.dys)
        nb_dtheta = len(self.dthetas)
        self.nb_joints = len(self.joints)

        # ————— build dense 3D array, falling back to nearest when missing —————
        self.data_array = [[[None]*nb_dtheta for _ in range(nb_dy)] for _ in range(nb_dx)]

        for ix, dx in enumerate(self.dxs):
            # what dy’s actually exist for this dx?
            available_dys = sorted(_data.get(dx, {}))
            if not available_dys:
                raise ValueError(f"No data at all for dx={dx:.3f}")

            for iy, dy in enumerate(self.dys):
                # pick the best dy
                if dy in available_dys:
                    chosen_dy = dy
                else:
                    chosen_dy = min(available_dys, key=lambda y: abs(y - dy))
                    print(f"WARNING: dx={dx:.3f}, dy={dy:.3f} missing, using nearest dy={chosen_dy:.3f}")

                # now what dtheta’s exist at (dx, chosen_dy)?
                available_dt = sorted(_data[dx][chosen_dy])

                for ith, dtheta in enumerate(self.dthetas):
                    # pick the best dtheta
                    if dtheta in available_dt:
                        chosen_dt = dtheta
                    else:
                        chosen_dt = min(available_dt, key=lambda t: abs(t - dtheta))
                        print(
                            f"WARNING: (dx={dx:.3f}, dy={dy:.3f}, dtheta={dtheta:.3f}) "
                            f"missing, using nearest dtheta={chosen_dt:.3f} "
                            f"(from dy={chosen_dy:.3f})"
                        )

                    self.data_array[ix][iy][ith] = _data[dx][chosen_dy][chosen_dt]

        print("[Poly ref data] Done processing")

    def vel_to_index(self, dx, dy, dtheta):

        dx = np.clip(dx, self.dx_range[0], self.dx_range[1])
        dy = np.clip(dy, self.dy_range[0], self.dy_range[1])
        dtheta = np.clip(dtheta, self.dtheta_range[0], self.dtheta_range[1])

        ix = np.argmin(np.abs(np.array(self.dxs) - dx))
        iy = np.argmin(np.abs(np.array(self.dys) - dy))
        itheta = np.argmin(np.abs(np.array(self.dthetas) - dtheta))

        return int(ix), int(iy), int(itheta)

    def sample_polynomial(self, t, coeffs):
        ret = []
        for c in coeffs:
            ret.append(np.polyval(np.flip(c), t))

        return ret

    def get_reference_motion(self, dx, dy, dtheta, i):
        ix, iy, itheta = self.vel_to_index(dx, dy, dtheta)
        t = i % self.nb_steps_in_period / self.nb_steps_in_period
        t = np.clip(t, 0.0, 1.0)  # safeguard
        ret = self.sample_polynomial(t, self.data_array[ix][iy][itheta])
        return ret
