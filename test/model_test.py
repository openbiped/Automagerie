# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for all models."""

from pathlib import Path
from typing import List

from absl.testing import absltest
from absl.testing import parameterized
import mujoco

# Internal import.


SCRIPT_DIR = Path(__file__).resolve().parent.parent
ROBOTS_DIR = SCRIPT_DIR / "robots"

MODEL_XMLS = []
for d in sorted(p for p in ROBOTS_DIR.iterdir() if p.is_dir()):
    hi, lo = d / "scene-high.xml", d / "scene.xml"
    path = hi if hi.exists() else lo if lo.exists() else None
    if path:
        MODEL_XMLS.append((d.name, path))
    else:
        print(f"⚠️  No scene file in {d}")

# Total simulation duration, in seconds.
_MAX_SIM_TIME = 0.1
# Scale for the pseudorandom control noise.
_NOISE_SCALE = 1.0


def _pseudorandom_ctrlnoise(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    i: int,
    noise: float,
) -> None:
  for j in range(model.nu):
    ctrlrange = model.actuator_ctrlrange[j]
    if model.actuator_ctrllimited[j]:
      center = 0.5 * (ctrlrange[1] + ctrlrange[0])
      radius = 0.5 * (ctrlrange[1] - ctrlrange[0])
    else:
      center = 0.0
      radius = 1.0
    data.ctrl[j] = center + radius * noise * (2*mujoco.mju_Halton(i, j+2) - 1)


class ModelsTest(parameterized.TestCase):
  """Tests that MuJoCo models load and do not emit warnings."""

  @parameterized.named_parameters(MODEL_XMLS)
  def test_compiles_and_steps(self, xml_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    i = 0
    while data.time < _MAX_SIM_TIME:
      _pseudorandom_ctrlnoise(model, data, i, _NOISE_SCALE)
      mujoco.mj_step(model, data)
      i += 1
    # Check no warnings were triggered during the simulation.
    if not all(data.warning.number == 0):
      warning_info = '\n'.join([
          f'{mujoco.mjtWarning(enum_value).name}: count={count}'
          for enum_value, count in enumerate(data.warning.number) if count
      ])
      self.fail(f'MuJoCo warning(s) encountered:\n{warning_info}')


if __name__ == '__main__':
  absltest.main()
