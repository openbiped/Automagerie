# Copyright 2025 Mimir Reynisson
# Copyright 2024 DeepMind Technologies Limited
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
"""Generate a markdown table with images of some of the models in Automagerie.

Requirements:
    curl -LsSf https://astral.sh/uv/install.sh | sh

Instructions:
    `uv run scripts/generate_assets.py` will create a markdown document called
    `gallery.md` with a table of images. Copy this table into README.md to
    display the images.
"""
import os
import enum
import math
import argparse
from pathlib import Path

from absl import app
import cv2
from dm_control import mjcf
from mdutils import mdutils
import numpy as np
from PIL import Image
import tqdm.auto

def create_arena(width, height):
  arena = mjcf.RootElement()
  arena.visual.quality.shadowsize = 8192
  arena.visual.headlight.diffuse = (0.6,) * 3
  arena.visual.headlight.ambient = (0.3,) * 3
  arena.visual.headlight.specular = (0.2,) * 3
  getattr(arena.visual, "global").offheight = width*2
  getattr(arena.visual, "global").offwidth = height*2
  arena.asset.add(
      "texture",
      type="skybox",
      builtin="gradient",
      height=height+12,
      width=width+12,
      rgb1="1 1 1",
      rgb2="1 1 1",
  )
  return arena

def choose_keyframe_id(root):
    keys = list(root.find_all('key'))
    name_to_idx = {getattr(k, 'name', None): i for i, k in enumerate(keys)}
    for wanted in ('snapshot', 'home'):
        if wanted in name_to_idx:
            return name_to_idx[wanted]
    return 0

SCRIPT_DIR = Path(__file__).resolve().parent.parent
ROBOTS_DIR = SCRIPT_DIR / "robots"

MODEL_XMLS = []
for d in sorted(p for p in ROBOTS_DIR.iterdir() if p.is_dir()):
    hi = d / "scene-high.xml"
    lo = d / "scene.xml"
    if hi.exists():
        MODEL_XMLS.append(hi)
    elif lo.exists():
        MODEL_XMLS.append(lo)
    else:
        print(f"⚠️  No scene file in {d}")
NAME_MAP = {p.parent.name: p.parent.name for p in MODEL_XMLS}

def main():
  parser = argparse.ArgumentParser(description="Generate robot gallery")
  parser.add_argument("--width", type=int, default=500, help="Image width in pixels")
  parser.add_argument("--height", type=int, default=500, help="Image height in pixels")
  parser.add_argument("--columns", type=int, default=5, help="Column width")
  args = parser.parse_args()

  paths = []
  pngs = []
  for xml in tqdm.auto.tqdm(MODEL_XMLS):
    try:
      robot_maker = xml.parent.stem
      robot_name = xml.stem
      robot = f"{robot_maker}/{robot_name}"

      snapshot_xml = mjcf.from_path(xml.as_posix(), escape_separators=True)
      kid = choose_keyframe_id(snapshot_xml)
      arena = create_arena(args.width, args.height)
      arena.include_copy(snapshot_xml, override_attributes=True)

      physics = mjcf.Physics.from_mjcf_model(arena)

      try:
        physics.reset(keyframe_id=kid)
      except:
        # try without any keyframe
        physics.reset()

      physics.forward()

      filename = f"robots/{robot_maker}/{robot_maker}.png"
      img = physics.render(height=args.width, width=args.height)
      Image.fromarray(img).save(filename)

      model_xml = mjcf.from_path(xml.as_posix(), escape_separators=True)
      for light in model_xml.find_all("light"):
        light.remove()
      for body in list(model_xml.find_all('body')):
        if getattr(body, 'name', None) == 'floor':
          body.remove()
      for geom in list(model_xml.find_all('geom')):
        if getattr(geom, 'name', None) == 'floor' or getattr(geom, 'type', None) == 'plane':
          geom.remove()

      arena = create_arena(args.width, args.height)
      arena.include_copy(model_xml, override_attributes=True)

      physics = mjcf.Physics.from_mjcf_model(arena)
      try:
        physics.reset(keyframe_id=0)
      except:
        physics.reset()

      physics.forward()

      img = physics.render(height=args.width, width=args.height)
      img = cv2.putText(
          img.copy(),
          f"{robot_maker}",
          (5, args.height-20),
          cv2.FONT_HERSHEY_SIMPLEX,
          1.3,
          (0, 0, 0),
          1,
          cv2.LINE_AA,
      )

      filename = f"assets/{robot_maker}.png"
      paths.append(filename)

      png = np.zeros((args.width, args.height, 4), dtype=np.uint8)
      u, v = np.where(np.all(img == 255, axis=-1))
      png[u, v, -1] = 0
      png[u, v, :3] = 0
      u, v = np.where(np.any(img != 255, axis=-1))
      png[u, v, :3] = img[u, v]
      png[u, v, -1] = 255
      pngs.append(png.copy())
      Image.fromarray(png).save(filename)
    except Exception as e:
      print(e)
      print(f"failed to load {xml.as_posix()}")

  n_models = len(paths)
  n_cols = args.columns
  n_rows = int(math.ceil(n_models / n_cols))
  table = []
  for r in range(n_rows):
    row = []
    for c in range(n_cols):
      i = r * n_cols + c
      if i >= n_models:
        row.append("")
      else:
        robot = os.path.splitext(os.path.basename(paths[i]))[0]   # 'assets/q1mini.png' -> 'q1mini'
        row.append(f"<a href='robots/{robot}'><img src='{paths[i]}' width=200></a>")
    table.extend(row)

  mdfile = mdutils.MdUtils(file_name="gallery")
  mdfile.new_table(columns=n_cols, rows=n_rows, text=table, text_align="center")
  mdfile.create_md_file()


if __name__ == "__main__":
  main()
