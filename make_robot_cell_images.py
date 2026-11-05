"""
Generate the images used by robot_cell.py.

Writes into robot_cell_images/:

* background.png  - the bench, conveyor, racks and footer, without any arms.
* arm{i}_{angle}_{gripper}_{load}_{height}.png - one transparent overlay per
  arm (i = 0, 1, 2) for each of its 16 states:
      angle    0 (facing the viewer / rack) or 180 (facing away / conveyor)
      gripper  open or closed
      load     holding or empty
      height   up or down

Every overlay is the full picture size, so robot_cell.py only has to paste
the three arm overlays for the current state on top of the background.

Run:  python make_robot_cell_images.py      (needs Pillow; uses robot_arms.py)
"""

from __future__ import annotations

import itertools
import math
import os

from PIL import Image, ImageDraw
from PIL.PngImagePlugin import PngInfo

import robot_arms as ra

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_cell_images")
ARM_X = (160, 400, 640)
LIFT = 46                         # how far "up" is above "down"
WARN = (255, 70, 70)


def downsample(img: Image.Image) -> Image.Image:
    """Shrink a super-sampled image, keeping transparent edges clean."""
    if img.mode == "RGBA":
        return img.convert("RGBa").resize((ra.W, ra.H), Image.LANCZOS).convert("RGBA")
    return img.resize((ra.W, ra.H), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Scene layout: each arm has a 5-slot rack straight in front of it (angle 0)
# and an empty carrier straight behind it on the conveyor (angle 180).
# ---------------------------------------------------------------------------
def make_rack(j: int) -> ra.Rack:
    rack = ra.Rack(ra.TUBE_COLOURS[j])
    rack.FIRST_X = -2 * rack.PITCH            # centre slot (2) is straight ahead
    rack.fill = [1.0, 1.0, 0.0, 1.0, 1.0]     # centre slot free for the arm
    return rack


def make_conveyor() -> ra.Conveyor:
    conv = ra.Conveyor()
    conv.PITCH = 120.0
    conv.X0 = ARM_X[0]                        # carriers 0, 2, 4 sit behind the arms
    conv.offset = 0.0
    conv.filled = lambda k, j: k % 2 == 1     # carriers between arms are full
    return conv


def draw_background() -> Image.Image:
    img = Image.new("RGB", (ra.W * ra.SS, ra.H * ra.SS), ra.BG)
    d = ImageDraw.Draw(img)
    S = ra.SS
    # wall grid and HUD panels
    for x in range(0, ra.W, 40):
        d.line([x * S, 0, x * S, 250 * S], fill=ra.GRID, width=S)
    for y in range(0, 250, 40):
        d.line([0, y * S, ra.W * S, y * S], fill=ra.GRID, width=S)
    for px in (22, 648):
        d.rounded_rectangle([px * S, 14 * S, (px + 130) * S, 78 * S], radius=4 * S,
                            outline=ra.HUD, width=S, fill=(14, 26, 44))
        for i, w in enumerate((60, 44, 50)):
            y = (22 + i * 6) * S
            d.line([(px + 8) * S, y, (px + 8 + w) * S, y], fill=(80, 140, 190), width=2 * S)
        pts = [((px + 8 + i) * S, (56 + 8 * math.sin(i / 14 + px)) * S)
               for i in range(0, 115, 2)]
        d.line(pts, fill=ra.HUD, width=S)
    make_conveyor().draw(d)
    # bench
    d.polygon([(40 * S, 250 * S), (760 * S, 250 * S), (800 * S, 358 * S), (0, 358 * S)],
              fill=ra.BENCH_TOP, outline=(90, 100, 140), width=S)
    d.rectangle([0, 358 * S, ra.W * S, 410 * S], fill=ra.BENCH_FRONT)
    d.line([0, 368 * S, ra.W * S, 368 * S], fill=ra.HUD, width=2 * S)
    for j, x in enumerate(ARM_X):
        P = ra.Projector(x)
        make_rack(j).draw(img, d, P)
        # controller box, to the right where the arm never reaches
        c0, c1 = P(85, ra.Rack.DEPTH, 60), P(111, ra.Rack.DEPTH, 0)
        d.rounded_rectangle([c0[0], c0[1], c1[0], c1[1]], radius=3 * S,
                            fill=(150, 156, 172), outline=(90, 96, 116), width=S)
        d.rounded_rectangle([c0[0] + 5 * S, c0[1] + 6 * S, c1[0] - 5 * S, c1[1] - 10 * S],
                            radius=2 * S, fill=(34, 38, 54))
    # footer band (text is added live by robot_cell.py)
    d.rectangle([0, 410 * S, ra.W * S, ra.H * S], fill=(10, 13, 24))
    return downsample(img)


# ---------------------------------------------------------------------------
# Arm overlays
# ---------------------------------------------------------------------------
def arm_pose(angle: int, raised: bool) -> ra.Pose:
    """Gripper target for a state, in the arm's own frame."""
    if angle == 0:                                     # facing the viewer: rack
        y = ra.Rack.DEPTH
        z = ra.Rack.TUBE_Z + ra.TUBE_LEN - 6
    else:                                              # facing away: conveyor
        y = ra.Conveyor.DEPTH
        z = ra.Conveyor.BELT_Z + ra.Conveyor.CARRIER_H - 4 + ra.TUBE_LEN - 6
    if raised:
        z += LIFT
    return ra.Pose.at(0.0, y, z)


def draw_arm(j: int, angle: int, gripper_open: bool, holding: bool,
             raised: bool) -> Image.Image:
    img = Image.new("RGBA", (ra.W * ra.SS, ra.H * ra.SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    arm = ra.RobotArm(j, ARM_X[j])
    arm.pose = arm_pose(angle, raised)
    arm.grip_open = 1.0 if gripper_open else 0.0
    # Holding with the gripper open is fine when the arm is down (the tube is
    # sitting in its slot, being released or about to be gripped), but when
    # the arm is up the tube has nothing holding it and falls.
    dropping = holding and gripper_open and raised
    arm.holding = ra.TUBE_COLOURS[j] if holding and not dropping else None
    arm.draw(d)
    if dropping:
        # An open gripper cannot hold a tube: show it slipping out, with a
        # red warning ring, so the impossible state is obvious on screen.
        gx, gy, gz = arm.forward()["grip"]
        drop = 16
        bottom = arm.P(gx, gy, gz + 6 - ra.TUBE_LEN - drop)
        ra.draw_tube(d, bottom, ra.TUBE_COLOURS[j])
        cx, cy = arm.P(gx, gy, gz - drop - 10)
        r = 34 * ra.SS
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=WARN, width=3 * ra.SS)
    return downsample(img)


def state_filename(j: int, angle: int, gripper_open: bool, holding: bool,
                   raised: bool) -> str:
    return "arm{}_{}_{}_{}_{}.png".format(
        j, angle, "open" if gripper_open else "closed",
        "holding" if holding else "empty", "up" if raised else "down")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    draw_background().save(os.path.join(OUT_DIR, "background.png"), optimize=True)
    count = 0
    for j in range(3):
        for angle, gripper_open, holding, raised in itertools.product(
                (0, 180), (True, False), (True, False), (True, False)):
            img = draw_arm(j, angle, gripper_open, holding, raised)
            # crop to the arm's own column to keep the files small; robot_cell.py
            # reads the offset back from the PNG's text chunk
            box = img.getbbox()
            img = img.crop(box)
            info = PngInfo()
            info.add_text("offset", f"{box[0]},{box[1]}")
            img.save(os.path.join(OUT_DIR, state_filename(j, angle, gripper_open, holding, raised)),
                     pnginfo=info, optimize=True)
            count += 1
    print(f"wrote background.png and {count} arm images to {OUT_DIR}")


if __name__ == "__main__":
    main()
