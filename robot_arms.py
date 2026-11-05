"""
WSA032 - Sample handling cell
=============================

Generates ``robot_arms.gif``: three robot arms moving sample tubes between a
conveyor and their bench racks.

Each arm is a 3-axis (yaw / shoulder / elbow) manipulator with a vertical
gripper.  The important point of the model is the *base yaw*:

* The shoulder and elbow only ever move the arm inside its own vertical
  "arm plane", described by a horizontal reach ``r`` and a height ``z``.
* The base (pillar) rotates that whole plane about the vertical pillar axis
  by the yaw angle ``theta``.  A point (r, z) in the arm plane therefore sits
  at world position (r*cos(theta), r*sin(theta), z).
* Only *after* that rotation are points projected onto the screen.
* The hinge pins (shoulder, elbow, wrist) lie along the axis
  (-sin(theta), cos(theta), 0), at right angles to the arm plane, and the
  links are solid bars with their width along that axis, so they turn too.

So when the pillar turns, the shoulder axis, the links, the elbow, the
gripper and any tube it holds all turn with it.  (The old animation solved
the arm in the flat screen plane, so the pillar turned but the arm's axis
didn't.)

World axes (per arm, origin at the centre of the pillar base):
    x  -> to the right along the bench
    y  -> away from the viewer (towards the conveyor)
    z  -> up

Run:  python robot_arms.py       (needs Pillow)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------
# Output / render settings
# --------------------------------------------------------------------------
W, H = 800, 480          # final GIF size
SS = 2                   # super-sampling factor for anti-aliasing
FRAME_MS = 45            # frame duration
OUTFILE = "robot_arms.gif"

# Oblique projection: depth (y) is drawn up-and-slightly-right on screen.
DEPTH_RISE = 0.45        # screen pixels up per unit of depth
DEPTH_SHEAR = 0.30       # screen pixels right per unit of depth
BASE_SCREEN_Y = 270      # screen y of every pillar base (z = 0, y = 0)
# Direction the camera looks along (into the scene) for this projection.
VIEW_DIR = (-DEPTH_SHEAR, 1.0, -DEPTH_RISE)

# Colours
BG = (12, 16, 30)
GRID = (24, 32, 54)
HUD = (56, 200, 240)
BENCH_TOP = (58, 66, 98)
BENCH_FRONT = (40, 46, 72)
STEEL = (122, 132, 156)
STEEL_DARK = (78, 86, 108)
ARM_WHITE = (232, 236, 244)
ARM_SHADE = (176, 184, 202)
ARM_EDGE = (60, 66, 86)
LINK_TOP = (206, 212, 226)  # the top/bottom faces of a link, lit less than its side
ORANGE = (255, 145, 30)
ORANGE_DARK = (150, 70, 10)
GLASS = (200, 210, 228)
TUBE_COLOURS = [(40, 200, 240), (225, 60, 170), (90, 220, 120)]

FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "DejaVuSansMono.ttf",
]


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    paths = FONT_PATHS
    if bold:
        paths = [p.replace("Mono.ttf", "Mono-Bold.ttf") for p in FONT_PATHS] + paths
    for p in paths:
        try:
            return ImageFont.truetype(p, size * SS)
        except OSError:
            continue
    return ImageFont.load_default()


def smooth(t: float) -> float:
    """Smoothstep easing 0..1 -> 0..1."""
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_angle(a: float, b: float, t: float) -> float:
    """Interpolate angles along the shortest path."""
    d = (b - a + math.pi) % (2 * math.pi) - math.pi
    return a + d * t


def convex_hull(points):
    """Monotone-chain convex hull of 2D screen points."""
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


# --------------------------------------------------------------------------
# Projection
# --------------------------------------------------------------------------
@dataclass
class Projector:
    """Maps world points (relative to an origin on screen) to screen pixels."""

    origin_x: float
    origin_y: float = BASE_SCREEN_Y

    def __call__(self, x: float, y: float, z: float) -> tuple[float, float]:
        sx = self.origin_x + x + DEPTH_SHEAR * y
        sy = self.origin_y - z - DEPTH_RISE * y
        return sx * SS, sy * SS


# --------------------------------------------------------------------------
# Scene objects
# --------------------------------------------------------------------------
TUBE_LEN = 44
TUBE_W = 10


def draw_tube(d: ImageDraw.ImageDraw, bottom: tuple[float, float], colour) -> None:
    """Draw a test tube standing upright with its bottom at a screen point."""
    bx, by = bottom
    w, L = TUBE_W * SS, TUBE_LEN * SS
    x0, x1 = bx - w / 2, bx + w / 2
    top = by - L
    d.rounded_rectangle([x0, top, x1, by], radius=w / 2, fill=(30, 36, 56),
                        outline=GLASS, width=SS)
    d.rounded_rectangle([x0 + SS, top + L * 0.38, x1 - SS, by - SS],
                        radius=w / 2 - SS, fill=colour)
    d.rectangle([x0 - SS, top - 2 * SS, x1 + SS, top + 4 * SS], fill=(52, 58, 80),
                outline=(96, 104, 130), width=SS)
    d.line([x0 + 2.5 * SS, top + 6 * SS, x0 + 2.5 * SS, by - 6 * SS],
           fill=(255, 255, 255), width=SS)


def draw_tube_clipped(img: Image.Image, bottom: tuple[float, float], colour,
                      clip_y: float) -> None:
    """Draw a tube but only the part above screen row `clip_y` (used for a tube
    rising up into a rack slot from the feeder underneath)."""
    bx, by = bottom
    pad = 4 * SS
    w, h = int(TUBE_W * SS + 2 * pad), int(TUBE_LEN * SS + 2 * pad)
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw_tube(ImageDraw.Draw(layer), (w / 2, h - pad), colour)
    left, top = int(bx - w / 2), int(by - (h - pad))
    keep = int(clip_y) - top
    if keep <= 0:
        return
    layer = layer.crop((0, 0, w, min(keep, h)))
    img.paste(layer, (left, top), layer)


class Rack:
    """A 5-slot tube rack on the bench in front-left of an arm.  A feeder under
    the bench pushes a fresh tube up into a slot after the arm empties it."""

    DEPTH = -130.0
    FIRST_X = -95.0
    PITCH = 20.0
    TUBE_Z = 10.0         # z of a seated tube's bottom
    TOP_Z = 14.0          # top surface of the rack block

    def __init__(self, colour) -> None:
        self.colour = colour
        self.fill = [1.0] * 5      # per slot: 0 = empty, 1 = seated, between = rising

    def slot_xy(self, k: int) -> tuple[float, float]:
        return self.FIRST_X + k * self.PITCH, self.DEPTH

    def draw(self, img: Image.Image, d: ImageDraw.ImageDraw, P: Projector) -> None:
        x0, y = self.slot_xy(0)
        x1, _ = self.slot_xy(len(self.fill) - 1)
        a = P(x0 - 16, y, self.TOP_Z)
        b = P(x1 + 16, y, 0)
        d.rectangle([a[0], a[1], b[0], b[1]], fill=(70, 78, 108),
                    outline=(110, 120, 150), width=SS)
        for k, f in enumerate(self.fill):
            sx, sy = P(*self.slot_xy(k), self.TUBE_Z)
            if f >= 1.0:
                draw_tube(d, (sx, sy), self.colour)
                continue
            d.ellipse([sx - 6 * SS, sy - 3 * SS, sx + 6 * SS, sy + 3 * SS], fill=(20, 24, 40))
            if f > 0.0:
                rise = (1.0 - smooth(f)) * TUBE_LEN
                draw_tube_clipped(img, P(*self.slot_xy(k), self.TUBE_Z - rise),
                                  self.colour, P(*self.slot_xy(k), self.TOP_Z)[1])

    def draw_controller(self, d: ImageDraw.ImageDraw, P: Projector) -> None:
        """Controller box in front-right of the arm (drawn after the arm,
        because it is nearer the viewer than the pillar)."""
        y = self.DEPTH
        c0 = P(62, y, 60)
        c1 = P(88, y, 0)
        d.rounded_rectangle([c0[0], c0[1], c1[0], c1[1]], radius=3 * SS,
                            fill=(150, 156, 172), outline=(90, 96, 116), width=SS)
        d.rounded_rectangle([c0[0] + 5 * SS, c0[1] + 6 * SS, c1[0] - 5 * SS,
                             c1[1] - 10 * SS], radius=2 * SS, fill=(34, 38, 54))


class Conveyor:
    """A continuously running belt carrying 3-slot carriers to the right.

    Carriers enter empty on the left; arm j drops a tube into slot j of each
    carrier as it passes, so carriers leave on the right holding one tube of
    each colour.  Positions are in absolute scene coordinates (x = 0 at the
    left edge of the picture), drawn with a projector whose origin is x = 0.
    """

    DEPTH = 90.0
    BELT_Z = 81.0
    CARRIER_H = 8.0
    PITCH = 160.0               # carrier spacing
    SLOT_DX = 20.0
    X0 = 0.0                    # set by Cell so the carriers line up with the arms

    def __init__(self) -> None:
        self.P = Projector(0)
        self.offset = 0.0
        self.filled = lambda k, j: False      # replaced by Cell

    @property
    def tube_z(self) -> float:
        return self.BELT_Z + self.CARRIER_H - 4

    def carrier_x(self, k: int, offset: float | None = None) -> float:
        """Scene x (before depth shear) of carrier k's centre."""
        return self.X0 + k * self.PITCH + (self.offset if offset is None else offset)

    def slot_x(self, k: int, j: int, offset: float | None = None) -> float:
        return self.carrier_x(k, offset) + (j - 1) * self.SLOT_DX

    def draw(self, d: ImageDraw.ImageDraw) -> None:
        P, y = self.P, self.DEPTH
        # legs
        for lx in range(60, W + 60, 120):
            a = P(lx - 3, y, self.BELT_Z - 14)
            b = P(lx + 3, y, -20)
            d.rectangle([a[0], a[1], b[0], b[1]], fill=(34, 40, 62))
        top = P(0, y, self.BELT_Z)[1]
        bot = P(0, y, self.BELT_Z - 30)[1]
        d.rectangle([0, top - 4 * SS, W * SS, top + 2 * SS], fill=STEEL)
        d.rectangle([0, top + 2 * SS, W * SS, bot], fill=(26, 32, 52))
        d.rectangle([0, bot, W * SS, bot + 6 * SS], fill=STEEL)
        # chevrons moving with the belt
        mid = (top + bot) / 2
        hh = (bot - top) * 0.3
        start = (self.offset % 40) - 40
        for cx in range(int(start), W + 40, 40):
            px = cx * SS
            d.line([px - 4 * SS, mid - hh, px + 4 * SS, mid, px - 4 * SS, mid + hh],
                   fill=(40, 140, 200), width=3 * SS)
        # carriers
        k0 = math.floor((-120 - self.X0 - self.offset) / self.PITCH)
        k1 = math.ceil((W + 120 - self.X0 - self.offset) / self.PITCH)
        for k in range(k0, k1 + 1):
            cx = self.carrier_x(k)
            a = P(cx - 38, y, self.BELT_Z + self.CARRIER_H + 12)
            b = P(cx + 38, y, self.BELT_Z)
            d.rounded_rectangle([a[0], a[1], b[0], b[1]], radius=3 * SS,
                                fill=(70, 78, 106), outline=(116, 126, 156), width=SS)
            for j in range(3):
                if self.filled(k, j):
                    draw_tube(d, P(self.slot_x(k, j), y, self.tube_z), TUBE_COLOURS[j])
                else:
                    sx, sy = P(self.slot_x(k, j), y, self.BELT_Z + 18)
                    d.ellipse([sx - 6 * SS, sy - 3 * SS, sx + 6 * SS, sy + 3 * SS],
                              fill=(20, 24, 40))


# --------------------------------------------------------------------------
# The robot arm
# --------------------------------------------------------------------------
@dataclass
class Pose:
    """Gripper target in the arm's cylindrical coordinates."""

    theta: float    # base yaw (rad), 0 = +x, +pi/2 = towards the conveyor
    r: float        # horizontal distance of the grip point from the pillar axis
    z: float        # height of the grip point

    @staticmethod
    def at(x: float, y: float, z: float) -> "Pose":
        return Pose(math.atan2(y, x), math.hypot(x, y), z)

    def blend(self, other: "Pose", t: float) -> "Pose":
        return Pose(lerp_angle(self.theta, other.theta, t),
                    lerp(self.r, other.r, t), lerp(self.z, other.z, t))


class RobotArm:
    PILLAR_R = 20
    PILLAR_H = 78
    SHOULDER_Z = 84
    L1 = 115         # upper arm
    L2 = 100         # forearm
    WRIST_DROP = 26  # wrist joint -> grip point (gripper hangs vertically)

    def __init__(self, index: int, base_x: float) -> None:
        self.index = index
        self.base_x = base_x
        self.P = Projector(base_x)
        self.pose = Pose(math.pi / 2, 120, 140)
        self.grip_open = 1.0          # 1 = open, 0 = closed on a tube
        self.holding: tuple | None = None

    # ----- kinematics -----------------------------------------------------
    def inverse(self, pose: Pose) -> tuple[float, float]:
        """Shoulder/elbow angles (in the arm plane) for a grip-point pose."""
        wr = pose.r
        wz = pose.z + self.WRIST_DROP - self.SHOULDER_Z
        dist = min(math.hypot(wr, wz), self.L1 + self.L2 - 1e-6)
        cos_e = (dist ** 2 - self.L1 ** 2 - self.L2 ** 2) / (2 * self.L1 * self.L2)
        elbow = -math.acos(max(-1.0, min(1.0, cos_e)))        # elbow-up solution
        shoulder = math.atan2(wz, wr) - math.atan2(
            self.L2 * math.sin(elbow), self.L1 + self.L2 * math.cos(elbow))
        return shoulder, elbow

    def forward(self) -> dict[str, tuple[float, float, float]]:
        """World positions of each joint, with the arm plane rotated by yaw."""
        th = self.pose.theta
        q1, q2 = self.inverse(self.pose)
        # positions in the arm plane: (r, z)
        plane = {
            "shoulder": (0.0, self.SHOULDER_Z),
            "elbow": (self.L1 * math.cos(q1), self.SHOULDER_Z + self.L1 * math.sin(q1)),
        }
        er, ez = plane["elbow"]
        plane["wrist"] = (er + self.L2 * math.cos(q1 + q2), ez + self.L2 * math.sin(q1 + q2))
        wr, wz = plane["wrist"]
        plane["grip"] = (wr, wz - self.WRIST_DROP)
        # rotate the plane about the pillar (z) axis by the base yaw
        c, s = math.cos(th), math.sin(th)
        return {k: (r * c, r * s, z) for k, (r, z) in plane.items()}

    # ----- drawing --------------------------------------------------------
    def _ring(self, z: float, radius: float) -> list[tuple[float, float]]:
        return [self.P(radius * math.cos(a), radius * math.sin(a), z)
                for a in (i * 2 * math.pi / 48 for i in range(48))]

    def _pillar(self, d) -> None:
        P, th = self.P, self.pose.theta
        d.polygon(self._ring(0, 40), fill=(205, 210, 222), outline=ARM_EDGE, width=SS)
        d.polygon(self._ring(4, 36), fill=(236, 239, 246))
        # cylinder body
        left, right = P(-self.PILLAR_R, 0, 6), P(self.PILLAR_R, 0, self.PILLAR_H)
        d.rectangle([left[0], right[1], right[0], left[1]], fill=ARM_WHITE)
        d.rectangle([right[0] - 9 * SS, right[1], right[0], left[1]], fill=ARM_SHADE)
        d.polygon(self._ring(6, self.PILLAR_R)[24:] + self._ring(6, self.PILLAR_R)[:1],
                  fill=ARM_WHITE)
        d.line([left[0], right[1], left[0], left[1]], fill=ARM_EDGE, width=SS)
        d.line([right[0], right[1], right[0], left[1]], fill=ARM_EDGE, width=SS)
        # rotating turret ring on top of the pillar
        d.polygon(self._ring(self.PILLAR_H, self.PILLAR_R + 3), fill=(214, 219, 230),
                  outline=ARM_EDGE, width=SS)
        # yaw indicator marks on the base plate and turret: they turn with the arm
        for z, rad, size in ((4, 34, 3.5), (self.PILLAR_H, self.PILLAR_R, 3)):
            mx, my = P(rad * math.cos(th), rad * math.sin(th), z)
            d.ellipse([mx - size * SS, my - size * SS, mx + size * SS, my + size * SS],
                      fill=ORANGE)
        mx, my = P(34 * math.cos(th + math.pi), 34 * math.sin(th + math.pi), 4)
        d.ellipse([mx - 2.5 * SS, my - 2.5 * SS, mx + 2.5 * SS, my + 2.5 * SS],
                  fill=HUD)

    # ----- hinges -----------------------------------------------------------
    # Every hinge pin (shoulder, elbow, wrist) is horizontal and perpendicular
    # to the arm plane, i.e. along (-sin(theta), cos(theta), 0).  The hinges are
    # drawn as real 3D cylinders / plates about that axis, so they turn with
    # the base yaw instead of always facing the camera.
    YOKE_HW = 14         # half-gap between the shoulder yoke cheek plates

    def _axes(self):
        th = self.pose.theta
        u = (math.cos(th), math.sin(th), 0.0)      # radial direction in the arm plane
        a = (-math.sin(th), math.cos(th), 0.0)     # hinge pin axis
        return u, a

    def _far_side(self) -> int:
        """+1 if the +axis end of a hinge points away from the viewer, else -1.
        VIEW_DIR is the world direction that projects to a single screen point."""
        _, a = self._axes()
        dot = a[0] * VIEW_DIR[0] + a[1] * VIEW_DIR[1]
        return 1 if dot >= 0 else -1

    def _disc(self, c, radius, offset=0.0, n=32):
        """Screen points of a circle lying in the arm plane, centred at world
        point c, shifted `offset` along the hinge axis."""
        u, a = self._axes()
        pts = []
        for i in range(n):
            ph = 2 * math.pi * i / n
            cr, cz = radius * math.cos(ph), radius * math.sin(ph)
            pts.append(self.P(c[0] + u[0] * cr + a[0] * offset,
                              c[1] + u[1] * cr + a[1] * offset,
                              c[2] + cz))
        return pts

    def _hinge(self, d, c, radius, half_width) -> None:
        """An orange hinge pin: a cylinder of `radius` along the hinge axis."""
        sign = self._far_side()
        far = self._disc(c, radius, sign * half_width)
        near = self._disc(c, radius, -sign * half_width)
        d.polygon(convex_hull(far + near), fill=(214, 112, 20), outline=ORANGE_DARK, width=SS)
        d.polygon(near, fill=ORANGE, outline=ORANGE_DARK, width=SS)
        d.polygon(self._disc(c, radius * 0.4, -sign * half_width), fill=(60, 34, 10))

    def _link(self, d, p0, p1, half_width, thick) -> None:
        """A 3D arm link from joint p0 to joint p1: a bar with rounded ends,
        `thick` deep in the arm plane and 2*half_width wide along the hinge
        axis, so it turns with the yaw just like the hinge pins."""
        sign = self._far_side()
        near = self._disc(p0, thick / 2, -sign * half_width, 24) + \
            self._disc(p1, thick / 2, -sign * half_width, 24)
        far = self._disc(p0, thick / 2, sign * half_width, 24) + \
            self._disc(p1, thick / 2, sign * half_width, 24)
        d.polygon(convex_hull(near + far), fill=LINK_TOP, outline=ARM_EDGE, width=SS)
        d.polygon(convex_hull(near), fill=ARM_WHITE, outline=ARM_EDGE, width=SS)

    def _yoke_cheek(self, d, side: int) -> None:
        """One shoulder bracket plate, standing on the turret in the arm plane."""
        u, a = self._axes()
        off = side * self.YOKE_HW
        sh = (0.0, 0.0, self.SHOULDER_Z)
        pts = self._disc(sh, 13, off)
        for r in (-15, 15):
            pts.append(self.P(u[0] * r + a[0] * off, u[1] * r + a[1] * off, self.PILLAR_H))
        d.polygon(convex_hull(pts), fill=ARM_SHADE, outline=ARM_EDGE, width=SS)

    def _gripper(self, d, joints) -> None:
        P, th = self.P, self.pose.theta
        wx, wy, wz = joints["wrist"]
        gx, gy, gz = joints["grip"]
        # fingers open/close along the arm direction (rotated by the yaw)
        u = (math.cos(th), math.sin(th))
        half = lerp(TUBE_W / 2 + 1, 11, self.grip_open)
        palm_z = wz - 10
        d.line([P(wx, wy, wz), P(wx, wy, palm_z)], fill=STEEL_DARK, width=6 * SS)
        pa = P(wx - u[0] * 13, wy - u[1] * 13, palm_z)
        pb = P(wx + u[0] * 13, wy + u[1] * 13, palm_z)
        d.line([pa, pb], fill=(170, 178, 196), width=6 * SS)
        if self.holding is not None:
            draw_tube(d, P(gx, gy, gz + 6 - TUBE_LEN), self.holding)
        for side in (-1, 1):
            fx, fy = gx + u[0] * half * side, gy + u[1] * half * side
            d.line([P(fx, fy, palm_z), P(fx, fy, gz - 4)], fill=(235, 238, 245),
                   width=3 * SS)
            d.line([P(fx, fy, palm_z), P(fx, fy, gz - 4)], fill=ARM_EDGE, width=SS)

    def draw(self, d: ImageDraw.ImageDraw) -> None:
        j = self.forward()
        far_side = self._far_side()
        self._pillar(d)
        self._yoke_cheek(d, far_side)            # cheek plate behind the arm
        self._link(d, j["shoulder"], j["elbow"], 10, 17)    # upper arm
        self._link(d, j["elbow"], j["wrist"], 8, 13)        # forearm
        self._gripper(d, j)
        self._yoke_cheek(d, -far_side)           # cheek plate in front of the arm
        self._hinge(d, j["shoulder"], 9, self.YOKE_HW + 3)
        self._hinge(d, j["elbow"], 8, 12)
        self._hinge(d, j["wrist"], 5, 9)


# --------------------------------------------------------------------------
# Motion planning
# --------------------------------------------------------------------------
# Every arm runs the same cycle of CYCLE frames, offset in time so that it
# meets a passing carrier.  The cycle (tau = frame within the cycle):
#
#   0-6    descend onto the next tube in the rack
#   6-9    close the gripper (tube lifted out of the rack at 9)
#   9-15   lift clear of the rack
#   15-29  swing round (base yaw) to above the moving carrier
#   29-35  descend into the carrier's empty slot, tracking the belt
#   35-38  open the gripper (tube left in the carrier at 38)
#   38-43  lift clear, still tracking the belt
#   43-58  swing back to above the next rack slot
#
# The rack slot that was emptied is refilled from underneath a little later.
# Everything is a pure function of the frame number, and after
# LOOP_CYCLES cycles the scene is exactly where it started, so the GIF loops.
LIFT = 46            # clearance above the tube tops when travelling
CYCLE = 58           # frames per pick-and-place cycle
PICK = 9             # tau at which the tube leaves the rack
RELEASE = 38         # tau at which the tube is left in the carrier
REFILL_DELAY = 18    # frames after a pick before the feeder starts to raise a tube
REFILL_TIME = 16     # frames for a new tube to rise into the slot
LOOP_CYCLES = 5      # one cycle per rack slot


class Cell:
    ARM_X = (160, 400, 640)

    def __init__(self) -> None:
        self.conveyor = Conveyor()
        self.arms = [RobotArm(i, x) for i, x in enumerate(self.ARM_X)]
        self.racks = [Rack(TUBE_COLOURS[i]) for i in range(3)]
        self.belt_speed = Conveyor.PITCH / CYCLE          # px per frame
        self.frame = 0
        self.font_lbl = load_font(13)
        self.font_title = load_font(14, bold=True)
        self.font_small = load_font(11)

        # Line the carriers up so arm 0 releases into slot 0 of carrier 0 at
        # frame RELEASE, then give each other arm the time offset (phase) at
        # which its own slot passes directly behind it.
        v = self.belt_speed
        self.conveyor.X0 = self.ARM_X[0] + Conveyor.SLOT_DX - v * RELEASE
        self.phase = []
        for j, x in enumerate(self.ARM_X):
            lag = (x - (j - 1) * Conveyor.SLOT_DX - self.conveyor.X0 - v * RELEASE) % Conveyor.PITCH
            self.phase.append(round(-lag / v) % CYCLE)
        self.conveyor.filled = self.carrier_filled

    # ---- timing helpers ----
    def release_frame(self, j: int, n: int) -> int:
        return n * CYCLE + RELEASE - self.phase[j]

    def target_carrier(self, j: int, n: int) -> int:
        """The carrier arm j drops its tube into during cycle n."""
        offset = self.belt_speed * self.release_frame(j, n)
        x = self.ARM_X[j] - (j - 1) * Conveyor.SLOT_DX - self.conveyor.X0 - offset
        return round(x / Conveyor.PITCH)

    def carrier_filled(self, k: int, j: int) -> bool:
        """Has arm j already put its tube into carrier k?"""
        v = self.belt_speed
        # frame at which carrier k's slot j is directly behind arm j
        f_exact = (self.ARM_X[j] - (j - 1) * Conveyor.SLOT_DX - self.conveyor.X0
                   - k * Conveyor.PITCH) / v
        n = round((f_exact - RELEASE + self.phase[j]) / CYCLE)
        return self.frame >= self.release_frame(j, n)

    def rack_fill(self, j: int, slot: int) -> float:
        """0 = empty, 1 = seated, in between = being raised by the feeder."""
        c = self.frame - PICK + self.phase[j]
        n = c // CYCLE                                   # latest cycle that has picked
        n -= (n - slot) % LOOP_CYCLES                    # latest one that used this slot
        since = self.frame - (n * CYCLE + PICK - self.phase[j])
        return max(0.0, min(1.0, (since - REFILL_DELAY) / REFILL_TIME))

    # ---- poses (in each arm's own frame) ----
    def rack_pose(self, j: int, slot: int, z_extra: float) -> Pose:
        x, y = self.racks[j].slot_xy(slot % LOOP_CYCLES)
        return Pose.at(x, y, Rack.TUBE_Z + TUBE_LEN - 6 + z_extra)

    def conveyor_pose(self, j: int, k: int, frame: float, z_extra: float) -> Pose:
        x = self.conveyor.slot_x(k, j, self.belt_speed * frame) - self.ARM_X[j]
        return Pose.at(x, Conveyor.DEPTH,
                       self.conveyor.tube_z + TUBE_LEN - 6 + z_extra)

    def update_arm(self, j: int) -> None:
        arm, f = self.arms[j], self.frame
        n, tau = divmod(f + self.phase[j], CYCLE)
        slot = n % LOOP_CYCLES
        k = self.target_carrier(j, n)
        R_up, R_dn = self.rack_pose(j, slot, LIFT), self.rack_pose(j, slot, 0)
        C_up, C_dn = self.conveyor_pose(j, k, f, LIFT), self.conveyor_pose(j, k, f, 0)

        def seg(t0, t1):
            return smooth((tau - t0) / (t1 - t0))

        if tau < 6:
            pose = R_up.blend(R_dn, seg(0, 6))
        elif tau < 9:
            pose = R_dn
        elif tau < 15:
            pose = R_dn.blend(R_up, seg(9, 15))
        elif tau < 29:
            pose = R_up.blend(C_up, seg(15, 29))
        elif tau < 35:
            pose = C_up.blend(C_dn, seg(29, 35))
        elif tau < 38:
            pose = C_dn
        elif tau < 43:
            pose = C_dn.blend(C_up, seg(38, 43))
        else:
            lifted = self.conveyor_pose(j, k, f - (tau - 43), LIFT)
            pose = lifted.blend(self.rack_pose(j, slot + 1, LIFT), seg(43, CYCLE))
        arm.pose = pose

        if tau < 6:
            arm.grip_open = 1.0
        elif tau < 9:
            arm.grip_open = 1.0 - seg(6, 9)
        elif tau < 35:
            arm.grip_open = 0.0
        elif tau < 38:
            arm.grip_open = seg(35, 38)
        else:
            arm.grip_open = 1.0
        arm.holding = TUBE_COLOURS[j] if PICK <= tau < RELEASE else None

    # ---- frame generation ----
    def render(self, frame: int) -> Image.Image:
        self.frame = frame
        self.conveyor.offset = self.belt_speed * frame
        for j, rack in enumerate(self.racks):
            rack.fill = [self.rack_fill(j, s) for s in range(len(rack.fill))]
            self.update_arm(j)

        img = Image.new("RGB", (W * SS, H * SS), BG)
        d = ImageDraw.Draw(img)
        self._background(d)
        self.conveyor.draw(d)
        self._bench(d)
        for arm, rack in zip(self.arms, self.racks):
            rack.draw(img, d, arm.P)
        for arm in self.arms:
            arm.draw(d)
        for arm, rack in zip(self.arms, self.racks):
            rack.draw_controller(d, arm.P)
        self._footer(d)
        return img.resize((W, H), Image.LANCZOS)

    def _background(self, d) -> None:
        for x in range(0, W, 40):
            d.line([x * SS, 0, x * SS, 250 * SS], fill=GRID, width=SS)
        for y in range(0, 250, 40):
            d.line([0, y * SS, W * SS, y * SS], fill=GRID, width=SS)
        loop_phase = 2 * math.pi * self.frame / (CYCLE * LOOP_CYCLES)
        for px, speed in ((22, 3), (648, 4)):
            d.rounded_rectangle([px * SS, 14 * SS, (px + 130) * SS, 78 * SS], radius=4 * SS,
                                outline=HUD, width=SS, fill=(14, 26, 44))
            for i, w in enumerate((60, 44, 50)):
                y = (22 + i * 6) * SS
                d.line([(px + 8) * SS, y, (px + 8 + w) * SS, y], fill=(80, 140, 190), width=2 * SS)
            pts = [((px + 8 + i) * SS,
                    (56 + 8 * math.sin(i / 14 + speed * loop_phase + px)) * SS)
                   for i in range(0, 115, 2)]
            d.line(pts, fill=HUD, width=SS)

    def _bench(self, d) -> None:
        d.polygon([(40 * SS, 250 * SS), (760 * SS, 250 * SS), (800 * SS, 358 * SS), (0, 358 * SS)],
                  fill=BENCH_TOP, outline=(90, 100, 140), width=SS)
        d.rectangle([0, 358 * SS, W * SS, 410 * SS], fill=BENCH_FRONT)
        d.line([0, 368 * SS, W * SS, 368 * SS], fill=HUD, width=2 * SS)
        for arm in self.arms:
            d.text((arm.base_x * SS, 390 * SS), f"arm[{arm.index}]", fill=HUD,
                   font=self.font_lbl, anchor="mm")

    def _footer(self, d) -> None:
        d.rectangle([0, 410 * SS, W * SS, H * SS], fill=(10, 13, 24))
        d.text((16 * SS, 425 * SS), "WSA032  ·  SAMPLE HANDLING CELL", fill=HUD,
               font=self.font_title)
        yaws = "  ".join(f"yaw[{a.index}]={math.degrees(a.pose.theta) % 360:3.0f}°" for a in self.arms)
        d.text((16 * SS, 449 * SS),
               f"3 robotic arms  ·  1 conveyor  ·  belt 0.20 m/s  ·  arms online: 3/3  ·  {yaws}",
               fill=(130, 140, 165), font=self.font_small)


def build_frames() -> list[Image.Image]:
    cell = Cell()
    return [cell.render(f) for f in range(CYCLE * LOOP_CYCLES)]


def main() -> None:
    frames = build_frames()
    pal = frames[0].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(OUTFILE, save_all=True, append_images=q[1:], duration=FRAME_MS,
              loop=0, optimize=True, disposal=1)
    print(f"wrote {OUTFILE}: {len(frames)} frames")


if __name__ == "__main__":
    main()
