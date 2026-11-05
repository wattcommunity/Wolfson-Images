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
from dataclasses import dataclass, field

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


@dataclass
class Rack:
    """A 5-slot tube rack on the bench in front-left of an arm."""

    colour: tuple
    slots: list = field(default_factory=lambda: [True] * 5)
    depth: float = -130.0
    first_x: float = -95.0
    pitch: float = 20.0
    tube_z: float = 10.0          # z of a seated tube's bottom

    def slot_xy(self, k: int) -> tuple[float, float]:
        return self.first_x + k * self.pitch, self.depth

    def draw(self, d: ImageDraw.ImageDraw, P: Projector) -> None:
        x0, y = self.slot_xy(0)
        x1, _ = self.slot_xy(len(self.slots) - 1)
        a = P(x0 - 16, y, 14)
        b = P(x1 + 16, y, 0)
        d.rectangle([a[0], a[1], b[0], b[1]], fill=(70, 78, 108),
                    outline=(110, 120, 150), width=SS)
        for k, full in enumerate(self.slots):
            sx, sy = P(*self.slot_xy(k), self.tube_z)
            if full:
                draw_tube(d, (sx, sy), self.colour)
            else:
                d.ellipse([sx - 6 * SS, sy - 3 * SS, sx + 6 * SS, sy + 3 * SS],
                          fill=(20, 24, 40))
    def draw_controller(self, d: ImageDraw.ImageDraw, P: Projector) -> None:
        """Controller box in front-right of the arm (drawn after the arm,
        because it is nearer the viewer than the pillar)."""
        y = self.depth
        c0 = P(62, y, 60)
        c1 = P(88, y, 0)
        d.rounded_rectangle([c0[0], c0[1], c1[0], c1[1]], radius=3 * SS,
                            fill=(150, 156, 172), outline=(90, 96, 116), width=SS)
        d.rounded_rectangle([c0[0] + 5 * SS, c0[1] + 6 * SS, c1[0] - 5 * SS,
                             c1[1] - 10 * SS], radius=2 * SS, fill=(34, 38, 54))


class Conveyor:
    """An indexing belt carrying 3-slot tube carriers to the right."""

    DEPTH = 90.0
    BELT_Z = 81.0
    CARRIER_H = 8.0
    PITCH = 120.0               # carrier spacing (two per arm spacing)
    SLOT_DX = 20.0

    def __init__(self) -> None:
        self.offset = 0.0
        self.empty: dict[int, int] = {}   # carrier index -> empty slot

    @property
    def tube_z(self) -> float:
        return self.BELT_Z + self.CARRIER_H - 4

    def carrier_x(self, k: int) -> float:
        """World/screen-x (before shear) of carrier k's centre."""
        return 40.0 + k * self.PITCH + self.offset

    def carrier_at(self, x: float) -> int:
        return round((x - 40.0 - self.offset) / self.PITCH)

    def draw(self, d: ImageDraw.ImageDraw, P: Projector) -> None:
        y = self.DEPTH
        # legs
        for lx in range(60, W, 120):
            a = P(lx - 3, y, self.BELT_Z - 14)
            b = P(lx + 3, y, -20)
            d.rectangle([a[0], a[1], b[0], b[1]], fill=(34, 40, 62))
        top = P(-200, y, self.BELT_Z)[1]
        bot = P(-200, y, self.BELT_Z - 30)[1]
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
        # carriers (drawn left to right)
        k0 = self.carrier_at(-200)
        k1 = self.carrier_at(W + 200)
        for k in range(k0 - 1, k1 + 2):
            cx = self.carrier_x(k)
            a = P(cx - 38, y, self.BELT_Z + self.CARRIER_H + 12)
            b = P(cx + 38, y, self.BELT_Z)
            d.rounded_rectangle([a[0], a[1], b[0], b[1]], radius=3 * SS,
                                fill=(70, 78, 106), outline=(116, 126, 156), width=SS)
            for j in range(3):
                if self.empty.get(k) == j:
                    sx, sy = P(cx + (j - 1) * self.SLOT_DX, y, self.BELT_Z + 18)
                    d.ellipse([sx - 6 * SS, sy - 3 * SS, sx + 6 * SS, sy + 3 * SS],
                              fill=(20, 24, 40))
                    continue
                draw_tube(d, P(cx + (j - 1) * self.SLOT_DX, y, self.tube_z),
                          TUBE_COLOURS[j])


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

    def _thick(self, d, a, b, width, fill, edge) -> None:
        (ax, ay), (bx, by) = a, b
        L = math.hypot(bx - ax, by - ay) or 1.0
        nx, ny = -(by - ay) / L * width / 2, (bx - ax) / L * width / 2
        poly = [(ax + nx, ay + ny), (bx + nx, by + ny), (bx - nx, by - ny), (ax - nx, ay - ny)]
        d.polygon(poly, fill=fill, outline=edge, width=SS)
        for (cx, cy) in (a, b):
            d.ellipse([cx - width / 2, cy - width / 2, cx + width / 2, cy + width / 2],
                      fill=fill, outline=edge, width=SS)
        d.line([(ax + nx * 0.55, ay + ny * 0.55), (bx + nx * 0.55, by + ny * 0.55)],
               fill=(255, 255, 255), width=SS)

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
        S, E, Wr = (self.P(*j[k]) for k in ("shoulder", "elbow", "wrist"))
        far_side = self._far_side()
        self._pillar(d)
        self._yoke_cheek(d, far_side)            # cheek plate behind the arm
        self._thick(d, S, E, 17 * SS, ARM_WHITE, ARM_EDGE)
        self._thick(d, E, Wr, 13 * SS, ARM_WHITE, ARM_EDGE)
        self._gripper(d, j)
        self._yoke_cheek(d, -far_side)           # cheek plate in front of the arm
        self._hinge(d, j["shoulder"], 9, self.YOKE_HW + 3)
        self._hinge(d, j["elbow"], 8, 10)
        self._hinge(d, j["wrist"], 5, 7)


# --------------------------------------------------------------------------
# Motion planning (a simple timeline of keyframed moves + events)
# --------------------------------------------------------------------------
LIFT = 46            # clearance above the tube tops when travelling


class Cell:
    ARM_X = (160, 400, 640)

    def __init__(self) -> None:
        self.conveyor = Conveyor()
        self.arms = [RobotArm(i, x) for i, x in enumerate(self.ARM_X)]
        self.racks = [Rack(TUBE_COLOURS[i]) for i in range(3)]
        for r in self.racks:
            r.slots[0] = False                       # one free slot to start with
        self.frame = 0
        self.font_lbl = load_font(13)
        self.font_title = load_font(14, bold=True)
        self.font_small = load_font(11)

    # ---- targets in each arm's own world frame ----
    def conveyor_pose(self, arm: RobotArm, z_extra: float) -> Pose:
        k = self.conveyor.carrier_at(arm.base_x)
        x = self.conveyor.carrier_x(k) + (arm.index - 1) * Conveyor.SLOT_DX - arm.base_x
        return Pose.at(x, Conveyor.DEPTH, self.conveyor.tube_z + TUBE_LEN - 6 + z_extra)

    def rack_pose(self, arm: RobotArm, slot: int, z_extra: float) -> Pose:
        rack = self.racks[arm.index]
        x, y = rack.slot_xy(slot)
        return Pose.at(x, y, rack.tube_z + TUBE_LEN - 6 + z_extra)

    # ---- frame generation ----
    def render(self) -> Image.Image:
        img = Image.new("RGB", (W * SS, H * SS), BG)
        d = ImageDraw.Draw(img)
        self._background(d)
        self.conveyor.draw(d, self.arms[0].P)
        self._bench(d)
        for arm, rack in zip(self.arms, self.racks):
            rack.draw(d, arm.P)
        for arm in self.arms:
            arm.draw(d)
        for arm, rack in zip(self.arms, self.racks):
            rack.draw_controller(d, arm.P)
        self._footer(d)
        self.frame += 1
        return img.resize((W, H), Image.LANCZOS)

    def _background(self, d) -> None:
        for x in range(0, W, 40):
            d.line([x * SS, 0, x * SS, 250 * SS], fill=GRID, width=SS)
        for y in range(0, 250, 40):
            d.line([0, y * SS, W * SS, y * SS], fill=GRID, width=SS)
        for px in (22, 648):
            d.rounded_rectangle([px * SS, 14 * SS, (px + 130) * SS, 78 * SS], radius=4 * SS,
                                outline=HUD, width=SS, fill=(14, 26, 44))
            for i, w in enumerate((60, 44, 50)):
                y = (22 + i * 6) * SS
                d.line([(px + 8) * SS, y, (px + 8 + w) * SS, y], fill=(80, 140, 190), width=2 * SS)
            pts = [((px + 8 + i) * SS,
                    (56 + 8 * math.sin(i / 14 + self.frame * 0.15 + px)) * SS)
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
    frames: list[Image.Image] = []

    def move(steps: int, targets, grip=None, belt: float = 0.0):
        """Move every arm from its current pose to targets[i] over `steps` frames."""
        starts = [a.pose for a in cell.arms]
        g0 = [a.grip_open for a in cell.arms]
        off0 = cell.conveyor.offset
        for s in range(1, steps + 1):
            t = smooth(s / steps)
            for i, a in enumerate(cell.arms):
                if targets is not None:
                    a.pose = starts[i].blend(targets[i], t)
                if grip is not None:
                    a.grip_open = lerp(g0[i], grip, t)
            cell.conveyor.offset = off0 + belt * t
            frames.append(cell.render())

    def each(fn):
        return [fn(a) for a in cell.arms]

    def set_conveyor_slot(empty: bool):
        for a in cell.arms:
            k = cell.conveyor.carrier_at(a.base_x)
            if empty:
                cell.conveyor.empty[k] = a.index
            else:
                cell.conveyor.empty.pop(k, None)

    # start hovering over the conveyor
    for a in cell.arms:
        a.pose = cell.conveyor_pose(a, LIFT)

    # Two cycles: the free rack slot alternates 0 -> 4 -> 0, so the GIF loops.
    for gap, take in ((0, 4), (4, 0)):
        # --- pick the incoming tube off the carrier
        move(7, each(lambda a: cell.conveyor_pose(a, 0)))
        move(3, None, grip=0.0)
        set_conveyor_slot(True)
        for a in cell.arms:
            a.holding = TUBE_COLOURS[a.index]
        move(6, each(lambda a: cell.conveyor_pose(a, LIFT)))
        # --- swing round (base yaw) to the rack and place it in the free slot
        move(14, each(lambda a: cell.rack_pose(a, gap, LIFT)))
        move(6, each(lambda a: cell.rack_pose(a, gap, 0)))
        move(3, None, grip=1.0)
        for a, r in zip(cell.arms, cell.racks):
            a.holding = None
            r.slots[gap] = True
        move(5, each(lambda a: cell.rack_pose(a, gap, LIFT * 0.6)))
        # --- pick a processed tube from the other end of the rack
        move(8, each(lambda a: cell.rack_pose(a, take, LIFT * 0.6)))
        move(5, each(lambda a: cell.rack_pose(a, take, 0)))
        move(3, None, grip=0.0)
        for a, r in zip(cell.arms, cell.racks):
            a.holding = TUBE_COLOURS[a.index]
            r.slots[take] = False
        move(6, each(lambda a: cell.rack_pose(a, take, LIFT)))
        # --- swing back to the conveyor and drop it in the carrier
        move(14, each(lambda a: cell.conveyor_pose(a, LIFT)))
        move(7, each(lambda a: cell.conveyor_pose(a, 0)))
        move(3, None, grip=1.0)
        for a in cell.arms:
            a.holding = None
        set_conveyor_slot(False)
        move(6, each(lambda a: cell.conveyor_pose(a, LIFT)))
        # --- index the belt one carrier pitch while the arms hover clear
        move(12, None, belt=Conveyor.PITCH)
    return frames


def main() -> None:
    frames = build_frames()
    pal = frames[0].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(OUTFILE, save_all=True, append_images=q[1:], duration=FRAME_MS,
              loop=0, optimize=True, disposal=1)
    print(f"wrote {OUTFILE}: {len(frames)} frames")


if __name__ == "__main__":
    main()
