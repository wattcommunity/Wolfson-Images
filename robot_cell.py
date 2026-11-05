"""
robot_cell - a three-arm sample handling cell you control from Python.

You never choose or draw images yourself.  Change an arm's state and the
picture updates on its own:

    from robot_cell import RobotCell

    cell = RobotCell()
    arm = cell.arms[0]

    arm.raised = False         # lower the arm
    arm.gripper_open = False   # close the gripper on the tube
    arm.holding = True         # it now has the tube
    arm.raised = True          # lift it out
    arm.angle = 180            # turn to face the conveyor

    cell.finish()              # keep the window open until you close it

Each arm (cell.arms[0], cell.arms[1], cell.arms[2]) has four values:

    angle         0 = facing you (towards its rack)
                  180 = facing away (towards the conveyor)
    gripper_open  True or False
    holding       True or False  (is there a tube in the gripper?)
    raised        True = up, False = down

Things a real arm cannot do are shown with a warning in red: lifting a
tube with the gripper open (it falls), or turning while lowered (it would
hit the rack or the conveyor).  Opening the gripper while the arm is down
is how a tube is released into a slot.

You can also write your own arm class with methods such as pick() and
place(): subclass RobotArm and pass it in with RobotCell(arm_class=MyArm).
"""

from __future__ import annotations

import os
import time

from PIL import Image, ImageDraw, ImageFont

IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_cell_images")
STATE_NAMES = ("angle", "gripper_open", "holding", "raised")
FONT_PATHS = ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "DejaVuSansMono.ttf",
              "consola.ttf", "Menlo.ttc")

HUD = (56, 200, 240)
TEXT = (130, 140, 165)
WARN = (255, 90, 90)


class RobotError(Exception):
    """Raised when an arm is given a value it cannot take."""


class RobotArm:
    """One robot arm.  Changing any of its four values redraws the cell."""

    def __init__(self, cell: "RobotCell", index: int) -> None:
        self._cell = cell
        self._index = index
        self.angle = 0
        self.gripper_open = True
        self.holding = False
        self.raised = True
        self._ready = True          # from now on, every change redraws the cell

    def __setattr__(self, name: str, value) -> None:
        if name not in STATE_NAMES:
            object.__setattr__(self, name, value)
            return
        value = self._check(name, value)
        old = self.__dict__.get(name)
        object.__setattr__(self, name, value)
        if self.__dict__.get("_ready") and value != old:
            warnings = []
            if name == "angle" and not self.raised:
                warnings.append(f"arm[{self._index}] turned while lowered: "
                                "it would hit the rack or the conveyor")
            self._cell._refresh(f"arm[{self._index}].{name} = {value!r}", warnings)

    def _check(self, name: str, value):
        if name == "angle":
            if value not in (0, 180) or isinstance(value, bool):
                raise RobotError(f"arm[{self._index}].angle must be 0 (facing you) "
                                 f"or 180 (facing away), not {value!r}")
            return int(value)
        if not isinstance(value, bool):
            raise RobotError(f"arm[{self._index}].{name} must be True or False, "
                             f"not {value!r}")
        return value

    @property
    def index(self) -> int:
        return self._index

    def problems(self) -> list[str]:
        """Things about the current state that a real arm could not do."""
        if self.holding and self.gripper_open and self.raised:
            return [f"arm[{self._index}] is up and holding a tube with the gripper "
                    "open: the tube would fall"]
        return []

    def describe(self) -> str:
        return "{}°  {}  {}  {}".format(
            self.angle, "open" if self.gripper_open else "closed",
            "holding" if self.holding else "empty", "up" if self.raised else "down")

    def __repr__(self) -> str:
        return (f"RobotArm({self._index}, angle={self.angle}, gripper_open="
                f"{self.gripper_open}, holding={self.holding}, raised={self.raised})")


class RobotCell:
    """The whole cell: three arms, a conveyor, and the window that shows them.

    delay      seconds to pause after each change, so you can watch each step
    window     False to run without a window (for example to only save a GIF)
    record     True to keep every picture, for save_gif()
    arm_class  your own subclass of RobotArm, if you have one
    """

    def __init__(self, delay: float = 0.5, window: bool = True, record: bool = False,
                 arm_class: type = RobotArm, verbose: bool = True) -> None:
        self.delay = delay
        self.verbose = verbose
        self._frames: list[Image.Image] = [] if record else None
        self._background = Image.open(os.path.join(IMAGE_DIR, "background.png")).convert("RGB")
        self._sprites: dict[str, tuple[Image.Image, tuple[int, int]]] = {}
        self._font = self._load_font(11)
        self._font_label = self._load_font(13)
        self._font_title = self._load_font(14)
        self._tk = None
        self._closed = False
        if window:
            self._open_window()
        self.arms = tuple(arm_class(self, i) for i in range(3))
        self._refresh("start", [])

    # ---- public ----
    def finish(self) -> None:
        """Keep the window open until it is closed."""
        if self._tk is not None and not self._closed:
            self._tk.mainloop()

    def save_gif(self, path: str, frame_ms: int | None = None) -> None:
        """Save every picture shown so far as an animated GIF (needs record=True)."""
        if not self._frames:
            raise RobotError("nothing recorded: create the cell with RobotCell(record=True)")
        ms = frame_ms or max(60, int(self.delay * 1000))
        self._frames[0].save(path, save_all=True, append_images=self._frames[1:],
                             duration=ms, loop=0)

    def snapshot(self) -> Image.Image:
        """The picture for the current state, as a Pillow image."""
        return self._compose("", [])

    # ---- drawing ----
    @staticmethod
    def _load_font(size: int):
        for p in FONT_PATHS:
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _sprite(self, arm: RobotArm):
        name = "arm{}_{}_{}_{}_{}.png".format(
            arm.index, arm.angle, "open" if arm.gripper_open else "closed",
            "holding" if arm.holding else "empty", "up" if arm.raised else "down")
        if name not in self._sprites:
            img = Image.open(os.path.join(IMAGE_DIR, name))
            x, y = (int(v) for v in img.info.get("offset", "0,0").split(","))
            self._sprites[name] = (img.convert("RGBA"), (x, y))
        return self._sprites[name]

    def _compose(self, action: str, warnings: list[str]) -> Image.Image:
        img = self._background.copy()
        for arm in self.arms:
            sprite, offset = self._sprite(arm)
            img.paste(sprite, offset, sprite)
        d = ImageDraw.Draw(img)
        for arm, x in zip(self.arms, (160, 400, 640)):
            colour = WARN if arm.problems() else HUD
            d.text((x, 381), f"arm[{arm.index}]", fill=colour, font=self._font_label, anchor="mm")
            d.text((x, 398), arm.describe(), fill=TEXT, font=self._font, anchor="mm")
        d.text((16, 418), "WSA032  ·  SAMPLE HANDLING CELL", fill=HUD, font=self._font_title)
        problems = warnings + [p for arm in self.arms for p in arm.problems()]
        if problems:
            d.text((16, 442), "! " + problems[0], fill=WARN, font=self._font)
        elif action:
            d.text((16, 442), action, fill=TEXT, font=self._font)
        if len(problems) > 1:
            d.text((16, 459), "! " + problems[1], fill=WARN, font=self._font)
        return img

    def _refresh(self, action: str, warnings: list[str]) -> None:
        if self._closed:
            raise SystemExit("robot cell window was closed")
        problems = warnings + [p for arm in self.arms for p in arm.problems()]
        if self.verbose and action != "start":
            print(action)
            for p in problems:
                print("  WARNING:", p)
        img = self._compose(action, warnings)
        if self._frames is not None:
            self._frames.append(img)
        if self._tk is not None:
            self._show(img)
            end = time.monotonic() + self.delay
            while time.monotonic() < end and not self._closed:
                self._tk.update()
                time.sleep(0.01)

    # ---- window ----
    def _open_window(self) -> None:
        try:
            import tkinter as tk
            from PIL import ImageTk
            self._tk = tk.Tk()
        except Exception as exc:                     # no tkinter, or no display
            print(f"robot_cell: cannot open a window ({exc}); running without one")
            self._tk = None
            return
        self._ImageTk = ImageTk
        self._tk.title("Robot cell")
        self._tk.resizable(False, False)
        self._label = tk.Label(self._tk, borderwidth=0)
        self._label.pack()
        self._tk.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self) -> None:
        self._closed = True
        self._tk.destroy()

    def _show(self, img: Image.Image) -> None:
        photo = self._ImageTk.PhotoImage(img)
        self._label.configure(image=photo)
        self._label.image = photo                    # keep a reference
        self._tk.update()
