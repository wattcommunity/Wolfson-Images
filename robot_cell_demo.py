"""
Example of a student program for robot_cell.

The student only changes the arm's state; robot_cell keeps the picture
up to date on its own.

Run:  python robot_cell_demo.py
"""

from robot_cell import RobotArm, RobotCell


class Picker(RobotArm):
    """An arm that knows how to move a tube from its rack to the conveyor."""

    def pick_from_rack(self):
        self.angle = 0              # face the rack
        self.raised = False         # go down to the tube
        self.gripper_open = False   # grab it
        self.holding = True
        self.raised = True          # lift it out

    def place_on_conveyor(self):
        self.angle = 180            # face the conveyor
        self.raised = False         # lower it into the carrier
        self.gripper_open = True    # release the tube into the carrier
        self.holding = False        # the arm no longer has it
        self.raised = True          # move clear


cell = RobotCell(delay=0.5, arm_class=Picker)

for arm in cell.arms:
    arm.gripper_open = True         # make sure every arm starts open
for arm in cell.arms:
    arm.pick_from_rack()
    arm.place_on_conveyor()

cell.finish()
