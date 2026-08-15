"""Simulation bridge: run the existing record/HIL loop against a simulator.

The simulator lives in a separate process (and a separate Python environment),
so this package only owns the client side -- calibration-faithful conversion
between LeRobot motor values and joint angles, plus a `Robot`-shaped proxy.
"""

from evo_rlt.sim.calib import (
    ArmCalibration,
    BimanualCalibration,
    CalibrationError,
    MotorCalibration,
    BODY_JOINTS,
    GRIPPER_JOINT,
    MOTOR_NAMES,
    URDF_JOINT_LIMITS,
)

__all__ = [
    "ArmCalibration",
    "BimanualCalibration",
    "CalibrationError",
    "MotorCalibration",
    "BODY_JOINTS",
    "GRIPPER_JOINT",
    "MOTOR_NAMES",
    "URDF_JOINT_LIMITS",
]
