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
    "SimRobot",
    "SimRobotConfig",
]

#: LeRobot 的机器人注册表按"配置类名去掉 Config"去找实现类,而且只在
#: **配置类所在包**(``evo_rlt.sim``)和 ``evo_rlt.sim.simrobot`` 里找。类本身在
#: ``evo_rlt.sim.sim_robot``,不在这里导出的话 ``make_robot_from_config`` 会报
#: "Could not locate device class 'SimRobot'" —— 录制管线就没法用仿真当机器人。
#:
#: 用惰性导入而不是直接 import:``sim_robot`` 会拉进 zmq 和 lerobot,而
#: ``calib`` 这半边是仿真进程也要用的(那个环境里没有这两个包)。
def __getattr__(name: str):
    if name in ("SimRobot", "SimRobotConfig"):
        from evo_rlt.sim import sim_robot

        return getattr(sim_robot, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
