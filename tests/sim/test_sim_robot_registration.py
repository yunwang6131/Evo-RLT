"""仿真机器人必须能被 LeRobot 的注册表找到。

盯的是这个具体故障:``record_full --robot.type=sim_bi_so_follower`` 报
"invalid choice"。record 脚本的 ``--robot.type`` 选项列表是从注册表**现场**
构建的,而注册发生在 import ``evo_rlt.sim.sim_robot`` 时 —— 没人 import 过就
不在列表里,而报错信息只说"choose from ...",完全看不出是导入问题。
"""

from __future__ import annotations

import pytest

pytest.importorskip("lerobot")


def test_register_puts_the_sim_robot_in_lerobot_choices():
    from lerobot.robots.config import RobotConfig

    from evo_rlt.adapters.lerobot import register

    register()
    assert "sim_bi_so_follower" in RobotConfig._choice_registry


def test_lazy_export_alone_is_not_enough():
    """``evo_rlt.sim`` 的惰性导出只在 getattr 时触发。

    argparse 建选项列表时不会去 getattr,所以注册必须在 ``register()`` 里
    显式 import —— 这条钉住"别把 registry.py 里那行 import 删了当无用代码"。
    """
    import inspect

    from evo_rlt.adapters.lerobot import registry

    src = inspect.getsource(registry.register)
    assert "evo_rlt.sim.sim_robot" in src, (
        "register() 不再 import sim_robot —— --robot.type=sim_bi_so_follower 会失效"
    )


def test_registry_resolves_the_config_to_the_class():
    """注册表按'配置类名去掉 Config'找实现类,只在配置类所在包里找。"""
    from lerobot.robots import make_robot_from_config

    from evo_rlt.adapters.lerobot import register
    from evo_rlt.sim.sim_robot import SimRobotConfig

    register()
    robot = make_robot_from_config(SimRobotConfig())
    assert type(robot).__name__ == "SimRobot"
    # 相机键要和真机录制的一致,否则数据集的图像列对不上
    assert set(robot.cameras) == {"left_wrist", "right_wrist", "right_front"}
    assert len(robot.action_features) == 12
