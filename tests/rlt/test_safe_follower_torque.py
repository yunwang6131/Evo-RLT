from __future__ import annotations

from types import SimpleNamespace

from evo_rlt.adapters.lerobot.record.common import install_safe_follower_torque_enable


class _FakeBus:
    def __init__(self, present: dict[str, int]) -> None:
        self.present = present
        self.goal = {name: 0 for name in present}
        self.events: list[tuple] = []

    def sync_read(self, data_name, motors=None, normalize=True, num_retry=0):
        self.events.append(("read", data_name))
        values = self.present if data_name == "Present_Position" else self.goal
        return values.copy()

    def sync_write(self, data_name, values, normalize=True, num_retry=0):
        self.events.append(("write", data_name, values.copy()))
        self.goal = values.copy()

    def enable_torque(self, motors=None, num_retry=0):
        self.events.append(("enable", motors, num_retry))


def test_primes_goal_from_present_before_enabling_single_follower() -> None:
    bus = _FakeBus({"joint": 2048, "gripper": 1800})
    robot = SimpleNamespace(bus=bus)
    install_safe_follower_torque_enable(robot)

    bus.enable_torque(num_retry=2)

    assert bus.goal == bus.present
    assert bus.events == [
        ("read", "Present_Position"),
        ("write", "Goal_Position", bus.present),
        ("read", "Goal_Position"),
        ("enable", None, 2),
    ]


def test_installs_on_both_arms_and_is_idempotent() -> None:
    left_bus = _FakeBus({"joint": 1000})
    right_bus = _FakeBus({"joint": 3000})
    robot = SimpleNamespace(
        left_arm=SimpleNamespace(bus=left_bus),
        right_arm=SimpleNamespace(bus=right_bus),
    )
    install_safe_follower_torque_enable(robot)
    first_left_wrapper = left_bus.enable_torque
    install_safe_follower_torque_enable(robot)

    assert left_bus.enable_torque is first_left_wrapper
    left_bus.enable_torque()
    right_bus.enable_torque()
    assert left_bus.goal == left_bus.present
    assert right_bus.goal == right_bus.present
