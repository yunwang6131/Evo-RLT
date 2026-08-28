from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

DEFAULT_SETUP_PATH = Path.home() / ".roboclaw/workspace/embodied/manifest.json"

#: 采集时写进数据集的任务描述(``--task`` 的默认值)。pi0.5 是语言条件的,这句话
#: 就是 prompt —— 推理时必须和训练时一致,所以定义在这里一份,四个子命令共用。
#: **刻意不带颜色词**:螺栓的颜色 0821 从灰改成了饱和蓝(理由见
#: ``task_scene.json`` 里 ``bolt._color_note`` 那段实测),而 0821 之前采的 37 条
#: episode 里它还是灰的 —— 不写颜色是唯一同时兼容两批数据的描述。而且螺套还有
#: 个红色端面嵌片,prompt 里写 red 会指到它身上去。
#: 描述**到插入为止**:仿真场景里没有目标放置区(``task_scene.json`` 只有
#: table/socket/bolt),实采的动作序列也是插完两手一松就结束(37 条 episode 的
#: 末帧夹爪值确认过)。prompt 里多一句没有对应动作的子目标,只会稀释语言条件。
DEFAULT_TASK = (
    "Pick up the hexagonal part with the right arm, pull the pin out of the platform "
    "with the left arm, align the pin with the hole in the hexagonal part, and insert "
    "the pin into the hole."
)

#: manifest 里的相对路径按仓库根解析,而不是按 cwd —— 这样 manifest 可以写
#: ``configs/calibration/...`` 或 ``data/bimanual`` 而不用写死 /home/xxx,
#: 换台机器也能用。
_REPO_ROOT = Path(__file__).resolve().parents[5]

#: 采集直接落到仓库内的训练目录。以前先写 ~/lerobot_data 暂存区、确认没问题再
#: 手动 ``cp -r`` 进 data/ 并 ``rm -rf`` 暂存区 —— 那一步只是把几十 G 视频搬来
#: 搬去,数据本身一个字节都不变。``data/`` 在 .gitignore 里,不会进 git。
DEFAULT_DATASET_ROOT = _REPO_ROOT / "data" / "bimanual"


def load_setup_json(path: str | None = None) -> dict[str, Any]:
    setup_path = Path(path).expanduser() if path else DEFAULT_SETUP_PATH
    with open(setup_path) as fh:
        return json.load(fh)


def _resolve_repo_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (_REPO_ROOT / path)


def resolve_dataset_root(setup: dict[str, Any]) -> Path:
    dataset_root = setup.get("datasets", {}).get("root", "")
    return _resolve_repo_path(dataset_root) if dataset_root else DEFAULT_DATASET_ROOT


def get_sorted_followers(setup: dict[str, Any]) -> list[dict[str, Any]]:
    followers = [arm for arm in setup["arms"] if "follower" in arm["type"]]
    followers.sort(key=lambda arm: 0 if "left" in arm.get("alias", "") else 1)
    return followers


def get_sorted_leaders(setup: dict[str, Any]) -> list[dict[str, Any]]:
    leaders = [arm for arm in setup["arms"] if "leader" in arm["type"]]
    leaders.sort(key=lambda arm: 0 if "left" in arm.get("alias", "") else 1)
    return leaders

log = logging.getLogger(__name__)

CAMERA_RENAME = {"left_wrist": "wrist", "right_wrist": "wrist", "right_front": "front"}
LEFT_CAMERA_ALIASES = {"left_wrist"}
RIGHT_CAMERA_ALIASES = {"right_wrist", "right_front"}
TELEOP_ID = "bimanual_leader"
# Single-arm (so101_follower / so101_leader) ids. Kept separate from the
# bimanual ids above so calibration staging never collides between modes.
FOLLOWER_ID_SINGLE = "so_follower"
TELEOP_ID_SINGLE = "so_leader"


def install_safe_follower_torque_enable(robot: Any) -> None:
    """Make follower torque-enable hold the measured pose, not a stale goal.

    STS3215 ``Goal_Position`` is RAM-backed and can read as zero after a
    power cycle while the arm is physically near the middle of its range.
    LeRobot's SOFollower.configure() enables torque before the first normal
    action is sent.  If that stale zero remains, all joints immediately try
    to chase it; the resulting inrush can trip the servo's input-voltage or
    overload protection while ``enable_torque()`` is writing the following
    ``Lock=1`` register.

    Wrap each follower bus instance so every future disabled->enabled
    transition first copies raw Present_Position to raw Goal_Position and
    verifies the write.  This is installed before connect/configure, and is
    deliberately follower-only: leader arms remain torque-disabled.
    """

    arms = [
        arm
        for arm in (getattr(robot, "left_arm", None), getattr(robot, "right_arm", None))
        if arm is not None
    ]
    if not arms and hasattr(robot, "bus"):
        arms = [robot]

    for arm in arms:
        bus = getattr(arm, "bus", None)
        if bus is None or getattr(bus, "_evo_rlt_safe_torque_enable", False):
            continue
        original_enable_torque = bus.enable_torque

        def safe_enable_torque(
            motors=None,
            num_retry: int = 0,
            *,
            _bus=bus,
            _original=original_enable_torque,
        ) -> None:
            present = _bus.sync_read(
                "Present_Position", motors=motors, normalize=False, num_retry=num_retry
            )
            _bus.sync_write(
                "Goal_Position", present, normalize=False, num_retry=num_retry
            )
            written = _bus.sync_read(
                "Goal_Position", motors=motors, normalize=False, num_retry=num_retry
            )
            if written != present:
                raise RuntimeError(
                    "Refusing to enable follower torque: failed to synchronize "
                    f"Goal_Position to Present_Position (present={present}, goal={written})"
                )
            log.info("Primed follower Goal_Position from Present_Position before torque enable")
            _original(motors, num_retry=num_retry)

        bus.enable_torque = safe_enable_torque
        bus._evo_rlt_safe_torque_enable = True


#: 冻结/解冻一条臂的按键。单人采双臂数据时,一只手当夹具、另一只手操作。
ARM_FREEZE_KEY = "p"

#: 解冻时从"冻结位姿"过渡回"主臂当前位姿"的时间(秒)。
#: **不能为 0**:冻结期间操作者的手一直在动,解冻那一刻两者可能差几十度,
#: 直接切过去从臂会猛窜一下 —— 既危险,也会在数据里留下一个非物理的跳变。
ARM_UNFREEZE_BLEND_S = 0.6


class ArmFreeze:
    """把一条臂钉在按下按键那一刻的位姿上,另一条臂照常跟随主臂。

    为什么要它:双臂任务单人采数据时,一只手不够用。冻结一条臂当夹具(比如
    举着螺套),就能腾出手专心操作另一条。

    **对数据的影响**:动作流仍是完整的 12 维,冻结那条臂的值是常数 —— 这是
    合法的动作序列,策略会学到"这条臂稳住不动"。但要清楚代价:这样采的数据里
    **不存在"两臂同时微调"的样本**,需要真正双臂协同的动作学不出来。
    顺序式任务(一只手固定、另一只手对准)则完全合适。

    RLT 那边本来就只对左臂做 RL(``rl_action_arms=left``,右臂掩到冻结的 VLA
    参考),所以冻结右臂和现有架构是一致的。

    三个状态:跟随 / 冻结 / 解冻过渡。过渡这一段不能省 —— 冻结期间操作者的手
    一直在动,解冻那一刻主臂可能已经和从臂差几十度,直接交回去从臂会猛窜一下。
    """

    FOLLOW, FROZEN, BLENDING = "follow", "frozen", "blending"

    def __init__(self, side: str = "right", blend_s: float = ARM_UNFREEZE_BLEND_S,
                 fps: float = 30.0, leader_lock: Any | None = None) -> None:
        #: `sim.feedback.LeaderLock`。给了就在冻结时把对应的**主臂**也锁住,
        #: 这样操作者的手带不走它,解冻时两者本来就一致,平滑过渡只是兜底。
        #: 不给也能用,只是解冻要靠过渡把差值滑掉。
        self.leader_lock = leader_lock
        self.side = side
        self.prefix = f"{side}_"
        self._blend_steps = max(1, int(round(blend_s * fps)))
        self._state = self.FOLLOW
        self._held: dict[str, float] = {}
        self._blend_left = 0
        #: 统计,跑完打印。全 0 说明这个功能没被用上,和"用了但没效果"要分得开。
        self.frozen_frames = 0
        self.toggles = 0
        #: 上次加锁时各舵机读回的力矩上限,排查"主臂锁不住"用。
        self._lock_readback: dict[str, int] = {}

    @property
    def frozen(self) -> bool:
        return self._state == self.FROZEN

    @property
    def state(self) -> str:
        return self._state

    def toggle(self, action: dict[str, Any]) -> str:
        """按键时调用,返回切换后的状态。过渡途中再按会重新冻结在当前位姿。"""
        self.toggles += 1
        if self._state == self.FROZEN:
            self._state = self.BLENDING
            self._blend_left = self._blend_steps
            if self.leader_lock is not None:
                self.leader_lock.unlock()
        else:
            self._held = {k: float(v) for k, v in action.items()
                          if k.startswith(self.prefix)}
            self._state = self.FROZEN
            self._blend_left = 0
            if self.leader_lock is not None:
                # 锁在**读到这条指令的位姿**上,和 _held 是同一时刻,两者不会错开
                self._lock_readback = self.leader_lock.lock()
        return self._state

    def apply(self, action: dict[str, Any]) -> dict[str, Any]:
        """每帧调用,返回实际要下发的动作。跟随状态下原样返回,不复制。"""
        if self._state == self.FOLLOW:
            return action
        out = dict(action)
        if self._state == self.FROZEN:
            for key, value in self._held.items():
                if key in out:
                    out[key] = value
            self.frozen_frames += 1
            return out
        # BLENDING:从冻结位姿线性滑回主臂位姿
        alpha = 1.0 - self._blend_left / self._blend_steps
        for key, value in self._held.items():
            if key in out:
                out[key] = value + (float(action[key]) - value) * alpha
        self._blend_left -= 1
        if self._blend_left <= 0:
            self._state = self.FOLLOW
            self._held = {}
        return out

    def release(self) -> None:
        """退出时调用:确保主臂不会带着力矩留在原地。"""
        if self.leader_lock is not None:
            self.leader_lock.unlock()

    def summary(self) -> str:
        if not self.toggles:
            return f"  {self.side} 臂冻结: 没用过"
        return (f"  {self.side} 臂冻结: 切换 {self.toggles} 次, "
                f"冻结 {self.frozen_frames} 帧")


@dataclass(frozen=True)
class RobotSetup:
    setup: dict[str, Any]
    followers: list[dict[str, Any]]
    leaders: list[dict[str, Any]]
    left_cameras: dict[str, Any]
    right_cameras: dict[str, Any]


@dataclass(frozen=True)
class RunPaths:
    dataset_name: str
    dataset_root: Path
    day_dir: Path
    log_file: Path


def verify_manifest_ports(setup: dict[str, Any]) -> None:
    """核对 manifest 里每条臂的端口确实是它自己的那块转接板。

    四条臂的电机 ID 都是 1~6,认错了**不会报错** —— 只会把一条臂的标定套到
    另一条身上,表现成"标定一团乱"。曾经真的发生过:manifest 写死 ttyACM 号,
    而左右主臂是反的(说 left=ttyACM1,实际 ttyACM1 是右主臂 5AAF220248),
    于是走录制管线和走 diagnostics/teleop_sim.py 拿到的标定不是同一份。

    判据是端口解析出来的**序列号**,不是 ttyACM 序号 —— 后者按插入顺序分配,
    重启或换插口就变。序列号是转接板固有的。
    """
    import os

    expected = {}
    arms_json = _REPO_ROOT / "configs" / "arms.json"
    if arms_json.is_file():
        expected = {
            alias: spec.get("serial")
            for alias, spec in json.loads(arms_json.read_text()).get("arms", {}).items()
        }
    if not expected:
        return

    problems = []
    for arm in setup.get("arms", []):
        alias, port = arm.get("alias"), arm.get("port")
        want = expected.get(alias)
        if not (alias and port and want):
            continue
        real = os.path.realpath(port)
        # by-id 路径里带序列号;裸 ttyACM 号则要反查 by-id 才知道是谁
        found = want in port
        if not found:
            by_id = Path("/dev/serial/by-id")
            if by_id.is_dir():
                for link in by_id.iterdir():
                    if os.path.realpath(link) == real:
                        found = want in link.name
                        break
                else:
                    continue          # 设备不在线,没法核对,交给连接那一步报错
            else:
                continue
        if not found:
            problems.append(f"  {alias}: manifest 给的 {port} 不是序列号 {want} 那块板")
    if problems:
        raise ValueError(
            "manifest 的端口和 configs/arms.json 的序列号对不上:\n"
            + "\n".join(problems)
            + "\n  四条臂的电机 ID 都一样,接错不会报错,只会把标定套到别的臂上。"
            "\n  端口建议直接写 /dev/serial/by-id/... 的稳定路径。"
        )


def load_robot_setup(setup_json: str | None, *, sim: bool = False) -> RobotSetup:
    """读 setup manifest。``sim=True`` 时机器人是仿真,不需要真机 follower。

    仿真模式下 follower 那一侧完全不接真臂:``SimRobot`` 自己从
    ``configs/calibration/robots/`` 读标定(见 ``SimRobotConfig.calibration_source_dir``),
    不走这里 stage 出来的目录,也没有串口可解析。leader 仍然是真的 —— 人用主臂
    遥操仿真,这正是要采的数据。
    """
    setup = load_setup_json(setup_json)
    verify_manifest_ports(setup)
    followers = get_sorted_followers(setup)
    leaders = get_sorted_leaders(setup)
    if sim:
        left_cameras, right_cameras = build_camera_configs(setup.get("cameras", []))
        return RobotSetup(setup, [], leaders, left_cameras, right_cameras)
    if len(followers) == 1:
        # Single-arm: cameras are not split left/right, so they are all
        # stored under `left_cameras`; `right_cameras` stays empty and
        # downstream single-arm branches never read it.
        cameras = build_single_arm_camera_config(setup.get("cameras", []))
        return RobotSetup(setup, followers, leaders, cameras, {})
    if len(followers) < 2:
        raise ValueError(
            f"Need 1 follower arm (single-arm) or 2 follower arms (bimanual), got {len(followers)}"
        )
    left_cameras, right_cameras = build_camera_configs(setup.get("cameras", []))
    return RobotSetup(setup, followers, leaders, left_cameras, right_cameras)


def build_single_arm_camera_config(cameras: list[dict[str, Any]]) -> dict[str, Any]:
    """Single-arm cameras keep their manifest alias as-is (no left_/right_ split)."""
    out: dict[str, Any] = {}
    for camera in cameras:
        alias = camera["alias"]
        camera_config: dict[str, Any] = {
            "type": "opencv",
            "index_or_path": camera["port"],
            "width": camera.get("width", 640),
            "height": camera.get("height", 480),
            "fps": camera.get("fps", 30),
        }
        if camera.get("fourcc"):
            camera_config["fourcc"] = camera["fourcc"]
        out[alias] = camera_config
    return out


def build_camera_configs(cameras: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    left_cameras: dict[str, Any] = {}
    right_cameras: dict[str, Any] = {}
    for camera in cameras:
        alias = camera["alias"]
        camera_config: dict[str, Any] = {
            "type": "opencv",
            "index_or_path": camera["port"],
            "width": camera.get("width", 640),
            "height": camera.get("height", 480),
            "fps": camera.get("fps", 30),
        }
        if camera.get("fourcc"):
            camera_config["fourcc"] = camera["fourcc"]
        target_name = CAMERA_RENAME.get(alias, alias)
        if alias in LEFT_CAMERA_ALIASES:
            left_cameras[target_name] = camera_config
        elif alias in RIGHT_CAMERA_ALIASES:
            right_cameras[target_name] = camera_config
    return left_cameras, right_cameras


def resolve_run_paths(setup: dict[str, Any], dataset_tag: str, dataset_prefix: str) -> RunPaths:
    now = datetime.now()
    date_folder = f"{now:%m%d}_{dataset_tag}"
    dataset_leaf = f"{dataset_prefix}_{now:%H%M%S}"
    day_dir = resolve_dataset_root(setup) / date_folder
    dataset_root = day_dir / dataset_leaf
    return RunPaths(
        dataset_name=f"local/{dataset_leaf}",
        dataset_root=dataset_root,
        day_dir=day_dir,
        log_file=day_dir / f"{dataset_leaf}.log",
    )


def configure_logging(log_file: Path, log_level: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)


def remove_existing_dataset(dataset_root: Path) -> None:
    if dataset_root.exists():
        log.info("Removing existing dataset dir: %s", dataset_root)
        shutil.rmtree(dataset_root)


def stage_arm_calibration(arm: dict[str, Any], dst: Path) -> None:
    """把 manifest 指定的标定文件拷到 LeRobot 认的位置和文件名。

    找不到就**报错**,不是警告。LeRobot 拿不到标定文件不会停 —— 它会当场
    进入"请把手臂推到行程两端"的重标流程,把整个采集会话废掉,而且录出来的
    新标定通常是错的(操作者没料到要标,随手动两下就回车)。这个后果比一条
    滚过去的 warning 严重得多。
    """
    calibration_file = arm.get("calibration_file")
    if calibration_file:
        src = _resolve_repo_path(calibration_file)
    else:
        serial = Path(arm["calibration_dir"]).name
        src = _resolve_repo_path(arm["calibration_dir"]) / f"{serial}.json"
    if src.exists():
        shutil.copy2(src, dst)
        log.info("Calibration staged: %s -> %s", src, dst)
        return
    raise FileNotFoundError(
        f"{arm.get('alias', '?')} 的标定文件不存在: {src}\n"
        f"  (来自 manifest 的 calibration_file/calibration_dir)\n"
        f"  本项目的标定在 configs/calibration/ 下,由 diagnostics/calibration.py 写入;\n"
        f"  manifest 里若指向 ~/.cache/huggingface/... 那是另一份,可能从没标过。\n"
        f"  用 diagnostics/calibration.py --status 看当前各臂用的是哪个文件。"
    )


def stage_follower_calibrations(followers: list[dict[str, Any]], cal_dir: str) -> None:
    if not followers:
        # 仿真模式:没有真机 follower 可 stage。SimRobot 自己去
        # configs/calibration/robots/ 读,不看这个目录。
        return
    if len(followers) == 1:
        stage_arm_calibration(followers[0], Path(cal_dir) / f"{FOLLOWER_ID_SINGLE}.json")
        return
    for side, arm in (("left", followers[0]), ("right", followers[1])):
        stage_arm_calibration(arm, Path(cal_dir) / f"bimanual_{side}.json")


def build_teleop_argv(leaders: list[dict[str, Any]], no_teleop: bool) -> list[str]:
    if no_teleop:
        log.warning("Teleop disabled by --no-teleop")
        return []
    if not leaders:
        log.warning("Teleop disabled: no leader arms configured")
        return []
    if len(leaders) == 1:
        log.info("Teleop enabled (single-arm): leader=%s", leaders[0]["port"])
        return [
            "--teleop.type=so101_leader",
            f"--teleop.port={leaders[0]['port']}",
            f"--teleop.id={TELEOP_ID_SINGLE}",
        ]
    if len(leaders) < 2:
        log.warning("Teleop disabled: need 2 leader arms, got %d", len(leaders))
        return []
    log.info("Teleop enabled: left=%s, right=%s", leaders[0]["port"], leaders[1]["port"])
    return [
        "--teleop.type=bi_so_leader",
        f"--teleop.left_arm_config.port={leaders[0]['port']}",
        "--teleop.left_arm_config.use_degrees=true",
        f"--teleop.right_arm_config.port={leaders[1]['port']}",
        "--teleop.right_arm_config.use_degrees=true",
        f"--teleop.id={TELEOP_ID}",
    ]


def stage_leader_calibrations(
    leaders: list[dict[str, Any]], teleop_argv: list[str]
) -> TemporaryDirectory[str] | None:
    if not teleop_argv:
        return None
    leader_cal_dir = TemporaryDirectory(prefix="record-leader-cal-")
    if len(leaders) == 1:
        stage_arm_calibration(leaders[0], Path(leader_cal_dir.name) / f"{TELEOP_ID_SINGLE}.json")
    else:
        for side, arm in (("left", leaders[0]), ("right", leaders[1])):
            stage_arm_calibration(arm, Path(leader_cal_dir.name) / f"{TELEOP_ID}_{side}.json")
    teleop_argv.append(f"--teleop.calibration_dir={leader_cal_dir.name}")
    return leader_cal_dir


def build_robot_argv(
    followers: list[dict[str, Any]],
    left_cameras: dict[str, Any],
    right_cameras: dict[str, Any],
    cal_dir: str,
    *,
    sim_endpoint: str | None = None,
) -> list[str]:
    """拼给 LeRobot 的 ``--robot.*`` 参数。

    ``sim_endpoint`` 非空时改用仿真:类型是 ``sim_bi_so_follower``(注册在
    ``evo_rlt.sim.sim_robot``,由 ``evo_rlt.sim`` 导出给 LeRobot 的注册表找)。
    仿真不需要 port,也不需要 calibration_dir —— 它直接读项目里的
    ``configs/calibration/robots/``,相机是渲染出来的,不占 /dev/video*。
    """
    if sim_endpoint is not None:
        return [
            "--robot.type=sim_bi_so_follower",
            "--robot.id=bimanual",
            f"--robot.endpoint={sim_endpoint}",
        ]
    if len(followers) == 1:
        return [
            "--robot.type=so101_follower",
            f"--robot.id={FOLLOWER_ID_SINGLE}",
            f"--robot.calibration_dir={cal_dir}",
            f"--robot.port={followers[0]['port']}",
            f"--robot.cameras={json.dumps(left_cameras)}",
        ]
    return [
        "--robot.type=bi_so_follower",
        "--robot.id=bimanual",
        f"--robot.calibration_dir={cal_dir}",
        f"--robot.left_arm_config.port={followers[0]['port']}",
        "--robot.left_arm_config.use_degrees=true",
        f"--robot.left_arm_config.cameras={json.dumps(left_cameras)}",
        f"--robot.right_arm_config.port={followers[1]['port']}",
        "--robot.right_arm_config.use_degrees=true",
        f"--robot.right_arm_config.cameras={json.dumps(right_cameras)}",
    ]


def build_dataset_argv(
    *,
    dataset_name: str,
    dataset_root: Path,
    task: str,
    num_episodes: int,
    episode_time_s: int,
    fps: int,
    vcodec: str,
    rename_map: dict[str, str] | None = None,
) -> list[str]:
    """Build the ``--dataset.*`` overrides for a record run.

    ``rename_map`` renames observation keys before they reach the policy. It
    exists for policies whose pretrained config fixes the camera names --
    SmolVLA's base expects ``observation.images.camera{1,2,3}`` while this rig
    records ``left_wrist``/``right_wrist``/``right_front``. The map used at
    rollout **must** be the one used at training, or each camera feeds the
    wrong input slot and the policy behaves like it was never trained.
    """
    argv = [
        f"--dataset.repo_id={dataset_name}",
        f"--dataset.root={dataset_root}",
        f"--dataset.single_task={task}",
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.episode_time_s={episode_time_s}",
        f"--dataset.fps={fps}",
        f"--dataset.vcodec={vcodec}",
        "--dataset.push_to_hub=false",
        f"--dataset.video_encoding_batch_size={num_episodes + 1}",
        "--dataset.streaming_encoding=true",
    ]
    if rename_map:
        import json as _json

        argv.append(f"--dataset.rename_map={_json.dumps(rename_map, separators=(',', ':'))}")
    return argv


def build_policy_overrides(
    *,
    policy_path: str | None,
    vla_path: str | None,
    rl_token_path: str | None,
    phase_mode: str | None = None,
    chunk_exec_steps: int | None = None,
) -> list[str]:
    if policy_path is None:
        return []
    overrides = [f"--policy.path={policy_path}"]
    if phase_mode is not None:
        overrides.append(f"--policy.phase_mode={phase_mode}")
    if chunk_exec_steps is not None:
        overrides.append(f"--policy.chunk_exec_steps={chunk_exec_steps}")
    if vla_path is not None:
        overrides.append(f"--policy.vla_pretrained_path={vla_path}")
    if rl_token_path is not None:
        overrides.append(f"--policy.rl_token_pretrained_path={rl_token_path}")
    return overrides


def build_rtc_argv(
    *,
    enabled: bool,
    execution_horizon: int,
    max_guidance_weight: float,
    prefix_attention_schedule: str,
    vla_execution_horizon: int | None,
    action_queue_size_to_get_new_actions: int | None,
) -> list[str]:
    argv = [
        f"--rlt.rtc_enabled={'true' if enabled else 'false'}",
        f"--rlt.rtc_execution_horizon={execution_horizon}",
        f"--rlt.rtc_max_guidance_weight={max_guidance_weight}",
        f"--rlt.rtc_prefix_attention_schedule={prefix_attention_schedule}",
    ]
    if vla_execution_horizon is not None:
        argv.append(f"--rlt.vla_rtc_execution_horizon={vla_execution_horizon}")
    if action_queue_size_to_get_new_actions is not None:
        argv.append(
            "--rlt.rtc_action_queue_size_to_get_new_actions="
            f"{action_queue_size_to_get_new_actions}"
        )
    return argv


def preflight_motor_connections(
    followers: list[dict[str, Any]],
    leaders: list[dict[str, Any]],
    cal_dir: str,
    leader_cal_dir: str | None,
) -> None:
    from lerobot.robots.bi_so_follower import BiSOFollower, BiSOFollowerConfig
    from lerobot.robots.so_follower import SOFollower, SOFollowerConfig, SOFollowerRobotConfig
    from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
    from lerobot.teleoperators.so_leader import SOLeader, SOLeaderConfig, SOLeaderTeleopConfig

    def disconnect(device: Any) -> None:
        for arm_name in ("left_arm", "right_arm"):
            arm = getattr(device, arm_name, None)
            if arm is not None and arm.is_connected:
                arm.disconnect()
        if getattr(device, "is_connected", False):
            device.disconnect()

    log.info("Preflight checking follower motor connections before loading policy")
    if len(followers) == 1:
        robot = SOFollower(
            SOFollowerRobotConfig(
                id=FOLLOWER_ID_SINGLE,
                calibration_dir=Path(cal_dir),
                port=followers[0]["port"],
            )
        )
    else:
        robot = BiSOFollower(
            BiSOFollowerConfig(
                id="bimanual",
                calibration_dir=Path(cal_dir),
                left_arm_config=SOFollowerConfig(port=followers[0]["port"], use_degrees=True),
                right_arm_config=SOFollowerConfig(port=followers[1]["port"], use_degrees=True),
            )
        )
    install_safe_follower_torque_enable(robot)
    try:
        robot.connect(calibrate=True)
        log.info("Preflight follower motor check passed")
    finally:
        disconnect(robot)

    if not leaders or leader_cal_dir is None:
        return

    log.info("Preflight checking leader motor connections before loading policy")
    if len(leaders) == 1:
        teleop = SOLeader(
            SOLeaderTeleopConfig(
                id=TELEOP_ID_SINGLE,
                calibration_dir=Path(leader_cal_dir),
                port=leaders[0]["port"],
            )
        )
    else:
        teleop = BiSOLeader(
            BiSOLeaderConfig(
                id=TELEOP_ID,
                calibration_dir=Path(leader_cal_dir),
                left_arm_config=SOLeaderConfig(port=leaders[0]["port"], use_degrees=True),
                right_arm_config=SOLeaderConfig(port=leaders[1]["port"], use_degrees=True),
            )
        )
    try:
        teleop.connect(calibrate=True)
        log.info("Preflight leader motor check passed")
    finally:
        disconnect(teleop)


def load_dataset_stats_from_pretrained(pretrained_path: str | Path) -> dict[str, dict[str, Any]] | None:
    """Load the (feature -> {stat_name: tensor}) dataset_stats dict bundled
    with a saved lerobot policy checkpoint's own preprocessor pipeline --
    i.e. the normalization the model was actually TRAINED with.

    For online RL (rlt_ac), the outer ChunkACPolicy wrapper is built fresh
    every session (no --policy.path of its own) against a brand-new,
    zero-episode dataset, so `make_pre_post_processors()`'s usual
    `dataset_stats` source (the recording dataset's own stats) is always
    empty. Without this, the frozen VLA -- which DOES expect properly
    normalized state/action, per its own saved
    `policy_preprocessor.json`/`*_normalizer_processor.safetensors` -- ends
    up fed effectively un-normalized (or default-normalized) observations
    despite loading the right weights, which reads as the VLA "acting
    randomly" even outside any RL involvement. This loads that checkpoint's
    real stats so they can be passed through instead.

    Returns None if the checkpoint has no normalizer_processor step (e.g. a
    non-lerobot-standard checkpoint, or one saved without normalization).
    """
    from safetensors.torch import load_file

    pretrained_path = Path(pretrained_path)
    preprocessor_json = pretrained_path / "policy_preprocessor.json"
    if not preprocessor_json.is_file():
        return None
    with open(preprocessor_json) as fh:
        spec = json.load(fh)
    state_file = next(
        (
            step.get("state_file")
            for step in spec.get("steps", [])
            if step.get("registry_name") == "normalizer_processor"
        ),
        None,
    )
    if not state_file:
        return None
    flat = load_file(str(pretrained_path / state_file))
    stats: dict[str, dict[str, Any]] = {}
    for key, tensor in flat.items():
        # Feature names themselves contain dots (observation.state,
        # observation.images.left_wrist); stat names (mean/std/min/max/
        # q01.../q99) never do, so the LAST dot always separates them.
        feature_name, stat_name = key.rsplit(".", 1)
        stats.setdefault(feature_name, {})[stat_name] = tensor
    return stats


def set_offline_env() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    _quiet_video_encoder_logs()


def _quiet_video_encoder_logs() -> None:
    """Suppress libx264's per-container startup banner (cpu caps, codec info)
    that streaming video encoding prints once per camera per episode chunk.
    PyAV's log callback is process-global, so setting it once here covers
    encoders created later in background threads too."""
    try:
        import av

        av.logging.set_level(av.logging.ERROR)
    except ImportError:
        pass
