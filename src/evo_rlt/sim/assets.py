"""Build the dual-arm MuJoCo scene from the SO-101 URDF.

The URDF and its meshes are vendored at ``third_party/SO101`` so the simulation
does not depend on a checkout that belongs to another project. The generated
MJCF still goes to a cache directory, since it is a build artifact.

Three conversion details are load-bearing, and each one fails loudly rather than
subtly if it regresses:

* MuJoCo's URDF reader defaults to ``strippath=true``, so the URDF's
  ``assets/foo.stl`` must be paired with ``meshdir="assets"`` -- setting both a
  path prefix and ``strippath=false`` yields ``assets/assets/foo.stl``.
* URDF has no actuators. Six ``position`` actuators are appended, named exactly
  after the joints so that the attached copies become ``left_shoulder_pan`` ...
  ``right_gripper`` -- the same keys `BimanualCalibration.action_to_rad` emits.
* ``base_link`` is fixed, so MuJoCo folds its geoms into the world body and
  there is nothing left to attach twice. They are re-wrapped into an explicit
  ``base_link`` body before the arm is used as a reusable model.

The wrist cameras are declared inside the single-arm model, so attaching it
twice produces ``left_wrist`` and ``right_wrist`` -- matching the camera keys
the real setup records.
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path

# Dual-mode import: as part of the evo_rlt package on the client side, or as a
# loose module inside the simulator process, whose interpreter has no torch and
# therefore cannot import `evo_rlt` at all (its __init__ pulls in evo_rlt.core).
try:
    from evo_rlt.sim.protocol import JOINT_NAMES, SIM_JOINT_LIMITS
except ImportError:  # pragma: no cover - exercised only in the sim environment
    from protocol import JOINT_NAMES, SIM_JOINT_LIMITS


def _home_qpos() -> list[float] | None:
    """复位姿态的关节角,左右各六个,顺序同 JOINT_ORDER。

    优先用 ``configs/home_pose.json`` 里的 ticks —— 那是真机录制时
    ``--go-home-positions`` 实际用的复位位置。没有该文件时退回标定零位。

    仿真复位若用 URDF 全零,看到的是完全不同的姿态(夹爪竖立、手臂伸直),
    遥操一接上就得先跑一大段才对得齐,viewer 里也会一开就是错的。
    """
    try:
        try:
            from evo_rlt.sim.arms import FOLLOWER_CALIBRATION_DIR, arm
            from evo_rlt.sim.calib import GRIPPER_JOINT, MOTOR_NAMES, BimanualCalibration
        except ImportError:  # 仿真进程里按模块直接加载
            from arms import FOLLOWER_CALIBRATION_DIR, arm
            from calib import GRIPPER_JOINT, MOTOR_NAMES, BimanualCalibration

        bridge = BimanualCalibration.from_dir(
            FOLLOWER_CALIBRATION_DIR,
            left_id=arm("left_follower").calibration_id,
            right_id=arm("right_follower").calibration_id,
        )
    except Exception as exc:
        print(f"[assets] 读不到标定,复位姿态退回 URDF 零位: {exc}")
        return None

    import json

    home_file = REPO_ROOT / "configs" / "home_pose.json"
    ticks = {}
    if home_file.is_file():
        ticks = json.loads(home_file.read_text()).get("ticks", {})

    qpos: list[float] = []
    clipped: list[str] = []
    for side in ("left", "right"):
        calib_arm = bridge.arm(side)
        for name in MOTOR_NAMES:
            key = f"{side}_{name}"
            if key in ticks:
                motor = calib_arm.motors[name]
                raw = ticks[key]
                # 旧标定下记录的 ticks 可能落在新标定行程之外,钳到行程内,
                # 否则复位就把关节顶在限位上。
                bounded = min(motor.range_max, max(motor.range_min, raw))
                if bounded != raw:
                    clipped.append(f"{key} {raw}->{bounded}")
                rad = calib_arm.ticks_to_rad(name, bounded)
            else:
                # 未列出的关节(wrist_roll)用标定零位
                rad = calib_arm.value_to_rad(
                    name, 50.0 if name == GRIPPER_JOINT else 0.0, clip=True
                )
            lo, hi = SIM_JOINT_LIMITS[name]
            qpos.append(min(hi, max(lo, rad)))
    if clipped:
        print(f"[assets] 复位 ticks 超出当前标定行程,已钳位: {clipped}")

    # free joint 的初始位姿也要写进 keyframe。keyframe 的 qpos 必须覆盖整个
    # nq,少写的部分 MuJoCo 会补零 —— 表现为零件复位到世界原点、直接掉地上。
    task = _load_task_config()
    if task is not None:
        for key in ("socket", "bolt"):
            spec = task[key]
            # free joint 的 qpos 会覆盖 body 上写的 euler,所以这里必须把同一个
            # 朝向换算成四元数写进去,否则物体会以未旋转的姿态复位。
            qpos += [float(v) for v in spec["pos"]] + _euler_to_quat(spec.get("euler"))
    return qpos

#: Repository root, resolved from this file so the defaults work regardless of
#: the working directory the simulator was launched from.
REPO_ROOT = Path(__file__).resolve().parents[3]

#: 物理步长(秒)。接触参数的稳定性约束以它为准(见 GraspConfig.validate),
#: 所以只能在这里改,不要在 <option> 里另写一个。
PHYSICS_TIMESTEP = 0.002

DEFAULT_URDF = REPO_ROOT / "third_party" / "SO101" / "so101_new_calib.urdf"
SO101_ASSET_DIR = REPO_ROOT / "third_party" / "SO101" / "assets"
DEFAULT_CACHE_DIR = Path("~/.cache/evo_rlt/sim_assets")

ARM_MODEL_FILE = "so101.xml"
SCENE_FILE = "scene.xml"

#: URDF 里机身是黄色、电机是黑色。只替换机身色,电机保持黑色便于分辨结构。
_BODY_RGBA = "1 0.82 0.12 1"

#: 左右臂上色以便一眼区分。调相机、看遥操录像时,两条同色的臂很容易认错边。
ARM_COLORS = {"left": "1 0.82 0.12 1", "right": "0.25 0.78 0.35 1"}

#: attach 顺序,必须和 protocol.JOINT_ORDER 的左右次序一致。
ARM_SIDES_ORDER = ("left", "right")

#: 天空渐变:上方微蓝、下方近白。
SKY_TOP = "0.78 0.85 0.94"
SKY_BOTTOM = "0.97 0.98 1.0"

#: 桌面颜色。两个值取得很接近,肉眼近乎纯色,但仍留有极弱的格纹 ——
#: 完全纯色的平面会让视觉策略失去深度和位移的参照,训练时反而更难。
TABLE_RGB1 = "0.72 0.58 0.40"
TABLE_RGB2 = "0.68 0.54 0.36"


#: 任务物体的 STL。SolidWorks 导出单位是毫米,MuJoCo 用米。
TASK_MESH_DIR = REPO_ROOT / "data"
MM_TO_M = 0.001
#: ``_wide`` 后缀的是 diagnostics/widen_holes.py 把孔壁外扩过的版本 —— 凸分解
#: 会把孔壁向内近似,不预先扩就装不进去。扩多少要让**分解后**的通径等于 CAD
#: 真值,不是想扩多大就扩多大:
#:   桌子小孔  CAD 5.00/6.00mm,扩 1.8mm,分解后 5.13/6.13mm
#:   螺套内孔  CAD 5.00mm,     扩 0.2mm(不扩的话分解后只剩 4.80,而螺母杆是
#:                             4.75 —— 单边间隙 0.05mm,接触求解的穿透量都比它大)
#: 螺套只用扩 0.2 而桌子要 1.8,是因为桌子那 255 块要摊在整张台面上,螺套的
#: 256 块全用在这一个小零件上,孔壁近似得细得多。
#: 视觉也用这份加宽网格(和桌子一样的约定),0.2mm 的差别肉眼看不出来。
TASK_MESHES = {
    "table": "桌子_h119_wide.STL",
    "socket": "螺套_no_range_wide.STL",
    "socket_insert": "螺套_no_range_白色端面嵌片.STL",
    "bolt": "螺栓.SLDPRT.STL",
}


#: 必须做凸分解的夹爪网格。MuJoCo 的 mesh 碰撞取**凸包**,这两片钳口的凸包会
#: 把它们之间的 V 形开口整个填实 —— 实测两片在任何开度下都互相嵌入 20.2mm,
#: 钳口在碰撞空间里根本没有缝。零件"夹在钳口之间"实为埋进两个实心体,穿透几十
#: 毫米,接触力一算就把它弹飞。这和桌面的孔被填平是同一个毛病,同一个解法。
#:
#: 分解产物由 diagnostics/decompose_mesh.py 生成到 <mesh>_hulls/。
JAW_COLLISION_MESHES = ("moving_jaw_so101_v1", "wrist_roll_follower_so101_v1")


@dataclass
class GraspConfig:
    """钳口与零件的接触参数。

    这些值决定"能不能抓住",而且互相耦合,拍脑袋填必然反复。默认值由
    ``diagnostics/grasp_test.py --sweep`` 扫出来,改动请连带重跑它。
    """

    #: 接触维数。3 = 只有滑动摩擦,零件会在钳口里绕接触法线打转然后滑出;
    #: 4 加上扭转摩擦,这是两点夹持能稳住的前提。
    condim: int = 4
    #: (滑动, 扭转, 滚动)。MuJoCo 默认 (1, 0.005, 1e-4) 是通用值,不是夹爪值。
    #: **这一组只套在钳口上** —— 它要涩,夹得住。零件和台面另有一组,
    #: 见 ``part_friction``。
    friction: tuple[float, float, float] = (1.5, 0.05, 0.0005)
    #: 零件和台面的 (滑动, 扭转, 滚动)。**和钳口那组分开**,因为两者要的值相反。
    #:
    #: 钳口要涩(夹得住),而"杆在孔里"要滑 —— 孔的单边间隙只有 0.34mm,杆歪约
    #: 1 度就楔住,μ 大了直接自锁,夹住了也拔不出来。实测纯轴向拉,拔出所需的力:
    #:     μ      0.5N   1N    2N    3N    5N      (数字=上升 mm,>=25 为拔出)
    #:     1.5     0.3   0.6   1.5   5.4  14.5     <- 旧值,5N 都拔不干净
    #:     0.8     0.3   0.7  12.3   拔出  拔出
    #:     0.5     0.3  11.9   拔出  拔出  拔出     <- 现值
    #:     0.3     4.3   拔出  拔出  拔出  拔出
    #: 带一点侧倾(手没对正)时差距更大:μ=1.5 给 5N 只上升 6.4mm,μ=0.5 给 3N 就出来。
    #:
    #: 0.5 也更接近真实材料:两个零件和台面都是 PETG,PETG 对 PETG 大约 0.3~0.5。
    #: 1.5 当初是为"夹得住"选的夹爪值,被连带套到了这些接触上。
    #:
    #: 夹持不受影响:MuJoCo 取两边**逐项最大**,钳口 1.5 × 零件 0.5 仍然取 1.5。
    part_friction: tuple[float, float, float] = (0.5, 0.017, 0.0005)
    #: 接触的 (时间常数, 阻尼比)。默认 0.02 偏软,轻零件会被压进去再弹出;
    #: 但**时间常数不能小于 2 倍步长**(见 validate),否则接触弹簧比积分器能
    #: 稳定处理的还硬,每次接触往系统里注入能量,零件被弹飞甚至穿过桌子。
    solref: tuple[float, float] = (0.01, 1.0)
    #: 接触阻抗 (d0, dmax, width)。MuJoCo 默认 (0.9, 0.95, 0.001) —— dmax=0.95
    #: 意味着约束**最硬也只有 95%**,剩下那点软度就是零件持续往里沉的量。
    #:
    #: 遥操实测(仿真器的穿透监视器打出来的,不是构造场景):零件被压住时穿透
    #: 一路涨到 6~7mm 才停,18.5mm 的台面板陷进去三分之一。施力扫描对得上:
    #:     力        0.5N   2N    5N   10N   28N
    #:     默认      -0.4  -0.8  -1.4  -2.7  -10.2 mm
    #:     现值      -0.1  -0.2  -0.4  -0.5   -0.8 mm
    #:
    #: 这同时是滑移的一部分来源:接触面一直在下陷,摩擦支撑就不稳。
    #: 注意 dmax 不能取到 1.0 —— 那是完全刚性,求解器会病态。
    solimp: tuple[float, float, float] = (0.98, 0.999, 0.0005)
    #: 摩擦阻抗与法向阻抗之比。1 时摩擦相对法向力太"软",锥形约束下必滑。
    impratio: float = 10.0
    #: 夹爪关节的力矩上限(N·m)。取 Follower 舵机的**额定**力矩,不是堵转。
    #:
    #: SO-101 Follower Pro 六轴统一是 12V STS3215 / 1:345,Feetech 给的是
    #: 额定 10 kg·cm、堵转 30 kg·cm,即 0.981 / 2.942 N·m。堵转是电机转不动、
    #: 接近最大电流(12V 时 2.7A)才到的值,不是正常工作力矩;位置伺服夹住零件
    #: 时会长期顶在上限上,所以这里取额定。
    #:
    #: 原值 3.0 是照抄堵转。力臂 0.035 m,换算出的夹持力实测:
    #:   3.00 N·m -> 86.0 N  <- 旧值。2.35g 的螺栓受 86N 是 36000 m/s²,
    #:                          一个 2ms 步长就到 72 m/s —— "夹准了还被弹开"
    #:                          和"零件穿过桌子"都是从这里来的
    #:   0.981 N·m -> 约 30 N
    gripper_force_limit: float = 0.981

    @classmethod
    def load(cls) -> GraspConfig:
        """套用 configs/grasp.json,缺文件就用默认值。"""
        import json

        path = REPO_ROOT / "configs" / "grasp.json"
        if not path.is_file():
            return cls()
        raw = json.loads(path.read_text())
        fields = {f.name for f in dataclass_fields(cls)}
        kwargs = {k: v for k, v in raw.items() if k in fields}
        for key in ("friction", "part_friction", "solref", "solimp"):
            if key in kwargs:
                kwargs[key] = tuple(kwargs[key])
        cfg = cls(**kwargs)
        cfg.validate(PHYSICS_TIMESTEP)
        return cfg

    def validate(self, timestep: float) -> None:
        """挡住会让接触求解发散的取值。

        MuJoCo 要求 ``solref`` 的时间常数至少是步长的 2 倍。小于这个数,接触
        弹簧比积分器能稳定处理的还硬,每次接触都往系统里注入能量 —— 现象是
        零件被夹爪弹飞,速度够高时还会一步跨过 18.5mm 的台面板直接穿过去。

        这条曾经被扫参扫出来过(0.002 == 步长):扫描的评分只看滑移量,不看
        稳定性,发散的配置在那一次测试里恰好没炸,就被选中写进了配置。所以
        约束要写在这里,而不是指望扫描网格里不出现坏值。
        """
        floor = 2.0 * timestep
        if self.solref[0] < floor:
            raise AssetBuildError(
                f"solref 时间常数 {self.solref[0]:g}s 小于 2 倍步长 {floor:g}s,"
                f"接触会发散。改 configs/grasp.json 里的 solref[0] >= {floor:g}"
            )
        if self.condim not in (1, 3, 4, 6):
            raise AssetBuildError(f"condim 只能是 1/3/4/6,给的是 {self.condim}")
        if min(self.friction) < 0:
            raise AssetBuildError(f"friction 不能为负: {self.friction}")
        if not 0.0 < self.solimp[0] <= self.solimp[1] < 1.0:
            raise AssetBuildError(
                f"solimp 要满足 0 < d0 <= dmax < 1,给的是 {self.solimp}。"
                "dmax=1.0 是完全刚性,求解器会病态。"
            )
        if min(self.part_friction) < 0:
            raise AssetBuildError(f"part_friction 不能为负: {self.part_friction}")
        # 只许比钳口低。调**高**过:摩擦锥变陡会让摩擦约束挤掉法向约束,合爪时
        # 零件嵌进钳口(4.0/0.2 实测 -7.19mm),而且不单调、挑不出安全值。
        if self.part_friction[0] > self.friction[0]:
            raise AssetBuildError(
                f"part_friction 的滑动摩擦 {self.part_friction[0]} 高于钳口的 "
                f"{self.friction[0]}。调高会让合爪时零件嵌进钳口,见 part_friction 的说明。"
            )

    def contact_attrs(self, friction: tuple[float, float, float] | None = None
                      ) -> dict[str, str]:
        """可直接塞进 <geom> 的接触属性。

        ``friction`` 留空时用钳口那一组;零件和台面传 ``part_friction``。
        """
        return {
            "condim": str(self.condim),
            "friction": " ".join(f"{v:g}" for v in (friction or self.friction)),
            "solref": " ".join(f"{v:g}" for v in self.solref),
            "solimp": " ".join(f"{v:g}" for v in self.solimp),
        }


def _load_task_config() -> dict | None:
    """读 configs/task_scene.json。没有就不放任务物体,只留两条臂。"""
    import json

    path = REPO_ROOT / "configs" / "task_scene.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _euler_to_quat(euler: list[float] | None) -> list[float]:
    """MJCF 默认的 XYZ 欧拉角转四元数 (w, x, y, z)。"""
    import math

    if not euler:
        return [1.0, 0.0, 0.0, 0.0]
    rx, ry, rz = (float(v) / 2.0 for v in euler)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return [
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
    ]


def _body_attrs(name: str, spec: dict) -> dict[str, str]:
    """物体 body 的 pos/euler 属性。euler 可选,缺省即不旋转。"""
    attrs = {"name": name, "pos": " ".join(f"{v:g}" for v in spec["pos"])}
    if "euler" in spec:
        attrs["euler"] = " ".join(f"{v:g}" for v in spec["euler"])
    return attrs


def _stl_volume(path: Path) -> float:
    """二进制 STL 的封闭体积。用来按体积给凸块分摊质量。

    散度定理:每个三角形对体积的贡献是 v0·(v1×v2)/6。这里只要各块的相对
    大小,单位无所谓。不引 trimesh/scipy —— 仿真进程里没装。
    """
    import struct

    raw = path.read_bytes()
    if raw[:5] == b"solid" and b"facet" in raw[:500]:
        raise AssetBuildError(f"{path} 是 ASCII STL,凸分解应该输出二进制")
    count = struct.unpack("<I", raw[80:84])[0]
    total = 0.0
    for i in range(count):
        vals = struct.unpack("<12f", raw[84 + i * 50 : 84 + i * 50 + 48])
        ax, ay, az, bx, by, bz, cx, cy, cz = vals[3:12]
        total += (
            ax * (by * cz - bz * cy)
            - ay * (bx * cz - bz * cx)
            + az * (bx * cy - by * cx)
        ) / 6.0
    return abs(total)


def _add_part_geoms(asset: ET.Element, body: ET.Element, name: str, mesh: str,
                    material: str, mass: float, contact: dict) -> None:
    """给零件加视觉网格 + 凸分解出的碰撞几何。

    两个零件都是非凸的,凸包会毁掉任务本身:
      螺栓  六角头 + 细杆,凸包是从头到杆尖的**圆锥**。杆在碰撞空间里越靠头
            越粗,插孔时锥面卡在孔口就停了 —— 实测复位后自己上浮 10.3mm。
      螺套  中间有内孔,凸包直接填实,螺栓永远插不进去 —— 而这正是任务本身。

    视觉仍用原始网格(contype=0,只画不碰),碰撞用凸块(group 3,不渲染)。
    质量按各凸块的体积分摊。全给第一块会把质心挪到那块的形心上,零件的转动
    惯量和平衡姿态都跟着错;每块都给全额则零件重 N 倍。
    """
    ET.SubElement(body, "geom", {
        "name": f"{name}_visual", "type": "mesh", "mesh": mesh,
        "material": material, "contype": "0", "conaffinity": "0",
        "group": "1", "mass": "0",
    })

    hull_dir = TASK_MESH_DIR / f"{Path(TASK_MESHES[name]).stem}_hulls"
    hulls = sorted(hull_dir.glob("*.STL")) if hull_dir.is_dir() else []
    if not hulls:
        raise AssetBuildError(
            f"缺少 {name} 的碰撞凸块 {hull_dir}。先跑:\n"
            f"  python diagnostics/decompose_mesh.py {TASK_MESH_DIR / TASK_MESHES[name]}"
            " --threshold 0.005"
        )
    volumes = [_stl_volume(h) for h in hulls]
    total = sum(volumes)
    if total <= 0:
        raise AssetBuildError(f"{name} 的凸块体积算出来是 {total},网格可能不封闭")

    scale = f"{MM_TO_M} {MM_TO_M} {MM_TO_M}"
    for i, (hull, vol) in enumerate(zip(hulls, volumes)):
        hull_mesh = f"{name}_hull{i}"
        ET.SubElement(asset, "mesh", {"name": hull_mesh, "file": str(hull), "scale": scale})
        ET.SubElement(body, "geom", {
            "name": f"{name}_geom" if i == 0 else f"{name}_geom{i}",
            "type": "mesh", "mesh": hull_mesh, "group": "3",
            "rgba": "0.5 0.5 0.5 0.3",
            "mass": f"{mass * vol / total:.8g}",
            **contact,
        })
    print(f"[assets] {name} 碰撞用 {len(hulls)} 个凸块,质量按体积分摊")


def _add_task_objects(root: ET.Element, worldbody: ET.Element, cfg_task: dict) -> None:
    """把桌面和两个待装配零件加进场景。

    桌子是静态几何(不加 joint),螺套和螺栓各给一个 free joint —— 插销任务要
    双臂各抓一个再对接,两者都必须可动。

    嵌片不是独立刚体:它和螺套来自同一装配坐标系(z 30.8~32mm 正好贴在螺套
    顶面),所以作为同一个 body 下的第二个 geom,跟着螺套一起运动。它只是端面
    的一个薄片,单独给它 free joint 会让它掉下来。
    """
    asset = root.find("asset")
    # 哑光材质:geom 本身不接受 specular/shininess,这两个是 material 的属性
    for mat, rgba in (
        ("mat_table", cfg_task["table"]["rgba"]),
        ("mat_socket", cfg_task["socket"]["rgba"]),
        ("mat_insert", cfg_task["socket"]["insert_rgba"]),
        ("mat_bolt", cfg_task["bolt"]["rgba"]),
    ):
        ET.SubElement(asset, "material", {
            "name": mat,
            "rgba": " ".join(f"{v:g}" for v in rgba),
            "specular": "0.05", "shininess": "0.05", "reflectance": "0",
        })

    scale = f"{MM_TO_M} {MM_TO_M} {MM_TO_M}"
    for name, filename in TASK_MESHES.items():
        mesh_path = TASK_MESH_DIR / filename
        if not mesh_path.is_file():
            raise AssetBuildError(f"缺少任务网格 {mesh_path}")
        ET.SubElement(
            asset, "mesh",
            {"name": f"task_{name}", "file": str(mesh_path), "scale": scale},
        )

    table = cfg_task["table"]
    pose = {"pos": " ".join(f"{v:g}" for v in table["pos"])}
    if "euler" in table:
        pose["euler"] = " ".join(f"{v:g}" for v in table["euler"])

    # 视觉:原始网格,不参与碰撞
    ET.SubElement(worldbody, "geom", {
        "name": "worktable_visual", "type": "mesh", "mesh": "task_table",
        "material": "mat_table", "contype": "0", "conaffinity": "0",
        "group": "1", "mass": "0", **pose,
    })

    # 碰撞:凸分解出的若干凸块。MuJoCo 的 mesh 碰撞取凸包,单个网格会把台面上的
    # 孔和凹槽填平 —— 螺栓插不进去反被顶出。拆成凸块后凹陷才真实存在。
    hull_dir = TASK_MESH_DIR / f"{Path(TASK_MESHES['table']).stem}_hulls"
    hulls = sorted(hull_dir.glob("*.STL")) if hull_dir.is_dir() else []
    if not hulls:
        raise AssetBuildError(
            f"缺少桌子的碰撞凸块 {hull_dir}。先跑:\n"
            f"  python diagnostics/decompose_mesh.py {TASK_MESH_DIR / TASK_MESHES['table']}"
        )
    # 零件也要用夹持用的接触参数:MuJoCo 对非 <pair> 接触取两边 friction 的
    # 逐项最大、condim 的最大,所以只给钳口设是不够的,零件这边也得配上。
    grasp = GraspConfig.load()
    # 台面和零件用 part_friction(滑),钳口在 _substitute_jaw_collision 里用
    # friction(涩)。两组分开的理由见 GraspConfig.part_friction。
    contact = grasp.contact_attrs(grasp.part_friction)

    scale_attr = f"{MM_TO_M} {MM_TO_M} {MM_TO_M}"
    for i, hull in enumerate(hulls):
        ET.SubElement(asset, "mesh", {
            "name": f"table_hull{i}", "file": str(hull), "scale": scale_attr,
        })
        # 台面同样显式写接触参数。留空的话拿的是 MuJoCo 默认
        # (condim=3 / friction=1 / solref=0.02),而零件和钳口是 4 / 1.5 / 0.01,
        # 每个"零件碰台面"的接触都由混合规则拼出第三套值(solref 取平均 0.015)。
        # 配置文件里写着一套、实际算的是另一套,查起来毫无线索。
        ET.SubElement(worldbody, "geom", {
            "name": f"worktable_col{i}", "type": "mesh", "mesh": f"table_hull{i}",
            "group": "3", "rgba": "0.5 0.5 0.5 0.3", **pose, **contact,
        })
    print(f"[assets] 桌面碰撞用 {len(hulls)} 个凸块(孔与凹槽得以保留)")

    socket = cfg_task["socket"]
    body = ET.SubElement(
        worldbody, "body",
        _body_attrs("socket", socket),
    )
    ET.SubElement(body, "freejoint", {"name": "socket_free"})
    _add_part_geoms(asset, body, "socket", "task_socket", "mat_socket",
                    socket["mass"], contact)
    ET.SubElement(body, "geom", {
        "name": "socket_insert_geom", "type": "mesh", "mesh": "task_socket_insert",
        "material": "mat_insert",
        # 端面薄片只是视觉标记,不参与碰撞也不计质量,否则会给刚体加一层
        # 几乎零厚度的碰撞面,接触求解容易抖
        "contype": "0", "conaffinity": "0", "mass": "0",
        # 抬 0.1mm。嵌片和螺套来自同一装配坐标系,两者都占 z 30.8~32.0,
        # **顶面精确共面** —— 直接放会 z-fighting,渲染出来是一片黑红交错的
        # 放射状花纹,看着像块贴在顶上的补丁。螺套 STL 顶部没有给嵌片留沉孔
        # (按 z 逐层量截面,z=20 和 z=31.9 都是 361.04 mm²,一路实心到顶),
        # 所以在不改 CAD 的前提下只能靠这个偏移把深度分开。
        # 嵌片 XY 比螺套小一圈(336 vs 361 mm²),抬起来后四周留一圈黑边,
        # 看上去是嵌在端面里的环,而不是扣在上面的盖。
        "pos": "0 0 0.0001",
    })

    bolt = cfg_task["bolt"]
    body = ET.SubElement(
        worldbody, "body",
        _body_attrs("bolt", bolt),
    )
    ET.SubElement(body, "freejoint", {"name": "bolt_free"})
    _add_part_geoms(asset, body, "bolt", "task_bolt", "mat_bolt",
                    bolt["mass"], contact)


class AssetBuildError(RuntimeError):
    """Raised when the URDF cannot be turned into a usable scene."""


@dataclass
class CameraPose:
    """A camera's placement, in the frame of whatever body it hangs off.

    Defaults are a plausible wrist-cam mount, **not** a measurement. Calibrate
    against the real rig before trusting any image the simulator produces.
    """

    pos: tuple[float, float, float]
    xyaxes: tuple[float, float, float, float, float, float]
    fovy: float = 58.0

    def as_attrib(self, name: str) -> dict[str, str]:
        return {
            "name": name,
            "pos": " ".join(f"{v:.6g}" for v in self.pos),
            "xyaxes": " ".join(f"{v:.6g}" for v in self.xyaxes),
            "fovy": f"{self.fovy:g}",
        }


@dataclass
class SceneConfig:
    """Layout of the two arms and the fixed camera.

    ``arm_separation`` and the camera poses describe *your* rig; they are the
    first things to re-measure when sim images stop matching real ones.
    """

    urdf_path: Path = DEFAULT_URDF
    cache_dir: Path = DEFAULT_CACHE_DIR
    #: 两个 follower 底座中心的距离(米)。场景里所有物体的摆位都以它为基准,
    #: 改了之后桌子和零件的相对可达性会变。
    arm_separation: float = 0.42
    table_height: float = 0.0
    #: 桌面颜色,改这两行即可换色(见 TABLE_RGB1 的说明)
    table_rgb1: str = TABLE_RGB1
    table_rgb2: str = TABLE_RGB2
    # 执行器增益按**稳态误差**选:kp=50 给出 0.27 度,真机是 0.31 度。
    # 响应滞后不靠压低增益去凑,而是用 action_delay_steps 的纯延迟补 ——
    # 压低 kp 会让重力下垂变大(kp=20 时 0.67 度、kp=5 时 6.5 度),那是
    # 真机没有的误差,因为真机舵机的 PID 带积分项。
    #
    # 也试过 mujoco.pid 插件,在这个双臂模型上不稳定,放弃。
    #
    # 加了 armature 之后用 outputs/sign_check.json 复核过,两者仍是最优:
    #   kv(=dampratio 换算)  1.0    1.2    1.414   1.7    2.0
    #   shoulder_pan RMSE   0.440  0.431  0.430   0.441  0.467  (度)
    #   wrist_flex   RMSE   1.375  1.348  1.335   1.344  1.384
    # 而且 dampratio=1 时仿真的 p99.5 关节速度和真机对得上(1.54 vs 1.54、
    # 0.81 vs 0.78、1.59 vs 1.54),不是靠压速度换来的低 RMSE。
    control_kp: float = 50.0
    control_dampratio: float = 1.0
    #: 关节的转子折算惯量(kg·m²)。**URDF 里一个 <dynamics> 都没有**,所以
    #: 转成 MJCF 后 armature/damping/frictionloss 全是 0 —— 关节只剩连杆自身的
    #: 惯量。夹爪那一节因此只有 1.6e-5 kg·m²(由执行器的 kv 反推),kp=50 的
    #: 位置伺服作用在这么小的惯量上,一步就能走完 5 度阶跃的 90%:钳口在数值上
    #: 等于"无质量、瞬移",撞到什么都是无穷大冲量。
    #:
    #: 取值由两个独立测量夹住,不是拍的:
    #:
    #: **下界**来自接触。夹爪压向台面下方 22mm(理想行为是顶住不动):
    #:     armature      穿透      峰值接触力
    #:     0            -5.58 mm   383 N     <- 悬崖这边
    #:     0.002        -0.35 mm    17 N
    #:     0.0075       -0.25 mm    19 N     0.002~0.04 之间行为完全一致,
    #:     0.02         -0.20 mm    18 N        是平台不是斜坡
    #:
    #: **上界和最优**来自真机。outputs/sign_check.json 是 209 秒配对遥操
    #: (commanded vs 从臂实测),重放对比 RMSE:
    #:     armature   0.004  0.005  0.006  0.0075  0.009  0.011
    #:     夹爪 RMSE  0.498  0.489  0.482  0.482   0.497  0.536  (度)
    #:
    #: 注意:手臂五轴对 armature 几乎不敏感(30Hz 遥操是准静态,惯量没被激励),
    #: 只有夹爪能分辨 —— 而且必须先把夹爪的 damping/frictionloss 定对,
    #: 否则那两项会把 armature 的信号吃掉,曲线看上去是平的。
    joint_armature: float = 0.007
    #: 手臂五轴的黏性阻尼和干摩擦(N·m·s/rad / N·m)。
    #: 真机数据上这两项都是**单调变差**,最优在 0:
    #:     damping        0 -> 0.496,  0.1 -> 0.510,  0.4 -> 0.556 (shoulder_pan RMSE)
    #:     frictionloss   0 -> 0.510,  0.12 -> 0.520, 0.3 -> 0.547
    joint_damping: float = 0.0
    joint_frictionloss: float = 0.0
    #: 夹爪关节的阻尼和干摩擦。和手臂不同,这两项都有**尖锐的极小值**:
    #:     damping        0 -> 0.852,  0.05 -> 0.788, 0.10 -> 0.514, 0.20 -> 2.131
    #:     frictionloss   0.22 -> 0.640, 0.26 -> 0.535, 0.30 -> 0.514, 0.34 -> 0.705
    #: 物理上讲得通(夹爪那套连杆的摩擦比直驱关节大得多),但要留个话:
    #: 无法排除 0.3 是在吸收夹爪标定的非线性 —— 夹爪走的是"开度百分比"映射
    #: (见 calib._gripper_value_to_rad),和其他五轴的机械角映射不是一回事。
    gripper_damping: float = 0.1
    gripper_frictionloss: float = 0.3
    #: 非夹爪关节的力矩上限(N·m),取 Follower 舵机的**堵转**力矩。
    #: SO-101 Follower Pro 六轴统一 12V STS3215 / 1:345,Feetech 标称
    #: 额定 10 kg·cm、堵转 30 kg·cm = 0.981 / 2.942 N·m。
    #: URDF 写的 effort=10 是堵转值的 3.4 倍 —— 仿真里手臂能用真机使不出的力
    #: 把自己推进桌子里。手臂取堵转(短时冲击是真实的),夹爪取额定
    #: (见 GraspConfig.gripper_force_limit:位置伺服夹住零件时会长期顶在上限)。
    arm_force_limit: float = 2.942
    wrist_camera: CameraPose = field(
        default_factory=lambda: CameraPose(
            pos=(0.0, -0.05, 0.03), xyaxes=(1, 0, 0, 0, 0.5, 0.87)
        )
    )
    front_camera: CameraPose = field(
        default_factory=lambda: CameraPose(
            pos=(0.55, -0.18, 0.35), xyaxes=(0, -1, 0, 0.55, 0, 0.83)
        )
    )
    #: 右腕相机。为 None 时沿用 ``wrist_camera`` —— 左右臂现在是两份独立模型,
    #: 所以两只腕相机可以分别标定,不再被迫共用一套外参。
    right_wrist_camera: CameraPose | None = None

    # Body the wrist cameras are welded to. Overridable because a different
    # gripper build may rename it.
    wrist_camera_body: str = "gripper_link"

    #: 是否套用 configs/cameras.json。相机标定工具要用自己正在调的值建场景,
    #: 读盘会把它覆盖掉,所以那里显式关掉。
    load_tuned_cameras: bool = True

    def resolved(self) -> SceneConfig:
        self.urdf_path = Path(self.urdf_path).expanduser().resolve()
        self.cache_dir = Path(self.cache_dir).expanduser()
        self._load_tuned_cameras()
        return self

    def _load_tuned_cameras(self) -> None:
        """套用 ``configs/cameras.json`` 里手动标定过的相机外参。

        默认值只是个能出图的占位,不是测量结果。有标定结果就必须用上,否则
        每次重建场景都会悄悄退回占位值,而图像看起来"正常",只是和真机对不上。
        """
        if not self.load_tuned_cameras:
            return
        config = REPO_ROOT / "configs" / "cameras.json"
        if not config.is_file():
            return
        import json

        tuned = json.loads(config.read_text())
        if "left_wrist" in tuned:
            entry = tuned["left_wrist"]
            self.wrist_camera = CameraPose(
                pos=tuple(entry["pos"]),
                xyaxes=tuple(entry["xyaxes"]),
                fovy=entry.get("fovy", 58.0),
            )
        if "right_wrist" in tuned:
            entry = tuned["right_wrist"]
            self.right_wrist_camera = CameraPose(
                pos=tuple(entry["pos"]),
                xyaxes=tuple(entry["xyaxes"]),
                fovy=entry.get("fovy", 58.0),
            )
        if "right_front" in tuned:
            entry = tuned["right_front"]
            self.front_camera = CameraPose(
                pos=tuple(entry["pos"]),
                xyaxes=tuple(entry["xyaxes"]),
                fovy=entry.get("fovy", 58.0),
            )


def _urdf_with_mujoco_tag(urdf_path: Path, work_dir: Path) -> Path:
    """Copy the URDF next to its meshes and inject the MuJoCo compiler tag.

    The tag is added to a copy so the source tree (often a shared checkout) is
    never modified.
    """
    mesh_dir = urdf_path.parent / "assets"
    if not mesh_dir.is_dir():
        raise AssetBuildError(
            f"expected meshes in {mesh_dir}; the URDF references 'assets/*.stl'"
        )

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    if root.find("mujoco") is None:
        mj = ET.Element("mujoco")
        # strippath=true (the default) drops the 'assets/' prefix that meshdir
        # then re-adds; balanceinertia guards against degenerate inertia tensors.
        # discardvisual defaults to true for URDF, which would throw away the
        # visual meshes -- leaving the cameras rendering collision geometry.
        ET.SubElement(
            mj,
            "compiler",
            {
                "meshdir": "assets",
                "strippath": "true",
                "balanceinertia": "true",
                "discardvisual": "false",
            },
        )
        root.insert(0, mj)

    work_dir.mkdir(parents=True, exist_ok=True)
    patched = work_dir / urdf_path.name
    tree.write(patched, encoding="utf-8", xml_declaration=True)

    # MuJoCo resolves meshdir relative to the XML, so the meshes must be
    # reachable from work_dir. A symlink keeps the 16 MB unduplicated.
    link = work_dir / "assets"
    if link.is_symlink() or link.exists():
        if link.is_symlink():
            link.unlink()
        else:
            shutil.rmtree(link)
    link.symlink_to(mesh_dir, target_is_directory=True)
    return patched


def _wrap_base_body(root: ET.Element) -> None:
    """Re-wrap world-level geoms into a ``base_link`` body so it can be attached."""
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise AssetBuildError("converted MJCF has no <worldbody>")
    if worldbody.find("body") is not None and worldbody.find("geom") is None:
        return  # already a proper body tree

    base = ET.Element("body", {"name": "base_link"})
    for child in list(worldbody):
        if child.tag in ("geom", "body", "site"):
            worldbody.remove(child)
            base.append(child)
    if len(base) == 0:
        raise AssetBuildError("no geoms or bodies found to form base_link")
    worldbody.append(base)


#: MuJoCo renders geom groups 0-2 by default; 3 is the conventional home for
#: collision-only geometry.
_COLLISION_GROUP = "3"


def _separate_visual_collision(root: ET.Element) -> int:
    """Move collision geoms out of the rendered groups.

    The URDF gives visual and collision the same mesh, so leaving both in a
    rendered group draws every part twice for an identical image. Splitting them
    halves the render cost now, and lets a simplified collision mesh be swapped
    in later without it showing up in the cameras.

    Returns the number of geoms reassigned.
    """
    moved = 0
    for body in root.iter("body"):
        for geom in body.findall("geom"):
            # The converter marks visual geoms non-colliding and puts them in
            # group 1; anything else is the collision copy.
            is_visual = geom.get("contype") == "0" and geom.get("conaffinity") == "0"
            if not is_visual:
                geom.set("group", _COLLISION_GROUP)
                moved += 1
    return moved


def _substitute_jaw_collision(root: ET.Element, grasp: GraspConfig) -> int:
    """把两片钳口的碰撞几何换成凸分解出的凸块,并套上夹持用的接触参数。

    见 ``JAW_COLLISION_MESHES`` 的说明:不换的话钳口之间在碰撞空间里是实心的,
    夹爪抓不住任何东西。视觉仍用原始网格(那份 ``contype=0``,这里不动)。

    返回替换掉的碰撞 geom 数量(左右各一份模型,每份应为 2)。
    """
    asset = root.find("asset")
    if asset is None:
        raise AssetBuildError("converted MJCF has no <asset>")

    contact = grasp.contact_attrs()
    replaced = 0
    for body in root.iter("body"):
        for geom in list(body.findall("geom")):
            mesh = geom.get("mesh")
            if mesh not in JAW_COLLISION_MESHES:
                continue
            if geom.get("contype") == "0" and geom.get("conaffinity") == "0":
                continue  # 视觉那份保持原始网格

            hull_dir = SO101_ASSET_DIR / f"{mesh}_hulls"
            hulls = sorted(hull_dir.glob("*.STL")) if hull_dir.is_dir() else []
            if not hulls:
                raise AssetBuildError(
                    f"缺少钳口 {mesh} 的碰撞凸块 {hull_dir}。先跑:\n"
                    f"  python diagnostics/decompose_mesh.py "
                    f"{SO101_ASSET_DIR / (mesh + '.stl')} --threshold 0.01"
                )

            # 原 geom 的位姿必须原样继承,否则钳口整体偏位
            placement = {k: v for k, v in geom.attrib.items() if k in ("pos", "quat", "rgba")}
            body.remove(geom)
            for i, hull in enumerate(hulls):
                hull_mesh = f"{mesh}_hull{i}"
                if asset.find(f"mesh[@name='{hull_mesh}']") is None:
                    ET.SubElement(asset, "mesh", {"name": hull_mesh, "file": str(hull)})
                ET.SubElement(body, "geom", {
                    "name": f"{mesh}_col{i}", "type": "mesh", "mesh": hull_mesh,
                    "group": _COLLISION_GROUP, **placement, **contact,
                })
            replaced += 1
            print(f"[assets] 钳口 {mesh} 碰撞换成 {len(hulls)} 个凸块")
    return replaced


def _limit_joint_force(root: ET.Element, gripper_limit: float, arm_limit: float) -> int:
    """把关节力矩上限收到真机水平,返回改动的关节数。

    URDF 给的 effort=10 N·m 是 STS3215 堵转(30 kg·cm ≈ 2.942 N·m)的 3.4 倍。
    仿真里舵机能使出真机使不出的力:夹爪把轻零件挤飞(抓取成功率虚高),
    手臂把自己推进桌子里。两处都要收,只收夹爪的话压桌那一路照旧。
    """
    changed = 0
    for joint in root.iter("joint"):
        name = joint.get("name")
        if name not in JOINT_NAMES:
            continue
        limit = gripper_limit if name == "gripper" else arm_limit
        joint.set("actuatorfrcrange", f"{-limit:g} {limit:g}")
        changed += 1
    return changed


def _apply_joint_dynamics(root: ET.Element, cfg: SceneConfig) -> int:
    """补上 URDF 缺失的转子惯量、阻尼和干摩擦。

    URDF 里一个 ``<dynamics>`` 都没有,转换出来的关节三项全是 0。armature=0
    的后果不是"少一点阻尼",而是伺服在数值上变成瞬移的刚性约束 —— 见
    ``SceneConfig.joint_armature`` 里记的实测对照。

    必须在 ``_add_actuators`` **之前**调用:position 执行器的 ``dampratio``
    是编译期按关节等效惯量换算成 kv 的,armature 后写进去 kv 就还是按旧惯量
    算的,伺服会欠阻尼。

    夹爪和手臂分开设:真机数据上手臂的阻尼/干摩擦最优都在 0,夹爪却各有一个
    尖锐的极小值(见 ``SceneConfig.gripper_damping``)。

    返回改动的关节数。
    """
    changed = 0
    for joint in root.iter("joint"):
        name = joint.get("name")
        if name not in JOINT_NAMES:
            continue
        gripper = name == "gripper"
        joint.set("armature", f"{cfg.joint_armature:g}")
        joint.set("damping",
                  f"{(cfg.gripper_damping if gripper else cfg.joint_damping):g}")
        joint.set("frictionloss",
                  f"{(cfg.gripper_frictionloss if gripper else cfg.joint_frictionloss):g}")
        changed += 1
    return changed


def _exclude_adjacent_collisions(root: ET.Element) -> int:
    """Disable contact between directly connected links.

    Neighbouring URDF links overlap by construction -- their collision meshes
    share the joint they pivot about -- and MuJoCo does not exclude a pair just
    because a joint connects them. Left in, the overlap resolves as a permanent
    penetration (~26 mm between base_link and shoulder_link) whose contact force
    drives the joint away from its target: shoulder_pan settles ~60 degrees off
    with everything else tracking fine.

    Returns the number of excluded pairs.
    """
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise AssetBuildError("converted MJCF has no <worldbody>")

    pairs: list[tuple[str, str]] = []

    def walk(body: ET.Element) -> None:
        name = body.get("name")
        for child in body.findall("body"):
            child_name = child.get("name")
            if name and child_name:
                pairs.append((name, child_name))
            walk(child)

    for body in worldbody.findall("body"):
        walk(body)

    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for parent, child in pairs:
        ET.SubElement(contact, "exclude", {"body1": parent, "body2": child})
    return len(pairs)


def _find_body(root: ET.Element, name: str) -> ET.Element | None:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    return None


def _apply_joint_limits(root: ET.Element, limits: dict[str, tuple[float, float]]) -> list[str]:
    """Override joint ranges where the URDF is narrower than the real arm.

    The clip in `calib.py` and the physics limit here must agree: if MuJoCo
    stops a joint short of what the bridge is willing to command, the simulated
    arm silently cannot reach poses the real one does.

    Returns the joints whose range was changed.
    """
    changed = []
    for joint in root.iter("joint"):
        name = joint.get("name")
        if name not in limits or joint.get("range") is None:
            continue
        lo, hi = limits[name]
        current = [float(v) for v in joint.get("range").split()]
        if abs(current[0] - lo) > 1e-6 or abs(current[1] - hi) > 1e-6:
            joint.set("range", f"{lo:.6g} {hi:.6g}")
            changed.append(f"{name}: [{current[0]:.3f}, {current[1]:.3f}] -> [{lo:.3f}, {hi:.3f}]")
    return changed


def _add_actuators(root: ET.Element, cfg: SceneConfig) -> None:
    if root.find("actuator") is not None:
        return
    actuator = ET.SubElement(root, "actuator")
    for name in JOINT_NAMES:
        ET.SubElement(
            actuator,
            "position",
            {
                "name": name,
                "joint": name,
                "kp": f"{cfg.control_kp:g}",
                "dampratio": f"{cfg.control_dampratio:g}",
            },
        )


def _colorize(root: ET.Element, rgba: str) -> int:
    """给机身上色,电机保持原本的黑色。

    左右臂同色时,无论是调相机还是回看遥操录像都极易认错边。
    """
    changed = 0
    for geom in root.iter("geom"):
        if geom.get("rgba") == _BODY_RGBA:
            geom.set("rgba", rgba)
            changed += 1
    return changed


def _add_wrist_camera(root: ET.Element, cfg: SceneConfig, side: str = "left") -> None:
    """把腕相机挂到夹爪上,加前缀后即为 ``left_wrist`` / ``right_wrist``。"""
    pose = cfg.wrist_camera
    if side == "right" and cfg.right_wrist_camera is not None:
        pose = cfg.right_wrist_camera
    body = _find_body(root, cfg.wrist_camera_body)
    if body is None:
        available = sorted(b.get("name", "?") for b in root.iter("body"))
        raise AssetBuildError(
            f"wrist camera body {cfg.wrist_camera_body!r} not found; have: {available}"
        )
    ET.SubElement(body, "camera", pose.as_attrib("wrist"))


def build_arm_model(cfg: SceneConfig, side: str = "left") -> Path:
    """把 URDF 转成带执行器和腕相机的单臂 MJCF。

    每侧生成一份独立文件:这样左右臂可以上不同颜色,腕相机也能分别标定 ——
    共用一份模型时两者都做不到。
    """
    import mujoco

    cfg = cfg.resolved()
    if not cfg.urdf_path.is_file():
        raise AssetBuildError(
            f"SO-101 URDF not found at {cfg.urdf_path}. Pass --urdf to point at your copy."
        )

    work_dir = cfg.cache_dir
    patched_urdf = _urdf_with_mujoco_tag(cfg.urdf_path, work_dir)

    try:
        model = mujoco.MjModel.from_xml_path(str(patched_urdf))
    except ValueError as exc:
        raise AssetBuildError(f"MuJoCo could not load {patched_urdf}: {exc}") from exc

    raw_xml = work_dir / "so101_raw.xml"
    mujoco.mj_saveLastXML(str(raw_xml), model)

    tree = ET.parse(raw_xml)
    root = tree.getroot()
    _wrap_base_body(root)
    if _separate_visual_collision(root) == 0:
        raise AssetBuildError(
            "found no collision geoms to separate; the URDF conversion likely "
            "discarded visual meshes, leaving nothing correct to render"
        )
    grasp = GraspConfig.load()
    if _substitute_jaw_collision(root, grasp) != len(JAW_COLLISION_MESHES):
        raise AssetBuildError(
            f"expected to replace {len(JAW_COLLISION_MESHES)} jaw collision geoms; "
            "without the convex decomposition the gripper cannot hold anything"
        )
    if _limit_joint_force(root, grasp.gripper_force_limit, cfg.arm_force_limit) != len(JOINT_NAMES):
        raise AssetBuildError(
            f"expected to cap the force of {len(JOINT_NAMES)} joints; URDF 的 "
            "effort=10 是舵机堵转的 3.4 倍,漏掉哪个都会把手臂推进桌子里"
        )
    if _exclude_adjacent_collisions(root) == 0:
        raise AssetBuildError("found no linked body pairs to exclude from contact")
    for change in _apply_joint_limits(root, SIM_JOINT_LIMITS):
        print(f"[assets] widened {change}")
    # 必须在 _add_actuators 之前:dampratio 换算成 kv 时要用到 armature
    if _apply_joint_dynamics(root, cfg) != len(JOINT_NAMES):
        raise AssetBuildError(
            f"expected to set joint dynamics on {len(JOINT_NAMES)} joints; "
            "armature=0 会让伺服在数值上变成瞬移的刚性约束"
        )
    _add_actuators(root, cfg)
    _add_wrist_camera(root, cfg, side)
    _colorize(root, ARM_COLORS[side])

    arm_path = work_dir / f"so101_{side}.xml"
    tree.write(arm_path, encoding="utf-8", xml_declaration=True)

    # Fail here rather than inside the server if the edits broke the model.
    check = mujoco.MjModel.from_xml_path(str(arm_path))
    if check.nu != len(JOINT_NAMES):
        raise AssetBuildError(f"expected {len(JOINT_NAMES)} actuators, built {check.nu}")
    return arm_path


def build_scene(cfg: SceneConfig | None = None) -> Path:
    """Build the dual-arm scene, returning the path to ``scene.xml``."""
    import mujoco

    cfg = (cfg or SceneConfig()).resolved()
    for side in ARM_SIDES_ORDER:
        build_arm_model(cfg, side)

    half = cfg.arm_separation / 2.0
    root = ET.Element("mujoco", {"model": "dual_so101"})
    ET.SubElement(root, "compiler", {"angle": "radian"})
    # impratio 抬高摩擦相对法向力的阻抗。默认 1 时摩擦太"软",夹住的零件会
    # 顺着钳口滑出去 —— 这是全局项,不能只对夹爪设,故放在这里。
    # cone=elliptic:MuJoCo 默认的 pyramidal 是摩擦锥的**四边形内近似**,对角
    # 方向的摩擦被系统性低估,夹持时零件就沿那个方向滑。换成准确的椭圆锥后,
    # 固定抓取位姿下抬升滑移从 27.4mm 降到 10.0mm —— **摩擦系数一点没动**,
    # 穿透也没变化(-0.45 vs -0.47mm)。
    #
    # 这比"把摩擦调大"好在两点:调大摩擦(4.0/0.2)虽然滑移也降到 9.2mm,但会让
    # 摩擦约束挤掉法向约束,合爪时零件嵌进钳口(实测 -7.19mm,遥操里就是零件
    # 穿过夹爪);而且那个失败是位姿相关的随机失败,挑不出安全值。
    #
    # 它是全局求解器选项、不是关节属性,所以对夹爪的开合阻力没有影响 ——
    # 实测空载合拢 90% 用时两种锥都是 408ms,逐位相同。
    ET.SubElement(root, "option", {
        "timestep": f"{PHYSICS_TIMESTEP:g}", "integrator": "implicitfast",
        "cone": "elliptic",
        "impratio": f"{GraspConfig.load().impratio:g}",
    })
    # 求解器栈。MuJoCo 默认按"典型接触数"估一个值,而这个场景的零件是几百个
    # 凸块(螺套 256、台面 255),诊断工具还会临时把 geom_margin 放大到 20mm 去
    # 探间隙 —— 那一下接触数能到 5700、约束 23000,默认栈直接 mj_stackAlloc
    # 溢出并**整个进程 FatalError 退出**,不是抛异常。正常遥操 margin 为 0 时
    # 用不到这么多,留着纯粹是不想让工具把仿真器打死。
    ET.SubElement(root, "size", {"memory": "256M"})

    asset = ET.SubElement(root, "asset")
    # 没有 skybox 时 MuJoCo 背景是纯黑,相机画面里天空一片死黑,既不像真实
    # 场景,也让浅色物体缺少对比参照。用淡蓝到白的渐变。
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "skybox",
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": f"{SKY_TOP}",
            "rgb2": f"{SKY_BOTTOM}",
            "width": "256",
            "height": "256",
        },
    )
    for side in ARM_SIDES_ORDER:
        ET.SubElement(asset, "model", {"name": f"so101_{side}", "file": f"so101_{side}.xml"})
    ET.SubElement(
        asset,
        "texture",
        {"name": "table", "type": "2d", "builtin": "checker", "width": "512", "height": "512",
         "rgb1": cfg.table_rgb1, "rgb2": cfg.table_rgb2},
    )
    ET.SubElement(
        asset,
        "material",
        # 完全不反光:桌面的高光和镜面反射会在腕相机里形成大片亮斑,盖住要看的
        # 东西,而且真实台面多为哑光。specular/shininess/reflectance 三者都要
        # 归零 —— 只关其中一个仍会留下可见的高光。
        {
            "name": "table",
            "texture": "table",
            "texrepeat": "12 12",
            "reflectance": "0",
            "specular": "0",
            "shininess": "0",
        },
    )

    # 三盏平行光 + 提亮环境光。原来那盏点光源(没有 directional)强度随距离衰减,
    # 在桌面上打出一块集中的亮斑,腕相机凑近时会直接过曝糊掉要看的东西。
    # 平行光没有衰减,从三个方向补,阴影柔和且各处亮度一致。
    ET.SubElement(root, "visual")
    visual = root.find("visual")
    # 亮度按渲染实测定:三路相机画面均值 127、过曝像素 0.5%。
    # 三盏平行光是叠加的,每盏看着不强,加上环境光很容易推到饱和 ——
    # 实测 ambient=0.5/headlight=0.3/light=0.35 时 45.6% 的像素过曝,
    # 而 0.2/0.1/0.15 又偏暗(均值 94)。
    ET.SubElement(
        visual,
        "headlight",
        {"ambient": "0.28 0.28 0.28", "diffuse": "0.15 0.15 0.15", "specular": "0.04 0.04 0.04"},
    )

    worldbody = ET.SubElement(root, "worldbody")
    for pos, direction in (
        ("0 0 2.0", "0 0 -1"),
        ("1.0 1.0 1.5", "-0.5 -0.5 -1"),
        ("-1.0 -0.6 1.5", "0.5 0.3 -1"),
    ):
        ET.SubElement(
            worldbody,
            "light",
            {"pos": pos, "dir": direction, "directional": "true",
             "diffuse": "0.2 0.2 0.2", "specular": "0.02 0.02 0.02"},
        )
    ET.SubElement(
        worldbody,
        "geom",
        {"name": "table", "type": "plane", "size": "1 1 0.01",
         "pos": f"0 0 {cfg.table_height:g}", "material": "table"},
    )

    for side, y in (("left", half), ("right", -half)):
        mount = ET.SubElement(
            worldbody,
            "body",
            {"name": f"{side}_mount", "pos": f"0 {y:g} {cfg.table_height:g}"},
        )
        ET.SubElement(
            mount,
            "attach",
            {"model": f"so101_{side}", "body": "base_link", "prefix": f"{side}_"},
        )

    task = _load_task_config()
    if task is not None:
        _add_task_objects(root, worldbody, task)
        print("[assets] 已加入任务物体: 桌面 + 螺套(含红色端面) + 螺栓")

    # Fixed third view. Named to match the real rig's `right_front` camera key.
    ET.SubElement(worldbody, "camera", cfg.front_camera.as_attrib("right_front"))

    home = _home_qpos()
    if home is not None:
        keyframe = ET.SubElement(root, "keyframe")
        ET.SubElement(
            keyframe,
            "key",
            {"name": "home", "qpos": " ".join(f"{v:.6f}" for v in home)},
        )
        print(f"[assets] 复位姿态已写入 keyframe (wrist_roll={home[4]:.3f} rad)")

    scene_path = cfg.cache_dir / SCENE_FILE
    ET.ElementTree(root).write(scene_path, encoding="utf-8", xml_declaration=True)

    check = mujoco.MjModel.from_xml_path(str(scene_path))
    expected_nu = 2 * len(JOINT_NAMES)
    if check.nu != expected_nu:
        raise AssetBuildError(f"expected {expected_nu} actuators in scene, built {check.nu}")
    return scene_path


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build the dual SO-101 MuJoCo scene.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--arm-separation", type=float, default=0.36)
    args = parser.parse_args()

    cfg = SceneConfig(
        urdf_path=args.urdf, cache_dir=args.cache_dir, arm_separation=args.arm_separation
    )
    path = build_scene(cfg)
    print(f"built scene: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
