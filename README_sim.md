## 环境
```bash
conda create -n rlt_sim python=3.11 -y
~/anaconda3/envs/rlt_sim/bin/pip install mujoco pyzmq numpy
~/anaconda3/envs/evo-rlt/bin/pip install pyzmq
```
## 哪些要仿真器开着

| 要（连 5555） | 不要（自己加载 scene.xml 跑） |
|---|---|
| `teleop_sim.py` | `grasp_test.py` `settle_objects.py` `check_hole_fit.py` |
| `reset_objects.py` | `decompose_mesh.py` `widen_holes.py` `tune_cameras.py` |

右边这些随时单独跑，不占终端。全程只要两个终端：仿真器 + 遥操。

## 重建场景

改了 `assets.py` 或 `configs/` 下的 `cameras.json` / `task_scene.json` /
`grasp.json` 之后，必须重建才生效：

```bash
~/anaconda3/envs/rlt_sim/bin/python src/evo_rlt/sim/mj_server.py --build --benchmark
```

`--benchmark` 建完报告耗时就退出。只写 `--build` 会接着起服务器占住终端。

## 启动（两个终端）

```bash
# 终端 1：仿真器，一直开着
~/anaconda3/envs/rlt_sim/bin/python src/evo_rlt/sim/mj_server.py --viewer --show-cameras

# 终端 2：遥操，用真机主臂驱动仿真
~/anaconda3/envs/evo-rlt/bin/python diagnostics/teleop_sim.py --duration 120 --save outputs/solo_check.json
```

跑起来后在终端 2 里按 `b` 复位零件，见下节。

## 零件复位

零件被碰歪了重摆，手臂不动，不打断遥操。**在终端 2 里按 `b`** 复位全部零件。

只占这一个键，其余键位留给后面 RLT 的人工干预。停止用 Ctrl-C。

stdin 不是终端时（输出被重定向）按键自动关闭，改用命令行：

```bash
~/anaconda3/envs/evo-rlt/bin/python diagnostics/reset_objects.py          # 全部
~/anaconda3/envs/evo-rlt/bin/python diagnostics/reset_objects.py bolt     # 只螺栓
~/anaconda3/envs/evo-rlt/bin/python diagnostics/reset_objects.py --list   # 有哪些零件
```

## 抓取

```bash
# 抓一次并出图
~/anaconda3/envs/rlt_sim/bin/python diagnostics/grasp_test.py --render
#   --object socket   改抓螺套
#   --arm right       改右臂

# 扫接触参数，写回 configs/grasp.json
~/anaconda3/envs/rlt_sim/bin/python diagnostics/grasp_test.py --sweep --apply
```

## 网格凸分解

MuJoCo 的 mesh 碰撞取凸包，孔、槽、钳口开口都会被填实。改了任何带凹特征的
STL 之后要重新分解。

```bash
~/anaconda3/envs/rlt_sim/bin/python diagnostics/decompose_mesh.py <stl> --threshold 0.005
~/anaconda3/envs/rlt_sim/bin/python diagnostics/widen_holes.py --extra-mm 2.5   # 分解会啃掉约 1.5mm 孔壁
~/anaconda3/envs/rlt_sim/bin/python diagnostics/settle_objects.py --apply       # 让物理找零件的平衡位姿
```

## 标定

```bash
# 查：哪条臂在哪个口、用的哪个标定文件、什么时候标的
~/anaconda3/envs/evo-rlt/bin/python diagnostics/calibration.py --status

# 标定
~/anaconda3/envs/evo-rlt/bin/python diagnostics/calibration.py --arm left_follower
~/anaconda3/envs/evo-rlt/bin/python diagnostics/calibration.py --arm right_follower
~/anaconda3/envs/evo-rlt/bin/python diagnostics/calibration.py --arm left_leader
~/anaconda3/envs/evo-rlt/bin/python diagnostics/calibration.py --arm right_leader

# 检查：行程统一 + 映射表 + 左右角度差
~/anaconda3/envs/evo-rlt/bin/python diagnostics/calibration.py --check
```

- 已有标定时输入 `c` 回车才是重标，直接回车沿用旧的
- 第 1 步摆到行程中间定零位，第 2 步各关节推到硬限位，`wrist_roll` 不用推
- 换了 USB 转接板才需要重认：`diagnostics/probe_arms.py --identify`

## 相机标定

真机画面和仿真画面并排，按键实时调，调到一致为止。

```bash
~/anaconda3/envs/rlt_sim/bin/python diagnostics/tune_cameras.py
#   --no-real   不开真机相机，只看仿真
```

```
Tab      切换相机 (left_wrist / right_wrist / right_front)
w / s    前 / 后          a / d    左 / 右          r / f    上 / 下
i / k    俯 / 仰          j / l    左转 / 右转      u / o    视场角 -/+
1-5      步长             m        真机实时/冻结
SPACE    保存到 configs/cameras.json      ESC   退出不保存
```

保存后重建场景才生效：`mj_server.py --build`

## 排错

```bash
# 四条臂通信，只读
~/anaconda3/envs/evo-rlt/bin/python diagnostics/probe_arms.py

# 关掉仿真器
~/anaconda3/envs/evo-rlt/bin/python -c "
import sys;sys.path.insert(0,'src')
from evo_rlt.sim.sim_robot import make_sim_robot
r=make_sim_robot();r.connect();r.shutdown_server()"

# 谁占着 5555
ss -lptn 'sport = :5555'

# 串口号
ls -l /dev/serial/by-id/
```

## 路径

```
third_party/SO101/                     URDF + STL
configs/calibration/                   项目快照，仿真读这里
~/.cache/.../robots/so_follower/       follower 标定产物
~/.cache/.../teleoperators/so_leader/  leader 标定产物
~/.cache/evo_rlt/sim_assets/           生成的 MJCF
```

## 待办
- 相机外参：在 `wrist_roll` 修正和复位姿态变更之前调的，需重标
- 遥操摆放工具：用主臂把零件摆好，写回 `configs/task_scene.json`
- milestone / reward 判据
- 撞桌检测：遥操时提示 + 数据里打 flag

交接说明见 `docs/SIM_HANDOFF.md`。