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

**重建之后必须把终端 1 的仿真器重启。** `--build` 只是重写 `scene.xml`，已经在跑的
服务器是启动时把场景读进内存的，不会重新加载 —— 你会对着一个旧场景调半天，而且
所有症状都指向"改动没生效"（夹不住、穿桌、穿钳口，正是修复前的样子）。
服务器启动时会打印 scene 路径，对一下时间戳：

```bash
ls -l ~/.cache/evo_rlt/sim_assets/scene.xml   # 比服务器启动时间新 = 要重启
```

## 启动（两个终端）

```bash
# 终端 1：仿真器，一直开着
~/anaconda3/envs/rlt_sim/bin/python src/evo_rlt/sim/mj_server.py --viewer --show-cameras

# 终端 2：遥操，用真机主臂驱动仿真
~/anaconda3/envs/evo-rlt/bin/python diagnostics/teleop_sim.py --duration 120 --save outputs/solo_check.json
```

跑起来后在终端 2 里按 `b` 复位零件，见下节。

## 力反馈（主臂能感觉到从臂被挡住）

从臂顶到桌子、或夹爪夹到东西合不动时，让主臂给手一个阻力。

```bash
~/anaconda3/envs/evo-rlt/bin/python diagnostics/teleop_sim.py --force-feedback
#   --fb-gain 0.3        主臂朝从臂位置移动的比例，越大越硬也越易振荡
#   --fb-deadband 2.0    死区，位置差小于它不出力
#   --fb-torque 15       主臂力矩上限，占满量程百分比
```

两个关键设计，都是踩过坑换来的：

1. **阻力接在从臂的「实测 − 指令」上**，不是「主臂位置 − 从臂位置」。后者在
   正常跟随时就差一个纯延迟（实测 3 步 = 100ms，以 86°/s 挥臂时差 8.6°），
   会被误判成「被挡住」，于是**全程**很大阻力。
2. **自由运动时主臂是断电的**，只有误差超过死区才通电。一直通着电的话，
   `Goal_Position` 每 33ms 才刷新一次，你一动伺服就往一个周期前的旧位置拽 ——
   这是恒定的黏滞阻力，和有没有被挡住无关，同样表现为「动起来十分费力」。

死区默认 5.0，取自真机 6258 帧实测的残余误差（对齐后 p95=2.33、p99=4.79）。

跑完会打印一张表：各关节真正出过力的步数、最大误差、通断次数。全 0 说明从臂
一次都没被挡住（手臂没碰到任何东西），不是功能坏了 —— 这两种情况没有这张表
分不开。通断次数很多说明死区取小了，主臂会咔咔响。

**通电时主臂会主动出力，而你的手正握着它。** 所以默认关闭，而且：

- **第一次先在 solo 模式试**（`--no-followers`，从臂是仿真），撞坏不了东西
- 手一直握住主臂，别撒手
- **感觉到嗡嗡震颤就是环路在振荡** —— 降 `--fb-gain` 或加大 `--fb-deadband`。
  这是 position-position 双边环，在 30 Hz + 实测约 100ms 纯延迟下本来就容易自激，
  所以默认值调得很保守（只有满力矩的 15%），是"提示性阻力"不是"硬墙"
- Ctrl-C 正常退出会自动断力矩；进程被 `kill -9` 不会，那种情况重新连一次即可

原理和参数含义见 `src/evo_rlt/sim/feedback.py` 的模块说明。

## 用仿真采 VLA 数据

主臂是真的，从臂是仿真。先起仿真器，再用 `--sim` 跑录制：

```bash
# 终端 1
~/anaconda3/envs/rlt_sim/bin/python src/evo_rlt/sim/mj_server.py --viewer --show-cameras

# 终端 2：和真机录制同一个入口，只多一个 --sim
~/anaconda3/envs/evo-rlt/bin/python -m evo_rlt.adapters.lerobot.record full \
    --initial-source teleop --sim --setup-json configs/my_so101_manifest.json \
    --task "Pick up the hexagonal part with the right arm, pull the pin out of the platform with the left arm, align the pin with the hole in the hexagonal part, and insert the pin into the hole." \
    --num-episodes 50 \
    --episode-time-s 300 \
    --reset-time-s 3 \
    --discard-unlabeled-episodes
```
# 回放数据仿真

# 终端 1：仿真器
~/anaconda3/envs/rlt_sim/bin/python src/evo_rlt/sim/mj_server.py --viewer

# 终端 2
~/anaconda3/envs/evo-rlt/bin/python -c "
from evo_rlt.adapters.lerobot.registry import register; register()
from lerobot.scripts.lerobot_replay import replay
replay()
" --robot.type=sim_bi_so_follower \
  --dataset.root=data/bimanual/0821_teleop_full/record_teleop_full_153426 \
  --dataset.repo_id=local/record_teleop_full_153426 \
  --dataset.episode=0
# 回放数据视频
PATH="$HOME/anaconda3/envs/evo-rlt/bin:$PATH" \
~/anaconda3/envs/evo-rlt/bin/lerobot-dataset-viz \
  --root data/bimanual/0821_teleop_full/record_teleop_full_153426 \
  --repo-id local/record_teleop_full_153426 \
  --episode-index 0


**复位时螺套会在圆形凹槽里随机摆放**（位置 + 朝向），这是 VLA 数据的初始位姿
多样性来源；不随机的话策略学到的是「走到那个固定位置」而不是「找到零件」。
范围在 `configs/task_scene.json` 的 `socket.reset_random` 里，实测约束见那里的
注释。要复现同一批位姿用 `mj_server.py --random-seed 0`；把 `radius` 设 0 即关闭。

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
~/anaconda3/envs/rlt_sim/bin/python diagnostics/widen_holes.py --extra-mm 1.8
~/anaconda3/envs/rlt_sim/bin/python diagnostics/decompose_mesh.py <stl> --threshold 0.005 --max-hulls 256
~/anaconda3/envs/rlt_sim/bin/python diagnostics/check_hole_fit.py                # 必须核对,见下
~/anaconda3/envs/rlt_sim/bin/python diagnostics/settle_objects.py --apply        # 让物理找零件的平衡位姿
```

**`--max-hulls` 一定要写。** 它默认只有 64,而桌子需要 255 块;用默认值分解出的
孔壁粗得多,同样的 `--extra-mm 1.8` 下孔 B 的通路半径从 6.13mm 掉到 5.28mm
(真值 6.00),孔凭空紧了 0.85mm。块数不够不会报错,只会让孔悄悄变形。

`--extra-mm` 是补偿凸分解啃掉的孔壁,**不是想扩多大就扩多大**:目标是让最终
通路半径等于 CAD 里的真实孔径。当前这组值的实测(螺栓杆半径 4.75mm):

| 孔 | CAD 真值 | 加宽后 STL | 分解后通路 | 偏差 |
|---|---|---|---|---|
| B (0.220, 0.055) 插螺栓 | 6.00 mm | 7.80 | 6.13 mm | +0.13 |
| A (0.220, 0.085) | 5.00 mm | 6.80 | 5.13 mm | +0.13 |

改了孔或换了分解参数,跑 `check_hole_fit.py` 对着这张表核一遍。

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
