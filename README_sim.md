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

当前蓝色螺栓数据对应的完整仿真环境已经封存为
`snapshots/sim/blue_screw_v1`。修改场景前后可核对，误改后可恢复：

```bash
evo-rlt-sim-snapshot verify
evo-rlt-sim-snapshot restore
```

恢复会先把现状备份到 `outputs/sim_snapshot_backups/`，然后只覆盖快照声明的
仿真文件。恢复后仍需重新 `--build` 并重启 `mj_server.py`。ACT 的数据合并、
训练和仿真评估命令见 `README_ACT.md`。

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
## 自动成功判据

在此之前 `episode_success` 只有人按键一条路,122 条源数据就是这么标的。人标不了
增广和脚本采集 —— 那些一次产出上千条,没有自动判据就只能全部当成功收下。

判据只看零件真值位姿,不看接触:孔半径 5.20mm、杆半径 4.75mm,单边间隙 0.45mm,
杆尖一旦越过孔口平面且横向偏移在间隙量级,几何上它就只能在孔里。参数在
`configs/task_success.json`,几何常量是从场景网格量出来的。

```python
from evo_rlt.sim import task_success as ts
config = ts.load_config()
state = ts.evaluate(robot.object_poses, config)   # 逐帧
ts.episode_succeeded(states)                       # 整条(要连续 10 帧成立)
ts.furthest_stage(states)                          # idle/socket_lifted/bolt_pulled/aligned/inserted
```

**`furthest_stage` 是排错用的,别忽略它。** rollout 全失败时,"完全没学会"和
"学会了但最后对不准"在成功率上都是 0%,该做的事却完全相反 —— 前者加多少数据
都没用。分阶段判据是唯一能把这两种分开的东西。

零件位姿现在跟着每一帧观测一起回来(协议 v2):

```python
robot.object_poses    # {"socket": [x,y,z,qw,qx,qy,qz], "bolt": [...]}
robot.ee_poses        # 两只夹爪的 gripper_link 位姿
robot.fk(qpos_batch)  # 批量正运动学
robot.ik(side, targets, seed)   # 批量逆运动学
```

**协议版本从 1 涨到 2,旧的仿真器进程必须重启**,否则客户端握手就会拒绝。这是
故意的:旧仿真器不回 `object_poses`,判据会静默地拿到空字典,那等于"每条都判
失败",而不是一个能看见的错误。

## 数据不够:用已有演示增广

122 条演示里真正随机的只有一件事 —— 螺套复位时落在凹槽里的位置和朝向。螺栓的
初始位姿是固定的。所以这 122 条是对一个三维随机量的 122 次采样,可以把每条搬到
新的螺套位姿上重放:抓取前的末端轨迹整体平移,抓到之后连零件一起平移,插入因此
发生在工作空间的另一处。最难的对准和插入那一段是逐帧照抄人的,不是脚本编的。

```bash
# 终端 1:仿真器(必须是重启过的,协议 v2)
~/anaconda3/envs/rlt_sim/bin/python src/evo_rlt/sim/mj_server.py

# 终端 2
conda activate evo-rlt
evo-rlt-sim-augment calibrate                      # 一次性,约 35 分钟
evo-rlt-sim-augment run --out-root data/bimanual/blue_screw_aug_v1 --per-source 8
```

### calibrate 在做什么

**源数据里没有记录螺套的初始位姿** —— 采集时零件坐标从没送出过仿真进程。这一步
先由抓取那一帧的夹爪位姿反推一个估计,再拿重放本身把它验证:摆上去跑一遍,自动
判据说成功才算数。跑不通的源演示不进标定文件,因为它的几何没被复现出来,拿去
增广只会批量产出失败。

反推靠的是"螺套在夹爪坐标系里的位置是个常量"(人每次都以同样的姿势去抓六棱柱)。
这个常量用最小二乘拟合,自检指标是:推出来的螺套位置该像一个半径 25mm 的均匀
圆盘。实测和理论分位数对得上(q50 16.0 / 17.7mm,86% 落在 25mm 内),说明抓取帧
找对了。**只让 z 对上的一维拟合会秩亏** —— 那条弯路走过,解出来的螺套落在离凹槽
106mm 的地方,数值上毫无异常。

### 三个实测出来的关键约束

**1. SO-101 只有 5 个本体关节,够不到任意 6D 位姿,缺的是绕世界 z 轴的偏航。**
对纯平移目标解 IK,姿态残差的转轴 z 分量恒为 0.94,大小 0.135 度/毫米。所以位移
只做平移;偏航残差在抓取后**反过来读回夹爪实际到达的位姿**,由它决定螺套摆在哪 ——
螺套的偏航本来就是均匀随机的,被改掉几度不损失任何东西,而孔轴是它自己的 z 轴,
绕 z 转多少都不动,插入完全不受影响。

IK 的姿态权重因此按世界坐标轴分开给,默认 `(1, 1, 0.02)`:位置 3 维 + 倾角 2 维
= 5,和自由度数正好相等。各向同性的小权重会让手腕倾角自己漂(同样平移下倾角
误差 0.46~2.11 度),分轴之后是 0.00 度。

**2. 钳口会把六角重新坐正,所以摆件误差和抓后位姿不是一一对应的。**
只改摆件位姿去修对不准,重放成功率只能从 32% 推到 42%。改成平移**抓稳之后的
右臂轨迹**才是精确的 —— 零件被刚性握着,手走多少它就走多少,横偏能收敛到 0.1mm,
成功率到 58%。修正量由一次不录的重放量出来,存在标定文件的 `hold_correction`。

**3. 剩下的失败几乎全是轴线夹角,不是横偏。** 典型是"横偏 0.11mm 夹角 19.1°"。
根因是螺套在钳口里的倾斜和源演示当时那一次不同(成功那条插入瞬间螺套倾 12.6°、
螺栓倾 12.7°、轴夹角 0.1°;失败那条螺套只倾 1.8° 而螺栓 13.3°)。**下一步就是把
握持修正从"只平移"扩到"带一个绕水平轴的转动"**,分轴 IK 已经能精确跟随倾角了,
缺的是测量与施加那一段。

### run 的产出

数据集的列、dtype、shape、names 逐字照抄源数据集的 `meta/info.json`,所以可以
直接和源数据合并后一起训。每条跑完过一遍自动判据,只有成功的写进去;结尾会报
保留率和失败都停在哪一级。

`--max-delta` 是单次位移上限(默认 30mm)。越大多样性越好,IK 的偏航残差也越大
(0.135 度/毫米),而摆件朝向虽然能吸收它,夹爪相对台面的姿态也跟着变,过大会蹭
到台面。

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
## ACT训练
conda activate evo-rlt
lerobot-train \
  --dataset.repo_id=local/blue_screw_sim_v1 \
  --dataset.root=/home/wangyun/Evo-RLT/data/bimanual/blue_screw_sim_v1 \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.chunk_size=100 \
  --policy.n_action_steps=10 \
  --output_dir=/home/wangyun/Evo-RLT/outputs/act_blue_screw_sim_v1 \
  --job_name=act_blue_screw_sim_v1 \
  --batch_size=8 \
  --steps=60000 \
  --save_freq=10000 \
  --log_freq=100 \
  --num_workers=4 \
  --wandb.enable=false

## ACT 评测
cd /home/wangyun/Evo-RLT
~/anaconda3/envs/rlt_sim/bin/python src/evo_rlt/sim/mj_server.py --viewer 

conda activate evo-rlt
cd /home/wangyun/Evo-RLT
evo-rlt-act rollout \
  --checkpoint outputs/act_blue_screw_sim_v1/checkpoints/060000_ensemble/pretrained_model \
  --num-episodes 30 \
  --episode-time-s 150

## smolVLA指令

conda activate evo-rlt
cd /home/wangyun/Evo-RLT
HF_HUB_OFFLINE=1 lerobot-train --config_path=configs/smolvla/train_config.json

### 接着已有 checkpoint 继续训
`--steps` 是从 0 算起的新总步数;`--scheduler.num_decay_steps` 必须一起改,
否则学习率停在谷底,多训的步数等于白跑。别加 `--policy.path`,那会让 resume 失效、从头开始。

HF_HUB_OFFLINE=1 lerobot-train --config_path=outputs/smolvla_blue_screw_sim_v1/checkpoints/last/pretrained_model/train_config.json --resume=true --steps=60000 --scheduler.num_decay_steps=60000

## smolVLA评测
conda activate evo-rlt
evo-rlt-smolvla rollout \
  --checkpoint outputs/smolvla_blue_screw_sim_v1/checkpoints/050000/pretrained_model \
  --num-episodes 10

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
