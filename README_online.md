# 在线 RL：milestone reward + 双 Replay Buffer
## 分别查看相机
ffplay -fflags nobuffer -flags low_delay -f v4l2 -framerate 30 -video_size 640x480 -input_format yuyv422 /dev/video0

ffplay -fflags nobuffer -flags low_delay -f v4l2 -framerate 30 -video_size 640x480 -input_format yuyv422 /dev/video2

ffplay -fflags nobuffer -flags low_delay -f v4l2 -framerate 30 -video_size 640x480 -input_format yuyv422 /dev/video6

top 25 right 0 left 0
# 看全部相机现在的 brightness
for d in /dev/video*; do v4l2-ctl -d $d --get-ctrl=brightness 2>/dev/null | sed "s|^|$d  |"; done

# 设置曝光度
v4l2-ctl -d /dev/video2 --set-ctrl=brightness=0

for d in 2 4; do v4l2-ctl -d /dev/video$d --set-ctrl=brightness=64; done

for d in /dev/video2 /dev/video4 /dev/video6; do v4l2-ctl -d $d --set-ctrl=brightness=0,contrast=36,saturation=64,white_balance_automatic=1; done

# 回读
for d in /dev/video2 /dev/video4 /dev/video6; do echo -n "$d: "; v4l2-ctl -d $d --get-ctrl=brightness,contrast,saturation,white_balance_automatic; done

## 权限
sudo chmod 666 /dev/ttyACM*
## 查看端口(4条臂逐个插拔确认)
ls -l /dev/ttyACM*
## 查找相机
lerobot-find-cameras opencv # or realsense for Intel Realsense cameras

## 0. 安装与环境

```bash
cd Evo-RLT
conda activate evo-rlt
python -m pip install -e ".[lerobot]"
export TORCHDYNAMO_DISABLE=1
```

## 1. 训练 RL Token

```bash
python -c 'from evo_rlt.adapters.lerobot import register; register(); from lerobot.scripts.lerobot_train import main; main()' \
  --dataset.repo_id=local/merged_screw_v1 \
  --dataset.root=data/bimanual/merged_screw_v1 \
  --policy.type=rlt_token \
  --policy.repo_id=local/pin_insert_rlt_token \
  --policy.push_to_hub=false \
  --policy.vla_pretrained_path=pretrained/pi05_full_ft/pretrained_model \
  --policy.vla_dtype=bfloat16 \
  --policy.rl_token_num_rl_tokens=1 \
  --policy.tokenizer_path=/home/wangyun/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c \
  --policy.token_pool_size=0 \
  --policy.device=cuda \
  --batch_size=2 \
  --steps=10000 \
  --save_freq=2000 \
  --eval_freq=0 \
  --tolerance_s=0.04 \
  --output_dir=outputs/pin_insert_rl_token \
  --job_name=pin_insert_rl_token
```

## 2. 标注离线 critical phase 与 milestone

```bash
conda activate evo-rlt
python diagnostics/critical_segment_labeler_cv.py \
  --dataset-root data/bimanual/merged_screw_v1
```

### 标注按键

```text
r          第一次：critical phase 开始；第二次：结束并标为 success
u          结束 critical phase 并标为 failure
m          在 milestone 已完成的第一帧标注 milestone；再次按可移动
Shift+m    清除 milestone
z          清空当前 episode 的 critical phase 和 milestone
x          标记当前 episode 没有 critical phase
Space      播放 / 暂停
a / d      前后移动 1 帧
A / D      前后移动 10 帧
1 / 2 / 3  0.25x / 0.5x / 1x
Enter      保存并进入下一条
b / n      不保存，切换上一条 / 下一条
q / Esc    退出
```

标签写入：

```text
data/bimanual/merged_screw_v1/meta/critical_segments.json
```

`milestone_frame` 必须统一标在事件完成的第一帧，例如“销钉完全拔出”的第一帧。失败轨迹如果已经达到 milestone，也应标注；没有达到则保持 `null`。

## 3. 构建固定 Offline Replay Buffer

离线和在线必须使用完全相同的 `chunk-length`、`rl-action-arms`、`milestone-reward`、`terminal-reward` 和 `time-decay`。下面的 cache 命令已与第 4 节的新训练命令对齐；修改其中任何一项时必须同步修改另一条命令并重新生成 cache。

```bash
evo-rlt-build-transition-cache-v2 \
  --demo-dataset-repo-id local/merged_screw_v1 \
  --demo-dataset-root data/bimanual/merged_screw_v1 \
  --rl-token-policy-path outputs/pin_insert_rl_token/checkpoints/010000/pretrained_model \
  --vla-pretrained-path /home/wangyun/Evo-RLT/pretrained/pretrained_model \
  --tokenizer-path /home/wangyun/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c \
  --output-dir outputs/pin_insert_offline_cache \
  --task-instruction "Pick up the black hexagonal part with the right arm, pull the gray pin out of the white platform with the left arm, align the gray pin with the hole in the side of the black hexagonal part, insert the gray pin into the hole, and place the assembled object in the red square area." \
  --chunk-length 25 \
  --frame-stride 2 \
  --rl-action-arms both \
  --milestone-reward 0.5 \
  --terminal-reward 2.0 \
  --time-decay 0.98 \
  --batch-size 32 \
  --num-workers 32 \
  --train-ratio 0.9 \
  --tolerance-s 0.04 \
  --device cuda
```

当前缓存会把成功的 offline 示范动作同时用于 Critic 和 Actor：Critic 学真实执行动作的回报，Actor 在其可控手臂维度上直接模仿示范。

## 4. 在线训练：Offline + Online 双 Buffer

```bash
evo-rlt-online-train \
  --setup-json configs/my_so101_manifest.json \
  --vla-path /home/wangyun/Evo-RLT/pretrained/pretrained_model \
  --rl-token-path outputs/pin_insert_rl_token/checkpoints/010000/pretrained_model \
  --tokenizer-path /home/wangyun/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c \
  --task "Pick up the black hexagonal part with the right arm, pull the gray pin out of the white platform with the left arm, align the gray pin with the hole in the side of the black hexagonal part, insert the gray pin into the hole, and place the assembled object in the red square area." \
  --num-episodes 300 \
  --chunk-length 25 \
  --chunk-exec-steps 25 \
  --rl-action-arms both \
  --actor-action-clip-delta 0.7 \
  --actor-slew-rate-limit 0.05 \
  --offline-cache-path outputs/pin_insert_offline_cache \
  --offline-batch-fraction 0.5 \
  --milestone-reward 0.5 \
  --terminal-reward 2.0 \
  --time-decay 0.98 \
  --beta 0.3 \
  --demo-bc-weight 1.0 \
  --gamma 0.9995 \
  --target-q-clip 3.0 \
  --target-q-min 0.0 \
  --rankq-margin 0.1 \
  --rankq-margin-relative \
  --warmup-episodes 5 \
  --min-warmup-transitions 1000 \
  --min-warmup-successes 3 \
  --min-warmup-failures 3 \
  --critic-layer-norm \
  --utd-ratio 2 \
  --save-every-episodes 10 \
  --wandb \
  --wandb-project rlt-both \
  --wandb-run-name run6  \
  --resume-from outputs/online_rl/0807_online_rl/eval_online_rl_111713/latest_online_state.pt \
  --save-dir outputs/online_rl/0807_online_rl/eval_online_rl_111713 \
  --wandb-run-id tki5f2gq \
  --wandb-resume must
```

## 4.5 离线训练：用已采集的数据训练可部署模型

```bash
python -m evo_rlt.cli.train_offline_rl \
  --online-state outputs/online_rl/0807_online_rl/eval_online_rl_111713/latest_online_state.pt \
  --output-dir outputs/offline_rl/run4 \
  --policy-path outputs/online_rl/0807_online_rl/eval_online_rl_111713/step_000150 \
  --warm-start \
  --gradient-steps 100000 \
  --batch-size 256 \
  --gamma 0.9995 \
  --beta 0.3 \
  --demo-bc-weight 1.0 \
  --target-q-clip 3.0 \
  --target-q-min 0.0 \
  --rankq-margin 0.1 \
  --rankq-margin-relative \
  --rankq-alpha-success 1.0 \
  --rankq-alpha-failure 1.0 \
  --actor-action-clip-delta 0.7 \
  --actor-slew-rate-limit 0.05 \
  --lr-actor 3e-5 \
  --lr-critic 1e-4 \
  --save-every 10000 \
  --wandb --wandb-project rlt-offline --wandb-run-name offline4
```

  --offline-cache-path outputs/pin_insert_offline_cache \
  --offline-batch-fraction 0.5 \
未显式传的超参从 `--policy-path` 的 `config.json` 继承（默认取第一个 `--online-state`
旁最新的 `step_*/`），那里是采集时的旧值，要用新值必须写出来。

`--online-state` 可重复传多次合并多轮采集。`--warm-start` 从 checkpoint 载入 Actor/Critic，
默认随机初始化。`--eval-every 0` 关闭 val 评估。

产出格式与在线相同，直接用第 7 节的命令部署：

```bash
evo-rlt-record full --policy-path outputs/offline_rl/run1/final ...
```

W&B 指标：`offline_rl/*`（同在线的 `online_rl/*`）加 `val/q_gap`、`val/expert_action_mse`、
`val/ref_action_mse`。

## 5. 自定义 Episode Reset 位置

把下面参数追加到在线训练命令：

```bash
  --go-home-positions '{"left_shoulder_pan.pos": 2038, "left_shoulder_lift.pos": 2081, "left_elbow_flex.pos": 3034, "left_wrist_flex.pos": 1142, "left_gripper.pos": 2164, "right_shoulder_pan.pos": 2066, "right_shoulder_lift.pos": 2160, "right_elbow_flex.pos": 2880, "right_wrist_flex.pos": 1066, "right_gripper.pos": 2209}'
```

## 6. 恢复在线训练

下面是历史 run `yiycy1r4` 的恢复命令，因此仍使用该 run 原来的
`time-decay=0.995` 和旧奖励尺度。它不能与第 3、4 节新建的 `time-decay=0.98`
cache/run 混用；新一轮训练不要使用这里的 `--resume-from`。

```bash
evo-rlt-online-train \
  --setup-json configs/my_so101_manifest.json \
  --vla-path /home/wangyun/Evo-RLT/pretrained/pretrained_model \
  --rl-token-path outputs/pin_insert_rl_token/checkpoints/010000/pretrained_model \
  --tokenizer-path /home/wangyun/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c \
  --task "Pick up the black hexagonal part with the right arm, pull the gray pin out of the white platform with the left arm, align the gray pin with the hole in the side of the black hexagonal part, insert the gray pin into the hole, and place the assembled object in the red square area." \
  --num-episodes 300 \
  --episode-time-s 3000 \
  --reset-time-s 15 \
  --fps 30 \
  --chunk-length 25 \
  --chunk-exec-steps 25 \
  --action-dim 12 \
  --proprio-dim 12 \
  --rl-action-arms both \
  --actor-action-clip-delta 0.1 \
  --actor-slew-rate-limit 0.03 \
  --actor-smoothness-weight 0.0 \
  --actor-hidden-dim 512 \
  --actor-num-layers 3 \
  --actor-activation relu \
  --actor-residual \
  --no-actor-layer-norm \
  --actor-fixed-std 0.0 \
  --critic-hidden-dim 512 \
  --critic-num-layers 3 \
  --critic-activation relu \
  --critic-residual \
  --critic-layer-norm \
  --rankq-alpha-success 1.0 \
  --rankq-alpha-failure 1.0 \
  --rankq-noise-scale 0.15 \
  --rankq-margin 0.1 \
  --target-noise-std 0.1 \
  --target-noise-clip 0.3 \
  --offline-cache-path outputs/pin_insert_offline_cache \
  --offline-batch-fraction 0.5 \
  --milestone-reward 0.5 \
  --terminal-reward 1.0 \
  --time-decay 0.995 \
  --beta 0.3 \
  --demo-bc-weight 1.0 \
  --gamma 0.9995 \
  --tau 0.005 \
  --actor-update-interval 2 \
  --warmup-episodes 5 \
  --critic-only-episodes 10 \
  --actor-unfreeze-ramp-episodes 10 \
  --min-warmup-transitions 1000 \
  --min-warmup-successes 3 \
  --min-warmup-failures 3 \
  --stratified-sampling \
  --replay-capacity 20000 \
  --batch-size 256 \
  --lr-actor 3e-5 \
  --lr-critic 1e-4 \
  --utd-ratio 2 \
  --max-updates-per-episode 1000 \
  --intervention-blend-time-s 0.3 \
  --vla-ref \
  --play-sounds \
  --go-home-time-s 3.0 \
  --go-home-gripper-value 100.0 \
  --save-every-episodes 10 \
  --resume-from outputs/online_rl/0805_online_rl/eval_online_rl_110435/latest_online_state.pt \
  --save-dir outputs/online_rl/0805_online_rl/eval_online_rl_110435 \
  --wandb \
  --wandb-project rlt-both \
  --wandb-run-name run1 \
  --wandb-run-id yiycy1r4 \
  --wandb-resume must
```


## 7. 评测

### 纯VLA
evo-rlt-record full --initial-source vla --setup-json configs/my_so101_manifest.json --policy-path /home/wangyun/Evo-RLT/pretrained/pretrained_model --task "Pick up the black hexagonal part with the right arm, pull the gray pin out of the white platform with the left arm, align the gray pin with the hole in the side of the black hexagonal part, insert the gray pin into the hole, and place the assembled object in the red square area." --dataset-tag pi05_baseline_eval --num-episodes 30 --episode-time-s 600 --reset-time-s 6 --fps 30 --vcodec h264 --no-teleop

### RLT

evo-rlt-record full --initial-source vla --setup-json configs/my_so101_manifest.json --policy-path outputs/offline_rl/run4/step_080000 --rl-token-path outputs/pin_insert_rl_token/checkpoints/010000/pretrained_model --task "Pick up the black hexagonal part with the right arm, pull the gray pin out of the white platform with the left arm, align the gray pin with the hole in the side of the black hexagonal part, insert the gray pin into the hole, and place the assembled object in the red square area." --split-critical-phase --no-rtc --num-episodes 30 --dataset-tag eval_pin_insert

## 8. 收集 PI0.5 Baseline

```bash
evo-rlt-record full \
  --initial-source vla \
  --setup-json configs/my_so101_manifest.json \
  --policy-path /home/wangyun/Evo-RLT/pretrained/pretrained_model \
  --task "Pick up the black hexagonal part with the right arm, pull the gray pin out of the white platform with the left arm, align the gray pin with the hole in the side of the black hexagonal part, insert the gray pin into the hole, and place the assembled object in the red square area." \
  --dataset-tag pi05_baseline_eval \
  --num-episodes 30 \
  --episode-time-s 600 \
  --reset-time-s 6 \
  --fps 30 \
  --vcodec h264
```

## 9. 从 Online Buffer 删除错误 Episode

### 预览

```bash
evo-rlt-prune-online-state \
  --state-path outputs/pin_insert_online_rl/latest_online_state.pt \
  --episode-id 41
```

### 覆盖原状态

```bash
evo-rlt-prune-online-state \
  --state-path outputs/pin_insert_online_rl/latest_online_state.pt \
  --episode-id 41 \
  --in-place
```

`--in-place` 会先生成 `latest_online_state.pt.bak`。

## 10. W&B 指标速查
### 训练状态

| W&B 指标 | 含义 |
|---|---|
| `online_rl/warmup_satisfied` | `1` 表示 warmup 的 episode、transition、成功和失败门槛均已满足。 |
| `online_rl/critic_only` | `1` 表示只更新 critic，actor 尚未解冻。 |
| `online_rl/new_transitions` | 当前 episode 新增的 Online Buffer transition 数。 |
| `online_rl/actual_updates` | 当前 episode 结束后实际执行的梯度更新次数。 |
| `online_rl/effective_utd` | `actual_updates / new_transitions`。 |
| `online_rl/training_time_s` | 当前 episode 结束后的训练耗时，单位为秒。 |

### Replay Buffer 与 Batch

| W&B 指标 | 含义 |
|---|---|
| `online_rl/buffer_transitions` | 当前 Online Replay Buffer 的 transition 数。 |
| `online_rl/offline_buffer_transitions` | 固定 Offline Replay Buffer 的 transition 数。 |
| `online_rl/offline_batch_size` | 每个训练 batch 实际抽取的 offline 样本数。 |
| `online_rl/online_batch_size` | 每个训练 batch 实际抽取的 online 样本数。 |
| `online_rl/buffer_successes` | Online Buffer 中已完成的成功 critical-phase episode 数。 |
| `online_rl/buffer_failures` | Online Buffer 中已完成的失败 critical-phase episode 数。 |

### Episode 与真机表现

| W&B 指标 | 含义 |
|---|---|
| `online_rl/episode_reward` | 最新完成 episode 的 milestone reward 与 terminal reward 总和。 |
| `online_rl/episode_intervened` | 最新 episode 是否发生人工干预：`1` 是，`0` 否。 |
| `online_rl/episode_autonomous_success` | 最新 episode 是否在无人工干预下成功。 |
| `online_rl/autonomous_episodes` | 累计无人工干预 episode 数。 |
| `online_rl/autonomous_successes` | 累计无人工干预成功数。 |
| `online_rl/autonomous_success_rate` | 累计自主成功率。 |
| `online_rl/autonomous_success_rate_rolling_20` | 最近 20 个已标注 episode 中的自主成功率。 |
| `online_rl/autonomous_episodes_rolling_20` | 最近 20 个已标注 episode 中无人工干预的数量。 |
| `online_rl/intervention_rate` | 累计发生人工干预的 episode 比例。 |
| `online_rl/intervention_rate_rolling_20` | 最近 20 个已标注 episode 的人工干预比例。 |

### Loss 与 Q 诊断

| W&B 指标 | 含义 |
|---|---|
| `online_rl/loss` | 最后一次 gradient update 的总 loss。 |
| `online_rl/loss_critic` | Twin Critic 的 TD MSE，加上启用时的 RankQ loss。 |
| `online_rl/loss_actor` | Actor 的 `-Q(s, actor(s)) + beta × BC`；只在发生 actor update 后记录。 |
| `online_rl/q_action_sensitivity` | 同一 state 下，`exec/noisy/very_noisy/random/permuted` 五种 action 的 Q 标准差，再对 batch 取平均。 |

`q_action_sensitivity` 的判断方式：

| 现象 | 含义 |
|---|---|
| 长期接近 `0` | Critic 几乎忽略 action，可能从 `Q(s,a)` 塌陷成 `V(s)`。 |
| 保持明显非零 | Critic 能区分 action，但不代表 Q 值没有高估。 |
| 快速增大，同时自主成功率下降 | Critic 可能产生错误的 action 偏好，需要结合 Q 绝对值进一步诊断。 |

当前日志不能直接判定 Q overestimation；`loss_critic` 很低或 `q_action_sensitivity` 非零都不能单独证明 Q 值准确。

### W&B 启动参数

| 参数 | 含义 |
|---|---|
| `--wandb` | 开启 W&B 日志。 |
| `--wandb-project` | W&B project 名称。 |
| `--wandb-run-name` | 当前 run 的显示名称。 |
| `--wandb-entity` | W&B 用户或团队名称；默认使用当前登录账户。 |
| `--wandb-run-id` | 固定 run ID，恢复同一个 W&B run 时使用。 |
| `--wandb-resume` | W&B 恢复策略；严格恢复使用 `must`。 |

## 11. Reward 与更新公式

```text
milestone = milestone_reward × time_decay ^ milestone前已关闭chunk数
terminal  = success × terminal_reward × time_decay ^ 结束时已关闭chunk数
episode_reward = milestone + terminal

updates = min(本episode新增在线transition数 × utd_ratio,
              max_updates_per_episode)
```

```text
warmup 默认门槛：
recorded_episodes >= 5
online transitions >= 1000
online success episodes >= 3
online failure episodes >= 3
```
