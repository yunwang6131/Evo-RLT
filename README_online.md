# 在线 RL：milestone reward + 双 Replay Buffer
## 分别查看相机
ffplay -fflags nobuffer -flags low_delay -f v4l2 -framerate 30 -video_size 640x480 -input_format yuyv422 /dev/video2

ffplay -fflags nobuffer -flags low_delay -f v4l2 -framerate 30 -video_size 640x480 -input_format yuyv422 /dev/video4

ffplay -fflags nobuffer -flags low_delay -f v4l2 -framerate 30 -video_size 640x480 -input_format yuyv422 /dev/video6
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
  --gamma 0.99 \
  --target-q-clip 3.0 \
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

### Critic 校准参数与健康指标

`--gamma 0.99`、`--target-q-clip 3.0`、`--rankq-margin-relative` 三项是针对 run6（150 集，
自主成功 3/86，介入率始终 0.43 不降）诊断出的两个问题设的，改前的取值分别是 0.9995、
100.0、绝对 margin：

- **Q 高估约 9 倍**：`Q(s,ref)=4.79`，而 buffer 里每集平均回报只有 0.536、史上最好一集
  1.675。`gamma=0.9995` 的有效视界是 `1/(1-γ)=2000` 个 chunk，实测每集只有 32.6 个，
  错配 60 倍，自举链上的正偏差无衰减累积；`target_q_clip=100.0` 这个本该兜底的钳位，
  对着 1.675 的真实回报上界从头到尾一次都没触发过。
- **Critic 把动作排序排反了**：`Q(人类接管动作)=2.843` 低于 `Q(actor 自己的动作)=3.179`，
  77% 的介入样本排序是错的——而人类接管贡献了 94.9% 的成功（56/59），actor 自主成功率
  只有 3.5%。原因是 `rankq_margin` 是绝对值 0.1，Q 漂到 4.8 后它只在约束自身信号的 2%。
  改成相对 margin 后它随 Q 尺度自动缩放。

每次更新会额外打印一行健康检查，wandb 上对应 `online_rl/q_vs_return_ratio`、
`online_rl/q_rank_margin`、`online_rl/q_rank_correct_frac`：

```text
Online RL critic health: Q(ref)=4.790 vs empirical_return=0.536 (ratio=8.94, want ~1) | q_rank_margin=-0.393 (want > 0)
```

`q_rank_margin` 转正之前，其它 loss 曲线再好看都没有意义——actor 的梯度方向本身是错的。

### 动作幅度上限：clip_delta 与 slew_rate_limit 的分工

`--actor-action-clip-delta 0.7` / `--actor-slew-rate-limit 0.05` 是从 0.2 / 0.03 改过来的。
原来的 0.2 太小，直接卡死了人类介入数据的学习：

- 逐元素统计人类介入修正 `|exec - ref|`：p50=0.13、p75=0.28、**p90=0.67**、p99=3.60。
  在 0.2 的上限下 **34.7% 的修正元素 actor 永远无法复现**——`project_action_delta` 用的是
  `ref + limit*tanh(...)`，值域严格是 `±limit`，不是软性建议。
- 而且偏移**不能跨 chunk 累积**：介入期间 `|next_ref − exec_end| = 0.146` 大于
  `|next_ref − ref_end| = 0.105`，说明 VLA 的 ref 每个 chunk 都退回自己的轨迹，
  不跟随机械臂实际到达的位置。
- 这些大修正也不是介入起始的抖动：它们在 chunk 内均匀分布（各时段 90%～97%），
  是持续的真实修正需求。
- 后果已经发生：在 demo 监督实际激活的元素上，actor 的 raw 残差 mean=0.2733，
  **46.1% 已经顶到并越过 0.2 的天花板**（全局均值 0.06 是被 92.6% 被 VLA anchor
  拉住的非 demo 元素稀释出来的假象）。

两个上限的分工：`clip_delta` 限制总幅度，`slew_rate_limit` 限制单步突变，谁更紧谁生效。
旧配置 `0.03 × 25 步 = 0.75` 的累积能力对上 `clip_delta=0.2`，**slew 从未生效过**；
反过来，slew 定得过小（如 0.02，25 步累积 0.5 < 0.7）会让它越权去限制总幅度，
把 clip 放宽的收益吃掉大半。新配置 `0.05 × 25 = 1.25 > 0.7`，两者各司其职。

slew 的取值基准是**人类介入时 actor-residual 的实际单步变化率**（人已经这样操作过机械臂，
硬件没问题）：p50=0.008、p75=0.015、p90=0.029、**p95=0.044**、p99=0.112。
`0.05` 覆盖其中约 96%，再往上收益递减（0.08 只多 4 个百分点，单步幅度却翻倍）。

注意训练侧和部署侧的 slew 锚点不同：部署时 `_last_actor_residual` 跨 chunk 延续
（`action_modifier.py`），训练时用零残差锚点（`losses.py` 的 `_apply_slew_rate_limit_flat`，
replay 未持久化 counterfactual residual，这是刻意的保守选择）。所以训练侧每个 chunk 从 0
重新爬，可复现比例是下界：clip=0.7 下 slew=0.02→64.6%、0.03→74.3%、**0.05→82.2%**、
无 slew→90.6%。

需要更强的训练时平滑约束时，用 `--actor-smoothness-weight`（软惩罚、有梯度，
docstring 里就写着是 slew 的训练期补充），而不是继续压小 slew——后者是拿可达范围换平滑。

> 这一项直接放大机械臂实际动作幅度，没有硬件 E-stop 兜底。第一轮跑的时候手放在
> leader arm 附近，`actor_deploy_scale` 的爬坡期（`--actor-unfreeze-ramp-episodes`）
> 尤其注意。

> 用旧 checkpoint `--resume-from` 时注意：critic 权重是在 `gamma=0.9995` 下训出来的，
> Q 停在 4.8 附近，换上新的 gamma 和 3.0 钳位后前若干集 TD loss 会明显偏大，那是 Q 被
> 拉回正确尺度的过渡期，不是发散。想避开这段过渡就重新开一轮训练。

### 在线操作按键

```text
r          第一次：进入 critical phase；第二次：success 并立即写入在线 buffer
u          failure 并立即写入在线 buffer
m          milestone，只在当前 critical phase 第一次按下时生效
i          左臂人工接管
Space      双臂人工接管；再次按下解除接管
s / f      整个记录 episode 成功 / 失败并结束
```

critical phase 在 `r/u` 后立即结束并切回 VLA；后续 VLA 动作不进入在线 RL buffer。Offline buffer 固定不变，Online buffer 持续增长；每个训练 batch 由 `offline-batch-fraction` 控制混合比例。Offline transition 参与 TD，并用真实 outcome 门控成功示范 BC，但不参与 RankQ；RankQ 的成功/失败排序信号只来自 Online buffer。warmup、success/failure 门槛和 UTD 只统计 Online buffer。warmup 期间 Actor 可以在后台使用成功 offline 示范做纯 BC，但 `actor_deploy_scale=0`，机械臂仍然 100% 执行 VLA；critic-only 期间同样不把已更新的 Actor 输出直接交给机械臂。critic-only 结束后，`actor_deploy_scale` 才在 `actor-unfreeze-ramp-episodes` 内从 0 逐步升到 1，同时 Actor 的 Q 更新频率逐步解冻。

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

```bash
evo-rlt-record full \
  --initial-source vla \
  --setup-json configs/my_so101_manifest.json \
  --policy-path outputs/pin_insert_online_rl/step_000100 \
  --vla-path /home/wangyun/Evo-RLT/pretrained/pretrained_model \
  --rl-token-path outputs/pin_insert_rl_token/checkpoints/010000/pretrained_model \
  --task "Pick up the black hexagonal part with the right arm, pull the gray pin out of the white platform with the left arm, align the gray pin with the hole in the side of the black hexagonal part, insert the gray pin into the hole, and place the assembled object in the red square area." \
  --split-critical-phase \
  --no-rtc \
  --num-episodes 30 \
  --dataset-tag eval_pin_insert
```

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
