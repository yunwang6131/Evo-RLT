# online RL 训练（真机，同步版）

新增 CLI: `evo-rlt-online-train`。每个 episode = 一段 critical phase（r 进入，r 或双击 r 结束+打标签），结束后立刻在当前 policy 上做梯度更新（更新次数按这个episode实际新增的transition数动态算，不是固定次数），不存盘重载，下一个 episode 直接用新权重。rollout 和训练不并发。

actor 用 zero-init 残差头（mu = ref + delta，delta 最后一层零初始化），没训练之前 actor 输出 == VLA 参考，第一轮就是安全的——不需要offline先训一版actor-critic再warm start，直接从零开始在线训也是安全的。

## 新任务完整步骤

```
task = "Pick up the black hexagonal part with the right arm, pull the gray pin out of the white platform with the left arm, align the gray pin with the hole in the side of the black hexagonal part, insert the gray pin into the hole, and place the assembled object in the red square area."
```

### 0. VLA checkpoint 就位

`pretrained/<vla_name>/pretrained_model` 放好。如果这个VLA还没在新任务上微调过（纯base pi0.5，没见过这个任务的demo），先采一批teleop demo做SFT微调；如果已经是针对这个任务微调过的checkpoint，跳过这步直接到第1步。

```bash
# 采集SFT用的teleop demo（跟README_dualarm_rlt.md流程一样，只是task换了）
evo-rlt-record full \
  --initial-source teleop \
  --setup-json configs/my_so101_manifest.json \
  --dataset-tag pin_insert_sft \
  --task "$task" \
  --num-episodes 50 \
  --episode-time-s 300 \
  --reset-time-s 6 \
  --fps 30 \
  --vcodec h264 \
  --discard-unlabeled-episodes

# 挪进统一训练目录、合并多个session（参考README_dualarm_rlt.md），假设最终产出：
#   data/bimanual/merged_pin_insert/

# SFT微调VLA
python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/merged_pin_insert \
  --dataset.root=data/bimanual/merged_pin_insert \
  --policy.path=pretrained/pi05_base/pretrained_model \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --batch_size=16 \
  --steps=30000 \
  --save_freq=5000 \
  --eval_freq=0 \
  --tolerance_s=0.04 \
  --output_dir=outputs/pin_insert_vla_ft \
  --job_name=pin_insert_vla_ft
```

### 1. 训练 RL Token（冻结VLA，只训一个encoder做特征压缩）

不需要成功/失败标签，用上面同一批（或另采一批）teleop demo即可。

```bash
python -c 'from evo_rlt.adapters.lerobot import register; register(); from lerobot.scripts.lerobot_train import main; main()' \
  --dataset.repo_id=local/merged_pin_insert \
  --dataset.root=data/bimanual/merged_pin_insert \
  --policy.type=rlt_token \
  --policy.repo_id=local/pin_insert_rlt_token \
  --policy.push_to_hub=false \
  --policy.vla_pretrained_path=outputs/pin_insert_vla_ft/checkpoints/last/pretrained_model \
  --policy.vla_dtype=bfloat16 \
  --policy.rl_token_num_rl_tokens=1 \
  --policy.tokenizer_path=/home/wangyun/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c \
  --policy.token_pool_size=0 \
  --policy.device=cuda \
  --batch_size=8 \
  --steps=10000 \
  --save_freq=2000 \
  --eval_freq=0 \
  --tolerance_s=0.04 \
  --output_dir=outputs/pin_insert_rl_token \
  --job_name=pin_insert_rl_token
```

到这里为止，跟离线流程（README_dualarm_rlt.md）是共用的——**在线训练不需要再做`build-transition-cache`和`train-actor-critic`这两步**，`evo-rlt-online-train`会现场构造一个全新初始化的actor/critic。

### 2. 跑在线RL训练

```bash
evo-rlt-online-train \
  --setup-json configs/my_so101_manifest.json \
  --vla-path outputs/pin_insert_vla_ft/checkpoints/last/pretrained_model \
  --rl-token-path outputs/pin_insert_rl_token/checkpoints/last/pretrained_model \
  --tokenizer-path /home/wangyun/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c \
  --task "$task" \
  --num-episodes 5 \
  --actor-action-clip-delta 0.05 \
  --save-dir outputs/pin_insert_online_rl \
  --save-every-episodes 5 \
  --dry-run   # 先跑一遍dry-run确认拼出来的argv和按键提示没问题，再去掉这行真跑
```

**第一次跑强烈建议：** `--num-episodes` 设小（3-5，上面示例用的5）、`--actor-action-clip-delta` 保守（0.05，比默认0.1更保守），全程手放在leader臂附近。确认actor从"等于VLA"平稳过渡、日志里能看到`Online RL update after episode ...`且loss不是NaN，再加大`--num-episodes`跑正式session（默认50）。

## 按键 + episode之间的reset窗口

```text
$task 里按 r 键：进入/结束 critical phase（单击=success，double-tap-window-s窗口内双击=failure）
space（或 -i）：teleop 接管（leader backdrive）
x（estop-key）：强制接管，效果同上，多一个热键保险
```

**每个episode结束后（不管成功还是失败），都会有一段`--reset-time-s`秒（默认15）的纯teleop窗口，VLA/RL都不发动作，只有leader臂能控制follower**，用来把物件摆回起始位置。这段时间不计入数据/replay buffer。这是必须的——如果失败的episode把pin插歪了或者夹爪停在奇怪的姿态，下一个episode一开始VLA（冻结的、没在这个失败状态上训练过）会立刻自主接管，对着这个没见过的状态乱动，没人会先手动摆好。reset窗口结束后才轮到下一个episode开始录制。

## 安全须知

- `estop-key`/`space` 不是硬件急停，只是让policy停止发新动作、交还给leader臂，不断电。手一直放在leader臂附近，物理断电开关在触手可及的地方。
- v1 不支持 RTC（`--rlt.rtc_enabled` 强制关闭）、不支持异步rollout/训练。
- Human intervention期间VLA/RLT推理是关掉的（省算力），所以state/ref用的是intervention开始前最近一次真实编码，不是intervention那一刻的实时编码；intervention拖得越久这个近似越粗糙。

## Checkpoint

- `--save-dir/step_NNNNNN/`：每`--save-every-episodes`存一次完整policy权重（actor/critic/target_critic），跟平时部署用的checkpoint格式一样。
- `--save-dir/latest_online_state.pt`：**每个episode**都原子写入一次，是一个内部一致的完整快照——actor/critic/target_critic权重、optimizer state（分开存）、完整replay buffer、已完成episode数、RNG状态全在一起（不是只存optimizer/buffer，那样会跟每5个episode才存一次的权重错位、没法配对复原）。
- 注意：现在只做到"存"，没有`--resume-checkpoint`这种直接加载`latest_online_state.pt`接着跑的入口，真要断点续训得手动写脚本加载。

## 这个任务的 critical phase 边界怎么画

`pick up hexagon（右臂，粗）`和`pull gray pin（左臂，精）`是同时进行的，没法在两者之间找一个干净的时间分界点。**在两个并发动作里较早开始的那个的起始时刻按r，一直保持到insert结束**，不用纠结哪一段才算"精细操作"——如果并发的那段粗操作本身不需要修正，critic不会给出让它偏离VLA的梯度，delta趋于0，等于VLA代劳，不会把它带坏。

## 诊断

跟离线训练共用同一套诊断脚本：

```bash
python diagnostics/check_config_consistency.py \
  --ac-config-dir outputs/pin_insert_online_rl/step_000005 \
  --cache-build-chunk-length 10 --cache-build-frame-stride 2 \
  --deploy-chunk-exec-steps 25 --deploy-phase-mode manual
```

## 关键参数速查

```text
--reset-time-s（默认15）   episode之间纯teleop的reset窗口，成功失败都会跑，不计入训练数据
--warmup-episodes / --min-warmup-transitions / --min-warmup-successes / --min-warmup-failures
    只有episode数、transition数、成功数、失败数全部达标才真正开始训练（不是任一条件满足就行）
--critic-only-episodes   warmup**实际满足门槛那一刻**算起，再N个episode只更新critic，actor
    继续冻结在=VLA的状态（不是warmup_episodes+critic_only_episodes这个固定offset——
    如果warmup因为成功/失败数不够拖到很后面才达标，固定offset会导致critic-only被跳过）
--utd-ratio（默认1）/ --max-updates-per-episode（默认200）
    更新次数 = min(本episode新增transition数 * utd_ratio, 上限)
--lr-actor(3e-5) / --lr-critic(1e-4)   actor学得慢，critic学得快
--actor-action-clip-delta   RL actor单步动作相对VLA参考的最大偏移，安全兜底
--actor-hidden-dim/--actor-num-layers/--critic-hidden-dim/--critic-num-layers（默认512/3层）
    对齐ac_paper_screw.yaml的复杂任务档位
--actor-fixed-std（默认0）   目前不生效——rollout和训练loss都只用actor的确定性均值，从不调用
    actor.sample()，这个参数先留着占位，不代表有随机探索
--stratified-sampling（默认开）   训练batch按成功/失败/人工干预/最近数据分层采样
--gamma/--beta/--tau/--actor-update-interval   TD3+BC超参，跟离线训练那套一样
```

## 其他保证

- rerecord/丢弃/中途异常的episode：这段时间收集的transition不会进全局replay buffer——先进episode内部的staging区，`flush_episode`（正常打标签结束）才会提交，rerecord/异常直接被下一次`start_episode`丢弃，不会污染训练数据。
- 训练梯度更新那一小段出异常（比如CUDA OOM）不会让后续session永久跑飞——`tau`/`actor_update_interval`这些临时改动用`try/finally`保护，异常发生也会恢复正常值。
