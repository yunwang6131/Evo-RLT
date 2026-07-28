# online RL 训练（真机，同步版）


actor 用 zero-init 残差头（mu = ref + delta，delta 最后一层零初始化），没训练之前 actor 输出 == VLA 参考，第一轮就是安全的——不需要offline先训一版actor-critic再warm start，直接从零开始在线训也是安全的。

## 新任务完整步骤

```
task = "Pick up the black hexagonal part with the right arm, pull the gray pin out of the white platform with the left arm, align the gray pin with the hole in the side of the black hexagonal part, insert the gray pin into the hole, and place the assembled object in the red square area."
```
### 1. 训练 RL Token（冻结VLA，只训一个encoder做特征压缩）

不需要成功/失败标签，用上面同一批或另采一批teleop demo都可。

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

到这里为止，跟离线流程（README_dualarm_rlt.md）是共用的——**在线训练不需要再做`build-transition-cache`和`train-actor-critic`这两步**，`evo-rlt-online-train`会现场构造一个全新初始化的actor/critic。

### 2. 跑在线RL训练

```bash
evo-rlt-online-train \
  --setup-json configs/my_so101_manifest.json \
  --vla-path pretrained/pi05_full_ft/pretrained_model \
  --rl-token-path outputs/pin_insert_rl_token/checkpoints/last/pretrained_model \
  --tokenizer-path /home/wangyun/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c \
  --task "Pick up the black hexagonal part with the right arm, pull the gray pin out of the white platform with the left arm, align the gray pin with the hole in the side of the black hexagonal part, insert the gray pin into the hole, and place the assembled object in the red square area." \
  --num-episodes 5 \
  --actor-action-clip-delta 0.05 \
  --save-dir outputs/pin_insert_online_rl \
  --save-every-episodes 5 \
```

--actor-action-clip-delta 用来限制 RLT Actor 输出相对于 VLA 参考动作的最大偏移
看到Online RL update after episode ...`且loss不是NaN，再加大`--num-episodes`跑正式session

## 按键 + episode之间的reset窗口

```text
r：进入/退出critical phase（单击=success，double-tap-window-s窗口内双击=failure）——
   只结束critical phase，不结束整个episode，reward就在这一刻写进replay buffer，
   之后自动切回VLA继续跑（比如把组装好的东西放到指定区域），这段VLA跑的不算训练数据
s / f：整个episode真正结束（等VLA把后续动作做完、确认整体OK了再按）
space（或 -i）：teleop 接管
  按r（第1次）→ 进入critical phase，actor开始控制
    ↓（这期间可以按0次、1次或多次space）
    按space → intervention开始，你用leader臂直接接管，actor被打断
    再按space → intervention结束，如果critical phase还没结束，actor自动恢复接管
    ↓（可以反复intervene任意次）
  按r（第2次，0.6s确认窗口内不再按）→ critical phase结束，判定success，
    reward在这一刻写进replay buffer；如果0.6s内又按了一次r，判定failure
    ↓
  VLA自动接管，继续做后面的步骤
    ↓
  按s或f → 整个episode真正结束
备用x：强制接管，效果同上，多一个热键保险
```

episode结束（按s/f）后依次发生两件事：

1. **go-home**（`--go-home-time-s`秒，默认3）：机械臂自动ramp回标定时手动摆的那个"中间位置"夹爪归到`--go-home-gripper-value`
2. **teleop reset窗口**（`--reset-time-s`秒，默认15）：纯teleop，用来手动把物件摆回去。

这两段时间都不计入数据/replay buffer。`--go-home-time-s 0`可以关掉第一步。


## 跑前检查

```bash
python diagnostics/check_config_consistency.py \
  --ac-config-dir outputs/pin_insert_online_rl/step_000005 \
  --cache-build-chunk-length 10 --cache-build-frame-stride 2 \
  --deploy-chunk-exec-steps 25 --deploy-phase-mode manual
```

## 关键参数速查

```text
--reset-time-s
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
