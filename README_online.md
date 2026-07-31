python -m pip install -e ".[lerobot]"
# online RL 训练（真机，同步版）

actor 用 zero-init 残差头（mu = ref + delta，delta 最后一层零初始化），没训练之前 actor 输出 == VLA 参考，第一轮就是安全的——不需要offline先训一版actor-critic再warm start，直接从零开始在线训也是安全的。

```
task = "Pick up the black hexagonal part with the right arm, pull the gray pin out of the white platform with the left arm, align the gray pin with the hole in the side of the black hexagonal part, insert the gray pin into the hole, and place the assembled object in the red square area."
```
### 1. 训练 RL Token（冻结VLA，只训一个encoder做特征压缩）

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

### 1.5 可选：把 VLA 示范作为固定 offline replay

需要用当前版本重新构建 cache。旧版 v2 cache 令
`exec_chunk == ref_chunk`，没有保留真实示范修正，不适合双 buffer 在线训练。
左臂 RL 模式下，新 builder 使用“示范左臂动作 + 当前 VLA 右臂动作”，
与在线部署的控制权限完全一致：

```bash
evo-rlt-build-transition-cache-v2 \
  --demo-dataset-repo-id local/merged_screw_v1 \
  --demo-dataset-root data/bimanual/merged_screw_v1 \
  --rl-token-policy-path outputs/pin_insert_rl_token/checkpoints/010000/pretrained_model \
  --vla-pretrained-path /home/wangyun/Evo-RLT/pretrained/pretrained_model \
  --tokenizer-path /home/wangyun/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c \
  --output-dir outputs/pin_insert_offline_cache \
  --task-instruction "Pick up the black hexagonal part with the right arm, pull the gray pin out of the white platform with the left arm, align the gray pin with the hole in the side of the black hexagonal part, insert the gray pin into the hole, and place the assembled object in the red square area." \
  --default-episode-success success \
  --chunk-length 10 \
  --frame-stride 2 \
  --rl-action-arms left \
  --batch-size 32 \
  --num-workers 2 \
  --train-ratio 0.9 \
  --tolerance-s 0.04 \
  --device cuda
```

如果原始 VLA 数据包含整段长任务，最好先裁出与在线 `r` 键范围一致的
插入 critical phase；否则大量抓取/搬运/放置状态会稀释离线 batch。没有
`episode_success` 元数据时，上面的命令把标准 VLA 示范视为成功；如果数据中
混有失败，必须先补正确标签，不能统一标成成功。

然后给在线训练增加：

```bash
  --offline-cache-path outputs/pin_insert_offline_cache \
  --offline-batch-fraction 0.5 \
```

每个 batch 默认一半来自固定离线示范，一半来自在线 replay。在线 warmup
的 transition/success/failure 门槛以及 UTD 更新预算仍然只统计在线数据；
离线成功示范不会让 actor 提前解冻。续训时必须传回同一个
`--offline-cache-path`。

### 2. 跑在线RL训练

```bash
evo-rlt-online-train \
  --setup-json configs/my_so101_manifest.json \
  --vla-path /home/wangyun/Evo-RLT/pretrained/pretrained_model \
  --rl-token-path outputs/pin_insert_rl_token/checkpoints/010000/pretrained_model \
  --tokenizer-path /home/wangyun/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c \
  --task "Pick up the black hexagonal part with the right arm, pull the gray pin out of the white platform with the left arm, align the gray pin with the hole in the side of the black hexagonal part, insert the gray pin into the hole, and place the assembled object in the red square area." \
  --num-episodes 200 \
  --actor-action-clip-delta 0.1 \
  --rl-action-arms left \
  --offline-cache-path outputs/pin_insert_offline_cache \
  --offline-batch-fraction 0.5 \
  --beta 0.3 \
  --warmup-episodes 5 \
  --min-warmup-transitions 1000 \
  --min-warmup-successes 3 \
  --min-warmup-failures 3  \
  --rl-action-arms left \
  --gamma 0.9995 \
  --wandb \
  --wandb-project rlt-left-only \
  --wandb-run-name run1 \
  --save-every-episodes 5
```

`--save-dir` 现在可以不传——不传会自动生成 `outputs/online_rl/<MMDD>_<dataset-tag>/<HHMMSS>/`（跟数据集文件夹用同一个时间戳，方便对应），每次全新跑都不会互相覆盖。**续训(`--resume-from`)时必须显式传 `--save-dir`**，指向原来那次跑用的目录，不然会直接报错拦住——续训不该新开一个跟历史checkpoint不连续的目录。

# 恢复训练，从明确选择的历史训练状态继续：

evo-rlt-online-train \
  --setup-json configs/my_so101_manifest.json \
  --vla-path /home/wangyun/Evo-RLT/pretrained/pretrained_model \
  --rl-token-path outputs/pin_insert_rl_token/checkpoints/010000/pretrained_model \
  --tokenizer-path /home/wangyun/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c \
  --task "Pick up the black hexagonal part with the right arm, pull the gray pin out of the white platform with the left arm, align the gray pin with the hole in the side of the black hexagonal part, insert the gray pin into the hole, and place the assembled object in the red square area." \
  --num-episodes 200 \
  --actor-action-clip-delta 0.1 \
  --resume-from outputs/pin_insert_online_rl/step_000100/online_state.pt \
  --save-dir outputs/pin_insert_online_rl \
  --wandb \
  --wandb-project pin-insert-rl \
  --wandb-run-id run1 \
  --wandb-resume must \
  --save-every-episodes 5


这个可以调reset之后的位置：
  --go-home-positions '{"left_shoulder_pan.pos": 2054, "left_shoulder_lift.pos": 2099, "left_elbow_flex.pos": 3041, "left_wrist_flex.pos": 1448, "left_gripper.pos": 1789, "right_shoulder_pan.pos": 2095, "right_shoulder_lift.pos": 2132, "right_elbow_flex.pos": 2984, "right_wrist_flex.pos": 1428, "right_gripper.pos": 1991}' \

`--num-episodes` 是续训后的总 episode 目标。例如从
`step_000100/online_state.pt` 恢复并传 `--num-episodes 200`，会从 100 继续到 200
--actor-action-clip-delta 用来限制 RLT Actor 输出相对于 VLA 参考动作的最大偏移

# 评测

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

phase mode可以选择：always_rl/always_vla/manual

# 收集SFT之后的pi05_baseline

evo-rlt-record full   --initial-source vla   --setup-json configs/my_so101_manifest.json   --policy-path /home/wangyun/Evo-RLT/pretrained/pretrained_model   --task "Pick up the black hexagonal part with the right arm, pull the gray pin out of the white platform with the left arm, align the gray pin with the hole in the side of the black hexagonal part, insert the gray pin into the hole, and place the assembled object in the red square area."   --dataset-tag pi05_baseline_eval   --num-episodes 30   --episode-time-s 600   --reset-time-s 6   --fps 30   --vcodec h264

# 删除bufferl里面的某个episode
会先把原文件备份成 latest_online_state.pt.bak，再把episode 41的所有transition从buffer里删掉、覆写回 latest_online_state.pt。跑完会打印删了多少条、buffer从多少条变成多少条。

evo-rlt-prune-online-state \
  --state-path outputs/pin_insert_online_rl/latest_online_state.pt \
  --episode-id 41 \
  --in-place

如果不想直接覆写、想先看看结果对不对，去掉 --in-place 就行，会另外写一个 latest_online_state.pt.pruned.pt，原文件不动：
evo-rlt-prune-online-state \
  --state-path outputs/pin_insert_online_rl/latest_online_state.pt \
  --episode-id 41

# 人工标注VLA数据集中的critical phase
conda activate evo-rlt
python diagnostics/critical_segment_labeler_cv.py --dataset-root data/bimanual/merged_screw_v1

r:不在critical里 → 标记当前帧为critical开始;已经在critical里 → 在当前帧收尾并存为成功
u:在critical里的时候按 → 在当前帧收尾并存为失败(不在critical里按它没反应)
z:重置——把这条episode当前标的段(不管是已经收尾的,还是刚按了r还没收尾的)清空,重新来
x:标记"这条没有critical片段"(原来的x,没变)

(space播放暂停、a/d单帧、A/D十帧、1/2/3慢放、enter保存下一条、b/n不保存切换、q退出)都没变。存的格式也从"多段列表"改成单个 segment: [start, end, "success"|"failure"] | null。

## 按键 + episode之间的reset窗口

r：第一次按进入critical phase，第二次按立即判定success
u：立即把当前critical phase判定为failure
   r的success和u的failure都只结束critical phase，不结束整个episode，
   reward就在这一刻写进replay buffer，
   之后自动切回VLA继续跑（比如把组装好的东西放到指定区域），这段VLA跑的不算训练数据
s / f：整个episode真正结束（等VLA把后续动作做完、确认整体OK了再按）
i：仅左臂 teleop 接管
space：无干预时接管左右双臂；任意干预状态下解除干预
  按r（第1次）→ 进入critical phase，actor开始控制
    ↓（这期间可以按0次、1次或多次space）
    按i → 仅左臂人工接管，右臂继续执行策略动作
    或按space → 左右双臂人工接管
    在任意干预状态再按space → intervention结束，actor自动恢复接管
    ↓（可以反复intervene任意次）
  按r（第2次）→ critical phase立即结束，判定success，
    reward在这一刻写进replay buffer；期间任意时刻按u则立即判定failure
    ↓
  VLA自动接管，继续做后面的步骤
    ↓
  按s或f → 整个episode真正结束
episode结束（按s/f）后依次发生两件事：

## warmup 门槛
# recorded_episodes >= 5 
至少完成 5 个有效 episode，避免刚启动、数据量过少时立刻训练。
# transitions >= 1000 
Replay Buffer 中至少要有 1000 条状态转移。一条 transition 大致包括：
当前状态 + 执行动作 + reward + 下一状态 + 是否结束
它不是图像帧数，也不是 episode 数。一个 episode 可以产生几十或几百条 transition，具体数量取决于 critical phase 持续时间和 transition 构造方式。
# successes >= 3 
Replay Buffer 中至少包含 3 个成功的 critical-phase episode。成功数据让 critic 学会哪些状态和动作具有较高价值
# failures >= 3 
至少包含 3 个失败的 critical-phase episode。失败数据让 critic 能够区分“好动作”和“坏动作”。
# critic only
warm_up结束之后会跑10个只更新crtic的不更新actor的

## 关键参数速查

--utd-ratio（默认1）/ --max-updates-per-episode（默认200）
    更新次数 = min(本episode新增transition数 * utd_ratio, 上限)
--lr-actor(3e-5) / --lr-critic(1e-4)   actor学得慢，critic学得快
--actor-action-clip-delta   RL actor单步动作相对VLA参考的最大偏移
--rl-action-arms（在线训练默认left）   left=仅左臂学习RL残差，右臂严格使用VLA动作；both=双臂都学习
--actor-hidden-dim/--actor-num-layers/--critic-hidden-dim/--critic-num-layers（默认512/3层）
    对齐ac_paper_screw.yaml的复杂任务档位
--actor-fixed-std（默认0）   目前不生效——rollout和训练loss都只用actor的确定性均值，从不调用
    actor.sample()，这个参数先留着占位，不代表有随机探索
--stratified-sampling（默认开）   训练batch按成功/失败/人工干预/最近数据分层采样
--gamma/--beta/--tau/--actor-update-interval   TD3+BC超参，跟离线训练那套一样
```
export TORCHDYNAMO_DISABLE=1


buffer包含：
  当前状态 state
  实际执行动作 exec_chunk
  VLA参考动作 ref_chunk
  奖励 reward
  下一状态 next_state
  是否结束 done
  是否人工接管 intervention
  episode编号
