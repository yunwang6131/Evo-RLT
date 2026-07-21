# task
Pick up the small white object and the black object from the yellow area, insert the white object into the black object, and place the assembly in the yellow square area.
# 权限
sudo chmod 666 /dev/ttyACM*
# 查看端口(4条臂逐个插拔确认)
ls -l /dev/ttyACM*
# 查找相机
lerobot-find-cameras opencv # or realsense for Intel Realsense cameras
# leader右
crw-rw---- 1 root dialout 166, 0  7月 14 13:44 /dev/ttyACM0
# leader左
crw-rw---- 1 root dialout 166, 1  7月 14 13:44 /dev/ttyACM1
# follers右
crw-rw---- 1 root dialout 166, 2  7月 14 13:45 /dev/ttyACM2
# follers左
crw-rw---- 1 root dialout 166, 3  7月 14 13:45 /dev/ttyACM3
# 标定 left_follower
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=left_follower_arm
# 标定 right_follower
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=right_follower_arm
# 标定 left_leader
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=left_leader_arm
# 标定 right_leader
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=right_leader_arm

# 采集数据(写到暂存区 ~/lerobot_data,不直接落到 data)
evo-rlt-record full   --initial-source teleop   --setup-json configs/my_so101_manifest.json   --dataset-tag screw_demo_v1   --task "Pick up the small white object and the black object from the yellow area, insert the white object into the black object, and place the assembly in the yellow square area."   --num-episodes 70   --episode-time-s 300   --reset-time-s 6  --fps 30   --vcodec h264   --discard-unlabeled-episodes

# 删除某条（先确认目录，再将实际绝对路径直接写入命令）
ls -td ~/lerobot_data/bimanual/*/record_teleop_full_*/

~/anaconda3/envs/evo-rlt/bin/lerobot-edit-dataset \
  --repo_id local/record_teleop_full_172817 --root ~/lerobot_data/bimanual/0716_screw_demo_v1/record_teleop_full_172817 \
  --new_repo_id local/record_teleop_full_172817 --new_root ~/lerobot_data/bimanual/0716_screw_demo_v1/record_teleop_full_172817 \
  --operation.type delete_episodes \
  --operation.episode_indices "[5]"

jq '{total_episodes, total_frames}' ~/lerobot_data/bimanual/0716_screw_demo_v1/record_teleop_full_172817/meta/info.json

# 查看保存了多少条(当前这一次 session)
jq '{total_episodes, total_frames, total_videos, fps}' ~/lerobot_data/bimanual/0716_screw_demo_v1/record_teleop_full_172817/meta/info.json

# 查看暂存区目前为止所有 session 累计采集了多少条(跨多次 --dataset-tag)
jq -s 'map(.total_episodes) | add' ~/lerobot_data/bimanual/*/record_teleop_full_*/meta/info.json

# 确认没问题后,可以删掉自动生成的备份目录(可选)
# rm -rf ~/lerobot_data/bimanual/0716_screw_demo_v1/record_teleop_full_172817_old


## 回放(从暂存区回看,确认没问题再决定要不要挪进统一训练目录)
# 真机物理重放:
mkdir -p /tmp/evo-rlt-bimanual-calibration

cp ~/.cache/huggingface/lerobot/calibration/robots/so_follower/left_follower_arm.json \
   /tmp/evo-rlt-bimanual-calibration/bimanual_follower_left.json

cp ~/.cache/huggingface/lerobot/calibration/robots/so_follower/right_follower_arm.json \
   /tmp/evo-rlt-bimanual-calibration/bimanual_follower_right.json

ls -td ~/lerobot_data/bimanual/*/record_teleop_full_*/

lerobot-replay \
  --robot.type=bi_so_follower \
  --robot.id=bimanual_follower \
  --robot.calibration_dir=/tmp/evo-rlt-bimanual-calibration \
  --robot.left_arm_config.port=/dev/ttyACM3 \
  --robot.right_arm_config.port=/dev/ttyACM2 \
  --robot.left_arm_config.use_degrees=true \
  --robot.right_arm_config.use_degrees=true \
  --dataset.root=~/lerobot_data/bimanual/0716_screw_demo_v1/record_teleop_full_172817 \
  --dataset.repo_id=local/record_teleop_full_172817 \
  --dataset.episode=0

# 只看录像不动机械臂:
lerobot-dataset-viz --root ~/lerobot_data/bimanual/0716_screw_demo_v1/record_teleop_full_172817 --repo-id local/record_teleop_full_172817 --episode-index 0

# data里面的回放视频
lerobot-dataset-viz \
  --root data/bimanual/0715_screw_demo_v1/record_teleop_full_163740 \
  --repo-id local/record_teleop_full_163740 \
  --episode-index 0
## 确认这次采集没问题后,手动挪进统一训练目录(不会自动执行,自己看着跑)
ls -td ~/lerobot_data/bimanual/*/record_teleop_full_*/
mkdir -p data/bimanual/0716_screw_demo_v1
cp -r ~/lerobot_data/bimanual/0716_screw_demo_v1/record_teleop_full_172817 data/bimanual/0716_screw_demo_v1/

# 确认统一训练目录里数据没问题后,可以清掉暂存区对应这次的记录(可选,自己决定)
rm -rf ~/lerobot_data/bimanual/0716_screw_demo_v1/record_teleop_full_172817

## 以下训练数据全部指向统一训练目录 data/bimanual,不是暂存区
ls -td data/bimanual/*/record_teleop_full_*/

# 查看SFT之后的pi05_baseline
cd /home/wangyun/Evo-RLT
conda activate evo-rlt

# 训练好的模型位于 pretrained/pi05_full_ft/pretrained_model
evo-rlt-record full   --initial-source vla   --setup-json configs/my_so101_manifest.json   --policy-path pretrained/pi05_full_ft/pretrained_model   --task "Pick up the small white object and the black object from the yellow area, insert the white object into the black object, and place the assembly in the yellow square area."   --dataset-tag pi05_baseline_eval   --num-episodes 10   --episode-time-s 60   --reset-time-s 6   --fps 30   --vcodec h264

# 合并 data/bimanual 下现在有的全部11个session，作为RL Token/transition cache用的demo数据集
lerobot-edit-dataset \
  --operation.type merge \
  --operation.repo_ids "[local/rtf_0715_145644,local/rtf_0715_153343,local/rtf_0715_160721,local/rtf_0715_163740,local/rtf_0716_131809,local/rtf_0716_132534,local/rtf_0716_134459,local/rtf_0716_140211,local/rtf_0716_144858,local/rtf_0716_151749,local/rtf_0716_172817]" \
  --operation.roots "[data/bimanual/0715_screw_demo_v1/record_teleop_full_145644,data/bimanual/0715_screw_demo_v1/record_teleop_full_153343,data/bimanual/0715_screw_demo_v1/record_teleop_full_160721,data/bimanual/0715_screw_demo_v1/record_teleop_full_163740,data/bimanual/0716_screw_demo_v1/record_teleop_full_131809,data/bimanual/0716_screw_demo_v1/record_teleop_full_132534,data/bimanual/0716_screw_demo_v1/record_teleop_full_134459,data/bimanual/0716_screw_demo_v1/record_teleop_full_140211,data/bimanual/0716_screw_demo_v1/record_teleop_full_144858,data/bimanual/0716_screw_demo_v1/record_teleop_full_151749,data/bimanual/0716_screw_demo_v1/record_teleop_full_172817]" \
  --new_repo_id local/merged_screw_v1 \
  --new_root data/bimanual/merged_screw_v1

# 核对合并后的总episode数
jq '{total_episodes, total_frames}' data/bimanual/merged_screw_v1/meta/info.json

# 训练 RL Token（数据集用上面合并出来的 merged_screw_v1）
python -c 'from evo_rlt.adapters.lerobot import register; register(); from lerobot.scripts.lerobot_train import main; main()' \
  --dataset.repo_id=local/merged_screw_v1 \
  --dataset.root=data/bimanual/merged_screw_v1 \
  --policy.type=rlt_token \
  --policy.repo_id=local/bimanual_rlt_token \
  --policy.push_to_hub=false \
  --policy.vla_pretrained_path=pretrained/pi05_full_ft/pretrained_model \
  --policy.vla_dtype=bfloat16 \
  --policy.rl_token_num_rl_tokens=4 \
  --policy.tokenizer_path=/home/wangyun/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c \
  --policy.token_pool_size=0 \
  --policy.device=cuda \
  --batch_size=8 \
  --steps=10000 \
  --save_freq=2000 \
  --eval_freq=0 \
  --tolerance_s=0.04 \
  --output_dir=outputs/bimanual_rl_token \
  --job_name=bimanual_rl_token

# 构建 transition cache
evo-rlt-build-transition-cache-v2 \
  --demo-dataset-repo-id local/merged_screw_v1 \
  --demo-dataset-root data/bimanual/merged_screw_v1 \
  --rl-token-policy-path outputs/bimanual_rl_token/checkpoints/last/pretrained_model \
  --vla-pretrained-path pretrained/pi05_full_ft/pretrained_model \
  --tokenizer-path /home/wangyun/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c \
  --output-dir outputs/bimanual_cache \
  --task-instruction "Pick up the small white object and the black object from the yellow area, insert the white object into the black object, and place the assembly in the yellow square area." \
  --chunk-length 10 \
  --frame-stride 2 \
  --batch-size 2 \
  --num-workers 2 \
  --train-ratio 0.9 \
  --tolerance-s 0.04 \
  --device cuda

# 训练 chunk actor-critic
python -c 'from evo_rlt.adapters.lerobot import register; register(); from lerobot.scripts.lerobot_train import main; main()' \
  --dataset.repo_id=outputs/bimanual_cache \
  --policy.type=rlt_ac \
  --policy.repo_id=local/bimanual_rlt_ac \
  --policy.push_to_hub=false \
  --policy.vla_pretrained_path=pretrained/pi05_full_ft/pretrained_model \
  --policy.rl_token_pretrained_path=outputs/bimanual_rl_token/checkpoints/last/pretrained_model \
  --policy.vla_dtype=bfloat16 \
  --policy.tokenizer_path=/home/wangyun/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c \
  --policy.rl_token_num_rl_tokens=4 \
  --policy.actor_hidden_dim=512 --policy.actor_num_layers=4 \
  --policy.actor_fixed_std=0.01 --policy.actor_ref_dropout_p=0.7 \
  --policy.actor_activation=silu --policy.actor_residual=true \
  --policy.critic_hidden_dim=512 --policy.critic_num_layers=4 \
  --policy.critic_activation=silu --policy.critic_residual=true \
  --policy.beta=5.0 --policy.tau=0.02 \
  --policy.chunk_length=10 \
  --policy.chunk_exec_steps=25 \
  --policy.phase_mode=always_rl \
  --policy.device=cuda \
  --batch_size=2 \
  --steps=50000 \
  --save_freq=5000 \
  --eval_freq=0 \
  --output_dir=outputs/bimanual_ac \
  --job_name=bimanual_rlt_ac

# 真机部署验证
evo-rlt-record collect \
  --setup-json configs/my_so101_manifest.json \
  --policy-path outputs/bimanual_ac/checkpoints/last/pretrained_model \
  --vla-path pretrained/pi05_full_ft/pretrained_model \
  --rl-token-path outputs/bimanual_rl_token/checkpoints/last/pretrained_model \
  --task "Pick up the small white object and the black object from the yellow area, insert the white object into the black object, and place the assembly in the yellow square area." \
  --dataset-tag rlt_ac_eval \
  --num-episodes 10 \
  --episode-time-s 3000 \
  --fps 30 \
  --vcodec h264 \
  --rlt-toggle-key r \
  --teleop-toggle-key i \
  --rtc-execution-horizon 10 \
  --vla-rtc-execution-horizon 25 \
  --rtc-action-queue-size-to-get-new-actions 40
  
r / r+r   结束整个 episode（成功/失败）
s / f     结束当前 RLT 阶段（成功/失败），不结束 episode
i/Space   人工干预，取决于 --teleop-toggle-key
←         丢弃并重录
Esc       停止录制
# 只推理不录制
evo-rlt-record collect \
  --inference-only \
  --setup-json configs/my_so101_manifest.json \
  --policy-path outputs/bimanual_ac/checkpoints/last/pretrained_model \
  --vla-path pretrained/pi05_full_ft/pretrained_model \
  --rl-token-path outputs/bimanual_rl_token/checkpoints/last/pretrained_model \
  --task "Pick up the small white object and the black object from the yellow area, insert the white object into the black object, and place the assembly in the yellow square area." \
  --episode-time-s 3000 \
  --fps 30 \
  --rlt-toggle-key r \
  --teleop-toggle-key i \
  --rtc-execution-horizon 10 \
  --vla-rtc-execution-horizon 25 \
  --rtc-action-queue-size-to-get-new-actions 40

## 诊断（RL比VLA差时排查，diagnostics/）
# 1. cache本身有没有问题（reward是不是全零、exec_chunk是不是等于ref_chunk）
python diagnostics/inspect_transition_cache.py --cache-dir outputs/bimanual_cache --splits train val

# 若发现 reward 全零：build_transition_cache_v2.py 曾经硬编码 reward_seq=0，从不读取
# episode_success（已修复，见下）。已生成的 cache 不用整个重跑（GPU 编码不受影响，只是
# reward 标签没写对），可以直接原地 patch：仅当数据集里所有 episode 都是成功样本时适用，
# 否则请改用修复后的 evo-rlt-build-transition-cache-v2 重新生成。
PYTHONPATH=src python src/evo_rlt/cli/patch_cache_terminal_reward.py \
  --cache-dir outputs/bimanual_cache --splits train val

# 2. 训练/部署config是否对得上（chunk_length、chunk_exec_steps、phase_mode等）
python diagnostics/check_config_consistency.py \
  --ac-config-dir outputs/bimanual_ac/checkpoints/last/pretrained_model \
  --cache-build-chunk-length 10 --cache-build-frame-stride 2 \
  --deploy-chunk-exec-steps 25 --deploy-phase-mode always_rl