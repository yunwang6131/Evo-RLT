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

# 合并 data/bimanual 下现在有的全部12个session数据集
# 使用仓库内脚本保留独立MP4，绕过LeRobot v0.5.1跨session拼接时的重复DTS错误
python -m evo_rlt.cli.merge_lerobot_datasets \
  --input-parent data/bimanual/0724_screw_demo_v1 \
  --output-repo-id local/merged_screw_v1 \
  --output-root data/bimanual/merged_screw_v1 \
  --repo-id-prefix local/rtf_0724_ \
  --overwrite

# 核对合并后的总episode数
jq '{total_episodes, total_frames}' data/bimanual/merged_screw_v1/meta/info.json

# 检查回放合并后的数据
conda activate evo-rlt

lerobot-dataset-viz \
  --root data/bimanual/merged_screw_v1 \
  --repo-id local/merged_screw_v1 \
  --episode-index 0 \
  --mode local \
  --display-compressed-images
有问题的episode

# 删除指定episode
conda activate evo-rlt

lerobot-edit-dataset \
  --repo_id local/merged_screw_v1 \
  --root data/bimanual/merged_screw_v1 \
  --new_repo_id local/merged_screw_v1 \
  --new_root data/bimanual/merged_screw_v1 \
  --operation.type delete_episodes \
  --operation.episode_indices "[174]"
# 收集SFT之后的pi05_baseline

evo-rlt-record full   --initial-source vla   --setup-json configs/my_so101_manifest.json   --policy-path /home/wangyun/Evo-RLT/pretrained/pretrained_model   --task "Pick up the black hexagonal part with the right arm, pull the gray pin out of the white platform with the left arm, align the gray pin with the hole in the side of the black hexagonal part, insert the gray pin into the hole, and place the assembled object in the red square area."   --dataset-tag pi05_baseline_eval   --num-episodes 30   --episode-time-s 600   --reset-time-s 6   --fps 30   --vcodec h264

# 训练 RL Token, 用VLA full采集的数据
python -c 'from evo_rlt.adapters.lerobot import register; register(); from lerobot.scripts.lerobot_train import main; main()' \
  --dataset.repo_id=local/merged_screw_v1 \
  --dataset.root=data/bimanual/merged_screw_v1 \
  --policy.type=rlt_token \
  --policy.repo_id=local/bimanual_rlt_token \
  --policy.push_to_hub=false \
  --policy.vla_pretrained_path=pretrained/pi05_full_ft/pretrained_model \
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
  --output_dir=outputs/bimanual_rl_token \
  --job_name=bimanual_rl_token

# 构建 transition cache
# 如果 <demo-dataset-root>/meta/critical_segments.json 存在（用
# diagnostics/critical_segment_labeler_cv.py 标过），默认会自动读取并只用每条
# episode 里标出来的 critical-phase 片段（含该片段自己的 success/failure），跟在线
# RL 的 critical phase 语义保持一致；没有该 label 的 episode 会被跳过。不想用的话传
# --no-critical-segments 回退成整集 episode_success 的老行为。
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
  --rl-action-arms left \
  --batch-size 32 \
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
  --policy.rl_token_num_rl_tokens=1 \
  --policy.actor_activation=silu --policy.actor_residual=true \
  --policy.critic_activation=silu --policy.critic_residual=true \
  --policy.chunk_length=10 \
  --policy.chunk_exec_steps=25 \
  --policy.phase_mode=always_rl \
  --policy.device=cuda \
  --batch_size=256 \
  --steps=50000 \
  --salt_always_rl_eval \
  --num-episodes 20 \
  --episode-time-s 3000 \
  --fps 30 \
  --vcodec h264 \
  --rtc-execution-horizon 10 \
  --vla-rtc-execution-horizon 25 \
  --rtc-action-queue-size-to-get-new-actions 40 \ve_freq=5000 \
  --eval_freq=0 \
  --output_dir=outputs/bimanual_ac \
  --job_name=bimanual_rlt_ac

# 全程RLT
evo-rlt-record full \
  --initial-source vla \
  --setup-json configs/my_so101_manifest.json \
  --policy-path outputs/bimanual_ac/checkpoints/050000/pretrained_model \
  --vla-path pretrained/pi05_full_ft/pretrained_model \
  --rl-token-path outputs/bimanual_rl_token/checkpoints/last/pretrained_model \
  --phase-mode always_rl \
  --task "Pick up the black hexagonal part with the right arm, pull the gray pin out of the white platform with the left arm, align the gray pin with the hole in the side of the black hexagonal part, insert the gray pin into the hole, and place the assembled object in the red square area." \
  --dataset-tag rlt_always_rl_eval \
  --num-episodes 20 \
  --episode-time-s 3000 \
  --reset-time-s 6 \
  --fps 30 \
  --vcodec h264 \
  --rtc-execution-horizon 10 \
  --vla-rtc-execution-horizon 25 \
  --rtc-action-queue-size-to-get-new-actions 40 \
  --no-teleop

s    完整 episode 成功并结束
f    完整 episode 失败并结束
Esc  停止采集

# VLA → 手动进入 RLT，并且只录制RLT
evo-rlt-record collect \
  --setup-json configs/my_so101_manifest.json \
  --policy-path outputs/bimanual_ac/checkpoints/050000/pretrained_model \
  --vla-path pretrained/pi05_full_ft/pretrained_model \
  --rl-token-path outputs/bimanual_rl_token/checkpoints/last/pretrained_model \
  --task "Pick up the small white object and the black object from the yellow area, insert the white object into the black object, and place the assembly in the yellow square area." \
  --dataset-tag rlt_critical_eval \
  --num-episodes 20 \
  --episode-time-s 3000 \
  --fps 30 \
  --vcodec h264 \frrirr
  --only-critical \
  --rlt-toggle-key r \
  --teleop-toggle-key space \
  --rtc-execution-horizon 10 \
  --vla-rtc-execution-horizon 25 \
  --rtc-action-queue-size-to-get-new-actions 40
  
 r进入核心，r退出核心，只记录核心
## 诊断（RL比VLA差时排查，diagnostics/）
# 1. cache本身有没有问题（reward是不是全零、exec_chunk是不是等于ref_chunk）
python diagnostics/inspect_transition_cache.py --cache-dir outputs/bimanual_cache --splits train val

# 整体流程
从采集数据VLA开始，然后pi05微调，微调VLA之后再RL token，然后用SFT的VLA采集full里面有成功失败和认为干预的，然后transition cache,然后actor critic，然后用得到的模型采集Critical 片段，然后用这个片段制成数据集，然后累加之前的数据transition cache,然后在之前的checkpoint上actor critic。然后把最后这几个采集到actor_critic的重复几遍

# new task
Pick up the black hexagonal part with the right arm, pull the gray pin out of the white platform with the left arm, align the gray pin with the hole in the side of the black hexagonal part, insert the gray pin into the hole, and place the assembled object in the red square area.
