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
wangyun@wangyun:~$ 

# 标定 left_follower
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=left_follower_arm
# 标定 right_follower
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=right_follower_arm
# 标定 left_leader
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=left_leader_arm
# 标定 right_leader
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=right_leader_arm

# 采集数据(写到暂存区 ~/lerobot_data,不直接落到 Evo-RLT/data)
evo-rlt-record full   --initial-source teleop   --setup-json /home/wangyun/Evo-RLT/configs/my_so101_manifest.json   --dataset-tag screw_demo_v1   --task "Pick up the small white object and the black object from the yellow area, insert the white object into the black object, and place the assembly in the yellow square area."   --num-episodes 70   --episode-time-s 300   --reset-time-s 6  --fps 30   --vcodec h264   --discard-unlabeled-episodes

# 删除某条
DATASET_ROOT=$(ls -td /home/wangyun/lerobot_data/bimanual/*/record_teleop_full_*/ | head -1)
DATASET_ID="local/$(basename "$DATASET_ROOT")"

/home/wangyun/anaconda3/envs/evo-rlt/bin/lerobot-edit-dataset \
  --repo_id "$DATASET_ID" --root "$DATASET_ROOT" \
  --new_repo_id "$DATASET_ID" --new_root "$DATASET_ROOT" \
  --operation.type delete_episodes \
  --operation.episode_indices "[5]"

jq '{total_episodes, total_frames}' "$DATASET_ROOT/meta/info.json"

# 查看保存了多少条(当前这一次 session)
jq '{total_episodes, total_frames, total_videos, fps}' "$DATASET_ROOT/meta/info.json"

# 查看暂存区目前为止所有 session 累计采集了多少条(跨多次 --dataset-tag)
for f in /home/wangyun/lerobot_data/bimanual/*/record_teleop_full_*/meta/info.json; do
  jq -r '.total_episodes' "$f"
done | paste -sd+ | bc

# 确认没问题后,可以删掉自动生成的备份目录(可选)
# rm -rf "${DATASET_ROOT%/}_old"


## 回放(从暂存区回看,确认没问题再决定要不要挪进统一训练目录)
# 真机物理重放:
CAL_DIR=$(mktemp -d)

cp ~/.cache/huggingface/lerobot/calibration/robots/so_follower/left_follower_arm.json \
   "$CAL_DIR/bimanual_follower_left.json"

cp ~/.cache/huggingface/lerobot/calibration/robots/so_follower/right_follower_arm.json \
   "$CAL_DIR/bimanual_follower_right.json"

D=$(ls -td /home/wangyun/lerobot_data/bimanual/*/record_teleop_full_*/ | head -1)

lerobot-replay \
  --robot.type=bi_so_follower \
  --robot.id=bimanual_follower \
  --robot.calibration_dir="$CAL_DIR" \
  --robot.left_arm_config.port=/dev/ttyACM3 \
  --robot.right_arm_config.port=/dev/ttyACM2 \
  --robot.left_arm_config.use_degrees=true \
  --robot.right_arm_config.use_degrees=true \
  --dataset.root="$D" \
  --dataset.repo_id="local/$(basename "$D")" \
  --dataset.episode=0

# 只看录像不动机械臂:
D=$(ls -td /home/wangyun/lerobot_data/bimanual/*/record_teleop_full_*/ | head -1); lerobot-dataset-viz --root "$D" --repo-id "local/$(basename "$D")" --episode-index 0

# data里面的回放视频
D=$(ls -td /home/wangyun/Evo-RLT/data/bimanual/0715_screw_demo_v1/record_teleop_full_*/ | head -1)
lerobot-dataset-viz \
  --root "$D" \
  --repo-id "local/$(basename "$D")" \
  --episode-index 0
## 确认这次采集没问题后,手动挪进统一训练目录(不会自动执行,自己看着跑)
LATEST=$(ls -td /home/wangyun/lerobot_data/bimanual/*/record_teleop_full_*/ | head -1)
TAG_DIR=$(basename $(dirname "$LATEST"))
mkdir -p "/home/wangyun/Evo-RLT/data/bimanual/$TAG_DIR"
cp -r "$LATEST" "/home/wangyun/Evo-RLT/data/bimanual/$TAG_DIR/"
echo "已复制到 /home/wangyun/Evo-RLT/data/bimanual/$TAG_DIR/$(basename $LATEST)"

# 确认统一训练目录里数据没问题后,可以清掉暂存区对应这次的记录(可选,自己决定)
# rm -rf "$LATEST"

## 以下训练用的 DATASET_ROOT/DATASET_ID 全部指向统一训练目录 data/bimanual,不是暂存区
DATASET_ROOT=$(ls -td /home/wangyun/Evo-RLT/data/bimanual/*/record_teleop_full_*/ | head -1)
DATASET_ID="local/$(basename $DATASET_ROOT)"
echo $DATASET_ROOT $DATASET_ID

# 微调 VLA
python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="$DATASET_ID" \
  --dataset.root="$DATASET_ROOT" \
  --policy.path=/home/wangyun/Evo-RLT/pretrained/pi05_screw_c_mix_cont15k_fp16/online_base_vla_0611_rec_20260610224621_it300.pt \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --batch_size=16 \
  --steps=30000 \
  --save_freq=5000 \
  --eval_freq=0 \
  --tolerance_s=0.04 \
  --output_dir=outputs/bimanual_vla_ft \
  --job_name=bimanual_vla_ft

# 训练 RL Token
python -c 'from evo_rlt.adapters.lerobot import register; register(); from lerobot.scripts.lerobot_train import main; main()' \
  --dataset.repo_id="$DATASET_ID" \
  --dataset.root="$DATASET_ROOT" \
  --policy.type=rlt_token \
  --policy.repo_id=local/bimanual_rlt_token \
  --policy.push_to_hub=false \
  --policy.vla_pretrained_path=outputs/bimanual_vla_ft/checkpoints/last/pretrained_model \
  --policy.vla_dtype=bfloat16 \
  --policy.rl_token_num_rl_tokens=4 \
  --policy.tokenizer_path=<PALIGEMMA_TOKENIZER_PATH> \
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
  --demo-dataset-repo-id "$DATASET_ID" \
  --demo-dataset-root "$DATASET_ROOT" \
  --rl-token-policy-path outputs/bimanual_rl_token/checkpoints/last/pretrained_model \
  --vla-pretrained-path outputs/bimanual_vla_ft/checkpoints/last/pretrained_model \
  --tokenizer-path <PALIGEMMA_TOKENIZER_PATH> \
  --output-dir outputs/bimanual_cache \
  --task-instruction "Insert the white cylinder into the black sleeve." \
  --chunk-length 10 \
  --frame-stride 2 \
  --batch-size 8 \
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
  --policy.vla_pretrained_path=outputs/bimanual_vla_ft/checkpoints/last/pretrained_model \
  --policy.rl_token_pretrained_path=outputs/bimanual_rl_token/checkpoints/last/pretrained_model \
  --policy.vla_dtype=bfloat16 \
  --policy.tokenizer_path=<PALIGEMMA_TOKENIZER_PATH> \
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
  --batch_size=64 \
  --steps=50000 \
  --save_freq=5000 \
  --eval_freq=0 \
  --output_dir=outputs/bimanual_ac \
  --job_name=bimanual_rlt_ac

# 真机部署验证
evo-rlt-record collect \
  --setup-json /home/wangyun/Evo-RLT/configs/my_so101_manifest.json \
  --policy-path outputs/bimanual_ac/checkpoints/last/pretrained_model \
  --vla-path outputs/bimanual_vla_ft/checkpoints/last/pretrained_model \
  --rl-token-path outputs/bimanual_rl_token/checkpoints/last/pretrained_model \
  --task "Insert the white cylinder into the black sleeve." \
  --dataset-tag bimanual_rlt_test \
  --num-episodes 5 \
  --episode-time-s 3000 \
  --fps 30 \
  --vcodec h264 \
  --rlt-toggle-key r \
  --teleop-toggle-key space
