# 权限
sudo chmod 666 /dev/ttyACM*
# 查看端口
ls -l /dev/ttyACM*
# 删除整个数据集
rm -rf "$HOME/lerobot_datasets/so101_single_arm_demo"
# 查找相机
lerobot-find-cameras opencv # or realsense for Intel Realsense cameras

# 带摄像头测试主臂从臂
lerobot-teleoperate     --robot.type=so101_follower     --robot.port=/dev/ttyACM1     --robot.id=my_awesome_follower_arm     --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, side: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}"     --teleop.type=so101_leader     --teleop.port=/dev/ttyACM0     --teleop.id=my_awesome_leader_arm     --display_data=true

# 采集数据
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras='{
    front: {
      type: opencv,
      index_or_path: 2,
      width: 640,
      height: 480,
      fps: 30,
      fourcc: "MJPG"
    },
    side: {
      type: opencv,
      index_or_path: 4,
      width: 640,
      height: 480,
      fps: 30,
      fourcc: "MJPG"
    }
  }' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=my_awesome_leader_arm \
  --display_data=false \
  --dataset.repo_id=local/so101_single_arm_demo \
  --dataset.root="$HOME/lerobot_datasets/so101_single_arm_demo" \
  --dataset.single_task="Pick up the white cylinder and insert it into the black pipe." \
  --dataset.num_episodes=20 \
  --dataset.episode_time_s=70 \
  --dataset.reset_time_s=10 \
  --dataset.fps=30 \
  --dataset.push_to_hub=false

  # 微调单臂 pi05
  python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/so101_single_arm_demo \
  --dataset.root="$HOME/lerobot_datasets/so101_single_arm_demo" \
  --policy.path=<PI05_BASE_CHECKPOINT> \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --batch_size=8 \
  --steps=30000 \
  --save_freq=5000 \
  --eval_freq=0 \
  --tolerance_s=0.04 \
  --output_dir=outputs/so101_single_vla \
  --job_name=so101_single_vla


  # 回放第一条
  lerobot-replay \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=my_awesome_follower_arm \
  --dataset.repo_id=local/so101_single_arm_demo \
  --dataset.root="$HOME/lerobot_datasets/so101_single_arm_demo" \
  --dataset.episode=15

  # 查看保存了多少条
  jq '{
  total_episodes,
  total_frames,
  total_videos,
  fps
}' "$HOME/lerobot_datasets/so101_single_arm_demo/meta/info.json"

# 继续采集
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras='{
    front: {
      type: opencv,
      index_or_path: 2,
      width: 640,
      height: 480,
      fps: 30,
      fourcc: "MJPG"
    },
    side: {
      type: opencv,
      index_or_path: 4,
      width: 640,
      height: 480,
      fps: 30,
      fourcc: "MJPG"
    }
  }' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=my_awesome_leader_arm \
  --display_data=false \
  --dataset.repo_id=local/so101_single_arm_demo \
  --dataset.root="$HOME/lerobot_datasets/so101_single_arm_demo" \
  --dataset.single_task="Pick up the white cylinder and insert it into the black pipe." \
  --dataset.num_episodes=30 \
  --dataset.episode_time_s=70 \
  --dataset.reset_time_s=10 \
  --dataset.fps=30 \
  --dataset.vcodec=h264 \ls -l /dev/ttyACM*
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --dataset.video_encoding_batch_size=1 \
  --dataset.push_to_hub=false \
  --resume=true

  # 删除某个episode
  lerobot-edit-dataset \
  --repo_id local/so101_single_arm_demo \
  --root "$HOME/lerobot_datasets/so101_single_arm_demo" \
  --operation.type delete_episodes \
  --operation.episode_indices "[15]"

## 双臂复现
# 遥操作测试（先不录制，确认左右臂和相机都正常）
lerobot-teleoperate \
  --robot.type=bi_so_follower \
  --robot.id=bimanual_follower \
  --robot.left_arm_config.port=<LEFT_FOLLOWER_PORT> \
  --robot.left_arm_config.cameras='{wrist: {type: opencv, index_or_path: <LEFT_WRIST_CAM_INDEX>, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
  --robot.right_arm_config.port=<RIGHT_FOLLOWER_PORT> \
  --robot.right_arm_config.cameras='{wrist: {type: opencv, index_or_path: <RIGHT_WRIST_CAM_INDEX>, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, front: {type: opencv, index_or_path: <FRONT_CAM_INDEX>, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
  --teleop.type=bi_so_leader \
  --teleop.id=bimanual_leader \
  --teleop.left_arm_config.port=<LEFT_LEADER_PORT> \
  --teleop.right_arm_config.port=<RIGHT_LEADER_PORT> \
  --display_data=true

