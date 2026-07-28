# 云端在线 RL 训练

## 云端

```bash
python runing_service/rlt_ac/online_serve.py \
  --host 0.0.0.0 --port 8600 \
  --auth-token "$(openssl rand -hex 16)" \
  --vla-path pretrained/pi05_full_ft/pretrained_model \
  --rl-token-path outputs/pin_insert_rl_token/checkpoints/last/pretrained_model \
  --tokenizer-path /path/to/paligemma-3b-pt-224/snapshots/xxx \
  --action-dim 12 --proprio-dim 12 --chunk-length 10 --chunk-exec-steps 25 \
  --actor-action-clip-delta 0.05 \
  --save-dir outputs/pin_insert_online_rl_remote \
  --save-every-episodes 5
```

## 本机

```bash
evo-rlt-online-train \
  --setup-json configs/my_so101_manifest.json \
  --task "Pick up the black hexagonal part with the right arm, pull the gray pin out of the white platform with the left arm, align the gray pin with the hole in the side of the black hexagonal part, insert the gray pin into the hole, and place the assembled object in the red square area." \
  --num-episodes 5 \
  --actor-action-clip-delta 0.05 \
  --save-dir outputs/pin_insert_online_rl \
  --remote-server http://192.168.3.71:8600
```

`--vla-path`/`--rl-token-path`/`--tokenizer-path` 在 `--remote-server` 模式下不再必填——本机不会加载这些文件，云端 `online_serve.py` 那边填了就够。

其余超参两边同名参数必须一致[README_online.md](README_online.md)，不变。
