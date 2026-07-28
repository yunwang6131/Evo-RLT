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
  --vla-path pretrained/pi05_full_ft/pretrained_model \
  --rl-token-path outputs/pin_insert_rl_token/checkpoints/last/pretrained_model \
  --tokenizer-path /path/to/paligemma-3b-pt-224/snapshots/xxx \
  --task "..." \
  --num-episodes 5 \
  --actor-action-clip-delta 0.05 \
  --save-dir outputs/pin_insert_online_rl \
  --remote-server http://127.0.0.1:8600 \
  --remote-token "<与云端 --auth-token 相同>"
```

其余超参两边同名参数必须一致（`--gamma` `--beta` `--tau` `--utd-ratio` `--warmup-episodes` `--lr-actor` `--lr-critic` `--actor-hidden-dim` 等）。按键逻辑、go-home、reset 窗口见 [README_online.md](README_online.md)，不变。
