"""Client-side plumbing for cloud/remote online RL training.

`remote_client.RemoteOnlineRLSession` is used by `backend.record()` (see
`online_rl.remote_server`) to swap the in-process VLA + `rlt_ac` policy for a
network call to a cloud-hosted service. The actual runnable cloud server
lives in `runing_service/rlt_ac/online_serve.py` (not part of this package)
and imports `evo_rlt.adapters.lerobot.record.online_trainer.OnlineRLTrainer`
plus this package's `codec` module, so the replay-buffer/TD3+BC training math
has exactly one implementation shared by both local and remote training.
"""
