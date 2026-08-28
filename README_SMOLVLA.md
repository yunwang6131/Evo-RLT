# SmolVLA on the blue-screw simulation dataset

Same dataset, same evaluation loop, same keys as [README_ACT.md](README_ACT.md) —
only the policy differs, so the two success rates are directly comparable. Both
CLIs read the *same* dataset profile, `configs/blue_screw_sim_v1.json` — one
file, so the sources, merged root, task string and expected shape cannot drift
apart and quietly stop the comparison from being one. Training settings live
separately in `configs/smolvla/train_config.json`, which `lerobot-train` reads
directly.

Like ACT, this path is standard LeRobot behavior cloning. It does **not** feed
into Evo-RLT's RL-token or actor-critic stages, which remain pi0.5-only.

## What is different from ACT

| | ACT | SmolVLA |
|---|---|---|
| init | trained from scratch | **fine-tuned from `lerobot/smolvla_base`** |
| params trained | all (~52M) | all 450M (vision encoder included) |
| camera keys | dataset names used as-is | renamed to `camera1/2/3` |
| `chunk_size` | 100 (free choice) | 50 (fixed by the base) |
| batch / VRAM | 8 / 6789 MiB | 4 / 12205 MiB |
| throughput | 3.7 step/s | 1.22 step/s (4.9 samples/s) |
| step budget | 60k ≈ 4.5 h | 20k ≈ 4.6 h |

Three of these are load-bearing and worth knowing before you change anything:

**Fine-tune, never `--policy.type=smolvla`.** That flag builds a randomly
initialised 450M model and trains it happily — the logs look identical and only
the success rate tells you the pretrained weights were never loaded. 122
demonstrations cannot train a VLA from scratch. `train_config.json` sets
`policy.pretrained_path`, and a test pins it.

**`chunk_size` stays at 50.** It sets the action expert's sequence length, so
overriding it discards the pretrained expert while still looking like a
fine-tune. `n_action_steps` is the one to change instead, pinned to 10 — the
base ships 50, i.e. 1.67 s of open-loop execution, which is far too long for a
contact-sensitive insertion.

**Nothing is frozen, and both flags say so.** `set_requires_grad()` in
`smolvlm_with_expert.py` freezes the *entire* VLM -- vision encoder included --
whenever `train_expert_only` is true, so setting only
`freeze_vision_encoder=false` trains no vision at all and gives no sign of it.
`train_config.json` sets **both** explicitly rather than inheriting them from
the base's config, so pointing `pretrained_path` at a different base cannot
silently change what is being trained. Setting both to true opts back into the
base's original expert-only recipe.

Measured cost of unfreezing on a 16 GB card (RTX 5080 Laptop):

| | batch 16 | batch 8 | batch 4 |
|---|---|---|---|
| everything trained | OOM (15811 MiB) | OOM (15651 MiB) | **12205 MiB, 1.22 step/s** |
| expert only, vision frozen | 6547 MiB, 1.78 step/s | -- | 2881 MiB, 6.5 step/s |

Sample throughput drops from 28/s to 4.9/s -- 5.7x slower. On a bigger card,
raise `batch_size` and move `steps` and `scheduler_decay_steps` together.

**The camera rename map must match between training and rollout.** The base's
config hard-codes `observation.images.camera{1,2,3}`; this rig records
`left_wrist` / `right_wrist` / `right_front`. Without a map `lerobot-train`
refuses to start (`Feature mismatch`). Which camera lands in which slot does not
matter — but if training and rollout disagree, every camera feeds the wrong
input and the policy behaves exactly as if it had never been trained. Both come
from one function reading one field in the profile, so they cannot diverge.

## 0. Dependencies

SmolVLA's processor imports `num2words` when it loads the tokenizer, and
lerobot's `pi` extra (what this repo installs) does not pull it in:

```bash
pip install -e ".[smolvla]"
```

`evo-rlt-smolvla check` verifies this up front rather than letting it surface
several minutes into a training run.

Warm the HuggingFace cache once. The base's weights and the processor live in
two different repos, and a missing processor fails late, inside `make_policy`,
as `OSError: Can't load processor for 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct'`:

```bash
python -c "
from huggingface_hub import snapshot_download; print(snapshot_download('lerobot/smolvla_base'))"
python -c "
from transformers import AutoProcessor
AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM2-500M-Video-Instruct'); print('processor ok')"
```

Afterwards `HF_HUB_OFFLINE=1` makes runs independent of the network, which is
what the numbers below were measured with.

## 1. Verify the simulator and the dataset

```bash
conda activate evo-rlt
evo-rlt-sim-snapshot verify
evo-rlt-smolvla check
```

The dataset is the merged blue-bolt set — 122 episodes, 83,247 frames, three
480×640 cameras, 12 state and 12 action dimensions. If it does not exist yet,
`evo-rlt-smolvla prepare` builds it from the same nine sessions ACT uses (and
`evo-rlt-act prepare` produces a byte-identical result; run either one).

## 2. Fine-tune

```bash
HF_HUB_OFFLINE=1 lerobot-train --config_path=configs/smolvla/train_config.json
```

That is lerobot's own entry point — every one of its flags works, and any flag
it gains later works too. The json pins what SmolVLA gets wrong *silently*:

| Setting | Value | Reason |
|---|---:|---|
| `policy.pretrained_path` | `lerobot/smolvla_base` | fine-tune. A bare `type` builds a **random** 450M model and trains it happily — identical logs, only the success rate tells you |
| `chunk_size` | 50 | fixed by the base; changing it discards the pretrained expert |
| `n_action_steps` | 10 | replan every 0.33 s instead of 1.67 s open-loop |
| `freeze_vision_encoder` / `train_expert_only` | both false | both, always — `train_expert_only` alone re-freezes the whole VLM |
| `load_vlm_weights` | false | weights come from `pretrained_path`; a second copy overwrites them |
| `num_expert_layers` / `prefix_length` / `pad_language_to` | 0 / 0 / `max_length` | **the base's values, not SmolVLAConfig's defaults** (-1 / -1 / `longest`) — see below |
| `rename_map` | right_front→camera1, … | the base hard-codes camera1/2/3 |
| batch size | 4 | measured ceiling on 16 GB with nothing frozen (12205 MiB) |
| steps / `scheduler_decay_steps` | 20,000 both | 80k samples ≈ 1 epoch, ~4.6 h; a mismatch strands the LR at its floor |

Those three middle fields matter because of how config loading differs.
`--policy.path=...` loads the base's `config.json` and inherits everything not
overridden; `--config_path=...` builds the policy config from this json alone,
so an omitted field falls back to the *dataclass* default — and for these three
that is not what the base uses. Omit them and the model changes shape while the
run looks normal. A test pins all three.

Overrides are plain lerobot flags, appended after `--config_path`:

```bash
lerobot-train --config_path=configs/smolvla/train_config.json \
  --steps=40000 --policy.scheduler_decay_steps=40000
lerobot-train --config_path=configs/smolvla/train_config.json --wandb.enable=true
# expert only: 5.7x faster, vision never adapts; batch 16 only fits when frozen
lerobot-train --config_path=configs/smolvla/train_config.json \
  --policy.freeze_vision_encoder=true --policy.train_expert_only=true --batch_size=16
```

Keep `steps` and `policy.scheduler_decay_steps` equal; a mismatch either decays
the learning rate to its floor long before training ends, or never finishes
decaying.

`output_dir` must not already exist — lerobot refuses rather than overwrite a
finished run. Move the old one aside or point `--output_dir` somewhere new.

### Continuing a run

20k steps is one epoch. To continue instead of paying for it again from the base:

```bash
lerobot-train \
  --config_path=outputs/smolvla_blue_screw_sim_v1/checkpoints/last/pretrained_model/train_config.json \
  --resume=true --steps=60000 --scheduler.num_decay_steps=60000
```

`--steps` is the **new total** counted from step 0, not the number to add.
Everything else — dataset, batch size, rename map, freeze flags — is restored
from that checkpoint's `train_config.json`, so overriding it has no effect;
changing the recipe needs a fresh run.

Two things make a naive resume useless:

**Never pass `--policy.path` with `--resume`.** `TrainPipelineConfig.validate()`
reads `if policy_path: ... elif self.resume: ...`, so the base path silently
wins and the run restarts from step 0 — the logs look like a resume.

**`--scheduler.num_decay_steps` must be stretched too.** The scheduler is
rebuilt from the **top-level** `scheduler` block, not from the policy's
`scheduler_decay_steps`, and its state restores `last_epoch` to the steps
already trained. Left at 20000 while training to 60000, the cosine is already
finished and all 40k extra steps run at `decay_lr`:

| | LR at resume | LR 100 steps later |
|---|---|---|
| `--steps=60000` alone | 2.5e-6 | 2.5e-6 (floor — trains nothing) |
| with `--scheduler.num_decay_steps=60000` | 2.5e-6 | 7.5e-5 |

## 3. Roll out in the simulator

Terminal 1 — the simulator, left running:

```bash
~/anaconda3/envs/rlt_sim/bin/python src/evo_rlt/sim/mj_server.py --viewer --show-cameras
```

Terminal 2:

```bash
evo-rlt-smolvla rollout \
  --checkpoint outputs/smolvla_blue_screw_sim_v1/checkpoints/060000/pretrained_model \
  --num-episodes 10
```

Point at a checkpoint directory containing `config.json`, the weights, and
`policy_preprocessor.json` / `policy_postprocessor.json` — not at the parent
`checkpoints/`. The command verifies the checkpoint is a complete *SmolVLA*
checkpoint (`type == "smolvla"`) before connecting.

Keys during evaluation, identical to ACT:

| key | effect |
|---|---|
| `s` | mark success **and end the episode** |
| `f` | mark failure **and end the episode** |
| `b` | put the parts back to their initial poses; arms untouched |

Every episode needs `s` or `f`; an unlabeled episode is rejected. Give the
terminal focus before pressing — the keyboard hook is global, but the MuJoCo
viewer will *also* react (`f` toggles contact-force arrows, `b` bounding boxes).

Between episodes the arms return to the home pose automatically and the parts
are re-randomised, so each rollout starts from a comparable state.

### Real-Time Chunking (on by default)

Without RTC every chunk is an independent draw from the flow-matching prior,
and on this checkpoint those draws are far apart: two samples of the *same*
observation differ about as much as either differs from the demonstration. So
re-planning every 10 steps swaps trajectories mid-motion — the arm steps
forward and back. RTC makes each new chunk an inpainting of the actions still
queued from the previous one, so the seam is constrained instead of resampled.

Measured on the 60k checkpoint, 6 segments x 40 steps (human demonstrations sit
at 0.070 for reference):

| per-step \|Δ\| | overall | inside a chunk | **at a chunk boundary** |
|---|---|---|---|
| without RTC | 0.382 | 0.347 | **0.895** |
| with RTC | 0.298 | 0.293 | **0.406** |

Boundary jumps drop 55%, overall jitter 22%. The remaining ~4x over human is
the policy's own sampling variance, not the seam — averaging 16 samples of one
observation brings jitter to 1.6x human, so that part is variance rather than
a training deficit.

`--no-rtc` restores independent sampling, which is what evaluations before this
was wired up were run with — use it for an A/B on the same checkpoint.

Two details worth knowing before changing anything:

**SmolVLA's `select_action()` refuses RTC** (`assert not self._rtc_enabled()`)
because the queue it keeps has no notion of a previous chunk. RTC is only
reachable through `predict_action_chunk()`, so
[rtc_chunk_runtime.py](src/evo_rlt/adapters/lerobot/policies/rtc_chunk_runtime.py)
drives the queue from outside. It re-plans at the same cadence as plain
chunking — once per `n_action_steps` — but *while actions remain queued*,
because those leftovers are the prefix that guides the next chunk.

**RTC has to leave `torch.inference_mode()`.** Guidance differentiates the
denoised chunk w.r.t. the latent, and inference mode is stronger than
`no_grad`: the `enable_grad()` inside `RTCProcessor.denoise_step` cannot lift
it, and tensors created under it are permanently barred from autograd. The
runtime exits inference mode and clones every tensor crossing in. Skip that and
the first guided chunk dies with `element 0 of tensors does not require grad` —
mid-rollout, not in any preflight. A test pins it.

Cost: the re-planning step goes from 128 ms to 244 ms, on 10% of control steps,
so roughly +10% wall clock over an episode. Which matters only because —

### Why `--episode-time-s` defaults to 150

`episode_time_s` is **wall clock**, while the simulator advances a fixed 1/fps
of sim time per control step. A policy whose control loop runs at 9 Hz therefore
gets only `150 × 9 / 30 = 45 s` of sim time out of a 150 s budget. Human
demonstrations run 22 s (median) to 39 s (longest), so a 45 s default sized for
a 30 Hz loop would cut episodes off before the task could finish. If SmolVLA's
loop turns out slower than ACT's, raise this further — and use the *same* value
for both policies, or the comparison hands one of them more chances.

RTC's +10% wall clock lands here: at a fixed 150 s budget an RTC rollout gets
correspondingly less sim time than a `--no-rtc` one. For an A/B between them,
either raise `--episode-time-s` by 10% for the RTC run or check that episodes
are ending on `s`/`f` rather than on the timeout.

## 4. Compare

```bash
python -c "
import pandas as pd, glob
for r in sorted(glob.glob('data/bimanual/*_smolvla_blue_screw_eval/*/')) + \
         sorted(glob.glob('data/bimanual/*_act_blue_screw_eval/*/')):
    fs = sorted(glob.glob(r + 'meta/episodes/**/*.parquet', recursive=True))
    if not fs: continue
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    print(f'{r:<70} n={len(d):<3} {dict(d[\"episode_success\"].value_counts())}')
"
```

Evaluate several checkpoints on the same number of resets rather than assuming
the last one is best, and keep `--num-episodes` and `--episode-time-s` identical
across policies.
