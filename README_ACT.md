# ACT on the blue-screw simulation dataset

This path trains standard LeRobot ACT behavior cloning. It does **not** feed
ACT into Evo-RLT's RL-token or actor-critic stages; those stages depend on
pi0.5 prefix tokens and remain pi0.5-only.

The pinned dataset profile is `configs/blue_screw_sim_v1.json`, shared with
SmolVLA so both are judged on the same episodes. It names only the
nine non-empty blue-bolt sessions: 122 successful episodes, 83,247 frames,
three 480×640 cameras, 12 state dimensions, and 12 action dimensions.

## 1. Verify the preserved simulator

```bash
conda activate evo-rlt
evo-rlt-sim-snapshot verify
```

If it reports a difference, restore the captured environment and rebuild:

```bash
evo-rlt-sim-snapshot restore
~/anaconda3/envs/rlt_sim/bin/python src/evo_rlt/sim/mj_server.py --build --benchmark
```

Restore always backs up the displaced files under
`outputs/sim_snapshot_backups/<timestamp>/`.

## 2. Validate and merge only the blue sessions

Inspect the nine source datasets without writing anything:

```bash
evo-rlt-act check --sources
```

Preview the exact lossless merge command:

```bash
evo-rlt-act prepare --dry-run
```

Create `data/bimanual/blue_screw_sim_v1`:

```bash
evo-rlt-act prepare
evo-rlt-act check
```

If that output already exists and is intentionally being regenerated, use
`evo-rlt-act prepare --overwrite`. Only that exact output directory is
removed. The nine source sessions are never changed.

The merge keeps source MP4 files separate to avoid duplicate-DTS failures at
session boundaries. Preflight rejects a camera mismatch, joint-order mismatch,
wrong dimensions/FPS, wrong totals, empty data, or any episode not labeled
`success`.

## 3. Train ACT

```bash
lerobot-train --config_path=configs/act/train_config.json
```

lerobot's own entry point — every flag it has works, including ones added
later. The json holds the pinned settings:

| Setting | Value | Reason |
|---|---:|---|
| `chunk_size` | 100 | 3.33 s training target at 30 FPS |
| `n_action_steps` | 10 | replan every 0.33 s instead of executing 3.33 s open-loop |
| batch size | 8 | conservative default for three 480×640 views |
| steps | 60,000 | first full run for 122 demonstrations |
| checkpoint interval | 10,000 | compare overfitting and rollout behavior |
| output | `outputs/act_blue_screw_sim_v1` | separate from pi0.5/RLT runs |

`n_action_steps` is the only one of these that differs from ACT's own default
(100), and it is the load-bearing one. Overrides are plain lerobot flags:

```bash
lerobot-train --config_path=configs/act/train_config.json --batch_size=4
lerobot-train --config_path=configs/act/train_config.json --steps=30000
lerobot-train --config_path=configs/act/train_config.json --wandb.enable=true
```

To continue a finished run rather than starting over:

```bash
lerobot-train \
  --config_path=outputs/act_blue_screw_sim_v1/checkpoints/last/pretrained_model/train_config.json \
  --resume=true --steps=90000
```

`--steps` is the new total counted from step 0. Never pass `--policy.path`
alongside `--resume`: `validate()` reads `if policy_path: ... elif self.resume:`,
so it silently wins and the run restarts from step 0. ACT needs nothing done
about the LR — its scheduler is `None`, a constant 1e-5 (SmolVLA's cosine decay
does need care; see [README_SMOLVLA.md](README_SMOLVLA.md)).

`output_dir` must not already exist — lerobot refuses rather than overwrite a
finished run.

Do not increase `n_action_steps` just to make inference cheaper: insertion is
contact-sensitive, so a long open-loop window can turn a visually small error
into a failed grasp or collision. Training `chunk_size` and deployment
`n_action_steps` serve different purposes.

## 4. Run the trained checkpoint in the preserved simulator

Start the simulator:

```bash
~/anaconda3/envs/rlt_sim/bin/python src/evo_rlt/sim/mj_server.py --viewer --show-cameras
```

In the training environment, point rollout at a numeric checkpoint:

```bash
evo-rlt-act rollout \
  --checkpoint outputs/act_blue_screw_sim_v1/checkpoints/060000/pretrained_model \
  --num-episodes 10
```

Use the checkpoint directory containing `config.json`, model weights,
`policy_preprocessor.json`, and `policy_postprocessor.json`; do not point at
the parent `checkpoints/` directory. The rollout command verifies that it is a
complete ACT checkpoint before connecting to the simulator.

During recorded evaluation, `s` marks success and `f` marks failure. `b` puts the
task parts back to their initial poses without moving the arms -- rollout runs
with no leader arms, so this is the only way to straighten a part the policy
knocked over. Results
are saved under the normal `data/bimanual/<date>_act_blue_screw_eval/` path.
For a useful comparison, evaluate the 10k, 20k, …, 60k checkpoints on the same
number of resets instead of assuming the last checkpoint is best.

## 5. Recovery boundaries

- The simulation snapshot includes source/config/calibration, the SO-101
  model, and all task meshes/convex hulls needed to rebuild `scene.xml`.
- It does not duplicate demonstrations or model checkpoints.
- `evo-rlt-sim-snapshot restore` changes only the listed simulation scopes;
  ACT, pi0.5, RLT code, datasets, and outputs are outside that boundary.
