# Blue screw simulation snapshot v1

This snapshot preserves the simulation used to record the 122 blue-bolt
episodes under `data/bimanual/0821_teleop_full`.

It has two parts:

- tracked configuration, calibration, SO-101 model, and simulator source are
  pinned to Git tag `sim-blue-screw-v1` (commit
  `387787a1fa6c82b597b60b7502405dd62c35ae6e`);
- ignored task meshes and convex-decomposition outputs are copied under
  `assets/` and verified by deterministic SHA-256 tree digests.

Check whether the working simulation still matches the snapshot:

```bash
evo-rlt-sim-snapshot verify
```

Restore it (the current files are backed up under
`outputs/sim_snapshot_backups/` first):

```bash
evo-rlt-sim-snapshot restore
```

Then rebuild and restart the simulator:

```bash
~/anaconda3/envs/rlt_sim/bin/python src/evo_rlt/sim/mj_server.py --build --benchmark
```

The dataset itself is deliberately not duplicated by this environment
snapshot.
