# Calibration snapshot

A copy of this machine's SO-101 calibration, taken 2026-08-14 from
`~/.cache/huggingface/lerobot/calibration/`. Vendored so the sim bridge does not
silently depend on a cache that other projects also write to.

```
robots/left_follower_arm.json          followers -- what SimRobot maps through
robots/right_follower_arm.json
teleoperators/left_leader_arm.json     leaders -- reference copy
teleoperators/right_leader_arm.json
```

`SimRobot` reads `robots/` by default. The simulator itself never sees these
files: calibration is resolved on the client side, and only radians cross the
process boundary.

## This is a snapshot, and snapshots go stale

**Recalibrating the real arms updates the live files and leaves this copy
behind.** Sim and real then sit in different pose spaces, and nothing about the
data will look obviously wrong. After any recalibration:

```bash
python diagnostics/check_sim_calib.py --check-drift
```

It diffs this snapshot against the live files field by field and prints the
refresh command if they have parted ways. Re-run the loopback check afterwards --
poses will have shifted.

## These files are machine-specific

They describe *these* two arms. `range_min` / `range_max` are where a human
pushed each joint during calibration, which is why the two arms disagree: a
transported value of 0 lands up to 15.8 degrees apart between left and right.
That per-arm difference is exactly what `calib.py` reproduces, and why mapping
values onto URDF limits instead would be wrong.

On different hardware, recalibrate and replace these.
