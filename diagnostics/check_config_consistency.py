#!/usr/bin/env python
"""Diff config.json fields between a trained rlt_ac checkpoint, its rlt_token
checkpoint, and (optionally) the flags actually used at cache-build/deploy time.
No torch/lerobot import needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ac-config-dir", required=True)
    p.add_argument("--rl-token-config-dir", default=None, help="defaults to rl_token_pretrained_path in ac config")
    p.add_argument("--cache-build-chunk-length", type=int, default=None)
    p.add_argument("--cache-build-frame-stride", type=int, default=None)
    p.add_argument("--deploy-chunk-exec-steps", type=int, default=None)
    p.add_argument("--deploy-phase-mode", default=None)
    return p.parse_args()


def load_config(config_dir: str) -> dict:
    return json.loads((Path(config_dir) / "config.json").read_text())


def check(label: str, expected, actual) -> None:
    status = "OK" if expected == actual else "MISMATCH"
    print(f"  [{status}] {label}: yours={expected!r} checkpoint={actual!r}")


def main() -> None:
    args = parse_args()
    ac_cfg = load_config(args.ac_config_dir)
    print(f"=== rlt_ac: {args.ac_config_dir} ===")
    for f in ["chunk_length", "chunk_exec_steps", "phase_mode", "action_dim", "proprio_dim",
              "rl_token_dim", "rl_token_num_rl_tokens", "vla_pretrained_path", "vla_dtype",
              "rl_token_pretrained_path", "tokenizer_path", "gamma", "beta", "tau"]:
        print(f"  {f} = {ac_cfg.get(f, '<missing>')!r}")

    rt_dir = args.rl_token_config_dir or ac_cfg.get("rl_token_pretrained_path")
    if rt_dir:
        rt_cfg = load_config(rt_dir)
        print(f"\n=== rlt_ac vs rlt_token ({rt_dir}) ===")
        for f in ["rl_token_dim", "rl_token_num_rl_tokens", "action_dim", "proprio_dim",
                  "token_pool_size", "vla_pretrained_path"]:
            check(f, ac_cfg.get(f), rt_cfg.get(f))

    print("\n=== cache-build / deploy args vs ac config ===")
    if args.cache_build_chunk_length is not None:
        check("chunk_length", args.cache_build_chunk_length, ac_cfg.get("chunk_length"))
    if args.cache_build_frame_stride is not None:
        cl = ac_cfg.get("chunk_length")
        ok = cl is not None and cl % args.cache_build_frame_stride == 0
        print(f"  [{'OK' if ok else 'MISMATCH'}] chunk_length({cl}) % frame_stride({args.cache_build_frame_stride}) == 0")
    if args.deploy_chunk_exec_steps is not None:
        check("chunk_exec_steps", args.deploy_chunk_exec_steps, ac_cfg.get("chunk_exec_steps"))
    if args.deploy_phase_mode is not None:
        check("phase_mode", args.deploy_phase_mode, ac_cfg.get("phase_mode"))


if __name__ == "__main__":
    main()
