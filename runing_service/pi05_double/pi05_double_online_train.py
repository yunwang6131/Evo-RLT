#!/usr/bin/env python3

"""双臂 SO-101 的同步 Evo-RLT 在线训练入口。

  python runing_service/pi05_double/pi05_double_online_train.py

   检查输出中的串口、相机、VLA、RL Token、tokenizer 和保存目录。当前默认
   串口及相机来自 configs/my_so101_manifest.json。

   python runing_service/pi05_double/pi05_double_online_train.py \\
     --num-episodes 20 \\
     --save-every-episodes 5 \\
     --reset-time-s 15 \\
     --run

    默认资源
VLA
    pretrained/pi05_full_ft/pretrained_model
RL Token
    outputs/bimanual_rl_token/checkpoints/last/pretrained_model
Robot manifest
    configs/my_so101_manifest.json
Online checkpoints
    outputs/pin_insert_online_rl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from evo_rlt.adapters.lerobot.record.online_cli import main as online_train_main  # noqa: E402


TASK = (
    "Pick up the black hexagonal part with the right arm, pull the gray pin out "
    "of the white platform with the left arm, align the gray pin with the hole "
    "in the side of the black hexagonal part, insert the gray pin into the hole, "
    "and place the assembled object in the red square area."
)
DEFAULT_TOKENIZER = Path(
    "/home/wangyun/.cache/huggingface/hub/"
    "models--google--paligemma-3b-pt-224/snapshots/"
    "35e4f46485b4d07967e7e9935bc3786aad50687c"
)


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Bimanual Evo-RLT online training configured from README_online.md.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--setup-json", type=Path, default=REPO_ROOT / "configs/my_so101_manifest.json")
    parser.add_argument(
        "--vla-path", type=Path, default=REPO_ROOT / "pretrained/pi05_full_ft/pretrained_model"
    )
    parser.add_argument(
        "--rl-token-path",
        type=Path,
        default=REPO_ROOT / "outputs/bimanual_rl_token/checkpoints/last/pretrained_model",
    )
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--save-dir", type=Path, default=REPO_ROOT / "outputs/pin_insert_online_rl")
    parser.add_argument("--dataset-tag", default="pin_insert_online_rl")
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--actor-action-clip-delta", type=float, default=0.05)
    parser.add_argument("--save-every-episodes", type=int, default=5)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run on real hardware. Without this flag only the generated configuration is printed.",
    )
    return parser.parse_known_args(argv)


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def require_checkpoint(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not (resolved / "config.json").is_file() or not (resolved / "model.safetensors").is_file():
        raise FileNotFoundError(f"{label} is not a complete pretrained_model directory: {resolved}")
    return resolved


def main(argv: list[str] | None = None) -> None:
    args, passthrough = parse_args(argv)
    setup = require_file(args.setup_json, "Robot manifest")
    vla = require_checkpoint(args.vla_path, "VLA checkpoint")
    rl_token = require_checkpoint(args.rl_token_path, "RL Token checkpoint")
    tokenizer = args.tokenizer_path.expanduser().resolve()
    if not tokenizer.is_dir():
        raise FileNotFoundError(f"Tokenizer snapshot not found: {tokenizer}")

    online_argv = [
        "--setup-json", str(setup),
        "--vla-path", str(vla),
        "--rl-token-path", str(rl_token),
        "--tokenizer-path", str(tokenizer),
        "--task", args.task,
        "--num-episodes", str(args.num_episodes),
        "--actor-action-clip-delta", str(args.actor_action_clip_delta),
        "--save-dir", str(args.save_dir.expanduser().resolve()),
        "--save-every-episodes", str(args.save_every_episodes),
        "--dataset-tag", args.dataset_tag,
        *passthrough,
    ]
    if not args.run:
        online_argv.append("--dry-run")
    online_train_main(online_argv)


if __name__ == "__main__":
    main()
