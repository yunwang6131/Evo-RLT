#!/usr/bin/env python3

"""Local-only entry point for the CRP PI05 RTC executor.

Reuses ``pi05_rtc_executor.py`` and pins the server to ``http://127.0.0.1:8002``.
"""

from __future__ import annotations

import argparse

import pi05_rtc_executor as executor

LOCAL_SERVER = "http://127.0.0.1:8002"


def parse_local_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CRP PI05 RTC executor against the local server on port 8002.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        default="",
        help="Language instruction. Falls back to server metadata default_task when empty.",
    )
    parser.add_argument(
        "--robot_address",
        "--robot_port",
        dest="robot_address",
        default="192.168.0.100",
        help="CRP controller IPv4 address; --robot_port is a compatibility alias.",
    )
    parser.add_argument("--top_camera", type=int, default=6)
    parser.add_argument("--wrist_camera", type=int, default=14)
    parser.add_argument(
        "--speed_ratio",
        type=int,
        default=None,
        help="Set CRP speed ratio on connect; omitted keeps the current setting.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=180,
        help="Control steps; 0 runs continuously until Ctrl+C.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--execution_horizon", type=int, default=10)
    parser.add_argument(
        "--rtc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace stale action tails with RTC; --no-rtc appends complete chunks.",
    )
    parser.add_argument(
        "--print-action-chunks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print every complete action chunk returned by the local server.",
    )
    parser.add_argument(
        "--actions_per_chunk",
        type=int,
        default=0,
        help="0 uses local server metadata.",
    )
    parser.add_argument("--request_timeout", type=float, default=60.0)
    parser.add_argument("--jpeg_quality", type=int, default=70)
    parser.add_argument("--max_consecutive_misses", type=int, default=10)
    parser.add_argument(
        "--allow_mock_server",
        action="store_true",
        help="Allow mock actions to reach the real arm (controlled bench tests only).",
    )
    parser.add_argument("--skip_confirm", action="store_true")
    args = parser.parse_args()
    args.server = LOCAL_SERVER
    args.token = ""
    return args


def main() -> None:
    executor.parse_args = parse_local_args
    try:
        executor.main()
    except RuntimeError as exc:
        if "Failed to obtain IRobotService from SDK" in str(exc):
            raise RuntimeError(
                "CRP SDK could not create IRobotService. This occurs before PI05/port-8002 "
                "communication. Check the CRP SDK license, controller address/network, "
                "controller service status, and whether another process owns the SDK session."
            ) from exc
        raise


if __name__ == "__main__":
    main()
