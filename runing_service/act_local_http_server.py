#!/usr/bin/env python3

"""Local-only ACT HTTP server.

The inference and wire-protocol implementation is shared with
``act_http_server.py``.  This entry point deliberately binds only to the
loopback interface on TCP port 8000, so it cannot be reached from another
machine and has no remote-service configuration.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from act_http_server import ACTHTTPServer, ACTService

LOCAL_HOST = "127.0.0.1"
LOCAL_PORT = 8001

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ACT inference locally on 127.0.0.1:8000.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--policy_path", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--mock", action="store_true", help="Run deterministic inference without a checkpoint."
    )
    parser.add_argument("--chunk_size", type=int, default=50, help="Mock ACT action horizon.")
    parser.add_argument("--latency_ms", type=float, default=0.0)
    parser.add_argument("--max_body_mb", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    service = ACTService(
        policy_path=args.policy_path,
        device=args.device,
        mock=args.mock,
        chunk_size=args.chunk_size,
        latency_ms=args.latency_ms,
    )
    server = ACTHTTPServer(
        (LOCAL_HOST, LOCAL_PORT),
        service,
        max_body_bytes=int(args.max_body_mb * 1024 * 1024),
    )
    logger.info("Local ACT server listening on http://%s:%d", LOCAL_HOST, LOCAL_PORT)
    logger.info("metadata=%s", service.metadata())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
