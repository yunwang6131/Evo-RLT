#!/usr/bin/env python
"""把任务零件放回初始位姿,手臂不动。

    python diagnostics/reset_objects.py              # 全部零件
    python diagnostics/reset_objects.py bolt         # 只放回螺栓
    python diagnostics/reset_objects.py --list       # 看场景里有哪些零件

采数据或调试时零件被碰歪了要重摆。整体 reset 会把手臂一起弹回复位姿态,遥操
到一半这么来一下,手上的主臂和仿真里的从臂就对不上了,所以单列这条命令。

另开一个终端跑即可,不用停遥操 —— ZMQ 的 REQ/REP 能同时接多个客户端,请求
排队处理。位姿取自场景的 home keyframe,和整体 reset 是同一份基准。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "evo_rlt" / "sim"))

from protocol import Command, DEFAULT_ENDPOINT, DEFAULT_TIMEOUT_S, Status  # noqa: E402


def request(endpoint: str, payload: dict, timeout_s: float) -> dict:
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
    sock.connect(endpoint)
    try:
        sock.send_json(payload)
        return sock.recv_json()
    except zmq.error.Again:
        raise SystemExit(
            f"{endpoint} 上没有仿真器响应。先启动:\n"
            "  python src/evo_rlt/sim/mj_server.py --viewer"
        ) from None
    finally:
        sock.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("objects", nargs="*", help="零件名,省略即全部")
    p.add_argument("--list", action="store_true", help="列出场景里的零件")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    args = p.parse_args()

    if args.list:
        reply = request(args.endpoint, {"command": Command.HANDSHAKE}, args.timeout)
        objs = reply.get("free_objects")
        if objs is None:
            raise SystemExit("仿真器没报告零件列表 —— 重启仿真器加载新版 mj_server.py")
        print("场景里的零件:", " ".join(objs) if objs else "(无)")
        return 0

    payload: dict = {"command": Command.RESET_OBJECTS}
    if args.objects:
        payload["objects"] = args.objects
    reply = request(args.endpoint, payload, args.timeout)
    if reply.get("status") != Status.OK:
        raise SystemExit(f"复位失败: {reply.get('error', reply)}")
    print("已放回:", " ".join(reply.get("objects_reset", [])) or "(无)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
