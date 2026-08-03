#!/usr/bin/env python
"""Lightweight local critical-segment labeler -- no web server, no browser.

Opens a single native Tkinter window (needs a local X display, or `ssh -X`)
showing the dataset's camera feeds for one episode at a time. You scrub
through the episode and mark the start/end frame of the critical
(fine-manipulation) sub-trajectory with keyboard shortcuts; each marked
segment also gets a success/failure outcome, since that outcome is what
`build_reward_seq` needs per segment once transitions are built only from
the cropped critical span instead of the whole episode.

This does not touch the source dataset. It only reads meta/episodes/*.parquet
for the episode->video/timestamp mapping and writes a small sidecar JSON.

Usage:
    conda activate evo-rlt
    python diagnostics/critical_segment_labeler_cv.py --dataset-root data/bimanual/merged_screw_v1

Output: <dataset-root>/meta/critical_segments.json
    {"<episode_index>": {"segment": [start_frame, end_frame, "success"|"failure"] | null,
                          "no_critical": bool, "milestone_frame": int | null,
                          "note": str, "updated_at": str}, ...}
(frame indices are relative to the start of the episode; one critical segment
per episode, same convention as the live `--only-critical` recorder: `r`
toggles entering/exiting critical phase, `u` exits as a failure instead.
`milestone_frame` is optional, mirroring the live recorder's separate
milestone_key: a single frame marking a one-time mid-phase shaping bonus
-- e.g. "pin pulled out" ahead of a separate final "inserted" judgment --
that build_transition_cache_v2 rewards regardless of whether the segment
itself ends success or failure. null means the demonstration never reached
it, same as never pressing the milestone key live.)

Keys:
    r               not in critical: mark critical-phase start at current frame
                    in critical: close it here and save as SUCCESS
    u               in critical: close it here and save as FAILURE (no-op otherwise)
    m               mark the milestone at the current frame (requires a
                    segment already started or closed this episode; press
                    again to move it). M (shift+m) clears it.
    space           play / pause
    a / d           step 1 frame back / forward
    A / D           step 10 frames back / forward (shift+a, shift+d)
    z               reset (clear the marked/in-progress segment and milestone for this episode)
    x               toggle "no critical segment in this episode"
    1 / 2 / 3       playback speed 0.25x / 0.5x / 1x
    enter           save this episode's label and go to the next episode
    b               previous episode (no save)
    n               next episode (no save)
    q / esc         quit
"""
from __future__ import annotations

import argparse
import json
import time
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk


def load_info(dataset_root: Path) -> dict:
    with open(dataset_root / "meta" / "info.json") as f:
        return json.load(f)


def load_episodes(dataset_root: Path, camera_keys: list[str]) -> list[dict]:
    files = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata parquet under {dataset_root}/meta/episodes")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values("episode_index").reset_index(drop=True)

    episodes = []
    for _, row in df.iterrows():
        task = row["tasks"]
        if hasattr(task, "__iter__") and not isinstance(task, str):
            task = ", ".join(str(t) for t in task)
        cameras = {}
        for key in camera_keys:
            cameras[key] = {
                "chunk_index": int(row[f"videos/{key}/chunk_index"]),
                "file_index": int(row[f"videos/{key}/file_index"]),
                "from_timestamp": float(row[f"videos/{key}/from_timestamp"]),
            }
        episodes.append(
            {
                "episode_index": int(row["episode_index"]),
                "length": int(row["length"]),
                "task": str(task),
                "cameras": cameras,
            }
        )
    return episodes


def load_labels(path: Path) -> dict:
    if path.is_file():
        with open(path) as f:
            return json.load(f)
    return {}


def save_labels(path: Path, labels: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    with open(tmp, "w") as f:
        json.dump(labels, f, indent=2, sort_keys=True)
    tmp.replace(path)


class EpisodePlayer:
    """Owns the per-camera cv2.VideoCapture handles for one episode."""

    def __init__(self, dataset_root: Path, episode: dict, fps: float):
        self.episode = episode
        self.fps = fps
        self.camera_keys = list(episode["cameras"].keys())
        self.caps: dict[str, cv2.VideoCapture] = {}
        self.start_frame_abs: dict[str, int] = {}
        for key, meta in episode["cameras"].items():
            video_path = (
                dataset_root
                / "videos"
                / key
                / f"chunk-{meta['chunk_index']:03d}"
                / f"file-{meta['file_index']:03d}.mp4"
            )
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {video_path}")
            self.caps[key] = cap
            self.start_frame_abs[key] = round(meta["from_timestamp"] * fps)
        self.local_frame = -1

    def seek_and_read(self, local_frame: int) -> dict:
        local_frame = max(0, min(self.episode["length"] - 1, local_frame))
        self.local_frame = local_frame
        frames = {}
        for key, cap in self.caps.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame_abs[key] + local_frame)
            ok, frame = cap.read()
            frames[key] = frame if ok else None
        return frames

    def advance_and_read(self) -> dict | None:
        if self.local_frame >= self.episode["length"] - 1:
            return None
        self.local_frame += 1
        frames = {}
        for key, cap in self.caps.items():
            ok, frame = cap.read()
            frames[key] = frame if ok else None
        return frames

    def release(self) -> None:
        for cap in self.caps.values():
            cap.release()


def compose_video_row(frames: dict, camera_keys: list[str], cam_w: int = 480) -> np.ndarray:
    imgs = []
    for key in camera_keys:
        frame = frames.get(key)
        if frame is None:
            img = np.zeros((int(cam_w * 0.75), cam_w, 3), dtype=np.uint8)
            cv2.putText(img, "no frame", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 220), 1, cv2.LINE_AA)
        else:
            h, w = frame.shape[:2]
            scale = cam_w / w
            img = cv2.resize(frame, (cam_w, int(h * scale)))
        cv2.putText(img, key, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        imgs.append(img)
    return np.hstack(imgs)


LEGEND_1 = "r: enter critical / exit=OK    u: exit=FAIL    m: mark milestone (M: clear)    space play/pause"
LEGEND_2 = "a/d -1/+1f  A/D -10/+10f  z reset  x no-critical  1/2/3 speed  ENTER save+next  b prev-ep  n next-ep  q quit"


def render(
    episodes, idx, player, frames, segment, no_critical, crit_start, milestone_frame,
    playing, rate, labeled,
) -> np.ndarray:
    ep = episodes[idx]
    video_row = compose_video_row(frames, player.camera_keys)
    W = video_row.shape[1]
    header_h, timeline_h, footer_h = 46, 22, 70
    canvas = np.full((video_row.shape[0] + header_h + timeline_h + footer_h, W, 3), 20, dtype=np.uint8)

    title = (
        f"Episode {ep['episode_index']}  ({idx + 1}/{len(episodes)})  "
        f"frame {player.local_frame}/{ep['length'] - 1}  speed={rate}x  {'PLAY' if playing else 'PAUSE'}"
    )
    cv2.putText(canvas, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if labeled:
        cv2.putText(canvas, "LABELED", (W - 130, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 210, 130), 1, cv2.LINE_AA)
    if crit_start is not None:
        cv2.putText(canvas, "IN CRITICAL", (W - 260, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 60), 1, cv2.LINE_AA)
    cv2.putText(canvas, ep["task"][:150], (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 180, 190), 1, cv2.LINE_AA)

    y0 = header_h
    canvas[y0 : y0 + video_row.shape[0], :] = video_row
    y1 = y0 + video_row.shape[0]

    length = ep["length"]

    def xf(f: int) -> int:
        return int(max(0, min(W - 1, f / length * W)))

    tl_y0, tl_y1 = y1 + 4, y1 + 4 + timeline_h
    cv2.rectangle(canvas, (0, tl_y0), (W, tl_y1), (45, 48, 54), -1)
    if segment is not None:
        s, e, outcome = segment
        color = (70, 190, 110) if outcome == "success" else (60, 60, 220)
        cv2.rectangle(canvas, (xf(s), tl_y0), (xf(e), tl_y1), color, -1)
    if crit_start is not None:
        s, e = sorted((crit_start, player.local_frame))
        cv2.rectangle(canvas, (xf(s), tl_y0), (xf(e), tl_y1), (255, 180, 60), 2)
    if milestone_frame is not None:
        mx = xf(milestone_frame)
        cv2.line(canvas, (mx, tl_y0), (mx, tl_y1), (60, 210, 230), 2)
    ph_x = xf(player.local_frame)
    cv2.line(canvas, (ph_x, tl_y0), (ph_x, tl_y1), (60, 60, 255), 2)

    fy = tl_y1 + 18
    if no_critical:
        seg_text = "NO CRITICAL SEGMENT for this episode"
    elif segment is not None:
        s, e, outcome = segment
        seg_text = f"[{s}, {e}]  {'OK' if outcome == 'success' else 'FAIL'}"
    elif crit_start is not None:
        seg_text = f"critical start marked at frame {crit_start}, press r (=OK) or u (=FAIL) to close"
    else:
        seg_text = "not marked yet -- press r at the critical-phase start frame"
    milestone_text = f"  |  milestone: {milestone_frame}" if milestone_frame is not None else "  |  milestone: (none, press m)"
    cv2.putText(canvas, ("segment: " + seg_text + milestone_text)[:200], (8, fy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 210, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, LEGEND_1, (8, fy + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 160, 170), 1, cv2.LINE_AA)
    cv2.putText(canvas, LEGEND_2, (8, fy + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 160, 170), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--labels-path", default=None)
    ap.add_argument("--start-index", type=int, default=None, help="Episode list index to start at (default: first unlabeled episode).")
    ap.add_argument("--cam-width", type=int, default=480, help="Per-camera display width in pixels.")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    labels_path = Path(args.labels_path).resolve() if args.labels_path else dataset_root / "meta" / "critical_segments.json"

    info = load_info(dataset_root)
    camera_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    if not camera_keys:
        raise RuntimeError(f"No video features found in {dataset_root}/meta/info.json")
    fps = float(info["fps"])

    episodes = load_episodes(dataset_root, camera_keys)
    labels = load_labels(labels_path)
    print(f"Loaded {len(episodes)} episodes, cameras={camera_keys}, fps={fps}")
    print(f"Labels file: {labels_path} ({len(labels)} already labeled)")
    print(LEGEND_1)
    print(LEGEND_2)

    if args.start_index is not None:
        idx = max(0, min(len(episodes) - 1, args.start_index))
    else:
        idx = 0
        for i, ep in enumerate(episodes):
            if str(ep["episode_index"]) not in labels:
                idx = i
                break

    state = {
        "segment": None, "no_critical": False, "crit_start": None,
        "milestone_frame": None, "playing": False, "rate": 1.0,
    }
    player = None

    def load_episode(new_idx: int) -> None:
        nonlocal player, idx
        if player is not None:
            player.release()
        idx = max(0, min(len(episodes) - 1, new_idx))
        player = EpisodePlayer(dataset_root, episodes[idx], fps)
        existing = labels.get(str(episodes[idx]["episode_index"]))
        state["segment"] = list(existing["segment"]) if existing and existing.get("segment") else None
        state["no_critical"] = bool(existing["no_critical"]) if existing else False
        state["crit_start"] = None
        state["milestone_frame"] = (
            existing.get("milestone_frame") if existing and existing.get("milestone_frame") is not None else None
        )
        state["playing"] = False

    load_episode(idx)
    frames = player.seek_and_read(0)

    def do_save() -> None:
        ep = episodes[idx]
        labels[str(ep["episode_index"])] = {
            "segment": state["segment"],
            "no_critical": state["no_critical"],
            "milestone_frame": state["milestone_frame"],
            "note": "",
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        save_labels(labels_path, labels)
        print(
            f"saved episode {ep['episode_index']}: segment={state['segment']} "
            f"no_critical={state['no_critical']} milestone_frame={state['milestone_frame']}"
        )

    # The evo-rlt env's opencv build is headless (no highgui window backend:
    # `cv2.getBuildInformation()` reports "GUI: NONE"), so cv2.imshow/waitKey
    # cannot open a window here. Tkinter (stdlib) + Pillow are confirmed
    # available in this env and don't need any package changes, so the
    # window/event-loop layer uses those instead; cv2 is still used for
    # video decoding and for drawing onto the numpy canvas (imgproc, not
    # highgui, so it works fine in a headless build).
    root = tk.Tk()
    root.title("critical-segment-labeler")
    img_label = tk.Label(root)
    img_label.pack()
    photo_holder: dict = {"img": None}

    def redraw() -> None:
        labeled = str(episodes[idx]["episode_index"]) in labels
        canvas = render(
            episodes, idx, player, frames, state["segment"], state["no_critical"],
            state["crit_start"], state["milestone_frame"], state["playing"], state["rate"], labeled,
        )
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        photo_holder["img"] = ImageTk.PhotoImage(Image.fromarray(rgb))
        img_label.configure(image=photo_holder["img"])

    def on_key(event: "tk.Event") -> None:
        nonlocal frames
        key, keysym = event.char, event.keysym
        if keysym == "Escape" or key == "q":
            root.destroy()
            return
        elif keysym == "space":
            state["playing"] = not state["playing"]
        elif key == "a":
            state["playing"] = False
            frames = player.seek_and_read(player.local_frame - 1)
        elif key == "d":
            state["playing"] = False
            frames = player.seek_and_read(player.local_frame + 1)
        elif key == "A":
            state["playing"] = False
            frames = player.seek_and_read(player.local_frame - 10)
        elif key == "D":
            state["playing"] = False
            frames = player.seek_and_read(player.local_frame + 10)
        elif key == "r":
            if state["crit_start"] is None:
                # (re)start: any previously finalized segment is discarded,
                # matching the live recorder where re-entering critical
                # phase means you're redoing it. Its milestone (if any)
                # belonged to that discarded segment, so it goes too.
                state["segment"] = None
                state["crit_start"] = player.local_frame
                state["no_critical"] = False
                state["milestone_frame"] = None
            else:
                s, e = sorted((state["crit_start"], player.local_frame))
                state["segment"] = [s, e, "success"]
                state["crit_start"] = None
        elif key == "u":
            if state["crit_start"] is not None:
                s, e = sorted((state["crit_start"], player.local_frame))
                state["segment"] = [s, e, "failure"]
                state["crit_start"] = None
        elif key == "m":
            # Mark/move the milestone to the current frame -- meaningful once
            # a segment has been started (in progress) or closed this
            # episode, mirroring the live recorder where milestone_key only
            # does anything mid-critical-phase attempt.
            if state["crit_start"] is not None or state["segment"] is not None:
                state["milestone_frame"] = player.local_frame
        elif key == "M":
            state["milestone_frame"] = None
        elif key == "z":
            state["segment"] = None
            state["crit_start"] = None
            state["milestone_frame"] = None
        elif key == "x":
            state["no_critical"] = not state["no_critical"]
            if state["no_critical"]:
                state["segment"] = None
                state["crit_start"] = None
                state["milestone_frame"] = None
        elif key in ("1", "2", "3"):
            state["rate"] = {"1": 0.25, "2": 0.5, "3": 1.0}[key]
        elif keysym in ("Return", "KP_Enter"):
            do_save()
            if idx + 1 < len(episodes):
                load_episode(idx + 1)
                frames = player.seek_and_read(0)
            else:
                print("All episodes reached the end of the list.")
        elif key == "b":
            load_episode(idx - 1)
            frames = player.seek_and_read(0)
        elif key == "n":
            load_episode(idx + 1)
            frames = player.seek_and_read(0)
        else:
            return
        redraw()

    def tick() -> None:
        nonlocal frames
        if state["playing"]:
            nf = player.advance_and_read()
            if nf is None:
                state["playing"] = False
            else:
                frames = nf
            redraw()
        delay = max(1, int(1000 / (fps * state["rate"]))) if state["playing"] else 40
        root.after(delay, tick)

    root.bind("<KeyPress>", on_key)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.focus_set()
    redraw()
    root.after(40, tick)
    try:
        root.mainloop()
    finally:
        player.release()


if __name__ == "__main__":
    main()
