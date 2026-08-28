"""Verify or restore the pinned blue-screw MuJoCo environment.

Tracked inputs are restored from the Git commit recorded in the snapshot
manifest. Task meshes live under the ignored ``data/`` tree, so the snapshot
also carries byte-for-byte archived copies of those inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = REPO_ROOT / "snapshots" / "sim" / "blue_screw_v1" / "manifest.json"


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def load_manifest(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported snapshot schema in {path}")
    commit = manifest.get("source_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError(f"Invalid source_commit in {path}")
    _git("cat-file", "-e", f"{commit}^{{commit}}")
    return manifest


def tree_digest(path: Path) -> str:
    """Hash one file or a tree, including relative names and file contents."""
    if not path.exists():
        return "MISSING"
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    digest = hashlib.sha256()
    for file_path in files:
        name = file_path.name if path.is_file() else file_path.relative_to(path).as_posix()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _tracked_scope_dirty(commit: str, scope: str) -> bool:
    result = _git("diff", "--quiet", commit, "--", scope, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.decode(errors="replace"))
    if result.returncode == 1:
        return True
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", scope).stdout.strip()
    return bool(untracked)


def verify(manifest: dict[str, Any], *, quiet: bool = False) -> bool:
    commit = manifest["source_commit"]
    clean = True
    for scope, expected_object in manifest["tracked_scopes"].items():
        actual_object = _git("rev-parse", f"{commit}:{scope}").stdout.decode().strip()
        object_ok = actual_object == expected_object
        working_ok = not _tracked_scope_dirty(commit, scope)
        ok = object_ok and working_ok
        clean &= ok
        if not quiet:
            print(f"[{'OK' if ok else 'CHANGED'}] tracked {scope}")

    for entry in manifest["archived_scopes"]:
        source = REPO_ROOT / entry["source"]
        archive = REPO_ROOT / entry["archive"]
        archive_digest = tree_digest(archive)
        if archive_digest != entry["sha256"]:
            raise RuntimeError(
                f"Snapshot archive is corrupt: {entry['archive']} expected {entry['sha256']}, "
                f"got {archive_digest}"
            )
        ok = tree_digest(source) == entry["sha256"]
        clean &= ok
        if not quiet:
            print(f"[{'OK' if ok else 'CHANGED'}] archived {entry['source']}")

    if not quiet:
        print("Simulation matches blue_screw_v1." if clean else "Simulation differs from blue_screw_v1.")
    return clean


def _backup(path: Path, backup_root: Path) -> None:
    if not path.exists():
        return
    destination = backup_root / _relative(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        shutil.copytree(path, destination)
    else:
        shutil.copy2(path, destination)


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _restore_tracked_scope(commit: str, scope: str) -> None:
    destination = REPO_ROOT / scope
    _remove(destination)
    names = _git("ls-tree", "-r", "--name-only", commit, "--", scope).stdout.decode().splitlines()
    if not names:
        raise RuntimeError(f"Snapshot commit has no files under tracked scope {scope}")
    for name in names:
        target = REPO_ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_git("show", f"{commit}:{name}").stdout)


def restore(manifest: dict[str, Any]) -> Path | None:
    if verify(manifest, quiet=True):
        print("Simulation already matches blue_screw_v1; nothing to restore.")
        return None

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_root = REPO_ROOT / "outputs" / "sim_snapshot_backups" / timestamp
    backup_root.mkdir(parents=True, exist_ok=False)
    commit = manifest["source_commit"]

    changed: list[str] = []
    for scope in manifest["tracked_scopes"]:
        if not _tracked_scope_dirty(commit, scope):
            continue
        source = REPO_ROOT / scope
        _backup(source, backup_root)
        _restore_tracked_scope(commit, scope)
        changed.append(scope)

    for entry in manifest["archived_scopes"]:
        source = REPO_ROOT / entry["source"]
        if tree_digest(source) == entry["sha256"]:
            continue
        archive = REPO_ROOT / entry["archive"]
        _backup(source, backup_root)
        _remove(source)
        source.parent.mkdir(parents=True, exist_ok=True)
        if archive.is_dir():
            shutil.copytree(archive, source)
        else:
            shutil.copy2(archive, source)
        changed.append(entry["source"])

    (backup_root / "restore.json").write_text(
        json.dumps(
            {"snapshot": manifest["name"], "source_commit": commit, "restored": changed},
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    if not verify(manifest, quiet=True):
        raise RuntimeError("Restore completed but verification still fails")
    print(f"Restored blue_screw_v1. Previous files: {backup_root}")
    print("Rebuild scene.xml and restart mj_server before using the simulator.")
    return backup_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "restore"), nargs="?", default="verify")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    if args.command == "verify":
        return 0 if verify(manifest) else 1
    restore(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
