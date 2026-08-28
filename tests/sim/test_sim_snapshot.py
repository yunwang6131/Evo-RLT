from __future__ import annotations

from pathlib import Path

from evo_rlt.cli.sim_snapshot import DEFAULT_MANIFEST, load_manifest, tree_digest, verify


def test_tree_digest_changes_with_content(tmp_path: Path) -> None:
    (tmp_path / "a.bin").write_bytes(b"a")
    before = tree_digest(tmp_path)
    (tmp_path / "a.bin").write_bytes(b"b")
    assert tree_digest(tmp_path) != before


def test_blue_screw_snapshot_is_sealed_and_matches_worktree() -> None:
    manifest = load_manifest(DEFAULT_MANIFEST)
    assert all(entry["sha256"] != "TO_BE_FILLED" for entry in manifest["archived_scopes"])
    assert verify(manifest, quiet=True)
