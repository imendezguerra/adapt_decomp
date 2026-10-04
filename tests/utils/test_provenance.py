"""Tests for adapt_decomp.utils.provenance: git state, patches, metadata and its YAML file."""

import shutil
import subprocess
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from adapt_decomp.utils import provenance

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")

STARTED = datetime(2026, 10, 4, 12, 0, 0, tzinfo=timezone.utc)


def _git(root, *args):
    """Run git in root, failing the test on error."""
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def repo(tmp_path):
    """A one-commit repository with a credentialed remote URL."""
    root = tmp_path / "proj"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "test")
    _git(root, "config", "commit.gpgsign", "false")
    _git(root, "config", "core.autocrlf", "false")
    (root / "a.txt").write_text("one\n")
    _git(root, "add", "a.txt")
    _git(root, "commit", "-q", "-m", "init")
    _git(root, "remote", "add", "origin", "https://user:secret@github.com/org/proj.git")
    return root


def _metadata(repo, **kwargs):
    """build_metadata with fixed times and a fixed command."""
    return provenance.build_metadata(
        command=["python", "-m", "benchmarks.fdsi", "apply", "--array-index", "2"],
        started=STARTED,
        finished=STARTED + timedelta(seconds=90),
        run_name="my run/3",
        reproduce=["python -m benchmarks.fdsi apply --task-index 3"],
        repo_dir=repo,
        **kwargs,
    )


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://user:token@github.com/org/repo.git", "https://github.com/org/repo.git"),
        ("http://token@host:8080/repo.git", "http://host:8080/repo.git"),
        ("ssh://git@github.com/org/repo.git", "ssh://github.com/org/repo.git"),
        ("git@github.com:org/repo.git", "git@github.com:org/repo.git"),
    ],
)
def test_sanitise_remote_url_strips_credentials(url, expected):
    assert provenance.sanitise_remote_url(url) == expected


def test_git_branch_name_keeps_only_valid_characters():
    assert provenance.git_branch_name("fdsi v1.1/apply:17") == "fdsi-v1.1-apply-17"


def test_git_state_of_a_clean_repository(repo):
    state = provenance.git_state(repo)

    assert state["commit"] == _git(repo, "rev-parse", "HEAD").strip()
    assert state["remote"] == "https://github.com/org/proj.git"
    assert state["branch"] is not None
    assert state["dirty"] is False
    assert state["diff"] == b""
    assert state["untracked"] == []


def test_git_state_of_a_dirty_repository(repo):
    (repo / "a.txt").write_text("two\n")
    (repo / "b.txt").write_text("new\n")

    state = provenance.git_state(repo)

    assert state["dirty"] is True
    assert b"+two" in state["diff"]
    assert state["untracked"] == ["b.txt"]


def test_git_state_outside_a_repository_is_none(tmp_path):
    outside = tmp_path / "not_a_repo"
    outside.mkdir()
    assert provenance.git_state(outside) is None


def test_save_patch_writes_each_distinct_diff_once(tmp_path):
    first = provenance.save_patch(b"diff one\n", tmp_path)
    again = provenance.save_patch(b"diff one\n", tmp_path)
    other = provenance.save_patch(b"diff two\n", tmp_path)

    assert first == again
    assert first != other
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([first.name, other.name])
    assert first.read_bytes() == b"diff one\n"


def test_build_metadata_of_a_clean_run(repo):
    metadata = _metadata(repo, extra={"task": {"stage": "apply"}})
    commit = _git(repo, "rev-parse", "HEAD").strip()

    assert next(iter(metadata)) == "task"  # caller sections come first
    for key in ("started_at", "finished_at", "run_time_s", "host", "os", "hardware", "python"):
        assert key in metadata
    assert metadata["started_at"] == "2026-10-04T12:00:00+00:00"
    assert metadata["run_time_s"] == 90.0
    assert metadata["git"]["dirty"] is False
    assert metadata["git"]["patch"] is None
    assert metadata["command"] == "python -m benchmarks.fdsi apply --array-index 2"
    assert metadata["reproduce"].splitlines() == [
        "git clone https://github.com/org/proj.git",
        "cd proj",
        f'git checkout -b "my-run-3" {commit}',
        "python -m benchmarks.fdsi apply --task-index 3",
    ]


def test_build_metadata_of_a_dirty_run_saves_an_applicable_patch(repo, tmp_path):
    (repo / "a.txt").write_text("two\n")
    metadata = _metadata(repo, patch_dir=repo / "outputs" / "patches")
    patch = metadata["git"]["patch"]

    assert metadata["git"]["dirty"] is True
    assert patch.startswith("outputs/patches/") and patch.endswith(".patch")
    assert f"git apply {patch}" in metadata["reproduce"].splitlines()

    # The reproduce lines rebuild the dirty tree: a clone at the commit plus the patch
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(repo), str(clone)], check=True)
    _git(clone, "config", "core.autocrlf", "false")
    _git(clone, "checkout", "-q", metadata["git"]["commit"])
    _git(clone, "apply", str(repo / patch))
    assert (clone / "a.txt").read_text() == "two\n"


def test_build_metadata_without_a_patch_dir_says_the_diff_was_not_saved(repo):
    (repo / "a.txt").write_text("two\n")
    metadata = _metadata(repo)

    assert metadata["git"]["patch"] is None
    assert "# uncommitted changes were not saved" in metadata["reproduce"]


def test_write_metadata_round_trips_numpy_values_and_writes_reproduce_as_a_block(tmp_path):
    path = tmp_path / "out.meta.yaml"
    provenance.write_metadata(
        path,
        {
            "digest": {
                "n_units": np.int64(9),
                "mean": np.float32(0.5),
                "roa": np.array([1.0, 0.5]),
            },
            "reproduce": "line one\nline two\n",
        },
    )

    assert provenance.read_metadata(path) == {
        "digest": {"n_units": 9, "mean": 0.5, "roa": [1.0, 0.5]},
        "reproduce": "line one\nline two\n",
    }
    assert "reproduce: |" in path.read_text()
    assert not list(tmp_path.glob("*.tmp"))
