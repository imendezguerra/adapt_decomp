"""Provenance of a produced result: when and where it ran, from which code, and how to redo it."""

import hashlib
import os
import re
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Union
from urllib.parse import urlsplit, urlunsplit

import yaml
from loguru import logger

from adapt_decomp.utils.system import describe_system
from adapt_decomp.utils.utils import to_yaml_safe

PATCH_HASH_CHARS = 16  # hex characters of a patch's SHA-256 used as its file name

# Commits already warned about for untracked files (once per process, not once per result)
_WARNED_UNTRACKED: Set[str] = set()


def _git(repo_dir: Path, *args: str) -> subprocess.CompletedProcess:
    """Run one git command in repo_dir, capturing its output as bytes.

    Args:
        repo_dir (Path): Directory inside the repository.
        *args (str): git arguments.

    Returns:
        subprocess.CompletedProcess: The finished process (never raises on a
        non-zero exit).
    """
    return subprocess.run(["git", "-C", str(repo_dir), *args], capture_output=True, check=False)


def sanitise_remote_url(url: str) -> str:
    """Strip credentials (user, password, token) from a remote URL.

    Args:
        url (str): Remote URL, e.g. "https://user:token@github.com/org/repo.git"
            or "git@github.com:org/repo.git".

    Returns:
        str: The URL without its userinfo for http(s)/ssh:// URLs; scp-like
        "git@host:path" URLs are returned unchanged (they hold no secret).
    """
    parts = urlsplit(url)
    if parts.scheme in ("http", "https", "ssh") and parts.hostname:
        netloc = parts.hostname + (f":{parts.port}" if parts.port else "")
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    return url


def git_state(repo_dir: Union[str, Path] = ".") -> Optional[Dict[str, Any]]:
    """The repository's remote, commit, branch and uncommitted changes.

    Args:
        repo_dir (Union[str, Path], optional): Directory inside the
            repository. Defaults to the current directory.

    Returns:
        Optional[Dict[str, Any]]: remote (credentials stripped, None without
        an "origin"), commit, branch (None when detached), dirty (tracked
        files differ from the commit), diff (bytes of "git diff HEAD
        --binary", empty when clean) and untracked (paths not ignored and not
        in the diff). None, with a warning, when git is missing or repo_dir is
        not in a repository.
    """
    repo_dir = Path(repo_dir)
    if shutil.which("git") is None:
        logger.warning("git is not installed, so no git state is recorded.")
        return None
    head = _git(repo_dir, "rev-parse", "HEAD")
    if head.returncode != 0:
        logger.warning(f"{repo_dir} is not in a git repository, so no git state is recorded.")
        return None

    remote = _git(repo_dir, "config", "--get", "remote.origin.url").stdout.decode().strip()
    branch = _git(repo_dir, "rev-parse", "--abbrev-ref", "HEAD").stdout.decode().strip()
    diff = _git(repo_dir, "diff", "HEAD", "--binary").stdout
    untracked = _git(repo_dir, "ls-files", "--others", "--exclude-standard").stdout.decode()
    return {
        "remote": sanitise_remote_url(remote) if remote else None,
        "commit": head.stdout.decode().strip(),
        "branch": None if branch == "HEAD" else branch,
        "dirty": bool(diff.strip()),
        "diff": diff,
        "untracked": [line for line in untracked.splitlines() if line],
    }


def save_patch(diff: bytes, patch_dir: Union[str, Path]) -> Path:
    """Write a diff once, named by its content hash, so identical diffs share one file.

    Args:
        diff (bytes): Output of "git diff HEAD --binary".
        patch_dir (Union[str, Path]): Directory for patch files.

    Returns:
        Path: The patch file, patch_dir / "<hash>.patch".
    """
    patch_dir = Path(patch_dir)
    patch_dir.mkdir(parents=True, exist_ok=True)
    path = patch_dir / f"{hashlib.sha256(diff).hexdigest()[:PATCH_HASH_CHARS]}.patch"
    if not path.exists():
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_bytes(diff)
        os.replace(tmp, path)
    return path


def git_branch_name(run_name: str) -> str:
    """Turn a run name into a valid git branch name.

    Args:
        run_name (str): Any run label.

    Returns:
        str: run_name with every character outside [A-Za-z0-9._-] replaced
        by "-".
    """
    return re.sub(r"[^A-Za-z0-9._-]", "-", run_name).strip("-.") or "run"


def build_metadata(
    *,
    command: Sequence[str],
    started: datetime,
    finished: datetime,
    run_name: str,
    reproduce: Sequence[str],
    repo_dir: Union[str, Path] = ".",
    patch_dir: Optional[Union[str, Path]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the metadata of one produced result.

    Args:
        command (Sequence[str]): The command as invoked, one argument per item.
        started (datetime): When the run started (timezone-aware).
        finished (datetime): When the run finished (timezone-aware).
        run_name (str): Label of this run, used as the reproduce branch name.
        reproduce (Sequence[str]): Shell lines that redo this result from a
            fresh checkout (environment, data, the command itself); the git
            clone/checkout lines are prepended here.
        repo_dir (Union[str, Path], optional): Directory inside the
            repository. Defaults to the current directory.
        patch_dir (Optional[Union[str, Path]], optional): Where to save the
            uncommitted diff when the tree is dirty. Defaults to None (the diff
            is not saved, and the reproduce lines say so).
        extra (Optional[Dict[str, Any]], optional): Caller sections placed
            first (e.g. the task, its inputs and an output digest). Defaults to
            None.

    Returns:
        Dict[str, Any]: YAML-safe metadata: extra's sections, started_at,
        finished_at, run_time_s, the describe_system() sections, git (None
        outside a repository), command and reproduce (one multi-line string).
    """
    repo_dir = Path(repo_dir)
    state = git_state(repo_dir)

    # Git section and the lines that rebuild this exact source tree
    git_section = None
    checkout: List[str] = []
    if state is not None:
        patch = None
        if state["dirty"] and patch_dir is not None:
            patch = Path(os.path.relpath(save_patch(state["diff"], patch_dir), repo_dir))
            patch = patch.as_posix()
        if state["untracked"] and state["commit"] not in _WARNED_UNTRACKED:
            _WARNED_UNTRACKED.add(state["commit"])
            logger.warning(
                f"{len(state['untracked'])} untracked file(s) are not captured by the git "
                "state; commit them for the result to be reproducible."
            )
        git_section = {
            "remote": state["remote"],
            "commit": state["commit"],
            "branch": state["branch"],
            "dirty": state["dirty"],
            "patch": patch,
            "untracked": state["untracked"],
        }
        repo_name = Path(urlsplit(state["remote"] or "").path).stem or repo_dir.resolve().name
        checkout = [
            f"git clone {state['remote'] or '<repository URL>'}",
            f"cd {repo_name}",
            f'git checkout -b "{git_branch_name(run_name)}" {state["commit"]}',
        ]
        if state["dirty"]:
            checkout.append(
                f"git apply {patch}" if patch else "# uncommitted changes were not saved"
            )

    return _yaml_safe_tree(
        {
            **(extra or {}),
            "started_at": started.isoformat(timespec="seconds"),
            "finished_at": finished.isoformat(timespec="seconds"),
            "run_time_s": round((finished - started).total_seconds(), 3),
            **describe_system(),
            "git": git_section,
            "command": shlex.join(command),
            "reproduce": "\n".join([*checkout, *reproduce]) + "\n",
        }
    )


def _yaml_safe_tree(value: Any) -> Any:
    """Recursively coerce a nested dict/list of values to YAML-safe types.

    Args:
        value (Any): A dict, list, tuple or leaf value.

    Returns:
        Any: The same structure with every leaf passed through to_yaml_safe.
    """
    if isinstance(value, dict):
        return {str(k): _yaml_safe_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe_tree(v) for v in value]
    return to_yaml_safe(value)


class _MetadataDumper(yaml.SafeDumper):
    """SafeDumper that writes multi-line strings as literal blocks."""


def _represent_str(dumper: yaml.SafeDumper, data: str) -> yaml.Node:
    """Represent multi-line strings in literal block style, others as usual.

    Args:
        dumper (yaml.SafeDumper): The active dumper.
        data (str): The string.

    Returns:
        yaml.Node: The scalar node.
    """
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_MetadataDumper.add_representer(str, _represent_str)


def write_metadata(path: Union[str, Path], metadata: Dict[str, Any]) -> None:
    """Write metadata as YAML, atomically (a reader never sees a partial file).

    Args:
        path (Union[str, Path]): Destination, e.g. "<output>.meta.yaml".
        metadata (Dict[str, Any]): From build_metadata().

    Returns:
        None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        yaml.dump(
            _yaml_safe_tree(metadata),
            f,
            Dumper=_MetadataDumper,
            sort_keys=False,
            allow_unicode=True,
        )
    os.replace(tmp, path)


def read_metadata(path: Union[str, Path]) -> Dict[str, Any]:
    """Read metadata written by write_metadata().

    Args:
        path (Union[str, Path]): The metadata file.

    Returns:
        Dict[str, Any]: The parsed metadata.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
