from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def resolve_project_id(hint: str) -> str:
    """Resolve a project identity from a path hint.

    Resolution order:
      1. Git repository root (auto-detected from hint path)
      2. Named workspace if hint is not a valid path
      3. Fallback: SHA-256 of the raw hint string (stable, collision-resistant)
    """
    path = Path(hint).expanduser().resolve()

    # Only attempt git/path resolution if the path actually exists on disk.
    # Non-existent paths are treated as explicit named workspaces.
    if path.exists():
        git_root = _find_git_root(path)
        if git_root:
            return _hash_path(git_root)
        return _hash_path(path)

    return _hash_string(hint)


def _find_git_root(start: Path) -> Path | None:
    """Walk up the directory tree to find a .git folder."""
    # Fast path: ask git directly (handles worktrees, submodules, etc.)
    try:
        result = subprocess.run(
            ["git", "-C", str(start if start.is_dir() else start.parent),
             "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback: manual walk (git not available)
    current = start if start.is_dir() else start.parent
    for parent in [current, *current.parents]:
        if (parent / ".git").exists():
            return parent

    return None


def _hash_path(path: Path) -> str:
    return _hash_string(str(path))


def _hash_string(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]
