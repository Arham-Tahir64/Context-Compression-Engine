from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx


def _base_url(args: argparse.Namespace) -> str:
    return args.base_url or f"http://{args.host}:{args.port}"


def _read_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
        if prompt:
            return prompt
    raise SystemExit("Provide a prompt via --prompt, --prompt-file, or stdin.")


def _read_text_file(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _format_file_block(project_root: Path, file_path: Path) -> str:
    relative = file_path.relative_to(project_root) if file_path.is_relative_to(project_root) else file_path
    content = file_path.read_text(encoding="utf-8")
    return f"[FILE] {relative}\n```\n{content}\n```"


def _format_notes_block(notes_path: Path) -> str:
    content = notes_path.read_text(encoding="utf-8")
    return f"[NOTES] {notes_path.name}\n{content}"


def _git_diff(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(project_root), "diff", "--no-ext-diff", "--relative"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr.strip() or "Failed to collect git diff.")
    return result.stdout.strip()


def build_recent_context(
    *,
    project_root: Path,
    files: list[str],
    include_git_diff: bool,
    notes_file: str | None,
) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    turn_index = 0

    if notes_file:
        turns.append(
            {
                "role": "user",
                "content": _format_notes_block(Path(notes_file)),
                "turn_index": turn_index,
            }
        )
        turn_index += 1

    for file_name in files:
        file_path = Path(file_name).expanduser().resolve()
        turns.append(
            {
                "role": "user",
                "content": _format_file_block(project_root, file_path),
                "turn_index": turn_index,
            }
        )
        turn_index += 1

    if include_git_diff:
        diff = _git_diff(project_root)
        if diff:
            turns.append(
                {
                    "role": "user",
                    "content": f"[GIT DIFF]\n```diff\n{diff}\n```",
                    "turn_index": turn_index,
                }
            )

    return turns


def build_compress_payload(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project).expanduser().resolve()
    return {
        "project_hint": str(project_root),
        "current_message": _read_prompt(args),
        "recent_context": build_recent_context(
            project_root=project_root,
            files=args.file or [],
            include_git_diff=args.include_git_diff,
            notes_file=args.notes_file,
        ),
        "metadata": {
            "tool": "cce-cli",
            "max_context_tokens": args.max_context_tokens,
        },
    }


def _print_response(payload: dict[str, Any], *, as_json: bool, field: str) -> None:
    if as_json:
        print(json.dumps(payload, indent=2))
        return
    print(payload[field])


def run_compress(args: argparse.Namespace) -> int:
    response = httpx.post(
        f"{_base_url(args)}/compress",
        json=build_compress_payload(args),
        timeout=args.timeout,
    )
    response.raise_for_status()
    _print_response(response.json(), as_json=args.json, field="optimized_prompt")
    return 0


def run_recall(args: argparse.Namespace) -> int:
    project_root = Path(args.project).expanduser().resolve()
    response = httpx.post(
        f"{_base_url(args)}/recall",
        json={
            "project_hint": str(project_root),
            "query": args.query,
            "max_tokens": args.max_tokens,
        },
        timeout=args.timeout,
    )
    response.raise_for_status()
    _print_response(response.json(), as_json=args.json, field="briefing")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generic CLI helper for the Context Compression Engine")
    parser.add_argument("--base-url", default=None, help="Override the server base URL")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=float, default=60.0)

    subparsers = parser.add_subparsers(dest="command", required=True)

    compress = subparsers.add_parser("compress", help="Send prompt + project context to /compress")
    compress.add_argument("--project", required=True, help="Absolute or relative path to the project root")
    compress.add_argument("--prompt", help="Prompt text to compress")
    compress.add_argument("--prompt-file", help="Read prompt text from a file")
    compress.add_argument("--file", action="append", help="Include a project file; may be repeated")
    compress.add_argument("--include-git-diff", action="store_true", help="Include git diff output")
    compress.add_argument("--notes-file", help="Include a notes/log text file")
    compress.add_argument("--max-context-tokens", type=int, default=8192)
    compress.add_argument("--json", action="store_true", help="Print the full JSON response")
    compress.set_defaults(func=run_compress)

    recall = subparsers.add_parser("recall", help="Fetch a project briefing from /recall")
    recall.add_argument("--project", required=True, help="Absolute or relative path to the project root")
    recall.add_argument("--query", default=None, help="Optional focus query")
    recall.add_argument("--max-tokens", type=int, default=2048)
    recall.add_argument("--json", action="store_true", help="Print the full JSON response")
    recall.set_defaults(func=run_recall)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
