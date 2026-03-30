from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from cce.cli import build_compress_payload, build_recent_context, main


def test_build_recent_context_includes_files_notes_and_diff(tmp_path, monkeypatch):
    project = tmp_path / "repo"
    project.mkdir()
    file_path = project / "app.py"
    file_path.write_text("print('hello')", encoding="utf-8")
    notes_path = tmp_path / "notes.txt"
    notes_path.write_text("Remember the auth refactor", encoding="utf-8")

    monkeypatch.setattr("cce.cli._git_diff", lambda project_root: "diff --git a/app.py b/app.py")

    context = build_recent_context(
        project_root=project,
        files=[str(file_path)],
        include_git_diff=True,
        notes_file=str(notes_path),
    )

    assert len(context) == 3
    assert context[0]["content"].startswith("[NOTES]")
    assert "[FILE] app.py" in context[1]["content"]
    assert "[GIT DIFF]" in context[2]["content"]


def test_build_compress_payload_uses_prompt_argument(tmp_path):
    project = tmp_path / "repo"
    project.mkdir()
    args = Namespace(
        project=str(project),
        prompt="Fix the auth bug",
        prompt_file=None,
        file=[],
        include_git_diff=False,
        notes_file=None,
        max_context_tokens=4096,
    )

    payload = build_compress_payload(args)
    assert payload["project_hint"] == str(project.resolve())
    assert payload["current_message"] == "Fix the auth bug"
    assert payload["recent_context"] == []


def test_cli_recall_prints_briefing(monkeypatch, capsys, tmp_path):
    project = tmp_path / "repo"
    project.mkdir()

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"briefing": "briefing text"}

    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: Response())

    exit_code = main(["recall", "--project", str(project)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.strip() == "briefing text"
