"""
File: claude_docs.py
Author: Zachary Hoffman
Last edited by: Claude

Purpose:
    The project's notes for Claude, as the tests that check their text read
    them: the root CLAUDE.md and every Markdown file under docs/claude/, in
    one fixed order. The root file is kept short and indexes the rest, so a
    test that looks for a sentence reads all of them as one text, and moving
    a paragraph from one notes file to another never breaks it.

Dependencies:
    The standard library only. Imported by the tests that read the notes
    (tests/test_backtester.py, tests/test_config.py, tests/test_main.py and
    tests/test_claude_docs.py); it is not a test module itself, like
    tests/dashboard_golden.py.

Notes:
    Each notes paragraph is one line, so a test can find a paragraph by how
    its line starts. notes_lines() keeps that: it joins the files with a
    blank line, never inside a line.
"""

import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
# The root file Claude Code sends with every request
ROOT_NOTES = REPO_ROOT / "CLAUDE.md"
# The detailed notes, read only when a task needs them
NOTES_DIR = REPO_ROOT / "docs" / "claude"


def notes_paths() -> list[pathlib.Path]:
    """
    The root CLAUDE.md, then every notes file under docs/claude/.

    Returns:
        list[pathlib.Path]: CLAUDE.md first, then each docs/claude/ Markdown
        file sorted by path, so every test reads them in the same order.
    """
    return [ROOT_NOTES, *sorted(NOTES_DIR.rglob("*.md"))]


def notes_text() -> str:
    """
    Every notes file's text, in notes_paths() order, as one string.

    Returns:
        str: the files' texts joined by a blank line.
    """
    return "\n\n".join(path.read_text(encoding="utf-8") for path in notes_paths())


def notes_lines() -> list[str]:
    """
    Every notes file's lines, in notes_paths() order.

    Returns:
        list[str]: the lines of notes_text().
    """
    return notes_text().splitlines()
