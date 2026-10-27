"""
File: test_port_claude_md_change.py
Author: Zachary Hoffman
Last edited by: Claude

Purpose:
    Tests scripts/port_claude_md_change.py, which moves a branch's edits of
    the old single-file CLAUDE.md into the notes that replaced it (the root
    CLAUDE.md and docs/claude/). Each test builds a small notes tree in its
    own tmp_path and an old and an edited single-file CLAUDE.md, and checks
    where each changed line lands.

Dependencies:
    scripts/port_claude_md_change.py, loaded from its path (scripts/ is not
    a package). No git and no network.

Notes:
    The script's main() is called with --old-file/--new-file, so no git
    revision is needed.
"""

import importlib.util
import pathlib

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "port_claude_md_change.py"
_spec = importlib.util.spec_from_file_location("port_claude_md_change", _SCRIPT)
port = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(port)

# The old single-file CLAUDE.md the tree below was split from
OLD = """# Old

## Notes

P1 first paragraph, which runs on for a while.

P2 second paragraph.

| `a.py` | row a |
| `b.py` | row b |

## What NOT To Do

- **Do not x.** Because.
"""


def _tree(tmp_path: pathlib.Path) -> pathlib.Path:
    """A repo with a short root and three notes files split from OLD."""
    (tmp_path / "CLAUDE.md").write_text("# Root\n\nShort.\n", encoding="utf-8")
    docs = tmp_path / "docs" / "claude"
    docs.mkdir(parents=True)
    (docs / "notes.md").write_text(
        "# Notes\n\nP1 first paragraph, which runs on for a while.\n\nP2 second paragraph.\n", encoding="utf-8")
    (docs / "map.md").write_text("# Map\n\n| `a.py` | row a |\n| `b.py` | row b |\n",
                                 encoding="utf-8")
    (docs / "what-not-to-do.md").write_text("# W\n\n- **Do not x.** Because.\n",
                                           encoding="utf-8")
    return tmp_path


def _run(tmp_path: pathlib.Path, new: str, *extra: str) -> int:
    """Ports OLD -> new into the tree at tmp_path."""
    (tmp_path / "old.md").write_text(OLD, encoding="utf-8")
    (tmp_path / "new.md").write_text(new, encoding="utf-8")
    return port.main(["--old-file", str(tmp_path / "old.md"), "--new-file",
                      str(tmp_path / "new.md"), "--repo", str(tmp_path), *extra])


def _read(tmp_path: pathlib.Path, name: str) -> str:
    return (tmp_path / "docs" / "claude" / name).read_text(encoding="utf-8")


class TestPort:
    def test_a_changed_paragraph_is_changed_where_it_now_lives(self, tmp_path):
        repo = _tree(tmp_path)
        assert _run(repo, OLD.replace("P2 second paragraph.", "P2 second, edited.")) == 0
        assert _read(repo, "notes.md") == "# Notes\n\nP1 first paragraph, which runs on for a while.\n\nP2 second, edited.\n"
        assert _read(repo, "map.md").endswith("| `b.py` | row b |\n")

    def test_an_added_paragraph_goes_after_the_one_it_followed(self, tmp_path):
        repo = _tree(tmp_path)
        new = OLD.replace("P1 first paragraph, which runs on for a while.\n", "P1 first paragraph, which runs on for a while.\n\nP1b new one.\n")
        assert _run(repo, new) == 0
        assert _read(repo, "notes.md") == (
            "# Notes\n\nP1 first paragraph, which runs on for a while.\n\nP1b new one.\n\nP2 second paragraph.\n")

    def test_an_added_row_goes_straight_after_the_row_it_followed(self, tmp_path):
        repo = _tree(tmp_path)
        new = OLD.replace("| `a.py` | row a |\n", "| `a.py` | row a |\n| `a2.py` | row a2 |\n")
        assert _run(repo, new) == 0
        assert _read(repo, "map.md") == (
            "# Map\n\n| `a.py` | row a |\n| `a2.py` | row a2 |\n| `b.py` | row b |\n")

    def test_a_removed_paragraph_goes_with_its_blank_line(self, tmp_path):
        repo = _tree(tmp_path)
        assert _run(repo, OLD.replace("P1 first paragraph, which runs on for a while.\n\n", "")) == 0
        assert _read(repo, "notes.md") == "# Notes\n\nP2 second paragraph.\n"

    def test_a_line_edited_on_main_too_is_reported_and_the_rest_still_written(
            self, tmp_path, capsys):
        repo = _tree(tmp_path)
        # main has since edited P1, so the branch's old P1 is not found as it was
        (repo / "docs" / "claude" / "notes.md").write_text(
            "# Notes\n\nP1 first paragraph, which runs on for a while. Main added this.\n\nP2 second paragraph.\n",
            encoding="utf-8")
        new = (OLD.replace("P1 first paragraph, which runs on for a while.", "P1 first paragraph, the branch's way.")
                  .replace("P2 second paragraph.", "P2 second, edited."))
        assert _run(repo, new) == 1
        out = capsys.readouterr().out
        assert "Could not place 1 line(s)" in out
        assert "edited on main too; see docs/claude/notes.md:3" in out
        assert "P2 second, edited." in _read(repo, "notes.md")

    def test_a_line_held_in_two_places_is_not_guessed(self, tmp_path, capsys):
        repo = _tree(tmp_path)
        (repo / "docs" / "claude" / "copy.md").write_text("# C\n\nP1 first paragraph, which runs on for a while.\n",
                                                         encoding="utf-8")
        assert _run(repo, OLD.replace("P1 first paragraph, which runs on for a while.", "P1 edited.")) == 1
        assert "Could not place 1 line(s)" in capsys.readouterr().out
        assert "P1 first paragraph, which runs on for a while." in _read(repo, "notes.md")
        assert "P1 first paragraph, which runs on for a while." in _read(repo, "copy.md")

    def test_a_dry_run_writes_nothing(self, tmp_path, capsys):
        repo = _tree(tmp_path)
        before = _read(repo, "notes.md")
        assert _run(repo, OLD.replace("P2 second paragraph.", "P2 edited."), "--dry-run") == 0
        assert _read(repo, "notes.md") == before
        assert capsys.readouterr().out.startswith("Would change: 1 lines replaced")

    def test_a_changed_rule_names_the_root_s_short_forms(self, tmp_path, capsys):
        repo = _tree(tmp_path)
        assert _run(repo, OLD.replace("Because.", "Because, and more.")) == 0
        assert _read(repo, "what-not-to-do.md").endswith("- **Do not x.** Because, and more.\n")
        out = capsys.readouterr().out
        assert "check the short forms there: ['docs/claude/what-not-to-do.md']" in out

    def test_it_needs_an_old_version(self, tmp_path):
        with pytest.raises(SystemExit) as exit_info:
            port.main(["--repo", str(_tree(tmp_path))])
        assert exit_info.value.code == 2
