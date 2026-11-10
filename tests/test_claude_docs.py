"""
File: test_claude_docs.py
Author: Zachary Hoffman
Last edited by: Claude

Purpose:
    Keeps the root CLAUDE.md short and its index of docs/claude/ complete.
    Claude Code sends CLAUDE.md with every request, and when the file
    changes on disk during a session it sends the whole new file again
    beside the old copy (anthropics/claude-code#98553). At 1.28 MB that
    overflowed long sessions, so its detail moved to docs/claude/, which
    Claude reads only when a task needs it. These tests fail when the root
    grows past its cap, when a docs/claude/ file is missing from the root's
    index (or the index names one that does not exist), or when a notes
    file would be loaded with every request after all. One more test keeps
    the notes and README.md from naming the same-title cap's command-line
    flag, which main.py refuses (the bot trades time-series pairs only), so
    no reader copies a command that exits 2; a dated decision record keeps
    the wording its decision was taken with, and is skipped.

Dependencies:
    tests/claude_docs.py, for where the notes live. Reads files only.

Notes:
    The checks read claude_docs.ROOT_NOTES, claude_docs.NOTES_DIR and
    claude_docs.REPO_ROOT when called, so a test can point them at a copy.
"""

import re

from . import claude_docs

# The largest the root CLAUDE.md may be. It was 24,718 bytes when this test
# was added; detail belongs in docs/claude/ (see its Documentation Standards).
ROOT_CLAUDE_MD_MAX_BYTES = 40_000

# The root's section that lists every docs/claude/ file
INDEX_HEADING = "## Where the Detail Is"

# The bold lead of a dated decision record: one that names a date and ends
# "decision record.", such as "**The live defaults of 2026-09-27 — decision
# record.**". An addendum's lead ("**Addendum to the 2026-09-27 decision
# record — ...**") ends otherwise, so an addendum is checked like any other
# paragraph
DATED_RECORD_LEAD = re.compile(
    r"\*\*[^*\n]*\b\d{4}-\d{2}-\d{2}\b[^*\n]* decision record\.\*\*")

# Every dated decision record the notes hold, by its bold lead. Adding a
# record means adding it here, so the records a test skips are always the
# ones meant, never more or fewer
DATED_RECORDS = ("**The live defaults of 2026-09-27 — decision record.**",)

# The same-title cap's command-line flag. main.py refuses it (exit 2) since
# the bot trades time-series pairs only, so no note may name it outside a
# dated decision record
RETIRED_SAME_TITLE_CAP_FLAG = "--same-title-size-cap"


def _index_paths(root_text: str) -> list[str]:
    """
    The docs/claude/ files the root's index table names, in its order.

    Args:
        root_text (str): the root CLAUDE.md's text.

    Returns:
        list[str]: each row's first cell, a path relative to docs/claude/.
    """
    assert INDEX_HEADING in root_text, f"CLAUDE.md has no {INDEX_HEADING!r} section"
    section = root_text.split(INDEX_HEADING, 1)[1].split("\n## ", 1)[0]
    return re.findall(r"^\| `([^`]+\.md)` \|", section, re.MULTILINE)


def _imports(root_text: str) -> list[str]:
    """
    The `@path` imports in a CLAUDE.md, outside code blocks and code spans.

    Claude Code loads an imported file with CLAUDE.md, at the same cost, and
    reads no import inside code (so "`@dataclass`" is not one).

    Args:
        root_text (str): the text to search.

    Returns:
        list[str]: each import as written, such as "@docs/claude/x.md".
    """
    text = re.sub(r"```.*?```", "", root_text, flags=re.DOTALL)
    text = re.sub(r"`[^`\n]*`", "", text)
    return re.findall(r"(?<![\w@])@[\w~.-]*[/.][\w~./-]*", text)


def _dated_record_lines(lines: list[str]) -> dict[int, str]:
    """
    The lines of one notes file that belong to a dated decision record.

    A record keeps the wording its decision was taken with, even where the
    code has changed since, so a test of what the notes name skips it. A
    record is the line that starts with its bold lead (DATED_RECORD_LEAD)
    and every line after it — its numbered items and its closing sentence —
    up to the next line that starts with a bold lead or a heading, or the
    end of the file. Nothing else is a record: not the paragraph before it,
    and not an addendum after it.

    Args:
        lines (list[str]): one file's lines, in order.

    Returns:
        dict[int, str]: each record line's 0-based index, mapped to the bold
            lead of the record it belongs to.
    """
    records, lead = {}, None
    for number, line in enumerate(lines):
        found = DATED_RECORD_LEAD.match(line)
        if found:
            lead = found.group(0)
        elif line.startswith(("**", "#")):
            lead = None
        if lead is not None:
            records[number] = lead
    return records


class TestRootClaudeMd:
    """The root CLAUDE.md is short, indexes every notes file, and is the only
    notes file sent with every request."""

    def test_the_root_claude_md_stays_small(self):
        size = len(claude_docs.ROOT_NOTES.read_bytes())
        assert size <= ROOT_CLAUDE_MD_MAX_BYTES, (
            f"CLAUDE.md is {size:,} bytes, over its {ROOT_CLAUDE_MD_MAX_BYTES:,}-byte cap. "
            "Claude Code sends it with every request and again whole each time it changes, "
            "so move detail into the docs/claude/ file for its area (its Documentation "
            "Standards say how) rather than raising the cap")

    def test_the_index_names_every_notes_file(self):
        named = _index_paths(claude_docs.ROOT_NOTES.read_text(encoding="utf-8"))
        assert len(named) == len(set(named)), "the index names a file twice"
        present = sorted(path.relative_to(claude_docs.NOTES_DIR).as_posix()
                         for path in claude_docs.NOTES_DIR.rglob("*.md"))
        assert present, "docs/claude/ holds no notes"
        missing = sorted(set(present) - set(named))
        assert not missing, f"docs/claude/ files the root's index does not name: {missing}"
        gone = sorted(set(named) - set(present))
        assert not gone, f"the root's index names files docs/claude/ does not have: {gone}"

    def test_no_notes_file_is_loaded_with_every_request(self):
        # A file named CLAUDE.md below the root loads whenever Claude reads a file
        # beside it, and an @-import loads with the root: either would put the
        # detail back into every request
        repo = claude_docs.REPO_ROOT
        nested = sorted(str(path.relative_to(repo))
                        for folder in (claude_docs.NOTES_DIR.parent, repo / "kalshi_betting",
                                       repo / "tests")
                        for path in folder.rglob("*") if path.name.lower() == "claude.md")
        assert not nested, f"notes files named CLAUDE.md: {nested}"
        imports = _imports(claude_docs.ROOT_NOTES.read_text(encoding="utf-8"))
        assert not imports, f"CLAUDE.md imports {imports}; name the file in the index instead"

    def test_the_import_search_reads_imports_and_not_code(self):
        # @dataclass in a code span, a commit trailer's address and a fenced
        # example are not imports; a bare @path is
        text = ("Use `@dataclass`. noreply@anthropic.com\n```\n@docs/x.md\n```\n"
                "See @docs/claude/overview.md and @README.md.")
        assert _imports(text) == ["@docs/claude/overview.md", "@README.md."]


class TestRetiredNames:
    """Neither the notes nor README.md name the same-title cap's old
    command-line flag (main.py refuses it), outside a dated decision record,
    which keeps the wording its decision was taken with."""

    def test_no_note_names_the_same_title_cap_flag_outside_a_dated_record(self):
        # The record rule on a sample: a record runs from its bold lead to the
        # next bold lead or heading, so its numbered items and closing sentence
        # are skipped, and neither the paragraph before it nor an addendum after
        # it is
        record = "**The live defaults of 2026-09-27 — decision record.**"
        sample = ["**A rule.** a", "", f"{record} b", "", "1. c", "   - d", "",
                  "Closing sentence.", "",
                  "**Addendum to the 2026-09-27 decision record — e.** f", "",
                  f"{record} g", "# A heading", "h"]
        assert _dated_record_lines(sample) == dict.fromkeys([*range(2, 9), 11], record)
        # Every notes file and README.md, each read on its own, so a record never
        # runs on into the next file
        paths = [*claude_docs.notes_paths(), claude_docs.REPO_ROOT / "README.md"]
        leads, named_in_a_record = set(), []
        for path in paths:
            lines = path.read_text(encoding="utf-8").splitlines()
            records = _dated_record_lines(lines)
            leads.update(records.values())
            for number, line in enumerate(lines):
                if RETIRED_SAME_TITLE_CAP_FLAG not in line:
                    continue
                assert number in records, (
                    f"{path.relative_to(claude_docs.REPO_ROOT)} line {number + 1} names "
                    f"{RETIRED_SAME_TITLE_CAP_FLAG}, which main.py refuses (exit 2): drop it, "
                    "or describe the old flag without its name")
                named_in_a_record.append(f"{path.name}:{number + 1}")
        # The skipped lines are exactly the dated records', never more or fewer ...
        assert sorted(leads) == sorted(DATED_RECORDS), (
            f"the notes' dated decision records are {sorted(leads)}, but DATED_RECORDS "
            f"names {sorted(DATED_RECORDS)}")
        # ... and the skip is used: the 2026-09-27 record keeps its one-run replay
        # command, which named the flag when the record was taken
        assert named_in_a_record, (
            f"no dated decision record names {RETIRED_SAME_TITLE_CAP_FLAG} any more, but a "
            "record keeps the wording its decision was taken with")
