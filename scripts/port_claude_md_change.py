"""
File: port_claude_md_change.py
Author: Zachary Hoffman
Last edited by: Claude

Purpose:
    Moves a branch's edits of the old, single-file CLAUDE.md into the notes
    that replaced it: the short root CLAUDE.md and the files under
    docs/claude/. On 2026-10-26 CLAUDE.md's detail was moved there word for
    word, one paragraph per line, so a branch that edited the old file
    changed whole lines that now sit in one of the notes files. This script
    compares the branch's CLAUDE.md before and after its edits, finds each
    changed line where it now lives, and makes the same change there: a
    changed line is replaced, a removed line is removed, and an added line
    goes after the line it followed in the old file.

    Typical use, on a branch that edited CLAUDE.md, when merging main (run
    the merge, the checkout and this script in ONE shell command, so a
    conflicted 1.28 MB CLAUDE.md is never left on disk between two requests
    of a running Claude Code session):

        base=$(git merge-base HEAD origin/main)
        git merge --no-commit origin/main; \
        git checkout origin/main -- CLAUDE.md; \
        python3 scripts/port_claude_md_change.py --base "$base" --head HEAD

Dependencies:
    The standard library and git. Imported by nothing; tests load it from
    its path (tests/test_port_claude_md_change.py).

Notes:
    A line it cannot place (not found in the notes, or found more than once)
    is listed with its old and new text, and the exit code is 1; every line
    it could place is still written. --dry-run writes nothing. A changed
    line that sits in docs/claude/what-not-to-do.md, key-patterns/ or
    known-gotchas/ may also have a short form in the root CLAUDE.md, which
    a person (or session) must then update by hand: the summary names those
    files.
"""

import argparse
import difflib
import pathlib
import subprocess
import sys
from dataclasses import dataclass, field

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


@dataclass
class Notes:
    """The notes files as lists of lines, keyed by path, and what changed."""

    files: dict[pathlib.Path, list[str]]
    changed: set[pathlib.Path] = field(default_factory=set)
    unplaced: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=lambda: {
        "replaced": 0, "removed": 0, "added": 0})

    def find(self, line: str) -> tuple[pathlib.Path, int] | None:
        """
        Where a line sits in the notes, if it sits in exactly one place.

        Args:
            line (str): the whole line to find.

        Returns:
            tuple[pathlib.Path, int] | None: the file and the line's index,
            or None when it is in no file or in more than one place.
        """
        hits = [(path, i) for path, lines in self.files.items()
                for i, text in enumerate(lines) if text == line]
        return hits[0] if len(hits) == 1 else None

    def near(self, line: str, repo: pathlib.Path) -> str:
        """
        Where a line that is not found as it stands probably sits now.

        A line both the branch and main edited is not found word for word;
        the notes line that starts the same way is most likely it.

        Args:
            line (str): the old line.
            repo (pathlib.Path): the repository root, to print paths from.

        Returns:
            str: " (edited on main too; see <file>:<line number>)", or "".
        """
        start = line[:60]
        hits = [f"{path.relative_to(repo)}:{i + 1}" for path, lines in self.files.items()
                for i, text in enumerate(lines) if len(start) >= 20 and text.startswith(start)]
        return f" (edited on main too; see {', '.join(hits)})" if hits else ""


def load_notes(repo: pathlib.Path) -> Notes:
    """
    Reads the root CLAUDE.md and every Markdown file under docs/claude/.

    Args:
        repo (pathlib.Path): the repository root.

    Returns:
        Notes: each file's lines, nothing changed yet.
    """
    paths = [repo / "CLAUDE.md", *sorted((repo / "docs" / "claude").rglob("*.md"))]
    return Notes({path: path.read_text(encoding="utf-8").split("\n") for path in paths})


def _show(rev: str, repo: pathlib.Path) -> list[str]:
    """CLAUDE.md as it stands at a git revision, as lines."""
    text = subprocess.run(["git", "show", f"{rev}:CLAUDE.md"], cwd=repo, check=True,
                          capture_output=True, text=True, encoding="utf-8").stdout
    return text.split("\n")


def _insert_after(notes: Notes, anchor: tuple[pathlib.Path, int] | None,
                  new_lines: list[str], context: str) -> tuple[pathlib.Path, int] | None:
    """
    Puts new lines after an anchor line, as paragraphs or as rows.

    When the anchor is followed by a blank line (a paragraph), each new line
    becomes a paragraph of its own, with a blank line before it; otherwise
    (a table row, a line of a code block) the new lines go straight after it.

    Args:
        notes (Notes): the notes, changed in place.
        anchor (tuple | None): where the line before them now sits.
        new_lines (list[str]): the lines to add, blank ones left out.
        context (str): the old line the new ones followed, for the report.

    Returns:
        tuple | None: where the last added line now sits, or None when the
        anchor could not be found (the lines are then reported unplaced).
    """
    if anchor is None:
        for line in new_lines:
            notes.unplaced.append(f"added after {context[:80]!r}: {line[:200]!r}")
        return None
    path, i = anchor
    lines = notes.files[path]
    paragraph = i + 1 < len(lines) and lines[i + 1] == ""
    block = [x for line in new_lines for x in (["", line] if paragraph else [line])]
    lines[i + 1:i + 1] = block
    notes.changed.add(path)
    notes.counts["added"] += len(new_lines)
    return path, i + len(block)


def port(old: list[str], new: list[str], notes: Notes,
         repo: pathlib.Path = REPO_ROOT) -> Notes:
    """
    Makes, in the notes, the line changes that turn old into new.

    Args:
        old (list[str]): the branch's CLAUDE.md before its edits.
        new (list[str]): the branch's CLAUDE.md after its edits.
        notes (Notes): the notes, changed in place.
        repo (pathlib.Path): the repository root, for the paths a report names.

    Returns:
        Notes: the same notes, with what changed and what could not be placed.
    """
    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        gone = [line for line in old[i1:i2] if line.strip()]
        added = [line for line in new[j1:j2] if line.strip()]
        last = None
        for k, line in enumerate(gone):
            where = notes.find(line)
            if where is None:
                pair = f" -> {added[k][:200]!r}" if k < len(added) else " (removed)"
                notes.unplaced.append(f"changed {line[:200]!r}{pair}{notes.near(line, repo)}")
                continue
            path, i = where
            if k < len(added):
                notes.files[path][i] = added[k]
                notes.counts["replaced"] += 1
                last = where
            else:
                lines = notes.files[path]
                # A removed paragraph takes its blank separator with it
                end = i + 2 if i + 1 < len(lines) and lines[i + 1] == "" else i + 1
                del lines[i:end]
                notes.counts["removed"] += 1
            notes.changed.add(path)
        extra = added[len(gone):]
        if extra:
            if last is None:
                # The nearest line before this change that the notes still hold
                before = [line for line in old[:i1] if line.strip()]
                anchor_line = before[-1] if before else ""
                last = notes.find(anchor_line) if anchor_line else None
            else:
                anchor_line = notes.files[last[0]][last[1]]
            _insert_after(notes, last, extra, anchor_line)
    return notes


def _shortened_rule_files(notes: Notes, repo: pathlib.Path) -> list[str]:
    """The changed notes files whose rules the root also states in short."""
    docs = repo / "docs" / "claude"
    return sorted(str(path.relative_to(repo)) for path in notes.changed
                  if path == docs / "what-not-to-do.md"
                  or path.parent in (docs / "key-patterns", docs / "known-gotchas"))


def main(argv: list[str] | None = None) -> int:
    """
    The command line: ports one CLAUDE.md change into the notes.

    Args:
        argv (list[str] | None): the arguments, sys.argv[1:] by default.

    Returns:
        int: 0 when every changed line was placed, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=(
        "Move a branch's edits of the old single-file CLAUDE.md into the root "
        "CLAUDE.md and docs/claude/."))
    parser.add_argument("--base", help="git revision of CLAUDE.md before the edits "
                        "(usually the branch's merge base with main)")
    parser.add_argument("--head", default="HEAD",
                        help="git revision of CLAUDE.md after the edits (default HEAD)")
    parser.add_argument("--old-file", type=pathlib.Path,
                        help="the old CLAUDE.md as a file, in place of --base")
    parser.add_argument("--new-file", type=pathlib.Path,
                        help="the edited CLAUDE.md as a file, in place of --head")
    parser.add_argument("--repo", type=pathlib.Path, default=REPO_ROOT,
                        help="the repository whose notes are changed")
    parser.add_argument("--dry-run", action="store_true", help="write nothing")
    args = parser.parse_args(argv)
    if args.old_file is None and args.base is None:
        parser.error("give --base REV or --old-file PATH")
    repo = args.repo.resolve()
    old = (args.old_file.read_text(encoding="utf-8").split("\n") if args.old_file
           else _show(args.base, repo))
    new = (args.new_file.read_text(encoding="utf-8").split("\n") if args.new_file
           else _show(args.head, repo))
    notes = port(old, new, load_notes(repo), repo)
    if not args.dry_run:
        for path in notes.changed:
            path.write_text("\n".join(notes.files[path]), encoding="utf-8")
    c = notes.counts
    print(f"{'Would change' if args.dry_run else 'Changed'}: {c['replaced']} lines replaced, "
          f"{c['added']} added, {c['removed']} removed, in "
          f"{sorted(str(p.relative_to(repo)) for p in notes.changed)}")
    shortened = _shortened_rule_files(notes, repo)
    if shortened:
        print("These files hold rules the root CLAUDE.md also states in short; check "
              f"the short forms there: {shortened}")
    if notes.unplaced:
        print(f"Could not place {len(notes.unplaced)} line(s); place them by hand:")
        for item in notes.unplaced:
            print(f"  {item}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
