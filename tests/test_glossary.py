"""
File: test_glossary.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Tests for glossary.py, the dictionary of the dashboards' scores: every
    term is complete, listed once and tagged with a known page, a page lists
    only its own terms, every cell is escaped, the Terms link points at the
    section's id, the heading clears the sticky filter bar, the tier-floor
    figures match config's, and the module imports only the standard library.
"""

import ast
import sys
from pathlib import Path

import pytest

from kalshi_betting import config, glossary


class TestTerms:
    def test_every_term_has_a_meaning_a_reason_and_known_pages(self):
        assert glossary.TERMS
        for term, meaning, reason, pages in glossary.TERMS:
            assert all(isinstance(cell, str) and cell.strip()
                       for cell in (term, meaning, reason)), term
            assert pages and set(pages) <= set(glossary.PAGES), term

    def test_terms_are_unique(self):
        names = [entry[0] for entry in glossary.TERMS]
        assert len(names) == len(set(names))

    def test_the_dictionary_stays_brief(self):
        assert 15 <= len(glossary.TERMS) <= 35

    def test_the_renamed_terms_are_named_as_asked(self):
        names = {entry[0] for entry in glossary.TERMS}
        for name in ("95% CI (under k̂)", "Calibration curve (reliability chart)",
                     "Split-half check (H1 / H2)", "Top event (return without it)",
                     "Ladder / cross-event pair", "Spread band", "Tier floors", "Fills",
                     "Portfolio value"):
            assert name in names, name
        for old in ("95% interval", "Reliability chart", "Split-half check",
                    "Without top event"):
            assert old not in names, old

    def test_the_tier_floor_text_matches_config(self):
        text = {t: m for t, m, _, _ in glossary.TERMS}["Tier floors"]
        short = f"{config.MIN_PRICE_DIFF_SHORT_GAP * 100:.0f} cents"
        long_ = f"{config.MIN_PRICE_DIFF_LONG_GAP * 100:.0f} cents"
        assert f"{config.SHORT_DEADLINE_GAP_DAYS} days apart or fewer" in text
        assert f"at least {short}" in text
        assert f"{config.SHORT_DEADLINE_GAP_DAYS + 1} to {config.MAX_DEADLINE_GAP_DAYS} " \
               "days apart" in text
        assert f"at least {long_}" in text


class TestPages:
    def test_a_page_lists_only_its_own_terms(self):
        backtest = {t for t, _, _ in glossary.terms_for(glossary.PAGE_BACKTEST)}
        data = {t for t, _, _ in glossary.terms_for(glossary.PAGE_DATA)}
        for term, _, _, pages in glossary.TERMS:
            assert (term in backtest) == (glossary.PAGE_BACKTEST in pages)
            assert (term in data) == (glossary.PAGE_DATA in pages)
        assert "Share settled Yes" in data and "Share settled Yes" not in backtest
        assert "Sharpe ratio" in backtest and "Sharpe ratio" not in data
        assert backtest & data  # shared terms exist

    def test_the_rendered_section_lists_only_its_pages_terms(self):
        backtest = glossary.terms_section_html(glossary.PAGE_BACKTEST)
        data = glossary.terms_section_html(glossary.PAGE_DATA)
        assert "Sharpe ratio" in backtest and "Sharpe ratio" not in data
        assert "Favourite–longshot gap" in data
        assert "Favourite–longshot gap" not in backtest
        assert "Brier score" in backtest and "Brier score" in data

    def test_display_order_follows_the_tuple(self):
        names = [t for t, _, _ in glossary.terms_for(glossary.PAGE_BACKTEST)]
        order = [entry[0] for entry in glossary.TERMS if entry[0] in names]
        assert names == order

    @pytest.mark.parametrize("call", [glossary.terms_for, glossary.terms_link_html,
                                      glossary.terms_section_html])
    def test_an_unknown_page_is_refused(self, call):
        with pytest.raises(ValueError):
            call("elsewhere")


class TestRendering:
    def test_every_cell_is_escaped(self, monkeypatch):
        monkeypatch.setattr(glossary, "TERMS",
                            (("a<b>&", "x<script>&y", "<i>z</i>", glossary._BOTH),))
        out = glossary.terms_section_html(glossary.PAGE_DATA)
        assert "a&lt;b&gt;&amp;" in out
        assert "x&lt;script&gt;&amp;y" in out
        assert "&lt;i&gt;z&lt;/i&gt;" in out
        assert "<script>" not in out
        assert "<i>" not in out

    def test_the_section_id_is_the_anchor_and_the_link_points_at_it(self):
        out = glossary.terms_section_html(glossary.PAGE_BACKTEST)
        assert out.count(f'id="{glossary.TERMS_ANCHOR}"') == 1
        link = glossary.terms_link_html(glossary.PAGE_BACKTEST)
        assert f'href="#{glossary.TERMS_ANCHOR}"' in link
        assert ">Terms</a>" in link

    def test_the_heading_clears_the_sticky_filter_bar(self):
        out = glossary.terms_section_html(glossary.PAGE_BACKTEST)
        head = out[:out.index("</div>")]
        assert "scroll-margin-top:220px" in head

    def test_the_section_has_three_short_columns_and_a_row_per_term(self):
        out = glossary.terms_section_html(glossary.PAGE_DATA)
        for head in ("Term", "What it means", "Why it helps"):
            assert f">{head}</th>" in out
        assert out.count("<tr><td") == len(glossary.terms_for(glossary.PAGE_DATA))

    def test_the_terms_are_read_when_rendered(self, monkeypatch):
        monkeypatch.setattr(glossary, "TERMS", (("one", "two", "three", glossary._DATA),))
        assert glossary.terms_section_html(glossary.PAGE_DATA).count("<tr><td") == 1
        assert glossary.terms_section_html(glossary.PAGE_BACKTEST).count("<tr><td") == 0


class TestIsolation:
    def test_it_imports_only_the_standard_library(self):
        tree = ast.parse(Path(glossary.__file__).read_text(encoding="utf-8"))
        stdlib = set(sys.stdlib_module_names)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, "relative import"
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            assert all(name in stdlib for name in names), names
