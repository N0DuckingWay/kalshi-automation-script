"""
Tests for kalshi_betting/eda.py: each cached day slice summarised once, and the merge.

Every test writes its own day slices with historical._DayStreamWriter into the
test's tmp_path; nothing reads the real backtest_cache/ or the network.
"""
import ast
import gzip
import html
import json
import logging
import logging.handlers
import math
import os
import re
import shutil
import subprocess
from concurrent.futures import Future
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

import pytest

from kalshi_betting import config, eda, glossary, historical

# 2026-08-17T00:00:00Z, a UTC midnight like Kalshi's archive cutoff
CUTOFF = 1786924800
SERIES = {
    "KXFED": ("Economics", ("Fed",)),
    "KXLAD": ("Science", ("Space",)),
}


def _record(ticker, event, settle, *, result="yes", title="Q?", subtitle="Yes",
            event_title="", open_time="2026-08-01T00:00:00Z",
            close_time="2026-08-10T00:00:00Z", structure="linear_cent", shard=0):
    """A record shaped like historical._market_to_dict's output."""
    return {
        "ticker": ticker, "event_ticker": event, "event_title": event_title,
        "title": title, "subtitle": subtitle, "result": result,
        "yes_ask_dollars": None, "no_ask_dollars": None, "yes_bid_dollars": None,
        "open_time": open_time, "close_time": close_time, "settlement_ts": settle,
        "status": "finalized", "price_level_structure": structure, "price_ranges": None,
        "exchange_index": shard,
    }


def _write(cache, kind, day, records, cutoff=CUTOFF):
    """Write one jsonl-v1 day slice and return its path."""
    folder = eda.SLICE_DIRS[kind]
    meta = ({"kind": "archive_created_day", "cutoff_ts": cutoff, "include_mve": True,
             "complete": True, "day": day} if kind == "archive"
            else {"kind": "live_settled_day", "include_mve": True, "complete": True,
                  "day": day})
    path = Path(cache) / folder / f"{day}.json.gz"
    with historical._DayStreamWriter(path, meta) as writer:
        writer.write_records(records)
        writer.commit()
    return path


def _build(cache, workers=1):
    """load_or_build_summaries, in this process unless asked otherwise."""
    return eda.load_or_build_summaries(cache, workers=workers)


def _merge(summaries, **kwargs):
    """merge with the test's series listing."""
    return eda.merge(summaries, series_categories=SERIES, **kwargs)


def _day_ts(day):
    """Epoch seconds of a UTC day's midnight."""
    return int(datetime.fromisoformat(day + "T00:00:00+00:00").timestamp())


def _key(week, combo, category, tag):
    return eda.KEY_SEPARATOR.join((week, "1" if combo else "0", category, tag))


# A Thursday and a Sunday of the week starting Monday 2026-08-10, and a later week
WK1 = "2026-08-10"
WK2 = "2026-08-17"


def _standard_cache(cache):
    """An archive slice and a live slice with hand-countable contents."""
    _write(cache, "archive", "2026-08-05", [
        _record("FED-A", "KXFED-26AUG", "2026-08-13T12:00:00Z", result="yes"),
        _record("FED-B", "KXFED-26AUG", "2026-08-16T23:59:59Z", result="no",
                subtitle="", open_time=None),
        _record("COMBO-1", "KXMVECROSSCATEGORY-S1", "2026-08-14T00:00:00Z",
                result="yes", title="Will X happen by Dec 31, 2026?"),
        # At the cutoff: belongs to the live endpoint, so left out of the archive count
        _record("FED-LATE", "KXFED-26AUG", "2026-08-17T00:00:00Z", result="yes"),
    ])
    _write(cache, "live", "2026-08-18", [
        _record("FED-C", "KXFED-26SEP", "2026-08-18T01:00:00Z", result="no",
                close_time="2026-08-18T00:00:00Z"),
        _record("COMBO-2", "KXMVESPORTSMULTIGAMEEXTENDED-S2", "2026-08-18T02:00:00Z",
                result="", structure="center_deci_edge_centi_cent", shard=1),
        _record("ODD", "ZZOTHER-1", "not a time", result="no"),
    ])


class TestCounts:
    def test_hand_counted_totals(self, tmp_path):
        _standard_cache(tmp_path)
        merged = _merge(_build(tmp_path))
        counts = merged["counts"]
        fed1 = counts[_key(WK1, False, "Economics", "Fed")]
        assert (fed1["n"], fed1["yes"], fed1["no"], fed1["other"]) == (2, 1, 1, 0)
        assert fed1["blank_subtitle"] == 1
        assert fed1["no_open"] == 1
        assert fed1["lifetime_unknown"] == 1
        # 2026-08-01 -> 2026-08-10 is 216 h: the [168, 720) bucket
        assert fed1["lifetime_h"][5] == 1
        # Settled 3 and 6.99 days after close: both in [72, 168)
        assert fed1["lag_h"][5] == 2
        assert fed1["regime"] == {"linear_cent": 2}
        assert fed1["shard"] == {"0": 2}
        combo1 = counts[_key(WK1, True, "Other", "General")]
        assert (combo1["n"], combo1["yes"]) == (1, 1)
        assert combo1["phrasing"] == {}
        fed2 = counts[_key(WK2, False, "Economics", "Fed")]
        assert (fed2["n"], fed2["no"]) == (1, 1)
        # Settled 1 h after close: the [1, 6) bucket
        assert fed2["lag_h"][2] == 1
        combo2 = counts[_key(WK2, True, "Other", "General")]
        assert (combo2["n"], combo2["other"]) == (1, 1)
        assert combo2["regime"] == {"center_deci_edge_centi_cent": 1}
        assert combo2["shard"] == {"1": 1}
        odd = counts[_key(eda.UNKNOWN_WEEK, False, "Other", "General")]
        assert (odd["n"], odd["no"], odd["lag_unknown"]) == (1, 1, 1)
        assert sum(c["n"] for c in counts.values()) == 6
        assert merged["records"] == 7
        assert merged["counted"] == 6
        assert merged["slices"] == {"archive": 1, "live": 1}
        assert merged["archive_cutoffs"] == [CUTOFF]
        assert merged["unreadable"] == []
        assert merged["coverage"]["archive_cutoff"] == "2026-08-17"
        assert merged["coverage"]["first_live_day"] == "2026-08-18"
        assert merged["coverage"]["gap_days"] == 1
        assert merged["coverage"]["note"].startswith("Gap:")

    def test_an_archive_record_at_or_after_its_cutoff_is_dropped_and_counted(self, tmp_path):
        _write(tmp_path, "archive", "2026-08-05", [
            _record("A", "KXFED-1", "2026-08-16T23:59:59Z"),
            _record("B", "KXFED-1", "2026-08-17T00:00:00Z"),
            _record("C", "KXFED-1", "2026-08-20T00:00:00Z"),
        ])
        (summary,) = _build(tmp_path)
        assert summary["counted"] == 1
        assert summary["dropped_after_cutoff"] == 2
        assert summary["key"]["cutoff_ts"] == CUTOFF

    def test_a_live_record_before_the_cutoff_is_still_counted(self, tmp_path):
        _write(tmp_path, "live", "2026-08-16", [_record("A", "KXFED-1", "2026-08-16T05:00:00Z")])
        (summary,) = _build(tmp_path)
        assert (summary["counted"], summary["dropped_after_cutoff"]) == (1, 0)
        assert summary["key"]["cutoff_ts"] is None

    def test_the_event_title_join_is_counted_before_and_after(self, tmp_path):
        (tmp_path / historical._EVENT_TITLES_CACHE.name).write_text(
            json.dumps({"KXFED-26AUG": "Fed decision in August"}))
        _write(tmp_path, "live", "2026-08-18", [
            _record("A", "KXFED-26AUG", "2026-08-18T01:00:00Z"),
            _record("B", "KXFED-26SEP", "2026-08-18T01:00:00Z"),
            _record("C", "KXFED-26OCT", "2026-08-18T01:00:00Z", event_title="Own"),
        ])
        c = _merge(_build(tmp_path))["counts"][_key(WK2, False, "Economics", "Fed")]
        assert (c["n"], c["blank_title_before"], c["blank_title_after"]) == (3, 2, 1)

    def test_phrasing_is_counted_for_non_combos_only(self, tmp_path):
        _write(tmp_path, "live", "2026-08-18", [
            _record("A", "KXFED-1", "2026-08-18T01:00:00Z",
                    title="Will the Fed cut by Dec 31, 2026?", subtitle=""),
            _record("B", "KXFED-1", "2026-08-18T01:00:00Z",
                    title="Fed rate on Dec 31, 2026?", subtitle=""),
            _record("C", "KXMVECROSSCATEGORY-S1", "2026-08-18T01:00:00Z",
                    title="Will the Fed cut by Dec 31, 2026?", subtitle=""),
        ])
        counts = _merge(_build(tmp_path))["counts"]
        assert counts[_key(WK2, False, "Economics", "Fed")]["phrasing"] == {
            "cumulative": 1, "snapshot": 1}
        assert counts[_key(WK2, True, "Other", "General")]["phrasing"] == {}

    def test_a_ladder_split_across_two_slices_is_one_ladder(self, tmp_path):
        _write(tmp_path, "archive", "2026-08-05", [
            _record("LAD-1", "KXLAD-26", "2026-08-10T00:00:00Z",
                    title="Will Starship launch by Oct 1, 2026?", subtitle=""),
        ])
        _write(tmp_path, "live", "2026-08-18", [
            _record("LAD-2", "KXLAD-26", "2026-08-18T00:00:00Z",
                    title="Will Starship launch by Oct 15, 2026?", subtitle=""),
            # One stated day alone is no ladder
            _record("SOLO", "KXLAD-27", "2026-08-18T00:00:00Z",
                    title="Will Starship launch by Nov 1, 2026?", subtitle=""),
        ])
        summaries = _build(tmp_path)
        assert all(len(days) == 1 for s in summaries for days in s["ladders"].values())
        merged = _merge(summaries)
        assert merged["ladder_events"] == 1
        assert merged["ladders"] == {"2026-10": {"events": 1, "rungs": 2}}


class TestReuse:
    def test_an_unchanged_slice_is_not_read_again(self, tmp_path, monkeypatch):
        _standard_cache(tmp_path)
        first = _build(tmp_path)

        def boom(*args, **kwargs):
            raise AssertionError("the slice was read again")

        monkeypatch.setattr(historical, "_day_store_iter", boom)
        assert _build(tmp_path) == first

    def _spy(self, monkeypatch):
        calls = []
        real = eda.summarise_slice

        def spy(path, kind, **kwargs):
            calls.append(Path(path).name)
            return real(path, kind, **kwargs)

        monkeypatch.setattr(eda, "summarise_slice", spy)
        return calls

    def test_a_new_mtime_rebuilds(self, tmp_path, monkeypatch):
        path = _write(tmp_path, "archive", "2026-08-05",
                      [_record("A", "KXFED-1", "2026-08-10T00:00:00Z")])
        _build(tmp_path)
        st = os.stat(path)
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        calls = self._spy(monkeypatch)
        _build(tmp_path)
        assert calls == ["2026-08-05.json.gz"]

    def test_a_new_size_rebuilds(self, tmp_path, monkeypatch):
        _write(tmp_path, "live", "2026-08-18", [_record("A", "KXFED-1", "2026-08-18T00:00:00Z")])
        _build(tmp_path)
        path = _write(tmp_path, "live", "2026-08-18", [
            _record("A", "KXFED-1", "2026-08-18T00:00:00Z"),
            _record("B", "KXFED-1", "2026-08-18T00:00:00Z")])
        calls = self._spy(monkeypatch)
        (summary,) = _build(tmp_path)
        assert calls == [path.name]
        assert summary["counted"] == 2

    def test_a_new_cutoff_rebuilds(self, tmp_path, monkeypatch):
        _write(tmp_path, "archive", "2026-08-05",
               [_record("A", "KXFED-1", "2026-08-10T00:00:00Z")])
        _build(tmp_path)
        # Same size and time, but the meta line now names another cutoff
        monkeypatch.setattr(eda, "_head_cutoff", lambda path: (True, CUTOFF + 86_400))
        calls = self._spy(monkeypatch)
        _build(tmp_path)
        assert calls == ["2026-08-05.json.gz"]

    def test_the_meta_cutoff_is_read_from_the_first_line(self, tmp_path):
        path = _write(tmp_path, "archive", "2026-08-05", [], cutoff=CUTOFF)
        assert eda._head_cutoff(path) == (True, CUTOFF)
        live = _write(tmp_path, "live", "2026-08-18", [])
        assert eda._head_cutoff(live) == (True, None)

    def test_a_version_bump_rebuilds(self, tmp_path, monkeypatch):
        _write(tmp_path, "live", "2026-08-18", [_record("A", "KXFED-1", "2026-08-18T00:00:00Z")])
        _build(tmp_path)
        monkeypatch.setattr(config, "EDA_SUMMARY_VERSION", config.EDA_SUMMARY_VERSION + 1)
        calls = self._spy(monkeypatch)
        (summary,) = _build(tmp_path)
        assert calls == ["2026-08-18.json.gz"]
        assert summary["key"]["version"] == config.EDA_SUMMARY_VERSION

    def test_summaries_are_saved_where_the_config_names(self, tmp_path):
        _write(tmp_path, "live", "2026-08-18", [_record("A", "KXFED-1", "2026-08-18T00:00:00Z")])
        _build(tmp_path)
        saved = tmp_path / config.EDA_SUMMARY_DIRNAME / "live" / "2026-08-18.json"
        assert json.loads(saved.read_text())["counted"] == 1
        assert not list(saved.parent.glob("*.tmp"))

    def test_a_removed_archive_slice_s_summary_is_ignored_and_deleted(self, tmp_path):
        _standard_cache(tmp_path)
        _build(tmp_path)
        (tmp_path / "archive_days" / "2026-08-05.json.gz").unlink()
        summaries = _build(tmp_path)
        assert [s["key"]["kind"] for s in summaries] == ["live"]
        assert not (tmp_path / config.EDA_SUMMARY_DIRNAME / "archive" / "2026-08-05.json").exists()
        assert _merge(summaries)["slices"] == {"live": 1, "archive": 0}

    def test_a_leftover_temporary_summary_is_deleted(self, tmp_path):
        _standard_cache(tmp_path)
        _build(tmp_path)
        tmp = tmp_path / config.EDA_SUMMARY_DIRNAME / "archive" / "2026-08-05.json.tmp"
        tmp.write_text("{")
        _build(tmp_path)
        assert not tmp.exists()

    @pytest.mark.parametrize("content", ["{not json", "[1, 2]", '{"key": 5}',
                                         '{"key": {"version": 2, "kind": 7}}'])
    def test_a_damaged_summary_is_just_rebuilt(self, tmp_path, content):
        _write(tmp_path, "archive", "2026-08-05", [_record("A", "KXFED-1", "2026-08-10T00:00:00Z")])
        _build(tmp_path)
        (tmp_path / config.EDA_SUMMARY_DIRNAME / "archive" / "2026-08-05.json").write_text(content)
        (summary,) = _build(tmp_path)
        assert summary["counted"] == 1

    def test_missing_slice_folders_are_empty(self, tmp_path):
        assert _build(tmp_path) == []
        assert _merge([])["counts"] == {}


class TestUnreadable:
    def test_a_damaged_slice_is_named_not_saved_and_the_rest_merge(self, tmp_path):
        _standard_cache(tmp_path)
        bad = tmp_path / "live_days" / "2026-08-19.json.gz"
        bad.write_bytes(b"this is not gzip")
        summaries = _build(tmp_path)
        named = [s for s in summaries if s["unreadable"]]
        assert len(named) == 1
        assert named[0]["unreadable"].startswith("2026-08-19.json.gz: ")
        assert "\n" not in named[0]["unreadable"]
        assert not (tmp_path / config.EDA_SUMMARY_DIRNAME / "live" / "2026-08-19.json").exists()
        merged = _merge(summaries)
        assert merged["counted"] == 6
        assert merged["unreadable"] == [f"live: {named[0]['unreadable']}"]

    def test_a_slice_cut_short_counts_nothing(self, tmp_path):
        path = _write(tmp_path, "live", "2026-08-18",
                      [_record(f"T{i}", "KXFED-1", "2026-08-18T00:00:00Z") for i in range(50)])
        path.write_bytes(path.read_bytes()[:-20])
        (summary,) = _build(tmp_path)
        assert summary["unreadable"]
        assert summary["counts"] == {} and summary["counted"] == 0

    def test_a_slice_that_changes_while_read_is_named(self, tmp_path, monkeypatch):
        _write(tmp_path, "live", "2026-08-18", [_record("A", "KXFED-1", "2026-08-18T00:00:00Z")])
        stats = iter([(10, 1), (11, 2)])
        monkeypatch.setattr(eda, "_slice_stat", lambda path: next(stats))
        (summary,) = _build(tmp_path)
        assert "changed while it was read" in summary["unreadable"]

    def test_a_count_that_raises_is_named_and_the_rest_go_on(self, tmp_path, monkeypatch):
        _standard_cache(tmp_path)
        real = eda.summarise_slice

        def flaky(path, kind, **kwargs):
            if kind == "live":
                raise RuntimeError("boom\nsecond line")
            return real(path, kind, **kwargs)

        monkeypatch.setattr(eda, "summarise_slice", flaky)
        summaries = _build(tmp_path)
        assert summaries[0]["unreadable"] == "2026-08-18.json.gz: RuntimeError: boom"
        assert summaries[1]["counted"] == 3


class TestCandles:
    def test_candle_markets_are_kept_and_survive_reuse(self, tmp_path, monkeypatch):
        _standard_cache(tmp_path)
        folder = tmp_path / historical._CANDLES_DIR.name
        folder.mkdir()
        for name in ("FED-A.json", "COMBO-2.json", "NOMARKET.json", "FED-A.json.tmp"):
            (folder / name).write_text("{}")
        candle_set = eda.candle_tickers(tmp_path)
        assert candle_set == frozenset({"FED-A", "COMBO-2", "NOMARKET"})
        merged = _merge(_build(tmp_path), candle_set=candle_set)
        assert merged["candles"]["FED-A"] == [
            "yes", "2026-08-10T00:00:00Z", "2026-08-13T12:00:00Z", "Economics", "Fed", False]
        assert merged["candles"]["COMBO-2"][3:] == ["Other", "General", True]
        assert (merged["candle_files"], merged["candles_unjoined"]) == (3, 1)
        monkeypatch.setattr(historical, "_day_store_iter",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-read")))
        again = _merge(_build(tmp_path), candle_set=candle_set)
        assert again["candles"] == merged["candles"]


class TestWorkers:
    def test_two_workers_merge_exactly_as_one(self, tmp_path):
        one, two = tmp_path / "one", tmp_path / "two"
        for cache in (one, two):
            _standard_cache(cache)
            (cache / historical._EVENT_TITLES_CACHE.name).write_text(
                json.dumps({"KXFED-26AUG": "Fed decision in August"}))
            (cache / historical._CANDLES_DIR.name).mkdir()
            (cache / historical._CANDLES_DIR.name / "FED-C.json").write_text("{}")
        merged_one = _merge(_build(one, workers=1))
        merged_two = _merge(_build(two, workers=2))
        assert merged_two == merged_one
        assert merged_two["candles"]["FED-C"][0] == "no"


class TestArchiveAndLiveCoverage:
    def _counted_live_days(self, merged):
        cov = merged["coverage"]
        return cov["first_live_day"], cov["last_live_day"]

    def test_a_deleted_live_day_the_archive_does_not_cover_is_still_counted(self, tmp_path):
        # The cutoff moves from 08-17 to 08-24 and a cache extension deletes the live
        # days before 08-24 without fetching the archive again (still stamped 08-17)
        _write(tmp_path, "archive", "2026-08-05",
               [_record("A", "KXFED-1", "2026-08-10T00:00:00Z")], cutoff=_day_ts("2026-08-17"))
        _write(tmp_path, "live", "2026-08-19", [_record("L19", "KXFED-1", "2026-08-19T03:00:00Z")])
        _write(tmp_path, "live", "2026-08-25", [_record("L25", "KXFED-1", "2026-08-25T03:00:00Z")])
        _build(tmp_path)
        (tmp_path / "live_days" / "2026-08-19.json.gz").unlink()
        summaries = _build(tmp_path)
        kept = [s for s in summaries if s.get("slice_gone")]
        assert [s["key"]["day"] for s in kept] == ["2026-08-19"]
        merged = _merge(summaries)
        assert merged["counted"] == 3
        assert self._counted_live_days(merged) == ("2026-08-19", "2026-08-25")
        assert merged["coverage"]["live_days_without_slice"] == 1
        assert (tmp_path / config.EDA_SUMMARY_DIRNAME / "live" / "2026-08-19.json").exists()

        # Once the archive is fetched again under the new cutoff, it covers 08-19
        _write(tmp_path, "archive", "2026-08-05",
               [_record("A", "KXFED-1", "2026-08-10T00:00:00Z"),
                _record("L19", "KXFED-1", "2026-08-19T03:00:00Z")],
               cutoff=_day_ts("2026-08-24"))
        summaries = _build(tmp_path)
        assert not (tmp_path / config.EDA_SUMMARY_DIRNAME / "live" / "2026-08-19.json").exists()
        merged = _merge(summaries)
        assert merged["counted"] == 3
        assert self._counted_live_days(merged) == ("2026-08-25", "2026-08-25")
        assert merged["coverage"]["gap_days"] == 1

    def test_an_interrupted_fetch_does_not_count_a_day_twice(self, tmp_path):
        # The archive was fetched again under 08-24, but the fetch stopped before
        # the live days before 08-24 were deleted
        _write(tmp_path, "archive", "2026-08-05",
               [_record("DUP", "KXFED-1", "2026-08-19T03:00:00Z")], cutoff=_day_ts("2026-08-24"))
        _write(tmp_path, "live", "2026-08-19", [_record("DUP", "KXFED-1", "2026-08-19T03:00:00Z")])
        _write(tmp_path, "live", "2026-08-24", [_record("NEW", "KXFED-1", "2026-08-24T03:00:00Z")])
        merged = _merge(_build(tmp_path))
        assert merged["counted"] == 2
        assert merged["coverage"]["live_days_skipped"] == 1
        assert self._counted_live_days(merged) == ("2026-08-24", "2026-08-24")
        assert merged["coverage"]["gap_days"] == 0
        assert merged["coverage"]["note"].startswith("No gap")

    def test_with_no_archive_every_live_day_counts(self, tmp_path):
        _write(tmp_path, "live", "2026-08-19", [_record("A", "KXFED-1", "2026-08-19T03:00:00Z")])
        merged = _merge(_build(tmp_path))
        assert merged["counted"] == 1
        assert merged["coverage"]["archive_cutoff"] is None


class TestListing:
    def test_a_slice_folder_that_cannot_be_listed_is_named_and_left_alone(self, tmp_path):
        _standard_cache(tmp_path)
        _build(tmp_path)
        saved = tmp_path / config.EDA_SUMMARY_DIRNAME / "live" / "2026-08-18.json"
        live = tmp_path / "live_days"
        for path in live.iterdir():
            path.unlink()
        live.rmdir()
        live.write_text("not a folder")
        summaries = _build(tmp_path)
        merged = _merge(summaries)
        assert len(merged["unreadable"]) == 1
        assert merged["unreadable"][0].startswith("live: live_days/: NotADirectoryError")
        assert saved.exists()
        assert merged["slices"] == {"live": 0, "archive": 1}

    def test_labels_are_applied_at_merge_time(self, tmp_path):
        _standard_cache(tmp_path)
        summaries = _build(tmp_path)
        assert all(k.count(eda.KEY_SEPARATOR) == 2 for s in summaries for k in s["counts"])
        refiled = eda.merge(summaries, series_categories={"KXFED": ("Finance", ())})
        assert _key(WK1, False, "Finance", "General") in refiled["counts"]
        assert _key(WK1, False, "Economics", "Fed") not in refiled["counts"]


class TestWorkerFallback:
    def test_a_pool_that_cannot_be_made_counts_in_this_process(self, tmp_path, monkeypatch, caplog):
        _standard_cache(tmp_path)

        def refuse(*args, **kwargs):
            raise PermissionError("Operation not permitted")

        monkeypatch.setattr(eda, "ProcessPoolExecutor", refuse)
        merged = _merge(_build(tmp_path, workers=2))
        assert merged["counted"] == 6 and merged["unreadable"] == []
        warnings = [r for r in caplog.records if "worker processes could not be used" in r.message]
        assert len(warnings) == 1
        assert "PermissionError" in warnings[0].message
        assert '__name__ == "__main__"' in warnings[0].message

    def test_workers_that_do_not_start_in_time_are_stopped(self, tmp_path, monkeypatch, caplog):
        _standard_cache(tmp_path)
        made = []

        class Process:
            killed = False

            def kill(self):
                self.killed = True

        class SilentPool:
            def __init__(self, **kwargs):
                self._processes = {1: Process()}
                self.shutdowns = []
                made.append(self)

            def submit(self, fn, *args):
                return Future()  # never finishes

            def shutdown(self, wait=True, cancel_futures=False):
                self.shutdowns.append((wait, cancel_futures))

        monkeypatch.setattr(eda, "ProcessPoolExecutor", SilentPool)
        monkeypatch.setattr(config, "EDA_WORKER_START_TIMEOUT_SECONDS", 0.05)
        merged = _merge(_build(tmp_path, workers=2))
        assert merged["counted"] == 6 and merged["unreadable"] == []
        (pool,) = made
        assert pool._processes[1].killed
        assert pool.shutdowns == [(False, True)]
        assert sum("worker processes could not be used" in r.message
                   for r in caplog.records) == 1


class TestIsolation:
    _FORBIDDEN = {"trader", "main", "scheduler", "run_lock", "defaults_server",
                  "v2_probe", "seller"}

    def _imports(self, path):
        names = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    names.update(alias.name for alias in node.names)
                    if node.module:
                        names.add(node.module)
                else:
                    names.add(node.module or "")
        return names

    def _project_names(self, names):
        package = Path(eda.__file__).parent
        local = {p.stem for p in package.glob("*.py")}
        return {n.split(".")[-1] for n in names
                if n.split(".")[-1] in local or n.startswith("kalshi_betting")}

    def test_eda_imports_only_the_standard_library_config_glossary_historical_and_scanner(self):
        package = Path(eda.__file__).parent
        tree = ast.parse((package / "eda.py").read_text())
        # Module level only: what a spawned worker loads when it imports eda
        top = set()
        for node in tree.body:
            holder = ast.Module(body=[node], type_ignores=[])
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                top |= self._imports_of(holder)
        assert self._project_names(top) == {"config", "glossary", "historical", "scanner"}
        assert not (top & self._FORBIDDEN)

    def test_the_only_local_imports_are_the_k_view_and_the_chart_library(self):
        package = Path(eda.__file__).parent
        tree = ast.parse((package / "eda.py").read_text())
        found = {}
        for func in ast.walk(tree):
            if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names = set()
                for node in ast.walk(func):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        names |= self._imports_of(ast.Module(body=[node], type_ignores=[]))
                if names:
                    found[func.name] = self._project_names(names) or names
        assert found == {"khat_view": {"backtester", "dashboard"},
                         "_plotly": {"plotly.graph_objects"}}
        assert not (self._imports(package / "eda.py") & self._FORBIDDEN)

    def _imports_of(self, module):
        names = set()
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    names.update(alias.name for alias in node.names)
                    if node.module:
                        names.add(node.module)
                else:
                    names.add(node.module or "")
        return names

    def test_no_module_imports_eda(self):
        package = Path(eda.__file__).parent
        for path in package.glob("*.py"):
            if path.name == "eda.py":
                continue
            assert "eda" not in self._imports(path), path.name


@pytest.mark.parametrize("value, bucket", [(-5, 0), (0.5, 0), (1, 1), (719.9, 5), (720, 6)])
def test_bucket_edges(value, bucket):
    assert eda._bucket(value, config.EDA_LIFETIME_EDGES_HOURS) == bucket


# ─── Candle calibration ───────────────────────────────────────────────────────

CLOSE = "2026-08-10T00:00:00Z"
CLOSE_TS = _day_ts("2026-08-10")


def _candle_file(cache, ticker, candles, *, fields=2, bare_list=False, raw=None):
    """Write <cache>/candlesticks/<ticker>.json in historical.fetch_candlesticks' format."""
    folder = Path(cache) / historical._CANDLES_DIR.name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{ticker}.json"
    if raw is not None:
        path.write_bytes(raw)
    elif bare_list:
        path.write_text(json.dumps(candles))
    else:
        body = {"open_ts": 0, "close_ts": CLOSE_TS, "period_interval": 60, "candles": candles}
        if fields is not None:
            body["fields"] = fields
        path.write_text(json.dumps(body))
    return path


def _c(ts, yes, no, volume=None):
    """One candle."""
    candle = {"ts": ts, "yes_ask_close": yes, "no_ask_close": no}
    if volume is not None:
        candle["volume"] = volume
    return candle


def _info(result="yes", category="Economics", tag="Fed", combo=False, close=CLOSE):
    """A merge()-shaped candle map entry."""
    return [result, close, "2026-08-11T00:00:00Z", category, tag, combo]


class TestSummariseCandles:
    def test_the_last_candle_at_or_before_close_minus_the_lead_is_used(self, tmp_path):
        _candle_file(tmp_path, "T1", [
            _c(CLOSE_TS - 100_000, 0.30, 0.72),        # 24 h lead: ask 0.30, mid 0.29
            _c(CLOSE_TS - 7200, 0.60, 0.50),           # older than the 1 h lead's candle
            _c(CLOSE_TS - 3600, 0.62, 0.40),           # ends exactly 1 h before: ask 0.62, mid 0.61
            _c(CLOSE_TS - 1800, 0.99, 0.02),           # ends after the 1 h target: never used
        ])
        out = eda.summarise_candles(tmp_path, {"T1": _info("yes")})
        one = out["calibration"]["1"]["Economics"]
        assert one["ask"]["bins"][12][:2] == [1, 1] and one["ask"]["n"] == 1
        assert one["ask"]["brier_sum"] == pytest.approx((0.62 - 1) ** 2)
        assert one["mid"]["bins"][12][:2] == [1, 1]
        assert one["mid"]["brier_sum"] == pytest.approx((0.61 - 1) ** 2)
        day = out["calibration"]["24"]["Economics"]
        assert day["ask"]["bins"][6][:2] == [1, 1]
        assert day["mid"]["bins"][5][:2] == [1, 1]
        assert day["mid"]["brier_sum"] == pytest.approx((0.29 - 1) ** 2)
        # Nothing ended 7 days before close: that lead has no price for this market
        assert "Economics" not in out["calibration"]["168"]
        assert out["coverage"]["168"] == {"markets": 1, "no_ask": 1, "no_mid": 1, "stale": 0,
                                          "wide": 0}
        assert out["coverage"]["1"] == {"markets": 1, "no_ask": 0, "no_mid": 0, "stale": 0,
                                       "wide": 0}

    def test_a_long_shot_has_an_ask_but_no_midpoint(self, tmp_path):
        before = CLOSE_TS - 7200
        _candle_file(tmp_path, "SHOT", [_c(before, 0.04, 0.99)])      # 4c ask, no bid: 4c wide
        _candle_file(tmp_path, "ONE", [_c(before, 1.0, 0.40)])        # a YES ask of 1.00: no quote
        _candle_file(tmp_path, "GOOD", [_c(before, 0.60, 0.50)])      # bid 0.50: 10c wide
        out = eda.summarise_candles(tmp_path, {t: _info("no") for t in ("SHOT", "ONE", "GOOD")})
        cell = out["calibration"]["1"]["Economics"]
        # SHOT and GOOD have a YES ask; only GOOD has a midpoint
        # (0.60 + 0.50) / 2 = 0.55, bin 11, settled No; ONE has neither
        assert cell["ask"]["n"] == 2
        assert cell["ask"]["bins"][0][:2] == [1, 0] and cell["ask"]["bins"][12][:2] == [1, 0]
        assert cell["ask"]["brier_sum"] == pytest.approx(0.04 ** 2 + 0.60 ** 2)
        assert cell["mid"]["n"] == 1 and cell["mid"]["bins"][11][:2] == [1, 0]
        assert cell["mid"]["brier_sum"] == pytest.approx(0.55 ** 2)
        assert out["coverage"]["1"] == {"markets": 3, "no_ask": 1, "no_mid": 2, "stale": 0,
                                        "wide": 0}

    def test_a_one_sided_book_is_priced_only_when_the_ask_alone_is_narrow(self, tmp_path):
        before = CLOSE_TS - 7200
        _candle_file(tmp_path, "FAV", [_c(before, 0.99, 0.99)])        # 99c ask, no bid
        _candle_file(tmp_path, "FAVBID", [_c(before, 0.99, 0.05)])     # bid 0.95: 4c wide
        _candle_file(tmp_path, "SHOT", [_c(before, 0.03, 0.99)])       # 3c ask, no bid
        _candle_file(tmp_path, "NOKEY", [{"ts": before, "yes_ask_close": 0.50}])
        out = eda.summarise_candles(tmp_path, {t: _info("yes") for t in
                                               ("FAV", "FAVBID", "SHOT", "NOKEY")})
        cell = out["calibration"]["1"]["Economics"]
        assert cell["ask"]["n"] == 2
        assert cell["ask"]["bins"][19][:2] == [1, 1] and cell["ask"]["bins"][0][:2] == [1, 1]
        assert cell["mid"]["n"] == 1 and cell["mid"]["bins"][19][:2] == [1, 1]   # FAVBID: 0.97
        # FAV (99c wide) and NOKEY (50c wide) are wide books, and are not "no ask"
        assert out["coverage"]["1"] == {"markets": 4, "no_ask": 0, "no_mid": 1, "stale": 0,
                                        "wide": 2}

    def test_a_book_wider_than_the_limit_has_no_price_at_all(self, tmp_path):
        before = CLOSE_TS - 7200
        _candle_file(tmp_path, "WIDE", [_c(before, 0.62, 0.50)])       # bid 0.50: 12c wide
        _candle_file(tmp_path, "EDGE", [_c(before, 0.60, 0.50)])       # 10c wide: counts
        _candle_file(tmp_path, "CROSSED", [_c(before, 0.40, 0.45)])   # bid 0.55 above the ask
        _candle_file(tmp_path, "TEN", [_c(before, 0.17, 0.93)])        # 10c wide, to float noise
        out = eda.summarise_candles(tmp_path, {t: _info("yes") for t in
                                               ("WIDE", "EDGE", "CROSSED", "TEN")})
        cell = out["calibration"]["1"]["Economics"]
        assert cell["ask"]["n"] == 3 and cell["mid"]["n"] == 3
        assert cell["ask"]["bins"][12][:2] == [1, 1] and cell["ask"]["bins"][8][:2] == [1, 1]
        assert cell["ask"]["bins"][3][:2] == [1, 1]
        assert out["coverage"]["1"]["wide"] == 1 and out["coverage"]["1"]["no_ask"] == 0
        # TEN is a hair over 10c wide in floating point (0.17 - (1 - 0.93) = 0.10000000000000009),
        # so it counts only because of the PRICE_EPSILON slack
        assert 0.17 - (1 - 0.93) > config.EDA_MAX_BOOK_WIDTH

    def test_a_bin_keeps_the_prices_in_it_not_the_bin_centre(self, tmp_path, monkeypatch):
        before = CLOSE_TS - 7200
        _candle_file(tmp_path, "A", [_c(before, 0.01, 0.99)])
        _candle_file(tmp_path, "B", [_c(before, 0.01, 0.99)])
        out = eda.summarise_candles(tmp_path, {t: _info("no") for t in ("A", "B")})
        slot = out["calibration"]["1"]["Economics"]["ask"]["bins"][0]
        assert slot[:2] == [2, 0] and slot[2] == pytest.approx(0.02)
        seen = _capture_charts(monkeypatch)
        eda._calibration_view(eda._plotly(), out, 1, "ask", [])
        (gap,), _ = seen["eda-cal-1-ask-gap"]
        assert list(gap.x) == [pytest.approx(0.01)]          # not the bin centre, 0.025
        assert list(gap.y) == [pytest.approx(-1.0)]          # 0% Yes minus 1c, not minus 2.5c
        (_fair, dots), _ = seen["eda-cal-1-ask-reliability"]
        assert list(dots.x) == [pytest.approx(0.01)] and list(dots.y) == [0.0]

    def test_facts_saved_before_the_width_rule_are_read_again(self, tmp_path):
        before = CLOSE_TS - 7200
        _candle_file(tmp_path, "FAV", [_c(before, 0.99, 0.99)])
        info = {"FAV": _info("yes")}
        eda.summarise_candles(tmp_path, info)
        path = tmp_path / config.EDA_SUMMARY_DIRNAME / "candles.json"
        saved = json.loads(path.read_text())
        assert saved["version"] == config.EDA_CANDLE_FACTS_VERSION >= 2
        saved["version"] = 1                                  # as an older build saved it
        path.write_text(json.dumps(saved))
        out = eda.summarise_candles(tmp_path, info)
        assert out["read"] == 1 and out["reused"] == 0
        # A fact with no "wide" list is refused too, whatever its version says
        del saved["files"]["FAV.json"]["wide"]
        saved["version"] = config.EDA_CANDLE_FACTS_VERSION
        path.write_text(json.dumps(saved))
        assert eda.summarise_candles(tmp_path, info)["read"] == 1

    def test_a_last_candle_older_than_the_maximum_age_gives_no_price(self, tmp_path):
        target = CLOSE_TS - 3600                      # the 1 h lead's moment
        age = int(config.EDA_PRICE_MAX_AGE_HOURS * 3600)
        _candle_file(tmp_path, "STALE", [_c(target - age, 0.60, 0.40)])    # exactly too old
        _candle_file(tmp_path, "FRESH", [_c(target - age + 1, 0.60, 0.40)])
        _candle_file(tmp_path, "NONE", [_c(CLOSE_TS - 10, 0.60, 0.40)])    # nothing before
        out = eda.summarise_candles(tmp_path, {t: _info("yes") for t in
                                               ("STALE", "FRESH", "NONE")})
        assert out["calibration"]["1"]["Economics"]["ask"]["n"] == 1
        assert out["coverage"]["1"] == {"markets": 3, "no_ask": 2, "no_mid": 2, "stale": 1,
                                       "wide": 0}

    def test_bins_and_brier_by_hand(self, tmp_path):
        before = CLOSE_TS - 7200
        _candle_file(tmp_path, "A", [_c(before, 0.60, 0.38)])    # ask 0.60, mid 0.61
        _candle_file(tmp_path, "B", [_c(before, 0.64, 0.38)])    # ask 0.64, mid 0.63, same bins
        _candle_file(tmp_path, "C", [_c(before, 0.12, 0.92)])    # ask 0.12, mid 0.10 (edge -> bin 2)
        _candle_file(tmp_path, "D", [_c(before, 0.30, 0.70)])    # mid 0.30, edge -> bin 6
        out = eda.summarise_candles(tmp_path, {
            "A": _info("yes"), "B": _info("no"), "C": _info("no", "Science", "Space"),
            "D": _info("")})
        eco = out["calibration"]["1"]["Economics"]
        assert eco["mid"]["bins"][12][:2] == [2, 1] and eco["mid"]["n"] == 2
        assert eco["mid"]["brier_sum"] == pytest.approx((0.61 - 1) ** 2 + 0.63 ** 2)
        assert eco["ask"]["bins"][12][:2] == [2, 1]
        assert eco["ask"]["brier_sum"] == pytest.approx((0.60 - 1) ** 2 + 0.64 ** 2)
        sci = out["calibration"]["1"]["Science"]
        assert sci["mid"]["bins"][2][:2] == [1, 0]
        assert sci["mid"]["brier_sum"] == pytest.approx(0.10 ** 2)
        assert sci["ask"]["bins"][2][:2] == [1, 0]
        # D has no yes/no result, so it is in no bin
        assert sum(slot[0] for slot in eco["mid"]["bins"]) == 2
        assert out["coverage"]["1"]["markets"] == 3
        assert out["bins"] == 20 and out["bin_width"] == config.EDA_CALIBRATION_BIN

    def test_volume_comes_only_from_current_version_files(self, tmp_path):
        before = CLOSE_TS - 7200
        _candle_file(tmp_path, "NEW1", [_c(before, 0.5, 0.5, 10), _c(before + 60, 0.5, 0.5, 5)])
        _candle_file(tmp_path, "NEW2", [_c(before, 0.5, 0.5, 7)])
        _candle_file(tmp_path, "OLD", [_c(before, 0.5, 0.5, 1000)], fields=None)
        _candle_file(tmp_path, "NOVOL", [_c(before, 0.5, 0.5)])
        _candle_file(tmp_path, "COMBO", [_c(before, 0.5, 0.5, 99)])
        out = eda.summarise_candles(tmp_path, {
            "NEW1": _info(), "NEW2": _info(category="Science", tag="Space"),
            "OLD": _info(), "NOVOL": _info(), "COMBO": _info(combo=True)})
        assert out["volume"] == {"Economics": 15.0, "Science": 7.0}
        assert out["volume_files"] == 2
        assert out["combos_left_out"] == 1
        assert "Other" not in out["calibration"]["1"]

    def test_files_are_counted_by_version_empty_unreadable_and_unjoined(self, tmp_path):
        before = CLOSE_TS - 7200
        _candle_file(tmp_path, "V2", [_c(before, 0.5, 0.5, 1)])
        _candle_file(tmp_path, "V1", [_c(before, 0.5, 0.5)], fields=None)
        _candle_file(tmp_path, "LIST", [_c(before, 0.5, 0.5)], bare_list=True)
        _candle_file(tmp_path, "EMPTY", [])
        _candle_file(tmp_path, "BAD", None, raw=b"{not json")
        _candle_file(tmp_path, "SHAPE", None, raw=b'{"fields": 2, "candles": 5}')
        _candle_file(tmp_path, "ORPHAN", [_c(before, 0.5, 0.5, 4)])
        (Path(tmp_path) / "candlesticks" / "ignored.txt").write_text("x")
        info = {t: _info() for t in ("V2", "V1", "LIST", "EMPTY", "BAD", "SHAPE")}
        out = eda.summarise_candles(tmp_path, info)
        assert out["files"] == 7
        assert out["by_fields"] == {"2": 3, "1": 2}
        assert out["empty"] == 1
        assert out["unreadable"] == 2
        assert out["unjoined"] == 1
        # The orphan's volume cannot be filed under a category
        assert out["volume"] == {"Economics": 1.0}
        assert out["folder_error"] is None

    def test_no_candle_folder_is_empty(self, tmp_path):
        out = eda.summarise_candles(tmp_path, {})
        assert out["files"] == 0 and out["calibration"]["1"] == {} and out["folder_error"] is None

    def test_the_folder_is_listed_once_and_never_globbed(self, tmp_path, monkeypatch):
        _candle_file(tmp_path, "A", [_c(CLOSE_TS - 7200, 0.5, 0.5)])
        calls = []
        real = os.scandir
        monkeypatch.setattr(os, "scandir", lambda p: (calls.append(str(p)), real(p))[1])
        eda.summarise_candles(tmp_path, {"A": _info()})
        assert calls == [str(tmp_path / "candlesticks")]

    def _spy(self, monkeypatch):
        calls = []
        real = eda._read_candle_file
        monkeypatch.setattr(eda, "_read_candle_file",
                            lambda *a, **k: (calls.append(a[0].name), real(*a, **k))[1])
        return calls

    def test_the_per_file_facts_are_reused_and_saved_where_the_config_names(
            self, tmp_path, monkeypatch):
        path = _candle_file(tmp_path, "A", [_c(CLOSE_TS - 7200, 0.60, 0.38, 3)])
        calls = self._spy(monkeypatch)
        first = eda.summarise_candles(tmp_path, {"A": _info()})
        assert (first["read"], first["reused"]) == (1, 0)
        saved = tmp_path / config.EDA_SUMMARY_DIRNAME / "candles.json"
        assert saved.is_file()
        assert not list(saved.parent.glob("*.tmp"))
        again = eda.summarise_candles(tmp_path, {"A": _info()})
        assert (again["read"], again["reused"]) == (0, 1)
        assert calls == [path.name]
        for key in ("calibration", "volume", "by_fields", "empty"):
            assert again[key] == first[key]

    def test_a_new_result_or_category_needs_no_re_read(self, tmp_path, monkeypatch):
        _candle_file(tmp_path, "A", [_c(CLOSE_TS - 7200, 0.60, 0.38)])
        eda.summarise_candles(tmp_path, {"A": _info("yes")})
        calls = self._spy(monkeypatch)
        out = eda.summarise_candles(tmp_path, {"A": _info("no", "Science", "Space")})
        assert calls == [] and out["reused"] == 1
        assert out["calibration"]["1"]["Science"]["ask"]["bins"][12][:2] == [1, 0]
        assert "Economics" not in out["calibration"]["1"]

    @pytest.mark.parametrize("change", ["size", "mtime", "close", "version"])
    def test_a_changed_key_re_reads_the_file(self, tmp_path, monkeypatch, change):
        path = _candle_file(tmp_path, "A", [_c(CLOSE_TS - 7200, 0.60, 0.38)])
        info = {"A": _info()}
        eda.summarise_candles(tmp_path, info)
        calls = self._spy(monkeypatch)
        if change == "size":
            _candle_file(tmp_path, "A", [_c(CLOSE_TS - 7200, 0.60, 0.38),
                                         _c(CLOSE_TS - 9000, 0.50, 0.50)])
        elif change == "mtime":
            st = path.stat()
            os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
        elif change == "close":
            info = {"A": _info(close="2026-08-10T01:00:00Z")}
        else:
            monkeypatch.setattr(config, "EDA_CANDLE_FACTS_VERSION",
                                config.EDA_CANDLE_FACTS_VERSION + 1)
        out = eda.summarise_candles(tmp_path, info)
        assert calls == ["A.json"] and out["read"] == 1

    def test_a_close_time_moves_the_price_that_is_read(self, tmp_path):
        _candle_file(tmp_path, "A", [_c(CLOSE_TS - 7200, 0.60, 0.38),
                                     _c(CLOSE_TS, 0.20, 0.82)])
        early = eda.summarise_candles(tmp_path, {"A": _info()})
        assert early["calibration"]["1"]["Economics"]["ask"]["bins"][12][0] == 1
        later = eda.summarise_candles(tmp_path, {"A": _info(close="2026-08-10T01:00:00Z")})
        # One hour later the candle ending at close is the last one at or before close - 1 h
        assert later["calibration"]["1"]["Economics"]["ask"]["bins"][4][0] == 1

    def test_a_vanished_file_is_dropped_from_the_saved_facts(self, tmp_path):
        a = _candle_file(tmp_path, "A", [_c(CLOSE_TS - 7200, 0.5, 0.5)])
        _candle_file(tmp_path, "B", [_c(CLOSE_TS - 7200, 0.5, 0.5)])
        eda.summarise_candles(tmp_path, {"A": _info(), "B": _info()})
        a.unlink()
        eda.summarise_candles(tmp_path, {"B": _info()})
        saved = json.loads((tmp_path / config.EDA_SUMMARY_DIRNAME / "candles.json").read_text())
        assert sorted(saved["files"]) == ["B.json"]

    @pytest.mark.parametrize("content", [b"", b"[]", b"{not json", b'{"version": 1}'])
    def test_a_damaged_fact_file_is_just_rebuilt(self, tmp_path, content):
        _candle_file(tmp_path, "A", [_c(CLOSE_TS - 7200, 0.5, 0.5)])
        saved = tmp_path / config.EDA_SUMMARY_DIRNAME / "candles.json"
        saved.parent.mkdir(parents=True)
        saved.write_bytes(content)
        out = eda.summarise_candles(tmp_path, {"A": _info()})
        assert out["read"] == 1 and out["files"] == 1

    def test_the_day_slice_version_does_not_touch_the_candle_facts(self, tmp_path, monkeypatch):
        _candle_file(tmp_path, "A", [_c(CLOSE_TS - 7200, 0.60, 0.38)])
        eda.summarise_candles(tmp_path, {"A": _info()})
        monkeypatch.setattr(config, "EDA_SUMMARY_VERSION", config.EDA_SUMMARY_VERSION + 1)
        assert eda.summarise_candles(tmp_path, {"A": _info()})["read"] == 0

    @pytest.mark.parametrize("change", [
        ("vol", "missing"), ("vol", "7"), ("vol", float("nan")), ("mids", ["x", None, None]),
        ("mids", [0.5, None]), ("mids", [1.5, None, None]), ("asks", [float("nan"), None, None]),
        ("asks", None), ("stale", [0, 1, 0]), ("stale", [False]), ("f", True), ("f", "2"),
        ("e", "no")])
    def test_a_badly_shaped_saved_fact_is_read_again_not_raised_on(self, tmp_path, change):
        _candle_file(tmp_path, "A", [_c(CLOSE_TS - 7200, 0.60, 0.38, 2)])
        first = eda.summarise_candles(tmp_path, {"A": _info()})
        saved_path = tmp_path / config.EDA_SUMMARY_DIRNAME / "candles.json"
        saved = json.loads(saved_path.read_text())
        key, value = change
        if value == "missing":
            del saved["files"]["A.json"][key]
        else:
            saved["files"]["A.json"][key] = value
        # allow_nan keeps a NaN in the file, as a hand-edit or a bug could leave it
        saved_path.write_text(json.dumps(saved))
        out = eda.summarise_candles(tmp_path, {"A": _info()})
        assert out["read"] == 1 and out["reused"] == 0
        assert out["calibration"] == first["calibration"]

    def test_a_save_that_fails_costs_nothing(self, tmp_path, monkeypatch, caplog):
        _candle_file(tmp_path, "A", [_c(CLOSE_TS - 7200, 0.5, 0.5)])

        def boom(path, summary):
            raise OSError("disk full")

        monkeypatch.setattr(eda, "_write_summary", boom)
        out = eda.summarise_candles(tmp_path, {"A": _info()})
        assert out["read"] == 1
        assert any("could not be saved" in r.message for r in caplog.records)


# ─── Inventory ────────────────────────────────────────────────────────────────

class TestInventory:
    def _tree(self, tmp_path):
        _write(tmp_path, "archive", "2026-08-05", [_record("A", "KXFED-1", "2026-08-06T00:00:00Z")])
        _write(tmp_path, "archive", "2026-08-07", [_record("B", "KXFED-1", "2026-08-08T00:00:00Z")])
        (tmp_path / "archive_days" / "notes.txt").write_text("hello")
        for name in ("A", "B", "C"):
            _candle_file(tmp_path, name, [_c(CLOSE_TS - 7200, 0.5, 0.5)])
        depth = tmp_path / config.DEPTH_SNAPSHOTS_DIRNAME
        depth.mkdir()
        for stamp in ("20261014T010203Z", "20261016T073903Z", "garbage.json.gz"):
            (depth / f"{stamp}.json.gz" if not stamp.endswith(".gz") else depth / stamp
             ).write_bytes(b"x")
        corpus = tmp_path / "settled_markets_2026-01-01_checkpoint-v3.jsonl.gz"
        with historical._DayStreamWriter(corpus, {
                "kind": "assembled", "start_date": "2026-01-01", "prefilter_tag": "checkpoint-v3",
                "assembled_at": "2026-10-01T00:00:00Z",
                "assembly_counts": {"settled": 5}, "other": 1}) as writer:
            writer.write_records([_record("A", "KXFED-1", "2026-08-06T00:00:00Z")])
            writer.commit()
        (tmp_path / "settled_markets_2025-01-01.jsonl.gz").write_bytes(b"this is not gzip")
        (tmp_path / "settled_markets_2024-01-01.json").write_text("{}")
        (tmp_path / "event_titles_v2.json").write_text("{}")
        (tmp_path / "scheduled_closes").mkdir()
        (tmp_path / "scheduled_closes" / "x.json").write_text("{}")
        (tmp_path / "mystery.bin").write_bytes(b"12345")
        (tmp_path / "event_titles.json").write_text("{}")
        (tmp_path / ".DS_Store").write_bytes(b"x")
        (tmp_path / ".claude").mkdir()
        (tmp_path / ".claude" / "settings.json").write_text("{}")

    def _rows(self, tmp_path):
        return {r["name"]: r for r in eda.inventory(tmp_path)}

    def test_a_fake_cache_tree(self, tmp_path):
        self._tree(tmp_path)
        rows = self._rows(tmp_path)
        archive = rows["archive_days"]
        assert archive["kind"] == "folder" and archive["files"] == 3
        assert (archive["first_day"], archive["last_day"]) == ("2026-08-05", "2026-08-07")
        assert archive["bytes"] == sum(f.stat().st_size for f in (tmp_path / "archive_days").iterdir())
        assert archive["description"] == eda.INVENTORY_DESCRIPTIONS["archive_days"]
        candles = rows["candlesticks"]
        assert candles["files"] == 3 and candles["first_day"] is None
        depth = rows[config.DEPTH_SNAPSHOTS_DIRNAME]
        assert depth["detail"] == {"snapshots": 2, "newest": "2026-10-16T07:39:03+00:00"}
        assert depth["files"] == 3
        titles = rows["event_titles_v2.json"]
        assert titles["kind"] == "file" and titles["bytes"] == 2
        assert titles["description"] == eda.INVENTORY_DESCRIPTIONS["event_titles_v2.json"]

    def test_unknown_entries_are_listed_not_skipped(self, tmp_path):
        self._tree(tmp_path)
        rows = self._rows(tmp_path)
        for name in ("scheduled_closes", "mystery.bin"):
            assert rows[name]["description"] == eda.INVENTORY_NOT_DESCRIBED
        assert rows["scheduled_closes"]["files"] == 1
        assert rows["mystery.bin"]["bytes"] == 5

    def test_hidden_entries_are_left_out_and_the_old_title_file_is_described(self, tmp_path):
        self._tree(tmp_path)
        rows = self._rows(tmp_path)
        assert ".DS_Store" not in rows and ".claude" not in rows
        assert rows["event_titles.json"]["description"] == \
            eda.INVENTORY_DESCRIPTIONS["event_titles.json"]

    def test_the_depth_folder_is_one_scandir_pass(self, tmp_path, monkeypatch):
        self._tree(tmp_path)
        calls = []
        real = os.scandir
        monkeypatch.setattr(os, "scandir", lambda p: (calls.append(str(p)), real(p))[1])
        eda.inventory(tmp_path)
        assert calls.count(str(tmp_path / config.DEPTH_SNAPSHOTS_DIRNAME)) == 1

    def test_an_assembled_corpus_shows_only_its_meta_line(self, tmp_path):
        self._tree(tmp_path)
        rows = self._rows(tmp_path)
        good = rows["settled_markets_2026-01-01_checkpoint-v3.jsonl.gz"]
        assert good["description"] == eda.INVENTORY_CORPUS_DESCRIPTION
        assert good["detail"] == {"start_date": "2026-01-01", "prefilter_tag": "checkpoint-v3",
                                  "assembled_at": "2026-10-01T00:00:00Z",
                                  "assembly_counts": {"settled": 5}}
        bad = rows["settled_markets_2025-01-01.jsonl.gz"]
        assert "meta_error" in bad["detail"] and bad["error"] is None
        legacy = rows["settled_markets_2024-01-01.json"]
        assert legacy["description"] == eda.INVENTORY_LEGACY_CORPUS_DESCRIPTION

    def test_the_corpus_is_not_read_past_its_first_line(self, tmp_path, monkeypatch):
        self._tree(tmp_path)
        reads = []
        real = gzip.open

        class Spy:
            def __init__(self, fh):
                self.fh = fh

            def readline(self, *a):
                reads.append("line")
                return self.fh.readline(*a)

            def read(self, *a):
                reads.append("read")
                return self.fh.read(*a)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                self.fh.close()

        monkeypatch.setattr(gzip, "open", lambda *a, **k: Spy(real(*a, **k)))
        eda.inventory(tmp_path)
        assert "read" not in reads and reads.count("line") == 2

    def test_the_candle_folder_is_one_scandir_pass(self, tmp_path, monkeypatch):
        self._tree(tmp_path)
        calls = []
        real = os.scandir
        monkeypatch.setattr(os, "scandir", lambda p: (calls.append(str(p)), real(p))[1])
        eda.inventory(tmp_path)
        assert calls.count(str(tmp_path / "candlesticks")) == 1

    def test_an_unreadable_folder_gets_a_row_saying_so(self, tmp_path, monkeypatch):
        self._tree(tmp_path)
        real = os.scandir

        def scandir(path):
            if str(path).endswith("archive_days"):
                raise PermissionError("denied")
            return real(path)

        monkeypatch.setattr(os, "scandir", scandir)
        rows = self._rows(tmp_path)
        assert rows["archive_days"]["error"].startswith("PermissionError")
        assert rows["archive_days"]["files"] is None
        assert rows["candlesticks"]["error"] is None

    def test_a_missing_cache_folder_is_one_error_row(self, tmp_path):
        rows = eda.inventory(tmp_path / "nope")
        assert len(rows) == 1 and rows[0]["error"]

    def test_an_empty_cache_has_no_rows(self, tmp_path):
        assert eda.inventory(tmp_path) == []


# ─── The k view ───────────────────────────────────────────────────────────────

def _obs(event, gap, implied, hit, category="Other"):
    from kalshi_betting import backtester
    return backtester.CalibrationObservation(gap, implied, hit, event, category)


def _sweep(observations, k=0.8):
    from kalshi_betting import backtester
    bucket = backtester._calibration_bucket("POOLED", 0.0, observations)
    calibration = backtester.IntervalCalibration(
        pooled=bucket, buckets=[], excluded_premise_violations=2,
        observations=tuple(observations))
    return SimpleNamespace(calibration=calibration, calibrations_by_band={},
                           primary=SimpleNamespace(k=k, spread_band=(0.0, 1.0)))


OBSERVATIONS = [
    _obs("KXFED-1", 3, 0.10, False),
    _obs("KXFED-1", 5, 0.12, True),
    _obs("KXFED-2", 10, 0.30, True),
    _obs("KXZZ-1", 20, 0.52, False),
    _obs("KXZZ-2", 25, 0.50, True),
    _obs("KXZZ-3", 12, -0.01, False),
]


class TestKhatView:
    def test_the_figures_equal_the_backtest_pages_for_the_same_observations(self):
        from kalshi_betting import backtester, dashboard
        view = eda.khat_view(_sweep(OBSERVATIONS), series_categories={
            "KXFED": ("Economics", ("Fed",))})
        pooled = backtester._calibration_bucket("POOLED", 0.0, tuple(OBSERVATIONS))
        low, high = dashboard._khat_interval(
            pooled.empirical_k, sum(o.implied for o in OBSERVATIONS),
            dashboard._khat_cluster_sums(OBSERVATIONS), 5)
        assert view["pooled"]["k"] == pooled.empirical_k
        assert view["pooled"]["rate"] == pooled.realised_rate
        assert view["pooled"]["implied"] == pooled.mean_implied
        assert (view["pooled"]["ci_low"], view["pooled"]["ci_high"]) == (low, high)
        assert view["pooled"]["n"] == 6 and view["pooled"]["events"] == 5
        assert view["k"] == 0.8 and view["observations"] == 6
        assert view["confidence_level"] == config.KHAT_CONFIDENCE_LEVEL
        assert view["excluded_premise_violations"] == 2
        # The page's own k-hat group for the whole population gives the same interval text
        stat = dashboard._khat_stat(OBSERVATIONS)
        assert stat["ci"] == f"95% CI: {low:.3f}–{high:.3f}"

    def test_mid_spread_bins_by_hand(self):
        view = eda.khat_view(_sweep(OBSERVATIONS))
        assert view["non_positive"] == 1
        bins = {b["lo"]: b for b in view["bins"]}
        assert sorted(bins) == [0.1, 0.3, 0.5]
        first = bins[0.1]
        assert (first["n"], first["in_between"], first["rate"]) == (2, 1, 0.5)
        assert first["hi"] == 0.15 and first["implied"] == pytest.approx(0.11)
        assert bins[0.3]["n"] == 1 and bins[0.3]["rate"] == 1.0
        # 0.50 is on a bin edge and goes in the bin above it
        assert (bins[0.5]["n"], bins[0.5]["in_between"]) == (2, 1)

    def test_gap_bands_and_categories(self):
        from kalshi_betting import backtester, dashboard
        view = eda.khat_view(_sweep(OBSERVATIONS), series_categories={
            "KXFED": ("Economics", ("Fed",))})
        assert [b["label"] for b in view["gap_bands"]] == ["0-7d", "8-15d", "16-30d"]
        zero_to_seven = view["gap_bands"][0]
        expect = backtester._calibration_bucket("0-7d", 0.0, OBSERVATIONS[:2])
        assert (zero_to_seven["n"], zero_to_seven["k"]) == (2, expect.empirical_k)
        assert zero_to_seven["events"] == 1
        # A single event gives no interval; two or more do
        assert zero_to_seven["ci_low"] is None
        by_name = {c["label"]: c for c in view["categories"]}
        assert sorted(by_name) == ["Economics", "Other"]
        assert by_name["Economics"]["n"] == 3 and by_name["Economics"]["events"] == 2
        members = OBSERVATIONS[3:]
        assert by_name["Other"]["n"] == 3
        low, high = dashboard._khat_interval(
            by_name["Other"]["k"], sum(o.implied for o in members),
            dashboard._khat_cluster_sums(members), 3)
        assert (by_name["Other"]["ci_low"], by_name["Other"]["ci_high"]) == (low, high)

    def test_nothing_to_show_is_none(self):
        assert eda.khat_view(None) is None
        assert eda.khat_view(SimpleNamespace(
            calibration=None, calibrations_by_band={},
            primary=SimpleNamespace(k=0.8, spread_band=(0.0, 1.0)))) is None
        assert eda.khat_view(_sweep([])) is None

    def test_a_non_finite_figure_becomes_none_and_the_view_is_strict_json(self):
        view = eda.khat_view(_sweep([_obs("KXA-1", 3, float("nan"), True),
                                     _obs("KXA-2", 4, 0.2, False)]))
        json.dumps(view, allow_nan=False)
        assert view["pooled"]["k"] is None and view["pooled"]["implied"] is None

    def test_a_missing_k_is_none(self):
        sweep = _sweep(OBSERVATIONS)
        sweep.primary = SimpleNamespace(k=float("nan"), spread_band=(0.0, 1.0))
        assert eda.khat_view(sweep)["k"] is None

    def test_save_and_load_round_trip(self, tmp_path):
        view = eda.khat_view(_sweep(OBSERVATIONS))
        eda.save_khat(tmp_path, view)
        path = tmp_path / config.EDA_SUMMARY_DIRNAME / "khat.json"
        assert path.is_file() and not list(path.parent.glob("*.tmp"))
        assert eda.load_khat(tmp_path) == view

    def test_nan_is_never_written(self, tmp_path):
        view = eda.khat_view(_sweep(OBSERVATIONS))
        eda.save_khat(tmp_path, view)
        before = (tmp_path / config.EDA_SUMMARY_DIRNAME / "khat.json").read_text()
        view["pooled"]["k"] = math.nan
        with pytest.raises(ValueError):
            eda.save_khat(tmp_path, view)
        assert (tmp_path / config.EDA_SUMMARY_DIRNAME / "khat.json").read_text() == before
        assert "NaN" not in before

    @pytest.mark.parametrize("content", [b"", b"[1]", b"{not json", b'{"k": NaN}', None])
    def test_a_missing_or_unreadable_file_loads_as_none(self, tmp_path, content):
        if content is not None:
            path = tmp_path / config.EDA_SUMMARY_DIRNAME / "khat.json"
            path.parent.mkdir(parents=True)
            path.write_bytes(content)
        assert eda.load_khat(tmp_path) is None

    def test_importing_eda_loads_neither_pandas_nor_plotly(self):
        import subprocess
        import sys
        root = str(Path(eda.__file__).parent.parent)
        code = (f"import sys; sys.path.insert(0, {root!r}); import kalshi_betting.eda; "
                "bad = [m for m in ('pandas', 'plotly', 'kalshi_betting.dashboard', "
                "'kalshi_betting.backtester') if m in sys.modules]; "
                "sys.exit(1 if bad else 0)")
        assert subprocess.run([sys.executable, "-c", code], cwd=root).returncode == 0


# ─── The Data page ────────────────────────────────────────────────────────────

class _Dom(HTMLParser):
    """Every element with an id (its tag and attributes) and every table's rows of cell text."""

    def __init__(self, text):
        super().__init__()
        self.ids = {}
        self.order = []
        self.tables = []
        self._row = None
        self._cell = None
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids[attrs["id"]] = (tag, attrs)
            self.order.append(attrs["id"])
        if tag == "table":
            self.tables.append([])
        elif tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = ""

    def handle_data(self, data):
        if self._cell is not None:
            self._cell += data

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            self._row.append(self._cell)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.tables[-1].append(self._row)
            self._row = None


def _page_cache(tmp_path):
    """A small cache: two day slices and two candle files that join to settled markets."""
    cache = tmp_path / "cache"
    _standard_cache(cache)
    before = CLOSE_TS - 7200
    _candle_file(cache, "FED-A", [_c(before, 0.62, 0.40, 10)])
    _candle_file(cache, "FED-B", [_c(before, 0.30, 0.72, 5)])
    return cache


def _page(tmp_path, cache=None, **kwargs):
    """Build the page into the test's folder and return (its path, its text)."""
    cache = cache if cache is not None else _page_cache(tmp_path)
    out = tmp_path / "out" / "eda_dashboard.html"
    kwargs.setdefault("series_categories", SERIES)
    path = eda.build_eda_page(cache_dir=cache, out_path=out, workers=1, **kwargs)
    assert path == out
    return path, out.read_text(encoding="utf-8")


def _config_of(text):
    """The switches and panels the page's script was given."""
    script = re.search(r'<script id="eda-script">(.*?)</script>', text, re.S).group(1)
    return json.loads(re.search(r"var cfg = (\{.*?\});\n", script, re.S).group(1))


class TestDataPage:
    def test_the_six_sections_come_in_order(self, tmp_path):
        _, text = _page(tmp_path)
        ids = _Dom(text).order
        wanted = ["sec-inventory", "sec-universe", "sec-calibration", "sec-ladders",
                  "sec-quality", glossary.TERMS_ANCHOR]
        assert [i for i in ids if i in wanted] == wanted
        headings = re.findall(r"<h2>(.*?)</h2>", text)
        assert headings == [eda.HEADING_INVENTORY, eda.HEADING_UNIVERSE,
                            eda.HEADING_CALIBRATION, eda.HEADING_LADDERS, eda.HEADING_QUALITY]

    def test_the_header_names_the_page_the_glossary_and_the_build(self, tmp_path):
        _, text = _page(tmp_path)
        assert "<h1>Data</h1>" in text
        assert glossary.terms_link_html(glossary.PAGE_DATA) in text
        built = re.search(r'<p class="note" id="built">(.*?)</p>', text).group(1)
        assert re.fullmatch(r"Built \d{4}-\d\d-\d\d \d\d:\d\d UTC from 2 day files "
                            r"\(7 records\)\.", built)
        assert config.LIVE_DASHBOARD_PLOTLY_URL in text
        assert f'integrity="{config.LIVE_DASHBOARD_PLOTLY_SRI}" crossorigin="anonymous"' in text

    def test_the_terms_are_the_data_pages_own(self, tmp_path):
        _, text = _page(tmp_path)
        terms = text[text.index(f'id="{glossary.TERMS_ANCHOR}"'):]
        data_terms = {t for t, _m, _r in glossary.terms_for(glossary.PAGE_DATA)}
        for term in data_terms:
            assert f">{html.escape(term)}<" in terms
        backtest_only = {t for t, _m, _r in glossary.terms_for(glossary.PAGE_BACKTEST)} - data_terms
        assert "Fills" in backtest_only
        for term in backtest_only:
            assert f">{html.escape(term)}<" not in terms

    def test_every_switch_has_its_buttons_and_exactly_one_view_shows(self, tmp_path):
        _, text = _page(tmp_path)
        dom = _Dom(text)
        cfg = _config_of(text)
        assert set(cfg["switches"]) == {"combo-universe", "combo-quality", "lead", "price"}
        labels = {name: [] for name in cfg["switches"]}
        for name, sw in cfg["switches"].items():
            for option, button in sw["options"].items():
                tag, attrs = dom.ids[button]
                assert tag == "button" and attrs["type"] == "button"
                assert attrs["aria-pressed"] == ("true" if option == sw["default"] else "false")
                labels[name].append(re.search(rf'id="{button}"[^>]*>(.*?)</button>',
                                              text).group(1))
        assert labels["combo-universe"] == ["Non-combo", "Combo", "All"]
        assert labels["combo-quality"] == ["Non-combo", "Combo", "All"]
        assert labels["lead"] == ["1 h", "24 h", "7 d"]
        assert labels["price"] == ["Yes ask", "Midpoint"]
        assert len(cfg["panels"]) == 3
        for panel in cfg["panels"]:
            shown = [k for k, v in panel["views"].items() if "hidden" not in dom.ids[v["div"]][1]]
            defaults = "|".join(cfg["switches"][s]["default"] for s in panel["switches"])
            assert shown == [defaults]
            for view in panel["views"].values():
                for chart in view["charts"]:
                    assert chart in dom.ids
        keys = {tuple(p["switches"]): sorted(p["views"]) for p in cfg["panels"]}
        assert keys[("combo-universe",)] == ["all", "combo", "noncombo"]
        assert keys[("lead", "price")] == sorted(f"{h}|{p}" for h in (1, 24, 168)
                                                 for p in ("ask", "mid"))

    def test_the_page_script_writes_no_text(self, tmp_path):
        _, text = _page(tmp_path)
        script = re.search(r'<script id="eda-script">(.*?)</script>', text, re.S).group(1)
        assert "innerHTML" not in script and "textContent" not in script
        assert len(script.splitlines()) < 45

    def test_the_inventory_lists_the_cache_with_plain_sizes(self, tmp_path):
        _, text = _page(tmp_path)
        table = _Dom(text).tables[0]
        assert table[0] == list(eda.INVENTORY_HEADERS)
        by_name = {row[0]: row for row in table[1:]}
        assert set(by_name) == {"archive_days/", "live_days/", "candlesticks/", "eda_days/"}
        assert by_name["candlesticks/"][1] == "2"
        assert by_name["archive_days/"][3] == "2026-08-05 to 2026-08-05"
        assert re.fullmatch(r"\d+(\.\d)? (B|KB|MB|GB)", by_name["live_days/"][2])
        assert eda.INVENTORY_NOTE in html.unescape(text)

    @pytest.mark.parametrize("count, shown", [
        (0, "0 B"), (999, "999 B"), (1000, "1.0 KB"), (28_400_000, "28.4 MB"),
        (24_600_000_000, "24.6 GB"), (None, "—"), (-1, "—"), (True, "—")])
    def test_sizes_are_written_plainly(self, count, shown):
        assert eda._human_bytes(count) == shown

    def test_histogram_buckets_are_named_from_the_config_edges(self):
        life = eda._bucket_labels(config.EDA_LIFETIME_EDGES_HOURS, eda.LIFETIME_FIRST_BUCKET)
        assert life == ["under 1 h", "1 h to 6 h", "6 h to 24 h", "24 h to 3 d", "3 d to 7 d",
                        "7 d to 30 d", "30 d or more"]
        lag = eda._bucket_labels(config.EDA_LAG_EDGES_HOURS, eda.LAG_FIRST_BUCKET)
        assert lag[0] == "before close" and lag[-1] == "7 d or more"
        assert len(life) == len(config.EDA_LIFETIME_EDGES_HOURS) + 1
        assert len(lag) == len(config.EDA_LAG_EDGES_HOURS) + 1

    def test_the_coverage_gap_is_shown_prominently(self, tmp_path):
        _, text = _page(tmp_path)
        assert '<p class="warn">Gap: the archive runs to 2026-08-17 but the first live day' in text

    def test_no_gap_is_a_grey_note(self, tmp_path):
        cache = tmp_path / "cache"
        _write(cache, "archive", "2026-08-05",
               [_record("FED-A", "KXFED-26AUG", "2026-08-13T12:00:00Z")])
        _write(cache, "live", "2026-08-17",
               [_record("FED-D", "KXFED-26SEP", "2026-08-17T01:00:00Z")])
        _, text = _page(tmp_path, cache)
        assert '<p class="note">No gap: the archive runs to 2026-08-17' in text
        assert "Gap: the archive" not in text

    def test_calibration_shows_the_markets_the_candles_priced(self, tmp_path):
        _, text = _page(tmp_path)
        tables = _Dom(text).tables
        brier = [t for t in tables if t[0] == list(eda.BRIER_HEADERS)]
        assert len(brier) == 2        # the 1 h ask and midpoint (the 24 h and 7 d have no price)
        assert brier[0][1][0] == "All" and brier[0][1][1] == "2"
        coverage = next(t for t in tables if t[0] == list(eda.COVERAGE_HEADERS))
        assert coverage[1] == ["1 h before close", "2", "2", "0", "0", "0", "0"]
        assert coverage[3][0] == "7 d before close"
        flat = html.unescape(text)
        for note in (eda.CALIBRATION_SAMPLE_NOTE, eda.CALIBRATION_ASK_NOTE,
                     eda.CALIBRATION_CLOSE_NOTE):
            assert note in flat

    def test_the_volume_chart_says_how_many_files_carry_volume(self, tmp_path):
        _, text = _page(tmp_path)
        assert "2 of the 2 candle files joined to a settled non-combo market carry it." in text
        assert 'id="eda-volume"' in text

    def test_a_category_name_is_escaped(self, tmp_path):
        listing = {"KXFED": ("Econ<b>omics", ("Fed",)), "KXLAD": ("Science", ("Space",))}
        _, text = _page(tmp_path, series_categories=listing)
        assert "Econ<b>omics" not in text
        assert "Econ&lt;b&gt;omics" in text

    def test_the_page_is_small_and_leaves_no_temporary_file(self, tmp_path):
        path, text = _page(tmp_path)
        assert len(text) < 200_000
        assert not list(path.parent.glob("*.tmp"))

    def test_a_default_page_goes_where_the_config_says(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path / "root")
        monkeypatch.setattr(config, "EDA_DASHBOARD_FILENAME", "elsewhere.html")
        monkeypatch.setattr(historical, "CACHE_DIR", _page_cache(tmp_path))
        path = eda.build_eda_page(series_categories=SERIES)
        assert path == tmp_path / "root" / "elsewhere.html" and path.is_file()


def _capture_charts(monkeypatch):
    """Record each chart the page builds as chart id -> (traces, keyword arguments)."""
    seen = {}
    real = eda._chart

    def spy(go, title, note, traces, chart_id, ids, **kwargs):
        seen[chart_id] = (traces, dict(kwargs, title=title, note=note))
        return real(go, title, note, traces, chart_id, ids, **kwargs)

    monkeypatch.setattr(eda, "_chart", spy)
    return seen


def _counts_for(rows):
    """merge()-shaped counts from (week, category, n, yes, no) rows, non-combo."""
    counts = {}
    for week, category, n, yes, no in rows:
        c = eda._new_counts()
        c.update(n=n, yes=yes, no=no)
        counts[eda.KEY_SEPARATOR.join((week, "0", category, "T"))] = c
    return counts


class TestDataPageCharts:
    def _view(self, monkeypatch, rows):
        seen = _capture_charts(monkeypatch)
        go = eda._plotly()
        agg = eda._aggregate(_counts_for(rows), ("0",))
        eda._universe_view(go, agg, "noncombo", [])
        return seen

    def test_a_month_with_few_markets_is_left_out_of_the_lines(self, monkeypatch):
        minimum = config.EDA_MIN_MARKETS_PER_POINT
        seen = self._view(monkeypatch, [
            ("2026-08-03", "Sports", minimum, 10, minimum - 10),          # exactly enough
            ("2026-09-07", "Sports", minimum - 1, 5, minimum - 6),        # one short: dropped
            ("2026-08-03", "Rare", 4, 4, 0),                              # a 100% spike: dropped
        ])
        traces, kwargs = seen["eda-noncombo-yes-time"]
        by_name = {t.name: t for t in traces}
        assert list(by_name["Sports"].x) == ["2026-08"]
        assert list(by_name["Rare"].x) == []
        assert str(minimum) in kwargs["note"] and "left out" in kwargs["note"]

    def test_the_share_by_category_is_a_sorted_horizontal_bar_with_n_in_the_hover(
            self, monkeypatch):
        seen = self._view(monkeypatch, [
            ("2026-08-03", "Small", 10, 5, 5), ("2026-08-03", "Big", 100, 20, 80),
            ("2026-08-03", "Open only", 7, 0, 0),         # nothing settled Yes or No
        ])
        (bar,), kwargs = seen["eda-noncombo-yes"]
        assert bar.orientation == "h"
        assert list(bar.y) == ["Big", "Small"]            # largest first; no empty category
        assert list(bar.x) == [20.0, 50.0] and list(bar.customdata) == [100, 10]
        assert "%{customdata" in bar.hovertemplate
        assert list(bar.text) == ["20.0% of 100", "50.0% of 10"]
        assert kwargs["yaxis"]["autorange"] == "reversed"

    def test_the_census_and_the_volume_are_horizontal_too(self, tmp_path, monkeypatch):
        seen = _capture_charts(monkeypatch)
        cache = _page_cache(tmp_path)
        _page(tmp_path, cache)
        volume, _ = seen["eda-volume"]
        assert volume[0].orientation == "h" and list(volume[0].x) == [15.0]
        assert all(t.orientation == "h" for t in seen["eda-census"][0])

    def test_the_legend_sits_above_the_plot_clear_of_the_axis_titles(self):
        go = eda._plotly()
        one = eda._chart(go, "T", None, [go.Bar(x=[1], y=[1])], "c1", [], x_title="Week")
        many = eda._chart(go, "T", None, [go.Bar(x=[1], y=[i]) for i in range(9)], "c2", [],
                          x_title="Week")
        assert '"yanchor":"bottom"' in many
        top = lambda html_: int(re.search(r'"margin":\{[^}]*"t":(\d+)', html_).group(1))  # noqa: E731
        assert top(many) >= top(one) + 2 * eda._LEGEND_ROW_PX

    def test_the_ladder_chart_counts_ladders(self, tmp_path, monkeypatch):
        seen = _capture_charts(monkeypatch)
        cache = tmp_path / "cache"
        _write(cache, "live", "2026-08-18", [
            _record("L-1", "KXLAD-1", "2026-08-18T01:00:00Z", title="Will it launch by Sep 1, 2026?"),
            _record("L-2", "KXLAD-1", "2026-08-18T02:00:00Z", title="Will it launch by Oct 1, 2026?")])
        _page(tmp_path, cache)
        assert seen["eda-ladders"][1]["y_title"] == "Ladders"

    @pytest.mark.parametrize("count, shown", [(1, "1 snapshot"), (2, "2 snapshots"),
                                              (1500, "1,500 snapshots")])
    def test_the_inventory_counts_snapshots_in_the_singular_too(self, count, shown):
        row = {"detail": {"snapshots": count}}
        assert eda._inventory_extra(row) == shown

    def test_the_terms_come_last_and_hold_only_the_data_pages_terms(self, tmp_path):
        _, text = _page(tmp_path)
        dom = _Dom(text)
        assert dom.order.index(glossary.TERMS_ANCHOR) > dom.order.index("sec-quality")
        terms_table = dom.tables[-1]
        assert [r[0] for r in terms_table[1:]] == [t for t, _m, _r in
                                                   glossary.terms_for(glossary.PAGE_DATA)]

    def test_the_width_note_sits_under_the_reliability_chart(self, tmp_path):
        _, text = _page(tmp_path)
        assert ("A price counts only where the gap between the Yes ask and the Yes bid was "
                "10c or less (a missing bid counts as 0), so one-sided quotes are left out."
                ) in html.unescape(text)
        assert "Wide book" in text


class TestDataPageSmallThings:
    def test_a_week_with_no_markets_is_skipped_not_divided_by(self):
        agg = eda._aggregate(_counts_for([("2026-08-03", "Sports", 10, 5, 5)]), ("0",))
        agg["quality"]["2026-08-10"] = [0] * (1 + len(eda.QUALITY_SERIES))
        html_ = eda._quality_view(eda._plotly(), agg, "noncombo", [])
        assert "2026-08-03" in html_ and "2026-08-10" not in html_

    def test_the_temporary_file_goes_whatever_stops_the_write(self, tmp_path, monkeypatch):
        def boom(src, dst):
            raise RuntimeError("stopped")

        monkeypatch.setattr(os, "replace", boom)
        out = tmp_path / "page.html"
        with pytest.raises(RuntimeError):
            eda._write_page(out, "x")
        assert not out.exists() and not list(tmp_path.glob("*.tmp"))

    def test_the_volume_note_counts_the_joined_non_combo_files(self, tmp_path):
        cache = _page_cache(tmp_path)
        _candle_file(cache, "UNJOINED", [_c(CLOSE_TS - 7200, 0.5, 0.5, 3)])
        _candle_file(cache, "COMBO-1", [_c(CLOSE_TS - 7200, 0.5, 0.5, 3)])
        _, text = _page(tmp_path, cache)
        # FED-A and FED-B carry volume; UNJOINED has no market, COMBO-1 is a combo
        assert "2 of the 2 candle files joined" in text

    def test_the_coverage_caption_explains_the_overlap(self, tmp_path):
        _, text = _page(tmp_path)
        flat = html.unescape(text)
        assert ("Stale ones are also counted under No ask, and every market with no ask is "
                "also under No midpoint; a wide book is in neither. Priced (ask) is Markets "
                "minus No ask minus Wide book.") in flat
        assert "Priced (ask)" in eda.COVERAGE_HEADERS

    @pytest.mark.parametrize("hours, short, long", [
        (1, "1 h", "1 h before close"), (24, "24 h", "24 h before close"),
        (168, "7 d", "7 d before close")])
    def test_leads_are_named_in_hours_then_days(self, hours, short, long):
        assert eda._short_lead(hours) == short and eda._lead_label(hours) == long

    def test_the_largest_categories_note_names_the_real_count(self, tmp_path):
        _, text = _page(tmp_path)
        assert f"the {eda._TOP_CATEGORIES} largest" in text


class TestDataPageKView:
    def test_with_no_sweep_and_nothing_saved_it_says_so(self, tmp_path):
        _, text = _page(tmp_path)
        assert eda.KHAT_NONE in text

    def test_the_table_equals_the_figures_of_the_k_view(self, tmp_path):
        sweep = _sweep(OBSERVATIONS)
        cache = _page_cache(tmp_path)
        _, text = _page(tmp_path, cache, sweep=sweep)
        view = eda.khat_view(sweep, series_categories=SERIES)
        rows = {row[0]: row for t in _Dom(text).tables if t[0][0] == "Group" for row in t[1:]}

        def expected(stat, label):
            low, high = stat["ci_low"], stat["ci_high"]
            return [label, f"{stat['n']:,}", f"{stat['events']:,}",
                    "—" if stat["k"] is None else f"{stat['k']:.3f}",
                    "—" if low is None or high is None else f"{low:.3f} to {high:.3f}"]

        assert rows[eda.KHAT_POOLED] == expected(view["pooled"], eda.KHAT_POOLED)
        assert view["gap_bands"] and view["categories"]
        for stat in view["gap_bands"] + view["categories"]:
            assert rows[stat["label"]] == expected(stat, stat["label"])
        headers = [t[0] for t in _Dom(text).tables if t[0][0] == "Group"][0]
        assert headers == ["Group", "Pairs", "Events", "k̂", "95% CI"]
        assert "mid spread of 0 or less left out" in text
        assert 'id="eda-khat"' in text

    def test_a_backtest_saves_its_figures_for_a_later_rebuild(self, tmp_path):
        cache = _page_cache(tmp_path)
        _page(tmp_path, cache, sweep=_sweep(OBSERVATIONS))
        saved = eda.load_khat(cache)
        assert saved == json.loads(json.dumps(
            eda.khat_view(_sweep(OBSERVATIONS), series_categories=SERIES)))
        _, text = _page(tmp_path, cache)        # no sweep: the saved view is shown
        assert eda.KHAT_NONE not in text and 'id="eda-khat"' in text

    def test_a_malformed_saved_view_costs_the_k_block_alone(self, tmp_path, caplog):
        cache = _page_cache(tmp_path)
        path = cache / config.EDA_SUMMARY_DIRNAME / "khat.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": config.EDA_KHAT_VERSION, "k": 0.8,
                                    "confidence_level": None, "observations": 3,
                                    "non_positive": 0, "pooled": {"label": "POOLED", "n": 3},
                                    "bins": [{"implied": 0.1}], "gap_bands": [],
                                    "categories": []}))
        with caplog.at_level(logging.WARNING):
            _, text = _page(tmp_path, cache)
        flat = html.unescape(text)
        assert 'id="eda-census"' in text                       # the rest of the section stays
        assert "The k figures could not be drawn (" in flat
        assert 'id="eda-khat"' not in text and "sec-quality" in text
        assert [r for r in caplog.records if "k view could not be drawn" in r.getMessage()]

    def test_the_saved_view_carries_a_version_and_a_mismatch_is_not_shown(self, tmp_path):
        view = eda.khat_view(_sweep(OBSERVATIONS), series_categories=SERIES)
        eda.save_khat(tmp_path, view)
        path = tmp_path / config.EDA_SUMMARY_DIRNAME / "khat.json"
        saved = json.loads(path.read_text())
        assert saved["version"] == config.EDA_KHAT_VERSION
        assert eda.load_khat(tmp_path) == json.loads(json.dumps(view))   # version taken off
        for changed in ({**saved, "version": config.EDA_KHAT_VERSION + 1},
                        {k: v for k, v in saved.items() if k != "version"}):
            path.write_text(json.dumps(changed))
            assert eda.load_khat(tmp_path) is None

    def test_the_wording_census_and_the_ladder_count(self, tmp_path):
        cache = tmp_path / "cache"
        _write(cache, "live", "2026-08-18", [
            _record("L-1", "KXLAD-1", "2026-08-18T01:00:00Z", title="Will it launch by Sep 1, 2026?"),
            _record("L-2", "KXLAD-1", "2026-08-18T02:00:00Z", title="Will it launch by Oct 1, 2026?"),
            _record("L-3", "KXFED-1", "2026-08-18T03:00:00Z", title="Q?")])
        _, text = _page(tmp_path, cache)
        assert 'id="eda-census"' in text and 'id="eda-ladders"' in text
        assert "Total: 1." in text


class TestDataPageNeverRaises:
    def test_a_missing_cache_folder_still_gives_a_page(self, tmp_path):
        out = tmp_path / "out" / "page.html"
        path = eda.build_eda_page(cache_dir=tmp_path / "nowhere", out_path=out,
                                  series_categories=SERIES)
        assert path == out
        text = out.read_text()
        assert "0 day files (0 records)" in text
        assert '<section id="sec-inventory">' in text

    def test_a_summary_step_that_raises_is_named_and_the_inventory_stays(self, tmp_path,
                                                                        monkeypatch, caplog):
        def boom(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(eda, "load_or_build_summaries", boom)
        with caplog.at_level(logging.WARNING):
            _, text = _page(tmp_path)
        flat = html.unescape(text)
        assert eda.FAILURES_HEAD in flat
        assert "Counting the day files: RuntimeError: boom" in flat
        assert "The day files could not be counted" in flat
        assert eda.NO_DAY_COUNTS in flat
        assert eda.NO_CANDLES in flat
        assert "archive_days/" in text                      # the inventory is still there
        records = [r for r in caplog.records if "counting the day files failed" in r.getMessage()]
        assert len(records) == 1 and records[0].exc_info

    def test_a_candle_step_that_raises_leaves_the_day_counts(self, tmp_path, monkeypatch):
        def boom(*args, **kwargs):
            raise OSError("disk gone")

        monkeypatch.setattr(eda, "summarise_candles", boom)
        _, text = _page(tmp_path)
        flat = html.unescape(text)
        assert "Reading the candle files: OSError: disk gone" in flat
        assert "from 2 day files (7 records)" in flat
        assert eda.NO_CANDLES in flat
        assert 'id="view-universe-noncombo"' in text and eda.NO_DAY_COUNTS not in flat

    def test_a_section_that_raises_leaves_the_others(self, tmp_path, monkeypatch):
        def boom(*args, **kwargs):
            raise ValueError("bad chart")

        monkeypatch.setattr(eda, "_calibration_section", boom)
        _, text = _page(tmp_path)
        flat = html.unescape(text)
        assert eda.SECTION_FAILED.format(section=eda.HEADING_CALIBRATION,
                                         reason="ValueError: bad chart") in flat
        assert "sec-universe" in text and 'id="eda-khat"' not in text and "sec-quality" in text
        # The failed section registered no switch the script would look for
        assert set(_config_of(text)["switches"]) == {"combo-universe", "combo-quality"}

    def test_without_the_chart_library_the_charts_say_so_and_the_inventory_stays(
            self, tmp_path, monkeypatch):
        def boom():
            raise ImportError("no plotly")

        monkeypatch.setattr(eda, "_plotly", boom)
        _, text = _page(tmp_path)
        flat = html.unescape(text)
        assert "Loading the chart library: ImportError: no plotly" in flat
        assert flat.count("the chart library could not be loaded") == 4
        assert "archive_days/" in text

    def test_a_listing_that_cannot_be_read_is_named(self, tmp_path, monkeypatch):
        def boom(client):
            raise RuntimeError("no listing")

        monkeypatch.setattr(historical, "load_series_categories", boom)
        out = tmp_path / "page.html"
        eda.build_eda_page(cache_dir=_page_cache(tmp_path), out_path=out)
        assert "Reading Kalshi&#x27;s series listing: RuntimeError: no listing" in out.read_text()

    def test_a_page_that_cannot_be_written_returns_none(self, tmp_path):
        blocker = tmp_path / "file"
        blocker.write_text("x")
        assert eda.build_eda_page(cache_dir=_page_cache(tmp_path),
                                  out_path=blocker / "page.html",
                                  series_categories=SERIES) is None

    def test_nothing_the_build_throws_escapes(self, tmp_path, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("everything is wrong")

        monkeypatch.setattr(eda, "_render", boom)
        out = tmp_path / "page.html"
        assert eda.build_eda_page(cache_dir=tmp_path, out_path=out,
                                  series_categories=SERIES) == out
        assert eda.PAGE_FAILED.format(reason="RuntimeError: everything is wrong") in \
            html.unescape(out.read_text())


class TestDataPageMain:
    def test_main_builds_the_page_and_prints_where(self, tmp_path, monkeypatch, capsys):
        cache = _page_cache(tmp_path)
        eda.save_khat(cache, eda.khat_view(_sweep(OBSERVATIONS), series_categories=SERIES))
        monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path / "root")
        monkeypatch.setattr(historical, "CACHE_DIR", cache)
        monkeypatch.setattr(eda, "_configure_logging", lambda: None)
        assert eda.main(["--workers", "1"]) == 0
        page = tmp_path / "root" / config.EDA_DASHBOARD_FILENAME
        assert capsys.readouterr().out.strip() == str(page)
        text = page.read_text()
        assert eda.KHAT_NONE not in text and 'id="eda-khat"' in text     # the saved view

    def test_main_reports_a_page_that_could_not_be_written(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path / "root")
        monkeypatch.setattr(eda, "_configure_logging", lambda: None)
        monkeypatch.setattr(eda, "build_eda_page", lambda **kw: None)
        assert eda.main([]) == 1
        assert eda.MAIN_NO_PAGE in capsys.readouterr().out

    def test_main_refuses_fewer_than_one_worker(self, capsys):
        with pytest.raises(SystemExit) as stopped:
            eda.main(["--workers", "0"])
        assert stopped.value.code == 2

    def test_the_default_worker_count_respects_the_config_maximum(self, monkeypatch):
        monkeypatch.setattr(os, "cpu_count", lambda: 64)
        assert eda._default_workers() == config.EDA_MAX_WORKERS
        monkeypatch.setattr(os, "cpu_count", lambda: 1)
        assert eda._default_workers() == 1

    def test_the_log_is_a_rotating_file_beside_the_other_logs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
        seen = {}
        monkeypatch.setattr(logging, "basicConfig", lambda **kw: seen.update(kw))
        eda._configure_logging()
        (handler,) = seen["handlers"]
        assert isinstance(handler, logging.handlers.RotatingFileHandler)
        assert Path(handler.baseFilename) == tmp_path / "kalshi_eda.log"
        assert not (tmp_path / "kalshi_eda.log").exists()        # delay=True


# ─── The page script, run under the live tab's JavaScript stand-in ────────────

_HARNESS = Path(__file__).parent / "js" / "live_harness.js"
_JSC = Path("/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc")


def _js_runtime():
    """node when installed, else macOS's JavaScriptCore shell; None when neither is."""
    node = shutil.which("node")
    if node:
        return node
    return str(_JSC) if _JSC.exists() else None


def _run_page_script(tmp_path, text, steps, *, plotly=True):
    """Run the page's script under tests/js/live_harness.js and return the snapshots."""
    runtime = _js_runtime()
    if runtime is None:
        pytest.skip("no JavaScript runtime (node or jsc) to run the page script with")
    ids = sorted(set(re.findall(r'\bid="([^"]+)"', text)))
    script = re.search(r'<script id="eda-script">(.*?)</script>', text, re.S).group(1)
    program = "\n".join([
        _HARNESS.read_text(encoding="utf-8"),
        *(f"__IDS[{json.dumps(i)}] = true;" for i in ids),
        f"__NO_PLOTLY = {json.dumps(not plotly)};",
        "__install();",
        script,
        f"__step({json.dumps(steps)}, 0);",
    ])
    path = tmp_path / "eda_run.js"
    path.write_text(program, encoding="utf-8")
    done = subprocess.run([runtime, str(path)], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


class TestDataPageScript:
    def test_clicking_combo_shows_its_view_hides_the_others_and_resizes_once(self, tmp_path):
        _, text = _page(tmp_path)
        cfg = _config_of(text)
        panel = next(p for p in cfg["panels"] if p["switches"] == ["combo-universe"])
        snap = _run_page_script(tmp_path, text, [["click", "btn-combo-universe-combo"],
                                                 ["snap", "after"]])["after"]
        el = snap["el"]
        assert el["view-universe-combo"]["hidden"] is False
        assert el["view-universe-noncombo"]["hidden"] is True
        assert el["view-universe-all"]["hidden"] is True
        pressed = {o: el[f"btn-combo-universe-{o}"]["attrs"]["aria-pressed"]
                   for o in ("noncombo", "combo", "all") if f"btn-combo-universe-{o}" in el}
        assert pressed == {"noncombo": "false", "combo": "true", "all": "false"}
        resized = [p["id"] for p in snap["plots"] if p["call"] == "resize"]
        assert resized == panel["views"]["combo"]["charts"] and len(set(resized)) == len(resized)
        assert resized                      # the combo view holds charts
        # The quality section's switch is its own
        assert "view-quality-combo" not in el

    def test_the_two_calibration_switches_pick_one_view_together(self, tmp_path):
        _, text = _page(tmp_path)
        snap = _run_page_script(tmp_path, text, [
            ["click", "btn-lead-168"], ["click", "btn-price-mid"], ["snap", "s"]])["s"]
        el = snap["el"]
        shown = [k for k, v in el.items() if k.startswith("view-cal-") and not v["hidden"]]
        assert shown == ["view-cal-168-mid"]
        assert len([k for k in el if k.startswith("view-cal-")]) == 6
        assert el["btn-lead-168"]["attrs"]["aria-pressed"] == "true"
        assert el["btn-lead-1"]["attrs"]["aria-pressed"] == "false"
        assert el["btn-price-mid"]["attrs"]["aria-pressed"] == "true"
        assert el["btn-price-ask"]["attrs"]["aria-pressed"] == "false"

    def test_without_plotly_the_views_still_switch(self, tmp_path):
        _, text = _page(tmp_path)
        snap = _run_page_script(tmp_path, text, [["click", "btn-combo-quality-all"],
                                                 ["snap", "s"]], plotly=False)["s"]
        assert snap["plots"] == []
        assert snap["el"]["view-quality-all"]["hidden"] is False
        assert snap["el"]["view-quality-noncombo"]["hidden"] is True
