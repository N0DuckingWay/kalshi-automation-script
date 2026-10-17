"""
Tests for kalshi_betting/eda.py: each cached day slice summarised once, and the merge.

Every test writes its own day slices with historical._DayStreamWriter into the
test's tmp_path; nothing reads the real backtest_cache/ or the network.
"""
import ast
import gzip
import json
import math
import os
from concurrent.futures import Future
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from kalshi_betting import config, eda, historical

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

    def test_eda_imports_only_the_standard_library_config_historical_and_scanner(self):
        package = Path(eda.__file__).parent
        tree = ast.parse((package / "eda.py").read_text())
        # Module level only: what a spawned worker loads when it imports eda
        top = set()
        for node in tree.body:
            holder = ast.Module(body=[node], type_ignores=[])
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                top |= self._imports_of(holder)
        assert self._project_names(top) == {"config", "historical", "scanner"}
        assert not (top & self._FORBIDDEN)

    def test_the_only_local_imports_are_khat_views_backtester_and_dashboard(self):
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
        assert found == {"khat_view": {"backtester", "dashboard"}}
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
        assert one["ask"]["bins"][12] == [1, 1] and one["ask"]["n"] == 1
        assert one["ask"]["brier_sum"] == pytest.approx((0.62 - 1) ** 2)
        assert one["mid"]["bins"][12] == [1, 1]
        assert one["mid"]["brier_sum"] == pytest.approx((0.61 - 1) ** 2)
        day = out["calibration"]["24"]["Economics"]
        assert day["ask"]["bins"][6] == [1, 1]
        assert day["mid"]["bins"][5] == [1, 1]
        assert day["mid"]["brier_sum"] == pytest.approx((0.29 - 1) ** 2)
        # Nothing ended 7 days before close: that lead has no price for this market
        assert "Economics" not in out["calibration"]["168"]
        assert out["coverage"]["168"] == {"markets": 1, "no_ask": 1, "no_mid": 1, "stale": 0}
        assert out["coverage"]["1"] == {"markets": 1, "no_ask": 0, "no_mid": 0, "stale": 0}

    def test_a_long_shot_has_an_ask_but_no_midpoint(self, tmp_path):
        before = CLOSE_TS - 7200
        _candle_file(tmp_path, "CLAMP", [_c(before, 0.60, 0.99)])
        _candle_file(tmp_path, "SHOT", [_c(before, 0.04, 0.99)])
        _candle_file(tmp_path, "ONE", [_c(before, 1.0, 0.40)])
        _candle_file(tmp_path, "MISSING", [{"ts": before, "yes_ask_close": 0.5}])
        _candle_file(tmp_path, "GOOD", [_c(before, 0.60, 0.50)])
        out = eda.summarise_candles(tmp_path, {t: _info("no") for t in
                                               ("CLAMP", "SHOT", "ONE", "MISSING", "GOOD")})
        cell = out["calibration"]["1"]["Economics"]
        # CLAMP, SHOT, MISSING and GOOD have a YES ask; only GOOD has a midpoint
        # (0.60 + 0.50) / 2 = 0.55, bin 11, settled No; ONE has neither
        assert cell["ask"]["n"] == 4
        assert cell["ask"]["bins"][0] == [1, 0] and cell["ask"]["bins"][12] == [2, 0]
        assert cell["ask"]["bins"][10] == [1, 0]
        assert cell["ask"]["brier_sum"] == pytest.approx(0.60 ** 2 + 0.04 ** 2 + 0.5 ** 2
                                                         + 0.60 ** 2)
        assert cell["mid"]["n"] == 1 and cell["mid"]["bins"][11] == [1, 0]
        assert cell["mid"]["brier_sum"] == pytest.approx(0.55 ** 2)
        assert out["coverage"]["1"] == {"markets": 5, "no_ask": 1, "no_mid": 4, "stale": 0}

    def test_a_last_candle_older_than_the_maximum_age_gives_no_price(self, tmp_path):
        target = CLOSE_TS - 3600                      # the 1 h lead's moment
        age = int(config.EDA_PRICE_MAX_AGE_HOURS * 3600)
        _candle_file(tmp_path, "STALE", [_c(target - age, 0.60, 0.40)])    # exactly too old
        _candle_file(tmp_path, "FRESH", [_c(target - age + 1, 0.60, 0.40)])
        _candle_file(tmp_path, "NONE", [_c(CLOSE_TS - 10, 0.60, 0.40)])    # nothing before
        out = eda.summarise_candles(tmp_path, {t: _info("yes") for t in
                                               ("STALE", "FRESH", "NONE")})
        assert out["calibration"]["1"]["Economics"]["ask"]["n"] == 1
        assert out["coverage"]["1"] == {"markets": 3, "no_ask": 2, "no_mid": 2, "stale": 1}

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
        assert eco["mid"]["bins"][12] == [2, 1] and eco["mid"]["n"] == 2
        assert eco["mid"]["brier_sum"] == pytest.approx((0.61 - 1) ** 2 + 0.63 ** 2)
        assert eco["ask"]["bins"][12] == [2, 1]
        assert eco["ask"]["brier_sum"] == pytest.approx((0.60 - 1) ** 2 + 0.64 ** 2)
        sci = out["calibration"]["1"]["Science"]
        assert sci["mid"]["bins"][2] == [1, 0]
        assert sci["mid"]["brier_sum"] == pytest.approx(0.10 ** 2)
        assert sci["ask"]["bins"][2] == [1, 0]
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
        assert out["calibration"]["1"]["Science"]["ask"]["bins"][12] == [1, 0]
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
