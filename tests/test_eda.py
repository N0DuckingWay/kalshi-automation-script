"""
Tests for kalshi_betting/eda.py: each cached day slice summarised once, and the merge.

Every test writes its own day slices with historical._DayStreamWriter into the
test's tmp_path; nothing reads the real backtest_cache/ or the network.
"""
import ast
import json
import os
from concurrent.futures import Future
from datetime import datetime
from pathlib import Path

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

    def test_eda_imports_only_the_standard_library_config_historical_and_scanner(self):
        package = Path(eda.__file__).parent
        names = self._imports(package / "eda.py")
        local = {p.stem for p in package.glob("*.py")}
        project = {n.split(".")[-1] for n in names if n.split(".")[-1] in local
                   or n.startswith("kalshi_betting")}
        assert project == {"config", "historical", "scanner"}
        assert not (names & self._FORBIDDEN)

    def test_no_module_imports_eda(self):
        package = Path(eda.__file__).parent
        for path in package.glob("*.py"):
            if path.name == "eda.py":
                continue
            assert "eda" not in self._imports(path), path.name


@pytest.mark.parametrize("value, bucket", [(-5, 0), (0.5, 0), (1, 1), (719.9, 5), (720, 6)])
def test_bucket_edges(value, bucket):
    assert eda._bucket(value, config.EDA_LIFETIME_EDGES_HOURS) == bucket
