"""
File: test_depth_model.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Tests for depth_model.py: the trailing-volume feature, fitting the depth
    table (medians, the fallback chain, rows that never fall with distance),
    the synthetic bid ladders and book (the top of the book reproduces the
    quotes and passes scanner's level parser without a warning), the model's
    pickle and digest, load_depth_model's never-raising loading of saved
    snapshots, snapshot() and the command line against a fake exchange, and
    that no live-trading module is imported by or imports this module.

Dependencies:
    Imports kalshi_betting.depth_model, config, historical and scanner. Fully
    offline: every exchange call is replaced, and historical.CACHE_DIR points
    at the test's tmp_path so nothing is written to the real backtest cache.

Notes:
    A snapshot reads candles through historical.fetch_candlesticks, which
    writes a per-ticker cache file, so every snapshot test replaces it.
"""
import ast
import gzip
import importlib
import inspect
import json
import logging
import os
import pickle
import pkgutil
import random
from datetime import UTC, datetime

import pytest

import kalshi_betting
from kalshi_betting import config, depth_model, historical, scanner
from kalshi_betting.depth_model import DepthModel

DISTANCES = config.DEPTH_MODEL_DISTANCES
STAMP = "2026-10-01T16:00:00Z"


def _levels(best, increments):
    """
    Bid levels with `increments` contracts at each table distance below `best`.

    Args:
        best (float): The best bid.
        increments (tuple): Contracts to rest at each of DEPTH_MODEL_DISTANCES
            below it; a 0 leaves that distance out.

    Returns:
        list: [[price string, quantity string], ...], best first.
    """
    return [[f"{round(best - d, 4):.4f}", str(inc)]
            for d, inc in zip(DISTANCES, increments, strict=True) if inc > 0]


def _record(volume, best, increments, *, side="yes", taken=STAMP, ticker="T"):
    """One snapshot record with a single ladder on `side`."""
    record = {"ticker": ticker, "taken_at": taken, "volume_24h": volume, "yes": [], "no": []}
    record[side] = _levels(best, increments)
    return record


def _model(overall, cells=None, volume_rows=None):
    """A hand-built model with the given rows."""
    return DepthModel(cells=cells or {}, volume_rows=volume_rows or {}, overall=overall,
                      snapshots=1, ladders=1, first_taken=STAMP, last_taken=STAMP, digest="x")


def _warnings(caplog):
    """The WARNING-or-worse records captured so far."""
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


class TestConstants:
    """The shipped table shape the module relies on."""

    def test_distances_start_at_the_best_bid_and_grow(self):
        assert config.DEPTH_MODEL_DISTANCES[0] == 0.0
        assert list(config.DEPTH_MODEL_DISTANCES) == sorted(set(config.DEPTH_MODEL_DISTANCES))

    def test_edges_grow(self):
        for edges in (config.DEPTH_MODEL_VOLUME_EDGES, config.DEPTH_MODEL_PRICE_EDGES):
            assert list(edges) == sorted(set(edges))

    def test_the_shipped_values(self):
        assert config.DEPTH_MODEL_VOLUME_EDGES == (1, 100, 1_000, 10_000)
        assert config.DEPTH_MODEL_PRICE_EDGES == (0.05, 0.20, 0.50, 0.80, 0.95)
        assert config.DEPTH_MODEL_QUANTILE == 0.5
        assert config.DEPTH_MODEL_MIN_LADDERS == 20
        assert config.DEPTH_SNAPSHOT_MARKETS == 2_000
        assert config.DEPTH_SNAPSHOTS_DIRNAME == "depth_snapshots"
        assert config.DEPTH_VOLUME_WINDOW_SECONDS == 86_400


class TestVolume24h:
    """The trailing-volume feature the snapshot and the backtest share."""

    WINDOW = config.DEPTH_VOLUME_WINDOW_SECONDS

    @staticmethod
    def _candles(*pairs):
        return [{"ts": ts, "volume": volume} for ts, volume in pairs]

    def test_none_when_no_candle_is_at_or_before_ts(self):
        assert depth_model.volume_24h([], 1_000) is None
        assert depth_model.volume_24h(self._candles((2_000, 5.0)), 1_000) is None

    def test_zero_when_the_market_has_earlier_candles_but_none_in_the_window(self):
        candles = self._candles((1_000, 9.0))
        assert depth_model.volume_24h(candles, 1_000 + self.WINDOW) == 0.0
        assert depth_model.volume_24h(candles, 1_000 + self.WINDOW + 3_600) == 0.0

    def test_the_window_includes_ts_and_excludes_its_far_edge(self):
        ts = 500_000
        candles = self._candles(
            (ts - self.WINDOW, 100.0),      # on the far edge: out
            (ts - self.WINDOW + 1, 1.0),    # just inside
            (ts - 3_600, 2.0),
            (ts, 4.0),                      # at ts: in
            (ts + 1, 1_000.0),              # after ts: out
        )
        assert depth_model.volume_24h(candles, ts) == 7.0

    def test_candles_without_volume_are_skipped(self):
        candles = self._candles((10, 3.0), (20, None), (30, 4.5))
        assert depth_model.volume_24h(candles, 30) == 7.5

    def test_none_when_every_candle_in_the_window_has_no_volume(self):
        ts = 1_000_000
        candles = self._candles((ts - self.WINDOW - 5, 50.0), (ts - 100, None), (ts, None))
        assert depth_model.volume_24h(candles, ts) is None

    def test_a_candle_with_no_volume_key_reads_as_unknown(self):
        assert depth_model.volume_24h([{"ts": 10}], 10) is None
        assert depth_model.volume_24h([{"ts": 10}, {"ts": 20, "volume": 2.0}], 20) == 2.0

    def test_a_nan_volume_is_skipped(self):
        candles = self._candles((10, float("nan")), (20, 2.0))
        assert depth_model.volume_24h(candles, 20) == 2.0

    def test_an_integer_too_large_for_a_float_is_skipped(self):
        candles = self._candles((10, 10**400), (20, 2.0))
        assert depth_model.volume_24h(candles, 20) == 2.0


class TestFit:
    """Fitting the table from snapshot records."""

    def test_a_ladder_becomes_cumulative_contracts_per_distance(self):
        record = {"ticker": "T", "taken_at": STAMP, "volume_24h": 5.0, "no": [],
                  "yes": [["0.50", "10"], ["0.49", "5"], ["0.45", "7"], ["0.30", "100"]]}
        model = depth_model.fit([record])
        assert model.overall == (10.0, 15.0, 15.0, 15.0, 22.0, 22.0, 122.0)
        assert model.ladders == 1 and model.snapshots == 1

    def test_each_non_empty_side_is_one_ladder(self):
        record = {"ticker": "T", "taken_at": STAMP, "volume_24h": 5.0,
                  "yes": [["0.50", "10"]], "no": [["0.40", "30"]]}
        model = depth_model.fit([record])
        assert model.ladders == 2
        assert model.overall[0] == 20.0   # the median of 10 and 30

    def test_the_best_bid_is_the_highest_price_with_contracts(self):
        record = {"ticker": "T", "taken_at": STAMP, "volume_24h": 5.0, "no": [],
                  "yes": [["0.60", "0"], ["0.50", "4"], ["0.49", "6"]]}
        assert depth_model.fit([record]).overall[:2] == (4.0, 10.0)

    def test_unusable_levels_are_skipped(self):
        record = {"ticker": "T", "taken_at": STAMP, "volume_24h": 5.0, "no": [],
                  "yes": [["x", "3"], ["0.50"], ["0.50", "-2"], ["1.20", "5"], None,
                          ["0.40", 6], [0.40, "nan"]]}
        assert depth_model.fit([record]).overall[0] == 6.0

    def test_a_number_too_large_for_a_float_is_unusable_not_an_error(self):
        huge = 10**400
        record = {"ticker": "T", "taken_at": STAMP, "volume_24h": 5.0, "no": [],
                  "yes": [["0.50", huge], [huge, "5"], ["0.40", 6]]}
        assert depth_model.fit([record]).overall[0] == 6.0
        assert depth_model.fit([dict(record, volume_24h=huge)]) is None

    def test_a_side_with_no_usable_level_is_no_ladder(self):
        record = {"ticker": "T", "taken_at": STAMP, "volume_24h": 5.0,
                  "yes": [["x", "3"]], "no": None}
        assert depth_model.fit([record]) is None

    def test_none_when_there_is_nothing_to_fit(self):
        assert depth_model.fit([]) is None
        assert depth_model.fit([{"ticker": "T", "taken_at": STAMP, "volume_24h": 5.0,
                                 "yes": [], "no": []}]) is None

    def test_a_record_with_no_volume_is_skipped(self):
        assert depth_model.fit([_record(None, 0.4, (5, 0, 0, 0, 0, 0, 0))]) is None
        model = depth_model.fit([_record(None, 0.4, (5, 0, 0, 0, 0, 0, 0)),
                                 _record(3.0, 0.4, (8, 0, 0, 0, 0, 0, 0))])
        assert model.ladders == 1 and model.overall[0] == 8.0

    def test_a_cell_reports_the_median_per_distance(self, monkeypatch):
        monkeypatch.setattr(config, "DEPTH_MODEL_MIN_LADDERS", 3)
        records = [
            _record(5.0, 0.40, (10, 0, 0, 0, 0, 0, 0)),
            _record(5.0, 0.40, (20, 0, 5, 0, 0, 0, 0)),
            _record(5.0, 0.40, (30, 40, 0, 0, 0, 0, 0)),
        ]
        model = depth_model.fit(records)
        # Volume 5 is bucket 1 (1-99); a best bid of 0.40 is band 2 (20-50c)
        assert model.cells[(1, 2)] == (20.0, 20.0, 25.0, 25.0, 25.0, 25.0, 25.0)
        assert model.cells[(1, 2)] == model.volume_rows[1] == model.overall

    def test_a_ladder_is_filed_by_volume_and_best_bid(self, monkeypatch):
        monkeypatch.setattr(config, "DEPTH_MODEL_MIN_LADDERS", 1)
        cases = [  # (volume, best bid, bucket, band)
            (0, 0.50, 0, 3), (0.5, 0.50, 0, 3), (1, 0.04, 1, 0), (99, 0.05, 1, 1),
            (100, 0.20, 2, 2), (999.9, 0.79, 2, 3), (1_000, 0.80, 3, 4),
            (9_999, 0.94, 3, 4), (10_000, 0.95, 4, 5), (50_000, 0.99, 4, 5),
        ]
        for volume, best, bucket, band in cases:
            model = depth_model.fit([_record(volume, best, (5, 0, 0, 0, 0, 0, 0))])
            assert list(model.cells) == [(bucket, band)], (volume, best)

    def test_the_fallback_chain_cell_then_volume_row_then_overall(self):
        def increments(first, second):
            return (first, second, 0, 0, 0, 0, 0)

        records = (
            # (bucket 2, band 2): a full cell
            [_record(150, 0.30, increments(10, 10)) for _ in range(25)]
            # (bucket 2, band 3): too thin for a cell; its volume row is full
            + [_record(150, 0.60, increments(100, 0)) for _ in range(5)]
            # bucket 1 and bucket 4: too thin for a volume row
            + [_record(5, 0.30, increments(1, 0)) for _ in range(4)]
            + [_record(20_000, 0.30, increments(7, 0)) for _ in range(3)]
            # bucket 3: a full row that moves the overall median away from bucket 2's
            + [_record(2_000, 0.90, increments(50, 0)) for _ in range(60)]
        )
        model = depth_model.fit(records)
        assert set(model.cells) == {(2, 2), (3, 4)}
        assert set(model.volume_rows) == {2, 3}
        assert model.overall[0] == 50.0 and model.volume_rows[2][0] == 10.0

        def first_level(volume, best):
            return depth_model.bid_ladder(model, best, volume)[0][1]

        assert first_level(150, 0.30) == 10.0     # its own cell
        assert first_level(150, 0.60) == 10.0     # thin cell: its volume row, not its own 100
        assert first_level(150, 0.10) == 10.0     # unseen cell of a seen bucket: the volume row
        assert first_level(5, 0.30) == 50.0       # thin volume row: overall, not its own 1
        assert first_level(20_000, 0.30) == 50.0  # likewise, not its own 7
        assert first_level(0, 0.30) == 50.0       # a bucket never seen: overall

    @pytest.mark.parametrize("quantile", [0.1, 0.5, 0.9])
    def test_rows_never_fall_as_the_distance_grows(self, monkeypatch, quantile):
        monkeypatch.setattr(config, "DEPTH_MODEL_QUANTILE", quantile)
        monkeypatch.setattr(config, "DEPTH_MODEL_MIN_LADDERS", 5)
        rng = random.Random(7)
        records = []
        for i in range(400):
            best = rng.choice([0.03, 0.10, 0.30, 0.60, 0.90, 0.97])
            sides = {}
            for side in ("yes", "no"):
                levels = [[f"{best - rng.randint(0, 25) / 100:.2f}", str(rng.randint(1, 400))]
                          for _ in range(rng.randint(1, 12))]
                sides[side] = [lv for lv in levels if float(lv[0]) > 0] or [[f"{best:.2f}", "1"]]
            records.append({"ticker": f"T{i}", "taken_at": STAMP,
                            "volume_24h": rng.choice([0, 5, 150, 2_000, 20_000]), **sides})
        model = depth_model.fit(records)
        rows = [model.overall, *model.cells.values(), *model.volume_rows.values()]
        assert len(model.cells) > 5
        for row in rows:
            assert len(row) == len(DISTANCES)
            assert all(a <= b for a, b in zip(row, row[1:], strict=False)), row

    def test_snapshots_and_dates_come_from_the_records_used(self):
        records = [
            _record(5, 0.4, (5, 0, 0, 0, 0, 0, 0), taken="2026-10-02T16:00:00Z"),
            _record(5, 0.4, (5, 0, 0, 0, 0, 0, 0), taken="2026-10-02T16:00:00Z"),
            _record(5, 0.4, (5, 0, 0, 0, 0, 0, 0), taken="2026-10-01T16:00:00Z"),
            _record(None, 0.4, (5, 0, 0, 0, 0, 0, 0), taken="2026-09-01T16:00:00Z"),
        ]
        model = depth_model.fit(records)
        assert model.snapshots == 2 and model.ladders == 3
        assert model.first_taken == "2026-10-01T16:00:00Z"
        assert model.last_taken == "2026-10-02T16:00:00Z"


class TestBidLadder:
    """Synthetic bid levels from a table row."""

    ROW = (10, 10, 20, 20, 30, 30, 50)

    def test_levels_that_add_nothing_are_dropped(self):
        levels = depth_model.bid_ladder(_model(self.ROW), 0.50, 5)
        assert levels == [[0.5, 10.0], [0.48, 10.0], [0.45, 10.0], [0.3, 20.0]]

    def test_the_ladder_stops_before_one_cent(self):
        model = _model(self.ROW)
        assert depth_model.bid_ladder(model, 0.12, 5) == [[0.12, 10.0], [0.1, 10.0], [0.07, 10.0]]
        # 0.03 - 0.02 is 0.01 (kept); 0.03 - 0.03 is 0 (stop)
        assert depth_model.bid_ladder(model, 0.03, 5) == [[0.03, 10.0], [0.01, 10.0]]
        assert depth_model.bid_ladder(model, 0.01, 5) == [[0.01, 10.0]]

    def test_every_level_is_positive_and_at_least_one_cent(self):
        model = _model((3, 4, 9, 9, 30, 31, 50))
        for cents in range(1, 100):
            levels = depth_model.bid_ladder(model, cents / 100, 5)
            assert levels and levels[0][0] == cents / 100
            prices = [p for p, _ in levels]
            assert prices == sorted(prices, reverse=True)
            assert all(p >= 0.01 for p in prices)
            assert all(qty > 0 for _, qty in levels)

    @pytest.mark.parametrize("best", [0.0, 0.009, 0.991, 1.0, -0.5, float("nan"), None, "x"])
    def test_empty_when_the_best_bid_is_outside_the_range(self, best):
        assert depth_model.bid_ladder(_model(self.ROW), best, 5) == []

    def test_empty_without_a_volume(self):
        assert depth_model.bid_ladder(_model(self.ROW), 0.5, None) == []
        assert depth_model.bid_ladder(_model(self.ROW), 0.5, float("nan")) == []

    def test_float_noise_in_the_best_bid_does_not_move_the_ladder(self):
        # 1 - 0.8 is 0.19999999999999996
        levels = depth_model.bid_ladder(_model(self.ROW), 1 - 0.8, 5)
        assert levels[0][0] == 0.2
        assert levels == depth_model.bid_ladder(_model(self.ROW), 0.2, 5)

    def test_it_reads_the_row_for_the_volume_and_price(self):
        model = _model(
            (1, 1, 1, 1, 1, 1, 1),
            cells={(2, 3): (5, 5, 5, 5, 5, 5, 5)},
            volume_rows={2: (3, 3, 3, 3, 3, 3, 3)},
        )
        assert depth_model.bid_ladder(model, 0.60, 150)[0][1] == 5.0   # cell
        assert depth_model.bid_ladder(model, 0.30, 150)[0][1] == 3.0   # volume row
        assert depth_model.bid_ladder(model, 0.60, 5)[0][1] == 1.0     # overall

    def test_the_increments_sum_to_the_table_row(self):
        model = _model((10, 15, 22, 40, 41, 60, 100))
        levels = depth_model.bid_ladder(model, 0.50, 5)
        assert sum(qty for _, qty in levels) == 100.0


class TestBook:
    """A synthetic book in scanner._fetch_orderbook's shape."""

    @staticmethod
    def _fitted():
        rng = random.Random(3)
        records = [
            _record(rng.choice([0, 5, 150]), rng.choice([0.2, 0.5, 0.8]),
                    tuple(rng.randint(0, 40) + (i == 0) for i in range(len(DISTANCES))),
                    side=rng.choice(["yes", "no"]))
            for _ in range(80)
        ]
        return depth_model.fit(records)

    @pytest.mark.parametrize("which", ["hand built", "fitted"])
    def test_the_top_of_the_book_reproduces_the_quotes(self, which, caplog):
        model = _model((10, 15, 20, 25, 40, 60, 100)) if which == "hand built" else self._fitted()
        caplog.set_level(logging.WARNING)
        for ask_cents in range(1, 100):
            for bid_cents in range(1, ask_cents + 1):
                yes_ask, yes_bid = ask_cents / 100, bid_cents / 100
                ob = depth_model.book(model, yes_ask, yes_bid, 5.0)
                assert set(ob) == {"yes", "no"}
                # Buying YES walks the NO bids: its cheapest ask is the YES ask
                yes_asks = scanner._bids_to_ask_levels(ob["no"])
                assert yes_asks and abs(yes_asks[0][0] - yes_ask) < 1e-9
                # Buying NO walks the YES bids: its cheapest ask is 1 - the YES bid
                no_asks = scanner._bids_to_ask_levels(ob["yes"])
                assert no_asks and abs(no_asks[0][0] - (1 - yes_bid)) < 1e-9
                for side in ("yes", "no"):
                    assert all(qty > 0 and price >= 0.01 for price, qty in ob[side])
        # The level parser dropped nothing, so it logged nothing
        assert _warnings(caplog) == []

    def test_the_no_ladder_starts_at_one_minus_the_yes_ask(self):
        # 1 - 0.3 is 0.7 only up to float noise
        ob = depth_model.book(_model((10, 15, 20, 25, 40, 60, 100)), 0.3, 0.25, 5.0)
        assert ob["no"][0][0] == 0.7 and ob["yes"][0][0] == 0.25

    def test_none_without_a_model_or_a_volume(self):
        assert depth_model.book(None, 0.4, 0.3, 5.0) is None
        assert depth_model.book(_model((1,) * 7), 0.4, 0.3, None) is None

    def test_a_side_whose_best_bid_is_out_of_range_is_empty(self):
        ob = depth_model.book(_model((1,) * 7), 0.995, 0.3, 5.0)
        assert ob["no"] == [] and ob["yes"] != []
        ob = depth_model.book(_model((1,) * 7), 0.4, 0.0, 5.0)
        assert ob["yes"] == [] and ob["no"] != []

    def test_missing_quotes_give_empty_sides(self):
        ob = depth_model.book(_model((1,) * 7), None, None, 5.0)
        assert ob == {"yes": [], "no": []}


class TestTheModelObject:
    """Pickling, equality and the digest."""

    RECORDS = [_record(150, 0.30, (10, 10, 0, 0, 0, 0, 0)) for _ in range(25)]

    def test_it_survives_pickling(self):
        model = depth_model.fit(self.RECORDS)
        copy = pickle.loads(pickle.dumps(model))
        assert copy == model and copy.digest == model.digest
        assert depth_model.bid_ladder(copy, 0.3, 150) == depth_model.bid_ladder(model, 0.3, 150)

    def test_equal_tables_share_a_digest_whatever_their_snapshots(self):
        first = depth_model.fit(self.RECORDS)
        other = depth_model.fit([dict(r, taken_at="2027-01-01T00:00:00Z") for r in self.RECORDS])
        assert first.digest == other.digest
        assert first.first_taken != other.first_taken

    def test_different_tables_have_different_digests(self):
        first = depth_model.fit(self.RECORDS)
        other = depth_model.fit([_record(150, 0.30, (11, 10, 0, 0, 0, 0, 0)) for _ in range(25)])
        assert first.digest != other.digest

    def test_it_hashes_by_its_digest(self):
        model = depth_model.fit(self.RECORDS)
        assert hash(model) == hash(model.digest)
        assert len({model, pickle.loads(pickle.dumps(model))}) == 1


@pytest.fixture
def snapshots(tmp_path, monkeypatch):
    """The snapshot folder under a tmp_path cache, which snapshot() and loading use."""
    monkeypatch.setattr(historical, "CACHE_DIR", tmp_path)
    return tmp_path / config.DEPTH_SNAPSHOTS_DIRNAME


def _write(path, records, *, fmt="depth-snapshot-v1"):
    """Write a snapshot file by hand."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": {"format": fmt, "taken_at": STAMP, "markets": len(records)},
               "records": records}
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


class TestLoadDepthModel:
    """Loading a model from the saved snapshots never raises."""

    GOOD = [_record(150, 0.30, (10, 10, 0, 0, 0, 0, 0), ticker=f"T{i}") for i in range(4)]

    def test_no_folder_is_none_with_one_warning(self, snapshots, caplog):
        assert not snapshots.exists()
        assert depth_model.load_depth_model() is None
        warnings = _warnings(caplog)
        assert len(warnings) == 1
        assert "no depth snapshot saved" in warnings[0].getMessage()
        assert "python3 -m kalshi_betting.depth_model snapshot" in warnings[0].getMessage()

    def test_an_empty_folder_is_none_with_one_warning(self, snapshots, caplog):
        snapshots.mkdir(parents=True)
        (snapshots / "notes.txt").write_text("not a snapshot")
        assert depth_model.load_depth_model() is None
        assert len(_warnings(caplog)) == 1
        assert "no depth snapshot saved" in _warnings(caplog)[0].getMessage()

    def test_an_unreadable_folder_is_not_reported_as_empty(self, snapshots, caplog):
        _write(snapshots / "20261002T160000Z.json.gz", self.GOOD)
        snapshots.chmod(0)
        try:
            if os.access(snapshots, os.R_OK):
                pytest.skip("folder permissions are not enforced here")
            assert depth_model.load_depth_model() is None
        finally:
            snapshots.chmod(0o700)
        warnings = _warnings(caplog)
        assert len(warnings) == 1
        assert "not loaded" in warnings[0].getMessage()
        assert "no depth snapshot saved" not in warnings[0].getMessage()

    def test_a_number_too_large_for_a_float_costs_a_record_not_the_model(self, snapshots,
                                                                         caplog):
        bad = _record(150, 0.30, (10, 0, 0, 0, 0, 0, 0), ticker="BAD")
        bad["yes"] = [["0.30", 10**400]]
        _write(snapshots / "20261002T160000Z.json.gz", self.GOOD + [bad])
        model = depth_model.load_depth_model()
        assert model is not None and model.ladders == 4
        assert _warnings(caplog) == []

    def test_a_market_saved_twice_in_one_hour_counts_once(self, snapshots):
        hour = [dict(r, taken_at="2026-10-02T16:05:30Z") for r in self.GOOD]
        rerun = [dict(r, taken_at="2026-10-02T16:59:01Z") for r in self.GOOD[:2]]
        later = [dict(r, taken_at="2026-10-02T17:00:00Z") for r in self.GOOD[:2]]
        _write(snapshots / "20261002T160530Z.json.gz", hour)
        _write(snapshots / "20261002T165901Z.json.gz", rerun)
        _write(snapshots / "20261002T170000Z.json.gz", later)
        model = depth_model.load_depth_model()
        # The rerun adds nothing; the next hour's two markets count again
        assert model.ladders == 6
        assert model.snapshots == 2

    def test_records_with_no_ticker_are_never_merged(self, snapshots):
        nameless = [dict(r, ticker="") for r in self.GOOD]
        _write(snapshots / "20261002T160530Z.json.gz", nameless)
        _write(snapshots / "20261002T165901Z.json.gz", nameless)
        assert depth_model.load_depth_model().ladders == 8

    def test_a_corrupt_file_is_skipped_with_a_warning_naming_it(self, snapshots, caplog):
        _write(snapshots / "20261002T160000Z.json.gz", self.GOOD)
        (snapshots / "20261001T160000Z.json.gz").write_bytes(b"this is not gzip")
        model = depth_model.load_depth_model()
        assert model is not None and model.ladders == 4
        warnings = _warnings(caplog)
        assert len(warnings) == 1
        assert "20261001T160000Z.json.gz" in warnings[0].getMessage()

    def test_other_unreadable_shapes_are_skipped_too(self, snapshots, caplog):
        _write(snapshots / "a_good.json.gz", self.GOOD)
        _write(snapshots / "b_wrong_format.json.gz", self.GOOD, fmt="something-else")
        with gzip.open(snapshots / "c_a_list.json.gz", "wt") as handle:
            json.dump([1, 2, 3], handle)
        with gzip.open(snapshots / "d_records_not_a_list.json.gz", "wt") as handle:
            json.dump({"meta": {"format": "depth-snapshot-v1"}, "records": "x"}, handle)
        (snapshots / "e_truncated.json.gz").write_bytes(
            gzip.compress(json.dumps({"meta": {}, "records": []}).encode())[:-6])
        model = depth_model.load_depth_model()
        assert model.ladders == 4
        text = " ".join(r.getMessage() for r in _warnings(caplog))
        for name in ("b_wrong_format", "c_a_list", "d_records_not_a_list", "e_truncated"):
            assert name in text
        assert "a_good" not in text

    def test_two_files_make_one_model(self, snapshots, caplog):
        caplog.set_level(logging.INFO)
        _write(snapshots / "20261001T160000Z.json.gz",
               [dict(r, taken_at="2026-10-01T16:00:00Z") for r in self.GOOD])
        _write(snapshots / "20261002T160000Z.json.gz",
               [dict(r, taken_at="2026-10-02T16:00:00Z") for r in self.GOOD[:2]])
        model = depth_model.load_depth_model()
        assert (model.snapshots, model.ladders) == (2, 6)
        assert (model.first_taken, model.last_taken) == (
            "2026-10-01T16:00:00Z", "2026-10-02T16:00:00Z")
        assert _warnings(caplog) == []
        info = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert info == ["Depth model: 2 snapshot(s), 6 ladders, taken "
                        "2026-10-01T16:00:00Z to 2026-10-02T16:00:00Z"]

    def test_a_temp_file_left_by_an_interrupted_write_is_ignored(self, snapshots):
        _write(snapshots / "20261002T160000Z.json.gz", self.GOOD)
        (snapshots / "20261003T160000Z.json.gz.tmp").write_bytes(b"half a file")
        assert depth_model.load_depth_model().ladders == 4

    def test_snapshots_with_no_volume_reading_give_none(self, snapshots, caplog):
        _write(snapshots / "20261002T160000Z.json.gz",
               [_record(None, 0.3, (5, 0, 0, 0, 0, 0, 0))])
        assert depth_model.load_depth_model() is None
        warnings = _warnings(caplog)
        assert len(warnings) == 1 and "volume reading" in warnings[0].getMessage()

    def test_it_never_raises(self, snapshots, monkeypatch, caplog):
        def boom():
            raise OSError("disk gone")
        monkeypatch.setattr(depth_model, "_snapshot_dir", boom)
        assert depth_model.load_depth_model() is None
        assert "disk gone" in _warnings(caplog)[0].getMessage()


def _market(ticker, event="KXA-26OCT02", ask="0.40"):
    """An open ApiMarket as the events listing returns it."""
    return scanner.ApiMarket(ticker=ticker, event_ticker=event, title="Will it?", subtitle="",
                             status="active", close_time=None, yes_ask_dollars=ask)


class FakeExchange:
    """Replaces the four exchange reads a snapshot makes and records the calls."""

    def __init__(self, monkeypatch, markets, books=None, candle_error=()):
        self.markets = markets
        self.books = books or {}
        self.candle_error = set(candle_error)
        self.listing_calls = []
        self.book_calls = []
        self.candle_calls = []
        monkeypatch.setattr(scanner, "fetch_shard_statuses", lambda client: None)
        monkeypatch.setattr(scanner, "fetch_open_events_with_markets", self._listing)
        monkeypatch.setattr(scanner, "_fetch_orderbook", self._book)
        monkeypatch.setattr(historical, "fetch_candlesticks", self._candles)

    def _listing(self, client, inactive_shards=None):
        self.listing_calls.append(inactive_shards)
        return list(self.markets)

    def _book(self, client, ticker):
        self.book_calls.append(ticker)
        return self.books.get(ticker, {"yes": [["0.38", "10"], ["0.37", "5"]],
                                       "no": [["0.60", "20"], ["0.55", "8"]]})

    def _candles(self, client, ticker, open_ts, close_ts, **kwargs):
        self.candle_calls.append((ticker, open_ts, close_ts, kwargs))
        if ticker in self.candle_error:
            raise RuntimeError("candle read failed")
        # One candle per hour of the window, 10 contracts each
        return [{"ts": ts, "yes_ask_close": 0.4, "no_ask_close": 0.6, "volume": 10.0}
                for ts in range(open_ts + 3_600, close_ts + 1, 3_600)]


NOW = datetime(2026, 10, 2, 16, 5, 30, tzinfo=UTC)


def _saved(snapshots):
    """The one snapshot file in the folder, parsed."""
    files = sorted(snapshots.glob("*.json.gz"))
    assert len(files) == 1
    with gzip.open(files[0], "rt", encoding="utf-8") as handle:
        return files[0], json.load(handle)


class TestSnapshot:
    """snapshot() against a fake exchange."""

    def _world(self, monkeypatch, **kwargs):
        markets = [_market(f"KXA-26OCT02-{i}") for i in range(1, 7)] + [
            _market("COMBO-1", event="KXMVECROSSCATEGORY-SHARD1-S1"),
            _market("NOASK-1", ask=None),
            _market("ZEROASK-1", ask="0.0000"),
            _market("FAILBOOK-1"),
            _market("EMPTYBOOK-1"),
        ]
        books = {"FAILBOOK-1": None, "EMPTYBOOK-1": {"yes": [], "no": []}}
        return FakeExchange(monkeypatch, markets, books, **kwargs)

    def test_it_saves_the_sampled_markets_books_and_volume(self, snapshots, monkeypatch):
        world = self._world(monkeypatch)
        path = depth_model.snapshot(object(), 100, now=NOW)
        saved_path, payload = _saved(snapshots)
        assert path == saved_path == snapshots / "20261002T160530Z.json.gz"
        assert payload["meta"] == {"format": "depth-snapshot-v1",
                                   "taken_at": "2026-10-02T16:05:30Z", "markets": 6}
        records = payload["records"]
        assert {r["ticker"] for r in records} == {f"KXA-26OCT02-{i}" for i in range(1, 7)}
        for record in records:
            assert record["taken_at"] == "2026-10-02T16:05:30Z"
            # The 24 hourly candles ending at the snapshot time, 10 contracts each
            assert record["volume_24h"] == 240.0
            assert record["yes"] == [["0.38", "10"], ["0.37", "5"]]
            assert record["no"] == [["0.60", "20"], ["0.55", "8"]]
        # A listing as a run reads it, with no shard dropped
        assert world.listing_calls == [set()]
        assert list(snapshots.glob("*.tmp")) == []

    def test_combo_markets_and_markets_with_no_ask_are_never_read(self, snapshots, monkeypatch):
        world = self._world(monkeypatch)
        depth_model.snapshot(object(), 100, now=NOW)
        assert set(world.book_calls) == {f"KXA-26OCT02-{i}" for i in range(1, 7)} | {
            "FAILBOOK-1", "EMPTYBOOK-1"}

    def test_candles_are_read_from_the_live_endpoint_over_25_hours(self, snapshots, monkeypatch):
        world = self._world(monkeypatch)
        depth_model.snapshot(object(), 100, now=NOW)
        now_ts = int(NOW.timestamp())
        by_ticker = {call[0]: call for call in world.candle_calls}
        # Neither a failed book nor an empty one is worth a candle read
        assert set(by_ticker) == {f"KXA-26OCT02-{i}" for i in range(1, 7)}
        for _ticker, open_ts, close_ts, kwargs in world.candle_calls:
            assert (open_ts, close_ts) == (now_ts - 25 * 3600, now_ts)
            assert kwargs["series"] == "KXA" and kwargs["live_first"] is True
            assert kwargs["use_cache"] is False

    def test_it_samples_at_most_the_requested_markets(self, snapshots, monkeypatch):
        FakeExchange(monkeypatch, [_market(f"KXA-26OCT02-{i}") for i in range(1, 7)])
        depth_model.snapshot(object(), 3, now=NOW)
        assert len(_saved(snapshots)[1]["records"]) == 3

    def test_the_sample_is_seeded_by_the_utc_hour(self, snapshots, monkeypatch):
        markets = [_market(f"KXB-26OCT02-{i:02d}", event="KXB-26OCT02") for i in range(12)]
        world = FakeExchange(monkeypatch, markets)
        depth_model.snapshot(object(), 4, now=NOW)
        first = sorted(world.book_calls)
        expected = random.Random(2026100216).sample(sorted(m.ticker for m in markets), 4)
        assert first == sorted(expected)
        # A rerun in the same hour picks the same markets, whatever order the listing came in
        world.markets = list(reversed(markets))
        world.book_calls.clear()
        depth_model.snapshot(object(), 4, now=NOW.replace(minute=59, second=1))
        assert sorted(world.book_calls) == first

    def test_a_rerun_in_the_same_hour_is_not_counted_twice(self, snapshots, monkeypatch):
        self._world(monkeypatch)
        depth_model.snapshot(object(), 100, now=NOW)
        depth_model.snapshot(object(), 100, now=NOW.replace(minute=59, second=1))
        assert len(list(snapshots.glob("*.json.gz"))) == 2
        model = depth_model.load_depth_model()
        assert (model.snapshots, model.ladders) == (1, 12)

    def test_a_failed_closing_fit_does_not_lose_the_saved_snapshot(self, snapshots,
                                                                  monkeypatch, caplog):
        self._world(monkeypatch)

        def boom(found):
            raise RuntimeError("fit exploded")
        monkeypatch.setattr(depth_model, "_build", boom)
        path = depth_model.snapshot(object(), 100, now=NOW)
        assert path.exists()
        warnings = [r.getMessage() for r in _warnings(caplog)]
        assert len(warnings) == 1 and "fit exploded" in warnings[0]
        assert "saved" in warnings[0]

    def test_a_naive_time_is_read_as_utc(self, snapshots, monkeypatch):
        self._world(monkeypatch)
        path = depth_model.snapshot(object(), 2, now=NOW.replace(tzinfo=None))
        assert path.name == "20261002T160530Z.json.gz"

    def test_a_market_whose_read_fails_is_skipped_with_a_warning(self, snapshots, monkeypatch,
                                                                 caplog):
        self._world(monkeypatch, candle_error={"KXA-26OCT02-3"})
        depth_model.snapshot(object(), 100, now=NOW)
        tickers = {r["ticker"] for r in _saved(snapshots)[1]["records"]}
        assert "KXA-26OCT02-3" not in tickers and len(tickers) == 5
        failed = [r.getMessage() for r in _warnings(caplog)]
        assert len(failed) == 1 and "KXA-26OCT02-3" in failed[0]
        assert "candle read failed" in failed[0]

    def test_nothing_to_sample_raises_and_saves_nothing(self, snapshots, monkeypatch):
        FakeExchange(monkeypatch, [_market("COMBO-1", event="KXMVECROSSCATEGORY-S1")])
        with pytest.raises(RuntimeError, match="to sample"):
            depth_model.snapshot(object(), 100, now=NOW)
        assert not snapshots.exists()

    def test_no_usable_book_raises_and_saves_nothing(self, snapshots, monkeypatch):
        FakeExchange(monkeypatch, [_market("KXA-26OCT02-1")], books={"KXA-26OCT02-1": None})
        with pytest.raises(RuntimeError, match="usable book"):
            depth_model.snapshot(object(), 100, now=NOW)
        assert not snapshots.exists()

    def test_markets_below_one_is_refused(self, snapshots, monkeypatch):
        FakeExchange(monkeypatch, [])
        with pytest.raises(ValueError):
            depth_model.snapshot(object(), 0, now=NOW)

    def test_the_saved_file_loads_into_a_model(self, snapshots, monkeypatch):
        self._world(monkeypatch)
        depth_model.snapshot(object(), 100, now=NOW)
        model = depth_model.load_depth_model()
        # Six markets with two ladders each
        assert (model.snapshots, model.ladders) == (1, 12)
        assert model.first_taken == model.last_taken == "2026-10-02T16:05:30Z"
        # 240 contracts a day is the 100-999 bucket. At the best bid the six yes ladders
        # rest 10 and the six no ladders 20, so the median is 15 on both sides
        book = depth_model.book(model, 0.4, 0.38, 240.0)
        assert book["no"][0] == [0.6, 15.0] and book["yes"][0] == [0.38, 15.0]

    def test_a_second_snapshot_joins_the_first(self, snapshots, monkeypatch):
        self._world(monkeypatch)
        depth_model.snapshot(object(), 100, now=NOW)
        depth_model.snapshot(object(), 100, now=datetime(2026, 10, 3, 16, 0, tzinfo=UTC))
        assert len(list(snapshots.glob("*.json.gz"))) == 2
        assert depth_model.load_depth_model().snapshots == 2

    def test_it_logs_progress_and_the_ladders_per_cell(self, snapshots, monkeypatch, caplog):
        caplog.set_level(logging.INFO)
        monkeypatch.setattr(config, "DEPTH_SNAPSHOT_PROGRESS_EVERY", 2)
        self._world(monkeypatch)
        depth_model.snapshot(object(), 100, now=NOW)
        text = [r.getMessage() for r in caplog.records]
        assert sum("markets read" in line for line in text) == 4   # 8 sampled, every 2
        assert any(line.startswith("Depth snapshot saved:") and "6 markets kept of 8" in line
                   for line in text)
        table = "\n".join(text)
        for label in ("Ladders per cell", "<5c", "5-20c", "95c+", "10,000+", "1,000-9,999"):
            assert label in table
        # The yes ladders (best 0.38) and no ladders (best 0.60) of the 100-999 bucket
        row = next(line for line in text if line.startswith("100-999"))
        assert row.split()[1:] == ["0", "0", "6", "6", "0", "0"]
        assert any(line.startswith("Depth model: 1 snapshot(s), 12 ladders") for line in text)

    def test_candles_without_volume_leave_the_model_empty_and_say_so(self, snapshots,
                                                                     monkeypatch, caplog):
        world = self._world(monkeypatch)
        monkeypatch.setattr(
            historical, "fetch_candlesticks",
            lambda client, ticker, open_ts, close_ts, **kw: [{"ts": close_ts, "volume": None}])
        depth_model.snapshot(object(), 100, now=NOW)
        assert world.listing_calls == [set()]
        assert any("no saved ladder has a volume reading" in r.getMessage()
                   for r in _warnings(caplog))


class TestMain:
    """The command line."""

    @pytest.fixture(autouse=True)
    def _no_logging_setup(self, monkeypatch):
        monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: None)

    def test_it_takes_one_subcommand(self):
        with pytest.raises(SystemExit) as caught:
            depth_model.main([])
        assert caught.value.code == 2

    def test_markets_must_be_at_least_one(self):
        with pytest.raises(SystemExit) as caught:
            depth_model.main(["snapshot", "--markets", "0"])
        assert caught.value.code == 2

    def test_it_snapshots_with_the_production_client(self, monkeypatch):
        client = object()
        calls = []
        monkeypatch.setattr(historical, "build_prod_live_client", lambda: client)
        monkeypatch.setattr(depth_model, "snapshot", lambda c, n: calls.append((c, n)))
        assert depth_model.main(["snapshot", "--markets", "7"]) == 0
        assert depth_model.main(["snapshot"]) == 0
        assert calls == [(client, 7), (client, config.DEPTH_SNAPSHOT_MARKETS)]

    def test_a_failed_snapshot_exits_one_with_an_error_line(self, monkeypatch, caplog):
        monkeypatch.setattr(historical, "build_prod_live_client", lambda: object())

        def failing(client, markets):
            raise RuntimeError("no open non-combo market with a YES ask to sample")
        monkeypatch.setattr(depth_model, "snapshot", failing)
        assert depth_model.main(["snapshot"]) == 1
        errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1 and "to sample" in errors[0]


class TestIsolation:
    """The module is read-only and backtest-only."""

    TREE = ast.parse(inspect.getsource(depth_model))

    def test_it_imports_no_order_path_or_report_module(self):
        imported = set()
        for node in ast.walk(self.TREE):
            if isinstance(node, ast.ImportFrom):
                imported |= {(node.module or "").split(".")[-1]}
                imported |= {a.name for a in node.names}
            elif isinstance(node, ast.Import):
                imported |= {a.name.split(".")[-1] for a in node.names}
        for name in ("trader", "main", "scheduler", "v2_probe", "defaults_server",
                     "backtester", "backtest", "dashboard", "strategy", "reporter"):
            assert name not in imported, name

    def test_no_live_module_imports_it(self):
        names = [m.name for m in pkgutil.iter_modules(kalshi_betting.__path__)]
        assert "depth_model" in names
        for name in names:
            if name in {"depth_model", "backtester", "backtest", "dashboard"}:
                continue
            module = importlib.import_module(f"kalshi_betting.{name}")
            for node in ast.walk(ast.parse(inspect.getsource(module))):
                if isinstance(node, ast.Import):
                    imported = {a.name.split(".")[-1] for a in node.names}
                elif isinstance(node, ast.ImportFrom):
                    imported = {(node.module or "").split(".")[-1]}
                    imported |= {a.name for a in node.names}
                else:
                    continue
                assert "depth_model" not in imported, f"kalshi_betting/{name}.py imports it"

    def test_it_sends_no_order(self):
        text = inspect.getsource(depth_model)
        for forbidden in ("create_order", "submit_order", "V2_ORDER_PATH", "signed_request_json"):
            assert forbidden not in text
