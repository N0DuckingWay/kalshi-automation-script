"""Tests for scanner.py normalize_title() — the core pair-detection function."""
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kalshi_betting.config import INCLUDE_MVE_MARKETS
from kalshi_betting.scanner import (
    CandidatePair,
    display_title,
    enrich_with_orderbook_prices,
    fetch_open_events_with_markets,
    find_same_title_pairs,
    find_time_series_pairs,
    normalize_title,
    pair_key,
    validate_pair_price,
)


class TestNormalizeTitle:
    def test_full_month_name_with_year(self):
        result = normalize_title("Will BTC exceed $80k by December 2025?")
        assert "december" not in result
        assert "2025" not in result
        assert "btc" in result
        assert "$80k" in result

    def test_full_month_name_with_date_and_year(self):
        result = normalize_title("Inflation rate by January 31, 2026")
        assert "january" not in result
        assert "31" not in result
        assert "2026" not in result

    def test_abbreviated_month_with_year(self):
        result = normalize_title("Fed funds rate Dec 2025")
        assert "dec" not in result
        assert "2025" not in result
        assert "fed funds rate" in result

    def test_iso_date(self):
        result = normalize_title("GDP report 2025-03-31")
        assert "2025" not in result
        assert "03" not in result
        assert "31" not in result
        assert "gdp report" in result

    def test_quarter(self):
        result = normalize_title("Fed funds rate Q2 2025")
        assert "q2" not in result
        assert "2025" not in result
        assert "fed funds rate" in result

    def test_quarter_without_year(self):
        result = normalize_title("Rate decision Q3")
        assert "q3" not in result
        assert "rate decision" in result

    def test_time_series_pair_normalizes_identically(self):
        # Two markets differing only in deadline must produce the same normalized string
        title_march = "Will BTC exceed $80k by March 2025?"
        title_june = "Will BTC exceed $80k by June 2025?"
        assert normalize_title(title_march) == normalize_title(title_june)

    def test_another_time_series_pair(self):
        a = "Will the Fed cut rates by January 2025?"
        b = "Will the Fed cut rates by June 2025?"
        assert normalize_title(a) == normalize_title(b)

    def test_no_dates_unchanged(self):
        title = "Will it rain tomorrow?"
        result = normalize_title(title)
        assert result == "will it rain tomorrow?"

    def test_output_is_lowercase(self):
        result = normalize_title("BTC PRICE ABOVE 80K")
        assert result == result.lower()

    def test_whitespace_collapsed(self):
        result = normalize_title("Will BTC exceed $80k by December 2025 ?")
        assert "  " not in result

    def test_abbreviated_month_day_sandbox_format(self):
        # "Apr 02" format used in Kalshi sandbox titles
        result = normalize_title("BTC price Apr 02")
        assert "apr" not in result

    def test_numeric_date(self):
        result = normalize_title("Price above 50k on 01/15/2026")
        assert "01" not in result or "2026" not in result

    def test_standalone_year(self):
        result = normalize_title("GDP growth in 2026")
        assert "2026" not in result
        assert "gdp growth" in result

    def test_content_preserved_after_stripping(self):
        # A title with many dates should still have meaningful content left
        result = normalize_title("Will the S&P 500 exceed 6000 by December 31, 2025?")
        assert len(result.strip()) > 5
        assert "s&p 500" in result or "s&p" in result or "500" in result

    def test_result_is_stripped(self):
        result = normalize_title("  Will BTC   exceed $80k by March 2025?  ")
        assert result == result.strip()


def _mock_market(
    *,
    ticker: str,
    event_ticker: str,
    title: str = "",
    subtitle: str = "",
    event_title: str | None = None,
    yes_ask: float = 0.50,
    no_ask: float = 0.50,
    close_time=None,
):
    """Build a SimpleNamespace Kalshi-market stand-in with the fields the scanner reads.

    SimpleNamespace (not MagicMock) is used so that `getattr(m, "_event_title", "")`
    returns the empty string when no event title was attached — MagicMock would
    auto-vivify a child mock and break the pair_key fallback.
    """
    from datetime import UTC, datetime
    attrs = {
        "ticker": ticker,
        "event_ticker": event_ticker,
        "title": title,
        "subtitle": subtitle,
        "yes_ask_dollars": str(yes_ask),
        "no_ask_dollars": str(no_ask),
        "yes_bid_dollars": str(max(yes_ask - 0.02, 0.01)),
        "close_time": close_time or datetime(2026, 6, 1, tzinfo=UTC),
    }
    if event_title is not None:
        attrs["_event_title"] = event_title
    return SimpleNamespace(**attrs)


class TestPairKey:
    def test_combines_event_and_market_title(self):
        m = _mock_market(ticker="T1", event_ticker="E1", title="Trump", event_title="2024 Election Winner")
        key = pair_key(m)
        assert "2024 Election Winner" in key
        assert "Trump" in key
        # Sanity — separator present so the two parts don't run together ambiguously
        assert "|" in key

    def test_falls_back_when_event_title_missing(self):
        m = _mock_market(ticker="T1", event_ticker="E1", title="Will BTC exceed $80k", event_title=None)
        assert pair_key(m) == "Will BTC exceed $80k"

    def test_falls_back_when_event_title_empty_string(self):
        m = _mock_market(ticker="T1", event_ticker="E1", title="Will BTC exceed $80k", event_title="")
        assert pair_key(m) == "Will BTC exceed $80k"


class TestDisplayTitle:
    def test_formats_with_event_prefix(self):
        m = _mock_market(ticker="T1", event_ticker="E1", title="Trump", event_title="2024 Election Winner")
        label = display_title(m)
        assert label == "2024 Election Winner: Trump"

    def test_falls_back_to_bare_title(self):
        m = _mock_market(ticker="T1", event_ticker="E1", title="Will BTC exceed $80k", event_title=None)
        assert display_title(m) == "Will BTC exceed $80k"


class TestSameTitleGrouping:
    """Validates that the combined-key grouping eliminates cross-event MVE collisions."""

    def test_cross_event_same_option_label_does_not_pair(self):
        # Two MVE markets with identical option label "Trump" but in completely
        # different events — must NOT be paired.
        mA = _mock_market(
            ticker="ELECT-TRUMP", event_ticker="ELECT-2024",
            title="Trump", event_title="2024 Election Winner",
            yes_ask=0.45, no_ask=0.55,
        )
        mB = _mock_market(
            ticker="TIME-TRUMP", event_ticker="TIME-2024",
            title="Trump", event_title="2024 Time Person of the Year",
            yes_ask=0.20, no_ask=0.80,
        )
        pairs = find_same_title_pairs([mA, mB])
        assert pairs == [], f"Expected no pairs across unrelated events; got {pairs}"

    def test_same_event_title_does_pair(self):
        # Two markets with identical event_title + market title but different
        # event_ticker — the legitimate same-title arbitrage case.
        mA = _mock_market(
            ticker="A1", event_ticker="EVT-A",
            title="Republicans control Senate after 2026", event_title="2026 Senate Control",
            yes_ask=0.30, no_ask=0.70,
        )
        mB = _mock_market(
            ticker="B1", event_ticker="EVT-B",
            title="Republicans control Senate after 2026", event_title="2026 Senate Control",
            yes_ask=0.40, no_ask=0.60,
        )
        pairs = find_same_title_pairs([mA, mB])
        # Should produce exactly one pair (best-per-group)
        assert len(pairs) == 1
        # Pair must reference both markets
        tickers = {pairs[0].market_a.ticker, pairs[0].market_b.ticker}
        assert tickers == {"A1", "B1"}

    def test_within_event_filter_still_applies(self):
        # Same event_title (so they group), same event_ticker (so within-event filter
        # rejects). This catches the multi-choice-options-in-the-same-event case.
        mA = _mock_market(
            ticker="X-TRUMP", event_ticker="MVE-2024",
            title="Trump", event_title="2024 Election Winner",
            yes_ask=0.45, no_ask=0.55,
        )
        mB = _mock_market(
            ticker="X-HARRIS", event_ticker="MVE-2024",
            title="Harris", event_title="2024 Election Winner",
            yes_ask=0.50, no_ask=0.50,
        )
        # Different market titles, but event_title shared. find_same_title_pairs
        # groups by (event_title, title, subtitle) — different titles mean they
        # land in different groups, so no pair regardless.
        assert find_same_title_pairs([mA, mB]) == []


class TestTimeSeriesGrouping:
    def test_cross_event_mve_option_label_does_not_pair_in_time_series(self):
        # Same option label, different events at different deadlines —
        # the time-series scanner must NOT pair these because the event titles differ.
        from datetime import UTC, datetime
        mA = _mock_market(
            ticker="ELECT-TRUMP-MAR", event_ticker="ELECT-MAR",
            title="Trump", event_title="2024 Election Winner",
            yes_ask=0.45, no_ask=0.55,
            close_time=datetime(2026, 3, 1, tzinfo=UTC),
        )
        mB = _mock_market(
            ticker="TIME-TRUMP-JUN", event_ticker="TIME-JUN",
            title="Trump", event_title="2024 Time Person of the Year",
            yes_ask=0.20, no_ask=0.80,
            close_time=datetime(2026, 3, 20, tzinfo=UTC),
        )
        # client is unused when `markets` is provided
        pairs = find_time_series_pairs(MagicMock(), held_tickers=set(), markets=[mA, mB])
        assert pairs == [], f"Expected no pairs across unrelated MVE events; got {pairs}"

    def test_pricier_later_contract_is_not_a_candidate(self):
        # Regression: the price-gap filter used to be abs(pA - pB) >= threshold,
        # which let a pair through (as untradeable) when the LATER-closing
        # contract was priced higher — normal term structure, not the anomaly
        # this strategy exploits (the arbitrage only exists when the EARLIER
        # contract is priced higher). Such a pair must not be a candidate at
        # all, directional only: pA - pB >= min_price_diff_for_gap(gap_days).
        from datetime import UTC, datetime
        mA = _mock_market(  # earlier-closing, CHEAPER — normal term structure
            ticker="EARLY", event_ticker="EVT-A",
            title="Will BTC exceed $80k",
            yes_ask=0.20, no_ask=0.80,
            close_time=datetime(2026, 3, 1, tzinfo=UTC),
        )
        mB = _mock_market(  # later-closing, more expensive
            ticker="LATE", event_ticker="EVT-B",
            title="Will BTC exceed $80k",
            yes_ask=0.45, no_ask=0.55,
            close_time=datetime(2026, 3, 20, tzinfo=UTC),
        )
        pairs = find_time_series_pairs(MagicMock(), held_tickers=set(), markets=[mA, mB])
        assert pairs == [], f"Pricier later contract must not be a candidate; got {pairs}"


def _ts_pair_markets(*, gap_days: int, pA: float, pB: float):
    """Build an earlier/later mock market pair sharing a title, gap_days apart.

    The earlier market carries YES ask pA (NO ask 1-pA) and the later one pB,
    so pA - pB is the directional price gap seen by find_time_series_pairs.
    """
    from datetime import UTC, datetime, timedelta
    early_close = datetime(2026, 3, 1, tzinfo=UTC)
    mA = _mock_market(
        ticker="EARLY", event_ticker="EVT-A",
        title="Will BTC exceed $80k",
        yes_ask=pA, no_ask=round(1.0 - pA, 4),
        close_time=early_close,
    )
    mB = _mock_market(
        ticker="LATE", event_ticker="EVT-B",
        title="Will BTC exceed $80k",
        yes_ask=pB, no_ask=round(1.0 - pB, 4),
        close_time=early_close + timedelta(days=gap_days),
    )
    return mA, mB


class TestTimeSeriesTieredThreshold:
    """The minimum price gap is tiered by deadline gap: 15% for gaps <= 15
    days, 30% for 16-30 days, and gaps > 30 days are never candidates."""

    def _scan(self, gap_days, pA, pB):
        mA, mB = _ts_pair_markets(gap_days=gap_days, pA=pA, pB=pB)
        return find_time_series_pairs(MagicMock(), held_tickers=set(), markets=[mA, mB])

    def test_short_gap_20pct_price_gap_accepted(self):
        # 10-day deadline gap → 15% tier; a 20% price gap qualifies
        pairs = self._scan(10, pA=0.50, pB=0.30)
        assert len(pairs) == 1
        assert pairs[0].market_a.ticker == "EARLY"

    def test_short_gap_10pct_price_gap_rejected(self):
        # 10-day deadline gap → 15% tier; a 10% price gap is below it
        assert self._scan(10, pA=0.40, pB=0.30) == []

    def test_long_gap_18pct_price_gap_rejected(self):
        # 20-day deadline gap → 30% tier; 18% would have passed the old flat
        # 15% threshold but must now be rejected
        assert self._scan(20, pA=0.48, pB=0.30) == []

    def test_long_gap_35pct_price_gap_accepted(self):
        # 20-day deadline gap → 30% tier; a 35% price gap clears it
        pairs = self._scan(20, pA=0.65, pB=0.30)
        assert len(pairs) == 1

    def test_boundary_15_day_gap_uses_short_tier(self):
        # Exactly 15 days is inclusive in the 15% tier — 20% qualifies
        pairs = self._scan(15, pA=0.50, pB=0.30)
        assert len(pairs) == 1

    def test_boundary_16_day_gap_uses_long_tier(self):
        # 16 days falls into the 30% tier — the same 20% gap now fails
        assert self._scan(16, pA=0.50, pB=0.30) == []

    def test_boundary_30_day_gap_still_allowed(self):
        # 30 days is the maximum allowed deadline gap; 35% clears the 30% tier
        pairs = self._scan(30, pA=0.65, pB=0.30)
        assert len(pairs) == 1

    def test_over_max_gap_rejected_regardless_of_price(self):
        # 35 days exceeds MAX_DEADLINE_GAP_DAYS — even a 40% price gap is out
        assert self._scan(35, pA=0.70, pB=0.30) == []

    def test_direction_still_rules_out_pricier_later_contract(self):
        # 10-day gap, later contract pricier by 35%: magnitude alone never
        # qualifies — the filter is directional (earlier must be expensive)
        assert self._scan(10, pA=0.30, pB=0.65) == []

    def test_same_event_ticker_never_pairs(self):
        # Two options inside the same multi-choice event share an event_ticker
        # and must not form a time-series pair, whatever the price gap
        from datetime import UTC, datetime
        mA = _mock_market(
            ticker="OPT-A", event_ticker="MVE-1",
            title="Will BTC exceed $80k",
            yes_ask=0.50, no_ask=0.50,
            close_time=datetime(2026, 3, 1, tzinfo=UTC),
        )
        mB = _mock_market(
            ticker="OPT-B", event_ticker="MVE-1",
            title="Will BTC exceed $80k",
            yes_ask=0.30, no_ask=0.70,
            close_time=datetime(2026, 3, 11, tzinfo=UTC),
        )
        assert find_time_series_pairs(MagicMock(), held_tickers=set(), markets=[mA, mB]) == []

    def test_different_titles_never_pair(self):
        # Markets asking different questions normalize to different keys and
        # are never grouped, whatever their prices or deadlines
        from datetime import UTC, datetime
        mA = _mock_market(
            ticker="BTC", event_ticker="EVT-A",
            title="Will BTC exceed $80k",
            yes_ask=0.50, no_ask=0.50,
            close_time=datetime(2026, 3, 1, tzinfo=UTC),
        )
        mB = _mock_market(
            ticker="ETH", event_ticker="EVT-B",
            title="Will ETH exceed $5k",
            yes_ask=0.30, no_ask=0.70,
            close_time=datetime(2026, 3, 11, tzinfo=UTC),
        )
        assert find_time_series_pairs(MagicMock(), held_tickers=set(), markets=[mA, mB]) == []

    def test_same_title_pairs_ignore_deadline_gap(self):
        # Same-title pairs keep the flat 5% threshold — a 6% divergence on
        # markets closing 20 days apart is still a candidate (the deadline gap
        # tiers apply only to time-series pairs)
        from datetime import UTC, datetime
        mA = _mock_market(
            ticker="A1", event_ticker="EVT-A",
            title="Republicans control Senate after 2026", event_title="2026 Senate Control",
            yes_ask=0.36, no_ask=0.64,
            close_time=datetime(2026, 3, 1, tzinfo=UTC),
        )
        mB = _mock_market(
            ticker="B1", event_ticker="EVT-B",
            title="Republicans control Senate after 2026", event_title="2026 Senate Control",
            yes_ask=0.30, no_ask=0.70,
            close_time=datetime(2026, 3, 21, tzinfo=UTC),
        )
        pairs = find_same_title_pairs([mA, mB])
        assert len(pairs) == 1


def _orderbook_client(*, nA_fill: float, pB_fill: float, qty: int = 100):
    """Mock KalshiClient whose order books offer depth at exactly one level.

    Buying NO on A consumes YES bids of A (NO ask = 1 - YES bid), and buying
    YES on B consumes NO bids of B (YES ask = 1 - NO bid) — so a YES bid of
    (1 - nA_fill) on A and a NO bid of (1 - pB_fill) on B yield qualifying
    depth priced at exactly nA_fill + pB_fill. Responses use the raw
    orderbook_fp JSON wire format (the SDK's modeled orderbook response can't
    deserialize live payloads anymore).
    """
    def fake_orderbook(ticker):
        if ticker == "EARLY":  # market A — YES bids become NO ask levels
            ob = {"yes_dollars": [[str(round(1.0 - nA_fill, 4)), str(qty)]],
                  "no_dollars": []}
        else:                  # market B — NO bids become YES ask levels
            ob = {"yes_dollars": [],
                  "no_dollars": [[str(round(1.0 - pB_fill, 4)), str(qty)]]}
        payload = {"orderbook_fp": ob}
        return SimpleNamespace(status=200, data=json.dumps(payload).encode("utf-8"))

    client = MagicMock()
    client.get_market_orderbook_without_preload_content = MagicMock(side_effect=fake_orderbook)
    return client


def _ts_candidate(*, gap_days: int, pA: float, pB: float, nA: float) -> CandidatePair:
    """Build a time_series CandidatePair whose legs close gap_days apart."""
    mA, mB = _ts_pair_markets(gap_days=gap_days, pA=pA, pB=pB)
    return CandidatePair(
        market_a=mA, market_b=mB,
        pA=pA, pB=pB, nA=nA,
        tradeable=True,
        canonical_title="will btc exceed $80k",
        pair_type="time_series",
    )


class TestOrderbookCeilingTieredByDeadlineGap:
    """enrich_with_orderbook_prices and validate_pair_price must apply the
    deadline-gap-tiered price-sum ceiling (0.85 for gaps <= 15 days, 0.70 for
    16-30 days), not the old flat 1 - 15% = 0.85."""

    def test_long_gap_depth_at_075_sum_marked_untradeable(self):
        # 20-day gap → ceiling 0.70. Depth priced at 0.45 + 0.30 = 0.75 would
        # have passed the old flat 0.85 ceiling but must now disqualify.
        pair = _ts_candidate(gap_days=20, pA=0.65, pB=0.30, nA=0.45)
        client = _orderbook_client(nA_fill=0.45, pB_fill=0.30)
        [enriched] = enrich_with_orderbook_prices(client, [pair])
        assert enriched.tradeable is False

    def test_short_gap_depth_at_080_sum_qualifies(self):
        # 10-day gap → ceiling 0.85. Depth priced at 0.45 + 0.35 = 0.80 qualifies
        # and the pair picks up the depth-weighted fill prices.
        pair = _ts_candidate(gap_days=10, pA=0.65, pB=0.35, nA=0.45)
        client = _orderbook_client(nA_fill=0.45, pB_fill=0.35)
        [enriched] = enrich_with_orderbook_prices(client, [pair])
        assert enriched.tradeable is True
        assert enriched.max_contracts == 100
        assert enriched.nA == pytest.approx(0.45)
        assert enriched.pB == pytest.approx(0.35)

    def test_validate_pair_price_rejects_long_gap_at_old_ceiling(self):
        # Pre-execution re-check applies the same tiered ceiling: a 20-day-gap
        # pair whose remaining depth sums to 0.80 no longer qualifies.
        pair = _ts_candidate(gap_days=20, pA=0.65, pB=0.35, nA=0.45)
        spec = SimpleNamespace(pair=pair, x=10)
        client = _orderbook_client(nA_fill=0.45, pB_fill=0.35)
        assert validate_pair_price(client, spec) is False

    def test_validate_pair_price_accepts_short_gap_at_same_depth(self):
        # Identical depth passes for a 10-day-gap pair (ceiling 0.85)
        pair = _ts_candidate(gap_days=10, pA=0.65, pB=0.35, nA=0.45)
        spec = SimpleNamespace(pair=pair, x=10)
        client = _orderbook_client(nA_fill=0.45, pB_fill=0.35)
        assert validate_pair_price(client, spec) is True


def _raw_page(events: list, cursor: str | None = None) -> SimpleNamespace:
    """Build a raw-response stand-in for a *_without_preload_content call.

    fetch_open_events_with_markets bypasses the SDK's broken Market model and
    parses the JSON body itself, so mocks provide (status, data-bytes) exactly
    like the SDK's RESTResponse.
    """
    payload = {"events": events, "cursor": cursor}
    return SimpleNamespace(status=200, data=json.dumps(payload).encode("utf-8"))


def _raw_market(ticker: str, title: str, status: str = "active", **extra) -> dict:
    m = {
        "ticker": ticker, "event_ticker": f"EVT-{ticker}", "title": title,
        "status": status, "close_time": "2026-06-01T00:00:00Z",
        "yes_ask_dollars": "0.50", "no_ask_dollars": "0.50", "yes_bid_dollars": "0.48",
    }
    m.update(extra)
    return m


class TestFetchOpenEventsMveStatusFilter:
    @pytest.mark.skipif(not INCLUDE_MVE_MARKETS, reason="MVE scanning disabled in config")
    def test_mve_active_markets_kept_others_dropped(self):
        # The API status string for an open market is "active" (allowed values:
        # initialized/active/closed/settled/determined — there is no "open").
        # Regression: comparing against "open" silently dropped every MVE market.
        mve_event = {"title": "2024 Election Winner", "markets": [
            _raw_market("MVE-ACT", "Trump", status="active"),
            _raw_market("MVE-SET", "Harris", status="settled"),
        ]}

        client = MagicMock()
        client.get_events_without_preload_content = MagicMock(return_value=_raw_page([]))
        client.get_multivariate_events_without_preload_content = MagicMock(
            return_value=_raw_page([mve_event])
        )

        markets = fetch_open_events_with_markets(client)
        tickers = {m.ticker for m in markets}
        assert "MVE-ACT" in tickers, "open MVE market (status='active') must be included"
        assert "MVE-SET" not in tickers, "settled MVE market must be excluded"
        # The parent event title must be attached for pair_key grouping
        assert markets[0]._event_title == "2024 Election Winner"

    def test_standard_events_also_filter_nested_market_status(self):
        # Regression: an "open" event can still nest a market that has already
        # resolved (e.g. one option in a multi-choice event settles while the
        # event itself stays open). status="open" on get_events() filters
        # EVENTS, not their nested markets — the standard (non-MVE) path must
        # apply the same "active" filter the MVE path already had, or a
        # stale settled market slips into pair detection.
        event = {"title": "Weather Event", "markets": [
            _raw_market("STD-ACT", "Rain tomorrow", status="active"),
            _raw_market("STD-SET", "Snow tomorrow", status="settled"),
        ]}

        client = MagicMock()
        client.get_events_without_preload_content = MagicMock(return_value=_raw_page([event]))
        client.get_multivariate_events_without_preload_content = MagicMock(
            return_value=_raw_page([])
        )

        markets = fetch_open_events_with_markets(client)
        tickers = {m.ticker for m in markets}
        assert "STD-ACT" in tickers, "open standard market (status='active') must be included"
        assert "STD-SET" not in tickers, "settled nested market must be excluded"

    def test_parsed_markets_carry_pipeline_fields(self):
        # The ApiMarket objects must expose everything downstream code reads:
        # prices as dollar strings, close_time as a tz-aware datetime, and the
        # falsy defaults (None prices, "" subtitle) the SDK model produced.
        event = {"title": "Weather Event", "markets": [
            _raw_market("STD-1", "Rain tomorrow",
                        yes_ask_dollars="0.35", no_ask_dollars="0.65",
                        close_time="2026-06-01T12:30:00Z"),
        ]}
        client = MagicMock()
        client.get_events_without_preload_content = MagicMock(return_value=_raw_page([event]))
        client.get_multivariate_events_without_preload_content = MagicMock(
            return_value=_raw_page([])
        )

        [m] = fetch_open_events_with_markets(client)
        assert float(m.yes_ask_dollars) == pytest.approx(0.35)
        assert float(m.no_ask_dollars) == pytest.approx(0.65)
        assert m.subtitle == ""                      # absent in raw JSON → ""
        assert m.close_time == datetime(2026, 6, 1, 12, 30, tzinfo=UTC)
        assert m.event_ticker == "EVT-STD-1"

    def test_pagination_follows_cursor(self):
        # Two pages on the standard endpoint: the fetch must request the second
        # page with the cursor from the first and concatenate the markets.
        page1 = _raw_page(
            [{"title": "E1", "markets": [_raw_market("PAGE1", "Q one")]}], cursor="CUR-2"
        )
        page2 = _raw_page(
            [{"title": "E2", "markets": [_raw_market("PAGE2", "Q two")]}], cursor=None
        )
        client = MagicMock()
        client.get_events_without_preload_content = MagicMock(side_effect=[page1, page2])
        client.get_multivariate_events_without_preload_content = MagicMock(
            return_value=_raw_page([])
        )

        markets = fetch_open_events_with_markets(client)
        assert {m.ticker for m in markets} == {"PAGE1", "PAGE2"}
        # Second call must have passed the cursor from page 1
        second_kwargs = client.get_events_without_preload_content.call_args_list[1].kwargs
        assert second_kwargs["cursor"] == "CUR-2"

    def test_get_held_tickers_parses_position_fp(self):
        # Positions arrive as raw JSON with position_fp strings (the SDK's
        # MarketPosition model can't deserialize live responses anymore).
        # Zero positions must be excluded; signed/fractional counts are held.
        from kalshi_betting.scanner import get_held_tickers

        payload = {"market_positions": [
            {"ticker": "HELD-NO", "position_fp": "-5"},
            {"ticker": "HELD-FRACTIONAL", "position_fp": "104.04"},
            {"ticker": "FLAT", "position_fp": "0"},
        ], "cursor": None}
        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(
            return_value=SimpleNamespace(status=200, data=json.dumps(payload).encode())
        )

        assert get_held_tickers(client) == {"HELD-NO", "HELD-FRACTIONAL"}

    @pytest.mark.skipif(not INCLUDE_MVE_MARKETS, reason="MVE scanning disabled in config")
    def test_mve_pull_bails_after_consecutive_marketless_pages(self):
        # The MVE listing is effectively unbounded and currently returns zero
        # nested markets; without the MVE_MAX_EMPTY_PAGES bail-out the fetch
        # pages forever. An endless supply of marketless MVE pages must stop
        # after exactly MVE_MAX_EMPTY_PAGES requests.
        from itertools import count

        from kalshi_betting.config import MVE_MAX_EMPTY_PAGES

        def endless_mve_pages(**kwargs):
            n = next(counter)
            return _raw_page(
                [{"title": f"MVE {n}", "markets": []}], cursor=f"CUR-{n}"
            )
        counter = count()

        client = MagicMock()
        client.get_events_without_preload_content = MagicMock(return_value=_raw_page([]))
        client.get_multivariate_events_without_preload_content = MagicMock(
            side_effect=endless_mve_pages
        )

        markets = fetch_open_events_with_markets(client)
        assert markets == []
        calls = client.get_multivariate_events_without_preload_content.call_count
        assert calls == MVE_MAX_EMPTY_PAGES, f"expected bail-out at {MVE_MAX_EMPTY_PAGES} pages, made {calls} calls"

    def test_non_2xx_raises_for_retry_helper(self):
        # The raw-response SDK variants do NOT raise on HTTP errors, so
        # _fetch_json_page must convert non-2xx statuses into ApiException —
        # otherwise api_call_with_retry can never see (and retry) 429/5xx.
        from kalshi_python_sync.exceptions import ApiException

        bad = SimpleNamespace(
            status=400,
            data=b'{"error":{"code":"bad_request"}}',
            getheaders=lambda: {},
            reason="Bad Request",
        )
        client = MagicMock()
        client.get_events_without_preload_content = MagicMock(return_value=bad)

        with pytest.raises(ApiException):
            fetch_open_events_with_markets(client)
