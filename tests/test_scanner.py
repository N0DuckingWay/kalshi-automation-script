"""Tests for scanner.py normalize_title() — the core pair-detection function."""
import dataclasses
import json
import logging
import random
import re
import sys
from dataclasses import replace as dc_replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, localcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from kalshi_python_sync.exceptions import ApiException
from urllib3.exceptions import ProtocolError

from kalshi_betting import _http, config, scanner
from kalshi_betting.config import (
    DEFAULT_EXCHANGE_INDEX,
    INCLUDE_MVE_MARKETS,
    MAX_DEADLINE_GAP_DAYS,
    PRICE_EPSILON,
    SAME_TITLE_LEG_SIDES,
    SPREAD_ABOVE_CEILING,
    SPREAD_BELOW_FLOOR,
    SPREAD_NOT_POSITIVE,
    TIME_SERIES_LEG_SIDES,
    LiveSettings,
    fee_per_pair_approx,
    kelly_budget,
    live_settings,
    live_time_series_floor,
    max_affordable_pairs,
    max_kelly_fraction,
    time_series_mid_spread,
    time_series_spread_refusal,
)
from kalshi_betting.scanner import (
    CandidatePair,
    HeldPair,
    HeldPosition,
    PriceRange,
    _bids_to_ask_levels,
    _fetch_orderbook,
    _filter_active_markets,
    _leg_ask_levels,
    _levels_with_edge_after_fee,
    _market_from_dict,
    _pair_max_sum,
    _pair_orderbooks,
    _pair_ticker,
    _parse_price_ranges,
    _reference_yes_ask,
    _shard_index,
    check_shard_coverage,
    deadline_gap_days,
    display_title,
    enrich_with_orderbook_prices,
    fetch_open_events_with_markets,
    fetch_shard_statuses,
    filter_markets_within_horizon,
    find_same_title_pairs,
    find_time_series_pairs,
    get_held_positions,
    held_pairs,
    inactive_shard_indexes,
    ladder_keys,
    leg_prices,
    leg_sides,
    market_ladder_keys,
    normalize_title,
    pair_gap_days,
    pair_held,
    pair_key,
    pair_ladder_keys,
    pair_mid_spread,
    prefix_fill_prices,
    resolve_held_ladders,
    tick_size_for_price,
    time_series_group_key,
    validate_pair_price,
)

# Balance handed to enrich_with_orderbook_prices. Deliberately far larger than
# any fixture book: the affordability cap is min(book depth, what the budget
# buys), so an ample balance makes it never bind and these tests keep exercising
# the depth path alone. Tests that mean to exercise the cap set their own.
_AMPLE_BALANCE_CENTS = 100_000_000


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

    def test_month_name_and_day_without_a_year(self):
        # The "by/before/until/through/after <Month>" pattern only consumes the
        # day when a YEAR follows it, so "by June 30" used to leave a bare "30"
        # (and "by July 31" a bare "31") behind and split two titles that
        # differ only in their deadline. The added pattern — the same
        # prepositions plus a full month name and a day — sits AHEAD of that
        # clause in _DATE_PATTERNS and takes the whole phrase.
        a = normalize_title("Will the S&P close above 6,000 by June 30?")
        b = normalize_title("Will the S&P close above 6,000 by July 31?")
        assert a == b
        assert "june" not in a
        assert "30" not in a

    def test_a_snapshot_date_with_no_deadline_preposition_is_kept(self):
        # The pattern above is anchored to the deadline preposition on
        # purpose. A bare "<Month> <day>" would also erase the date from
        # SNAPSHOT titles, merging two markets that ask about two different
        # days into one time-series group — the premise violation DR-67 now
        # screens at pair formation on the legs' wording (this pattern only
        # controls whether the two titles share a group). These two must
        # stay apart, as they did before the pattern was added.
        a = normalize_title("Highest temperature in NYC on June 30")
        b = normalize_title("Highest temperature in NYC on July 1")
        assert a != b
        assert "june 30" in a


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

    def test_appends_the_outcome_label(self):
        # DR-17: two strikes of one daily family share a title and differ only
        # in the subtitle, so the Excel "Market A"/"Market B" cells rendered
        # the same string for both legs of a cross-strike pair.
        m = _mock_market(
            ticker="KXBTCD-26SEP1517-T82749.99", event_ticker="KXBTCD-26SEP1517",
            title="Bitcoin price on Sep 15, 2026?", subtitle="$82,750 or above",
            event_title="BTC price on Sep 15, 2026 at 5pm EDT?",
        )
        assert display_title(m) == (
            "BTC price on Sep 15, 2026 at 5pm EDT?: "
            "Bitcoin price on Sep 15, 2026? — $82,750 or above"
        )

    def test_does_not_repeat_a_subtitle_that_is_already_the_label(self):
        # market_title() falls back to the subtitle when the title is empty, so
        # appending it again would render "Trump — Trump".
        m = _mock_market(
            ticker="T1", event_ticker="E1", title="", subtitle="Trump",
            event_title="2024 Election Winner",
        )
        assert display_title(m) == "2024 Election Winner: Trump"

    def test_no_subtitle_leaves_the_label_untouched(self):
        m = _mock_market(
            ticker="T1", event_ticker="E1", title="Trump", subtitle="",
            event_title="2024 Election Winner",
        )
        assert display_title(m) == "2024 Election Winner: Trump"


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
        #
        # RE-PINNED (DR-02/DR-54): the fixture used to put both markets on
        # EVT-A/EVT-B, which share the series prefix "EVT". That modelled two
        # instances of ONE recurring fixture, and the assertion below passed
        # only because the finder had no way to tell. The event tickers now
        # name two DIFFERENT series — the shape the 95% co-resolution prior was
        # built for, one question listed by two independent series — so the
        # assertion holds for the reason it always claimed to.
        mA = _mock_market(
            ticker="A1", event_ticker="EVA-1",
            title="Republicans control Senate after 2026", event_title="2026 Senate Control",
            yes_ask=0.30, no_ask=0.70,
        )
        mB = _mock_market(
            ticker="B1", event_ticker="EVB-1",
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

    def test_pricier_earlier_contract_is_not_a_candidate(self):
        # The price-gap filter is directional: pB - pA >= threshold, where B
        # is the LATER-closing contract. A pricier EARLIER contract carries no
        # market-implied in-between probability for the strategy to dispute,
        # so it must not be a candidate at all (not even as an untradeable
        # placeholder that could win the group's one-pair slot).
        #
        # RE-PINNED (DR-02/DR-54): both markets carried the SAME raw title with
        # no subtitle and no event title on EVT-A/EVT-B — one series — so the
        # one-series conjunct rejected the pair before the directional filter
        # was ever reached, and the assertion went green even with
        # min_price_diff_for_gap stubbed to -1.0. Each title now names its own
        # deadline (both still normalize to "will btc exceed $80k by", so the
        # two stay in ONE group), exactly as _ts_pair_markets does, which puts
        # the directional price filter back in charge of rejecting this pair.
        from datetime import UTC, datetime
        mA = _mock_market(  # earlier-closing, PRICIER — nothing to dispute
            ticker="EARLY", event_ticker="EVT-A",
            title="Will BTC exceed $80k by March 01, 2026",
            yes_ask=0.45, no_ask=0.55,
            close_time=datetime(2026, 3, 1, tzinfo=UTC),
        )
        mB = _mock_market(  # later-closing, cheaper
            ticker="LATE", event_ticker="EVT-B",
            title="Will BTC exceed $80k by March 20, 2026",
            yes_ask=0.20, no_ask=0.80,
            close_time=datetime(2026, 3, 20, tzinfo=UTC),
        )
        pairs = find_time_series_pairs(MagicMock(), held_tickers=set(), markets=[mA, mB])
        assert pairs == [], f"Pricier earlier contract must not be a candidate; got {pairs}"


class TestOutcomeDiscriminator:
    """DR-01: the time-series group key carries the market's outcome label.

    A daily price family lists dozens of strikes under ONE title, with the
    strike only in the subtitle. Keyed on the date-stripped title alone, every
    strike of every deadline landed in a single group, and the one-best-pair
    rule (largest pB - pA) then selected the highest earlier strike against the
    lowest later one — two different questions sized as one cumulative pair.
    time_series_group_key() appends the normalized subtitle, so a group now
    holds one OUTCOME at several deadlines.
    """

    _EARLY_CLOSE = datetime(2026, 9, 14, 21, tzinfo=UTC)
    _LATE_CLOSE = datetime(2026, 9, 18, 21, tzinfo=UTC)
    # (subtitle, earlier-event YES ask, later-event YES ask). Every same-strike
    # gap is 0.30, comfortably over the 15% short tier for this 4-day deadline
    # gap, so grouping — not price — is what decides which pairs appear.
    _STRIKES = (
        ("$180 or above", 0.10, 0.40),
        ("$190 or above", 0.09, 0.39),
        ("$200 or above", 0.08, 0.38),
        ("$210 or above", 0.07, 0.37),
    )

    def _family(self, *, strike_in_subtitle: bool = True,
                snapshot_wording: bool = False) -> list:
        """Two deadline events of one daily family, four strikes each.

        Both titles normalize to the same string, so with an empty subtitle the
        whole family collapses into one group — which is exactly the shape
        `strike_in_subtitle=False` reproduces.

        The titles are CUMULATIVE ("above $X by <date>"), because DR-01 is
        about the outcome label in the grouping key and must keep being tested
        on a family the deadline rule admits. `snapshot_wording=True` returns
        the original "Solana price ON <date>" shape — a real KXSOLD family,
        which that rule now refuses outright (see
        test_a_snapshot_family_forms_no_pairs_at_all).
        """
        early = "on Sep 14, 2026" if snapshot_wording else "by Sep 14, 2026"
        late = "on Sep 18, 2026" if snapshot_wording else "by Sep 18, 2026"
        markets = []
        for i, (strike, pA, pB) in enumerate(self._STRIKES):
            markets.append(_mock_market(
                ticker=f"KXSOLD-26SEP14-T{i}", event_ticker="KXSOLD-26SEP14",
                title=f"Solana price {early}?",
                event_title=f"Solana price {early}?",
                subtitle=strike if strike_in_subtitle else "",
                yes_ask=pA, no_ask=round(1.0 - pA, 4),
                close_time=self._EARLY_CLOSE,
            ))
            markets.append(_mock_market(
                ticker=f"KXSOLD-26SEP18-T{i}", event_ticker="KXSOLD-26SEP18",
                title=f"Solana price {late}?",
                event_title=f"Solana price {late}?",
                subtitle=strike if strike_in_subtitle else "",
                yes_ask=pB, no_ask=round(1.0 - pB, 4),
                close_time=self._LATE_CLOSE,
            ))
        return markets

    def test_a_snapshot_family_forms_no_pairs_at_all(self):
        # The wording this fixture ORIGINALLY carried, and a real KXSOLD
        # family: "Solana price ON Sep 14/18, 2026?". Both titles normalize to
        # "solana price on ?" so all eight markets still GROUP together — the
        # grouping key is deliberately untouched — but no pair survives,
        # because SOL >= $180 on Sep 14 does not imply SOL >= $180 on Sep 18
        # and the trade has no premise. This is the counterpart of the
        # cross-strike pins below: they check WHICH cumulative pairs form,
        # this checks that a snapshot family forms none.
        assert find_time_series_pairs(
            MagicMock(), held_tickers=set(),
            markets=self._family(snapshot_wording=True),
        ) == []

    def test_one_pair_per_strike_and_no_cross_strike_pair(self):
        pairs = find_time_series_pairs(
            MagicMock(), held_tickers=set(), markets=self._family(),
        )
        assert len(pairs) == len(self._STRIKES)
        for p in pairs:
            assert p.market_a.subtitle == p.market_b.subtitle, (
                f"cross-strike pair: {p.market_a.ticker} vs {p.market_b.ticker}"
            )
            # market_a is the EARLIER contract, as find_time_series_pairs sorts
            assert p.market_a.event_ticker == "KXSOLD-26SEP14"
            assert p.market_b.event_ticker == "KXSOLD-26SEP18"
        assert {p.market_a.subtitle for p in pairs} == {s for s, _, _ in self._STRIKES}

    def test_without_the_discriminator_the_family_collapses_to_one_pair(self):
        # The defect, reproduced: with no outcome label to key on, the eight
        # markets are one group and the best-pair rule picks the WIDEST strike
        # mismatch — the cheapest earlier contract ($210, pA 0.07) against the
        # dearest later one ($180, pB 0.40).
        pairs = find_time_series_pairs(
            MagicMock(), held_tickers=set(),
            markets=self._family(strike_in_subtitle=False),
        )
        assert len(pairs) == 1
        assert pairs[0].market_a.ticker == "KXSOLD-26SEP14-T3"
        assert pairs[0].market_b.ticker == "KXSOLD-26SEP18-T0"

    def test_mve_option_label_at_two_deadlines_still_pairs(self):
        # The subtitle must not break the case it was added to protect: one
        # option label across two deadline events of the same event series.
        #
        # RE-PINNED (DR-02/DR-54): both event titles used to read
        # "Presidential Election Winner", so the raw (title, subtitle, event
        # title) triple was identical on both legs and the deadline lived only
        # in the event ticker — which the time-series conjunct now refuses,
        # because identical wording across one series is two fixtures, not one
        # question at two deadlines. Each event title now names its own
        # deadline, exactly as a real two-deadline family does; normalize_title
        # strips those dates, so both legs still collapse into ONE group key
        # ("presidential election winner by | trump | donald trump") and the
        # pair is still formed — the key itself gains the stranded "by".
        mA = _mock_market(
            ticker="MAR-TRUMP", event_ticker="ELECT-MAR", title="Trump",
            event_title="Presidential Election Winner by March 2026",
            subtitle="Donald Trump",
            yes_ask=0.30, no_ask=0.70, close_time=datetime(2026, 3, 1, tzinfo=UTC),
        )
        mB = _mock_market(
            ticker="JUN-TRUMP", event_ticker="ELECT-JUN", title="Trump",
            event_title="Presidential Election Winner by June 2026",
            subtitle="Donald Trump",
            yes_ask=0.60, no_ask=0.40, close_time=datetime(2026, 3, 11, tzinfo=UTC),
        )
        pairs = find_time_series_pairs(MagicMock(), held_tickers=set(), markets=[mA, mB])
        assert len(pairs) == 1

    def test_trailing_punctuation_in_the_outcome_label_is_one_key(self):
        # Each event title names its own deadline, as a real two-deadline
        # family does — without one the wording states no deadline anywhere and
        # the cumulative-deadline rule refuses the pair before the subtitle
        # normalization under test is ever reached. Same re-pinning
        # test_mve_option_label_at_two_deadlines_still_pairs needed for DR-02.
        mA = _mock_market(
            ticker="MAR-TRUMP", event_ticker="ELECT-MAR", title="Trump",
            event_title="Presidential Election Winner by March 2026",
            subtitle="Donald Trump",
            yes_ask=0.30, no_ask=0.70, close_time=datetime(2026, 3, 1, tzinfo=UTC),
        )
        mB = _mock_market(
            ticker="JUN-TRUMP", event_ticker="ELECT-JUN", title="Trump",
            event_title="Presidential Election Winner by June 2026",
            subtitle="Donald Trump.",
            yes_ask=0.60, no_ask=0.40, close_time=datetime(2026, 3, 11, tzinfo=UTC),
        )
        pairs = find_time_series_pairs(MagicMock(), held_tickers=set(), markets=[mA, mB])
        assert len(pairs) == 1

    def test_dated_outcome_label_is_one_key(self):
        # A deadline spelled inside the OUTCOME label is still a deadline: the
        # explicit-date subset strips "June 30"/"July 31" so one strike at two
        # deadlines stays one group.
        mA = _mock_market(
            ticker="JUN-80K", event_ticker="EVT-JUN", title="Will BTC hit a new high",
            event_title="BTC milestones", subtitle="$80,000 by June 30",
            yes_ask=0.30, no_ask=0.70, close_time=datetime(2026, 6, 30, tzinfo=UTC),
        )
        mB = _mock_market(
            ticker="JUL-80K", event_ticker="EVT-JUL", title="Will BTC hit a new high",
            event_title="BTC milestones", subtitle="$80,000 by July 31",
            yes_ask=0.60, no_ask=0.40, close_time=datetime(2026, 7, 10, tzinfo=UTC),
        )
        pairs = find_time_series_pairs(MagicMock(), held_tickers=set(), markets=[mA, mB])
        assert len(pairs) == 1

    def test_strike_spelled_in_the_title_still_pairs(self):
        # The cumulative case the strategy exists for — the strike is in the
        # TITLE, the subtitle is empty on both legs, so the key is unchanged.
        mA = _mock_market(
            ticker="MAR-80K", event_ticker="EVT-MAR",
            title="Will BTC exceed $80k by March 2026?",
            yes_ask=0.30, no_ask=0.70, close_time=datetime(2026, 3, 1, tzinfo=UTC),
        )
        mB = _mock_market(
            ticker="JUN-80K", event_ticker="EVT-JUN",
            title="Will BTC exceed $80k by June 2026?",
            yes_ask=0.60, no_ask=0.40, close_time=datetime(2026, 3, 11, tzinfo=UTC),
        )
        pairs = find_time_series_pairs(MagicMock(), held_tickers=set(), markets=[mA, mB])
        assert len(pairs) == 1


class TestOneEventSeriesIsTwoFixtures:
    """DR-02 / DR-54: identical wording across two events of ONE series.

    A Kalshi event ticker is a series prefix followed by an instance stamp, so
    two events of one series are two instances of one recurring fixture — two
    ball games, two 15-minute price windows, two combos. Identical wording
    across them is the same question asked about two DIFFERENT events, not one
    question listed twice, so the SAME_TITLE_CO_RESOLVE_PROB prior does not
    apply: the sweep's own NPB pair was quoted at yes asks 0.97 and 0.01.

    Both finders must refuse it. Gating only find_same_title_pairs would merely
    relabel the trade: the same two tickers also clear the time-series filter
    (gap 1 day, pB - pA = 0.96), and main._dedup_pairs only ever dropped that
    copy because a same-title copy existed.
    """

    # The real pair the 2026-09-15 prod dry run selected and sized: one
    # fixture listed on two game days, identical title, identical outcome
    # label, closes 24 h apart. yes_ask 0.97 vs 0.01 — a 96-point
    # "divergence" that is simply two different games.
    _NPB_TITLE = "Fukuoka Hawks vs Orix Buffaloes: First Inning Run?"
    _NPB_EVENT_TITLE = "Fukuoka Hawks vs Orix Buffaloes: First Inning Run"

    @classmethod
    def _npb_markets(cls):
        close = datetime(2026, 9, 17, 9, tzinfo=UTC)
        mA = _mock_market(
            ticker="KXNPBRFI-26SEP160500FUKORI-Y",
            event_ticker="KXNPBRFI-26SEP160500FUKORI",
            title=cls._NPB_TITLE, subtitle="Yes", event_title=cls._NPB_EVENT_TITLE,
            yes_ask=0.97, no_ask=0.03,
            close_time=close + timedelta(days=1),
        )
        mB = _mock_market(
            ticker="KXNPBRFI-26SEP150500FUKORI-Y",
            event_ticker="KXNPBRFI-26SEP150500FUKORI",
            title=cls._NPB_TITLE, subtitle="Yes", event_title=cls._NPB_EVENT_TITLE,
            yes_ask=0.01, no_ask=0.99,
            close_time=close,
        )
        return mA, mB

    def test_same_title_finder_rejects_two_game_days_of_one_fixture(self):
        mA, mB = self._npb_markets()
        assert find_same_title_pairs([mA, mB]) == []

    def test_time_series_finder_rejects_the_same_two_markets(self):
        # Without this the same-title gate would only RELABEL the trade: the
        # markets close 1 day apart and pB - pA = 0.96 clears the 15% tier.
        mA, mB = self._npb_markets()
        assert find_time_series_pairs(MagicMock(), held_tickers=set(), markets=[mA, mB]) == []

    def test_dedup_has_nothing_left_to_relabel(self):
        # End state on the live path: both finders come back empty, so the
        # merge main performs produces no trade at all.
        from kalshi_betting.main import _dedup_pairs
        mA, mB = self._npb_markets()
        same = find_same_title_pairs([mA, mB])
        ts = find_time_series_pairs(MagicMock(), held_tickers=set(), markets=[mA, mB])
        assert _dedup_pairs(same, ts) == []

    def test_the_skip_is_reported_once_with_a_count(self, caplog):
        mA, mB = self._npb_markets()
        with caplog.at_level(logging.INFO):
            find_same_title_pairs([mA, mB])
        lines = [r.getMessage() for r in caplog.records
                 if "two instances of one event series" in r.getMessage()]
        assert len(lines) == 1
        assert lines[0].endswith(": 1")

    def test_the_time_series_skip_is_reported_once_per_candidate(self, caplog):
        # M10: the time-series finder's one-series conjunct used to `continue`
        # with no count, so a run whose time-series zero it caused logged no
        # cause at all (the same-title finder has always counted its twin).
        # THREE game days of one fixture make three candidates, so a counter
        # bumped once per GROUP (which would read 1) cannot pass.
        mA, mB = self._npb_markets()
        mC = _mock_market(
            ticker="KXNPBRFI-26SEP140500FUKORI-Y",
            event_ticker="KXNPBRFI-26SEP140500FUKORI",
            title=self._NPB_TITLE, subtitle="Yes", event_title=self._NPB_EVENT_TITLE,
            yes_ask=0.50, no_ask=0.50,
            close_time=mB.close_time - timedelta(days=1),
        )
        with caplog.at_level(logging.INFO):
            assert find_time_series_pairs(
                MagicMock(), held_tickers=set(), markets=[mA, mB, mC],
            ) == []
        assert [m for m in caplog.messages if "two instances of one event series" in m] == [
            "Time-series candidates skipped as two instances of one event series "
            "(identical wording, different fixture; counted before the gap and "
            "price filters): 3"
        ]

    def test_the_time_series_skip_is_silent_at_zero(self, caplog):
        # control: a dated pair of one series is not identical wording, so
        # the conjunct never fires and nothing is reported.
        mA, mB = self._sold_family("by")
        with caplog.at_level(logging.INFO):
            assert len(find_time_series_pairs(
                MagicMock(), held_tickers=set(), markets=[mA, mB],
            )) == 1
        assert "one event series" not in caplog.text

    def test_a_same_event_same_title_skip_is_reported_once_per_candidate(self, caplog):
        # M10: two markets of ONE event under one (event_title, title,
        # subtitle) key are that event's own markets, not one question listed
        # by two events. The skip was this finder's one uncounted refusal.
        # Three such markets make three candidates, none of which reaches the
        # series rule — so the two counters cannot stand in for each other.
        markets = [
            _mock_market(ticker=f"KXDUP-26-{i}", event_ticker="KXDUP-26",
                         title="Q", subtitle="Yes", event_title="E",
                         yes_ask=round(0.2 + 0.2 * i, 2),
                         no_ask=round(0.8 - 0.2 * i, 2))
            for i in range(3)
        ]
        with caplog.at_level(logging.INFO):
            assert find_same_title_pairs(markets) == []
        assert caplog.messages.count(
            "Same-title candidates skipped because both markets carry the same "
            "event ticker (one event's own markets, not one question listed by "
            "two events): 3"
        ) == 1
        assert "one event series" not in caplog.text

    def test_two_matches_of_one_table_tennis_player_are_rejected(self):
        # Two matches of one player within half an hour of each other,
        # which settled yes and no — the shape a close-time gate would have
        # admitted, which is why the rule is fixture identity instead.
        close = datetime(2026, 9, 10, 18, 35, tzinfo=UTC)
        mA = _mock_market(
            ticker="KXTTELITEMATCH-26SEP101835KKAMOL-MOL",
            event_ticker="KXTTELITEMATCH-26SEP101835KKAMOL",
            title="Michal Olbrycht wins", subtitle="Yes",
            event_title="TT Elite Series", yes_ask=0.60, no_ask=0.40,
            close_time=close,
        )
        mB = _mock_market(
            ticker="KXTTELITEMATCH-26SEP101810MOLJMI-MOL",
            event_ticker="KXTTELITEMATCH-26SEP101810MOLJMI",
            title="Michal Olbrycht wins", subtitle="Yes",
            event_title="TT Elite Series", yes_ask=0.20, no_ask=0.80,
            close_time=close - timedelta(minutes=25),
        )
        assert find_same_title_pairs([mA, mB]) == []
        assert find_time_series_pairs(MagicMock(), held_tickers=set(), markets=[mA, mB]) == []

    def test_two_fifteen_minute_price_windows_are_rejected(self):
        # Two consecutive intraday windows of one series: identical wording,
        # zero-day deadline gap, which is a valid short-tier gap.
        close = datetime(2026, 9, 15, 14, tzinfo=UTC)
        mA = _mock_market(
            ticker="KXNATGASMAX-26SEP1514-T3.10", event_ticker="KXNATGASMAX-26SEP1514",
            title="Natural gas price high?", subtitle="$3.10 or above",
            event_title="Natural gas price high", yes_ask=0.20, no_ask=0.80,
            close_time=close,
        )
        mB = _mock_market(
            ticker="KXNATGASMAX-26SEP1515-T3.10", event_ticker="KXNATGASMAX-26SEP1515",
            title="Natural gas price high?", subtitle="$3.10 or above",
            event_title="Natural gas price high", yes_ask=0.80, no_ask=0.20,
            close_time=close + timedelta(minutes=15),
        )
        assert find_same_title_pairs([mA, mB]) == []
        assert find_time_series_pairs(MagicMock(), held_tickers=set(), markets=[mA, mB]) == []

    def test_two_combo_events_are_rejected(self):
        # DR-54: a combo ticket's wording names its legs but never its date, so
        # one wording recurs across fixture instances and two tickets with
        # identical leg wording are two DIFFERENT tickets. Since DR-55 these
        # two do not resolve to a literal prefix at all: both event tickers
        # start with config.MVE_SERIES_FAMILY_PREFIX, so event_series answers
        # "KXMVE" for each and _same_series sees one family. (Before DR-55 the
        # same verdict came from the literal "KXMVECROSSCATEGORY" prefix —
        # which is why the cross-prefix case below needed its own test.)
        close = datetime(2026, 9, 15, 20, tzinfo=UTC)
        mA = _mock_market(
            ticker="KXMVECROSSCATEGORY-SHARD1-S6471E4699E9-Y",
            event_ticker="KXMVECROSSCATEGORY-SHARD1-S6471E4699E9",
            title="Parlay", subtitle="All legs hit",
            event_title="Cross-category combo", yes_ask=0.45, no_ask=0.55,
            close_time=close,
        )
        mB = _mock_market(
            ticker="KXMVECROSSCATEGORY-SHARD1-S93FFD638F77-Y",
            event_ticker="KXMVECROSSCATEGORY-SHARD1-S93FFD638F77",
            title="Parlay", subtitle="All legs hit",
            event_title="Cross-category combo", yes_ask=0.20, no_ask=0.80,
            close_time=close + timedelta(minutes=2),
        )
        assert find_same_title_pairs([mA, mB]) == []
        assert find_time_series_pairs(MagicMock(), held_tickers=set(), markets=[mA, mB]) == []

    @staticmethod
    def _cross_prefix_combo_markets():
        # DR-55: the SAME combo wording listed under two DIFFERENT KXMVE*
        # series. Measured in backtest_cache/live_days/2026-09-14.json.gz,
        # 10,643 distinct (event_title, title, subtitle) wordings appear under
        # both KXMVECROSSCATEGORY and KXMVECROSSCATEGORY0. Before the family
        # collapse these read as two different series, so the one-series rule
        # never fired and both finders formed the pair.
        #
        # The prices are deliberately chosen so that NOTHING ELSE rejects the
        # pair: the YES asks diverge 0.25 (>= SAME_TITLE_MIN_PRICE_DIFF for the
        # same-title finder) and, with the earlier leg the cheaper one, give
        # pB - pA = 0.25 at a zero-day gap (pA + nB = 0.75) for the time-series
        # finder, which both the tier-on rule and config.py's admit.
        close = datetime(2026, 9, 15, 20, tzinfo=UTC)
        earlier = _mock_market(
            ticker="KXMVECROSSCATEGORY-SHARD1-S6471E4699E9-Y",
            event_ticker="KXMVECROSSCATEGORY-SHARD1-S6471E4699E9",
            title="Parlay", subtitle="All legs hit",
            event_title="Cross-category combo", yes_ask=0.20, no_ask=0.80,
            close_time=close,
        )
        later = _mock_market(
            ticker="KXMVECROSSCATEGORY0-SHARD1-S93FFD638F77-Y",
            event_ticker="KXMVECROSSCATEGORY0-SHARD1-S93FFD638F77",
            title="Parlay", subtitle="All legs hit",
            event_title="Cross-category combo", yes_ask=0.45, no_ask=0.55,
            close_time=close + timedelta(hours=2),
        )
        return earlier, later

    def test_same_title_finder_rejects_two_combo_series(self):
        earlier, later = self._cross_prefix_combo_markets()
        # Guard that the fixture is the shape the rule must catch: identical
        # wording, two literal prefixes, one collapsed series.
        assert scanner._identical_wording(earlier, later) is True
        assert earlier.event_ticker.split("-")[0] != later.event_ticker.split("-")[0]
        assert scanner._same_series(earlier, later) is True
        assert find_same_title_pairs([earlier, later]) == []

    def test_time_series_finder_rejects_two_combo_series(self):
        # This direction matters on its own: _same_series is also the second
        # half of find_time_series_pairs' skip conjunct, so gating only the
        # same-title finder would relabel the trade rather than remove it.
        earlier, later = self._cross_prefix_combo_markets()
        assert find_time_series_pairs(
            MagicMock(), held_tickers=set(), markets=[earlier, later]
        ) == []

    def test_two_different_series_asking_one_question_still_pair(self):
        # The premise of the same-title strategy, and the only shape that
        # survives the rule: one question listed by two INDEPENDENT series.
        mA = _mock_market(
            ticker="KXFEDDEC-26-T25", event_ticker="KXFEDDEC-26",
            title="Fed cuts rates in December?", subtitle="Yes",
            event_title="Fed December decision", yes_ask=0.40, no_ask=0.60,
        )
        mB = _mock_market(
            ticker="FEDCUTDEC-26-T25", event_ticker="FEDCUTDEC-26",
            title="Fed cuts rates in December?", subtitle="Yes",
            event_title="Fed December decision", yes_ask=0.30, no_ask=0.70,
        )
        pairs = find_same_title_pairs([mA, mB])
        assert len(pairs) == 1
        assert {pairs[0].market_a.ticker, pairs[0].market_b.ticker} == {
            "KXFEDDEC-26-T25", "FEDCUTDEC-26-T25"
        }

    @staticmethod
    def _sold_family(preposition: str) -> tuple:
        """One KXSOLD strike listed by two deadline events of ONE series."""
        mA = _mock_market(
            ticker="KXSOLD-26SEP14-T180", event_ticker="KXSOLD-26SEP14",
            title=f"Solana price {preposition} Sep 14, 2026?", subtitle="$180 or above",
            event_title="Solana price", yes_ask=0.30, no_ask=0.70,
            close_time=datetime(2026, 9, 14, 21, tzinfo=UTC),
        )
        mB = _mock_market(
            ticker="KXSOLD-26SEP18-T180", event_ticker="KXSOLD-26SEP18",
            title=f"Solana price {preposition} Sep 18, 2026?", subtitle="$180 or above",
            event_title="Solana price", yes_ask=0.60, no_ask=0.40,
            close_time=datetime(2026, 9, 18, 21, tzinfo=UTC),
        )
        return mA, mB

    def test_a_dated_pair_of_one_series_is_untouched(self):
        # The deadline lives IN the wording, so _identical_wording is False and
        # the time-series conjunct never fires — a daily family's two deadline
        # events are two events of one series and must keep pairing.
        #
        # Re-pinned on CUMULATIVE wording: this asserts what the ONE-SERIES
        # rule does, and it must keep doing it on a pair the deadline rule
        # admits, or the assertion would pass for the wrong reason.
        mA, mB = self._sold_family("by")
        pairs = find_time_series_pairs(MagicMock(), held_tickers=set(), markets=[mA, mB])
        assert len(pairs) == 1
        assert (pairs[0].market_a.ticker, pairs[0].market_b.ticker) == (
            "KXSOLD-26SEP14-T180", "KXSOLD-26SEP18-T180"
        )

    def test_the_snapshot_spelling_of_that_same_family_is_now_refused(self):
        # The question the comment on this test used to defer. A
        # "price ON <date>" family is a SNAPSHOT family, not a
        # cumulative-deadline one (SOL >= $180 on Sep 14 does not imply
        # SOL >= $180 on Sep 18), so there is no in-between mass to dispute and
        # nesting is what would make the fair value of YES-A + NO-B at most
        # $1 — here it doesn't. The one-series rule still does not fire — the
        # two legs are worded differently — so this
        # is the deadline rule's verdict alone, on the very fixture that used
        # to document the gap.
        mA, mB = self._sold_family("on")
        assert scanner._identical_wording(mA, mB) is False
        assert find_time_series_pairs(
            MagicMock(), held_tickers=set(), markets=[mA, mB],
        ) == []

    def test_skipping_the_widest_candidate_promotes_the_next_one(self):
        # The rule is not only subtractive. Both finders keep ONE best pair per
        # group, so removing a group's top candidate promotes the runner-up:
        # this scanner can now propose a pair on tickers the previous code
        # never proposed at all.
        #
        # A and B are identically worded on one series (KXZ) and were the
        # group's pick at gap 0.67 — and its same-title pick too. C dates its
        # own deadline, so A/C survives the conjunct and wins the slot at gap
        # 0.30. That promotion is intended: A/C is the pair the group actually
        # supports once the look-alike is gone.
        base = datetime(2026, 12, 1, tzinfo=UTC)
        look_alike_title = "Will Z happen by December 1, 2026?"
        mA = _mock_market(
            ticker="KXZ-A-T", event_ticker="KXZ-A", title=look_alike_title,
            event_title="Recurring E", yes_ask=0.30, no_ask=0.70,
            close_time=base,
        )
        mB = _mock_market(
            ticker="KXZ-B-T", event_ticker="KXZ-B", title=look_alike_title,
            event_title="Recurring E", yes_ask=0.97, no_ask=0.03,
            close_time=base + timedelta(days=1),
        )
        mC = _mock_market(
            ticker="KXZ-C-T", event_ticker="KXZ-C",
            title="Will Z happen by December 11, 2026?",
            event_title="Recurring E", yes_ask=0.60, no_ask=0.40,
            close_time=base + timedelta(days=10),
        )
        [pair] = find_time_series_pairs(
            MagicMock(), held_tickers=set(), markets=[mA, mB, mC],
        )
        assert (pair.market_a.ticker, pair.market_b.ticker) == ("KXZ-A-T", "KXZ-C-T")
        assert pair.pB - pair.pA == pytest.approx(0.30)
        # The look-alike is gone from the same-title side too, so dedup has
        # nothing to prefer over the promoted time-series pair.
        assert find_same_title_pairs([mA, mB, mC]) == []

    def test_a_differing_subtitle_breaks_the_identical_wording_half(self):
        # GUARD on the conjunct's wording half: same series, same title, but
        # the outcome labels differ, so the raw triple does not match and
        # _identical_wording is False — the series alone never rejects a
        # time-series candidate.
        mA, mB = self._npb_markets()
        mB.subtitle = "No"
        assert scanner._same_series(mA, mB) is True
        assert scanner._identical_wording(mA, mB) is False


class TestEventSeries:
    """The one-series primitives on their own: scanner.event_series (the series
    identity — the literal prefix, with every combo (KXMVE*) prefix collapsed
    onto one family), _same_series (fixture identity, failing CLOSED on an unreadable
    ticker) and _identical_wording (the raw title/subtitle/event-title
    triple), plus the silent-at-zero half of find_same_title_pairs' skip
    summary."""

    @pytest.mark.parametrize("ticker,expected", [
        ("KXNPBRFI-26SEP160500FUKORI", "KXNPBRFI"),
        ("KXTTELITEMATCH-26SEP101835KKAMOL", "KXTTELITEMATCH"),
        # DR-55: all four real combo prefixes collapse onto the KXMVE family.
        # Kalshi lists combos under several series, so the literal prefix split
        # them into four and the one-series rule never fired between two of
        # them. Census of backtest_cache/event_titles.json, 2026-09-16:
        # KXMVECROSSCATEGORY 2,960,840 | KXMVESPORTSMULTIGAMEEXTENDED 906,157 |
        # KXMVECROSSCATEGORY0 94,230 | KXMVENBASINGLEGAME 61.
        ("KXMVECROSSCATEGORY-SHARD1-S6471E4699E9", "KXMVE"),
        ("KXMVECROSSCATEGORY0-SHARD1-S93FFD638F77", "KXMVE"),
        ("KXMVESPORTSMULTIGAMEEXTENDED-SHARD1-SA1B2C3D4E5F", "KXMVE"),
        ("KXMVENBASINGLEGAME-26SEP15LALBOS", "KXMVE"),
        # The uncollapsed path stays pinned: a non-MVE ticker is its own series.
        ("KXSOLD-26SEP14", "KXSOLD"),
    ])
    def test_real_event_tickers(self, ticker, expected):
        assert scanner.event_series(ticker) == expected

    def test_the_combo_family_collapses_but_neighbours_do_not(self):
        # GUARD on the collapse's blast radius: it keys off the family prefix,
        # so two combo series answer with one identity while ordinary series —
        # including one that stops one letter short of the family — are
        # untouched. Only KXMVE* collapses: no KXMV* prefix that is not KXMVE*
        # exists in the census, so the family name is the narrowest one
        # covering all four.
        assert scanner.event_series("KXMVECROSSCATEGORY-A") == scanner.event_series(
            "KXMVESPORTSMULTIGAMEEXTENDED-B"
        )
        assert scanner.event_series("KXNPBRFI-A") != scanner.event_series("KXSOLD-B")
        # The KXMV/KXMVE boundary itself: the family literal is narrow, so a
        # prefix that stops one letter short is NOT collapsed.
        assert scanner.event_series("KXMVPAWARD-26") == "KXMVPAWARD"
        # Case is normalized before the family test, so a lower-cased combo
        # ticker collapses too.
        assert scanner.event_series("kxmvecrosscategory0-shard1-x") == "KXMVE"

    def test_hyphenless_ticker_is_its_own_series(self):
        assert scanner.event_series("KXSOLD") == "KXSOLD"

    def test_case_is_normalized(self):
        assert scanner.event_series("kxsold-26sep14") == "KXSOLD"

    def test_surrounding_whitespace_is_trimmed(self):
        # The prefix is stripped after the split, so a padded ticker still
        # resolves to the same series as the clean one.
        assert scanner.event_series("  kxnpbrfi -26SEP160500FUKORI") == "KXNPBRFI"
        assert scanner.event_series("   ") == ""

    def test_empty_and_non_string_read_as_unknown(self):
        assert scanner.event_series("") == ""
        assert scanner.event_series(None) == ""
        assert scanner.event_series(MagicMock().event_ticker) == ""
        # A ticker that is only a hyphen has no prefix either
        assert scanner.event_series("-26SEP14") == ""

    def test_same_series_fails_closed_on_an_unreadable_ticker(self):
        # A pair whose fixture identity cannot be read must NOT be priced on
        # the co-resolution prior, so an unknown series reads as "same".
        known = SimpleNamespace(event_ticker="KXSOLD-26SEP14")
        blank = SimpleNamespace(event_ticker="")
        missing = SimpleNamespace(event_ticker=None)
        assert scanner._same_series(known, blank) is True
        assert scanner._same_series(blank, known) is True
        assert scanner._same_series(known, missing) is True
        assert scanner._same_series(blank, missing) is True

    def test_same_series_is_false_only_for_two_known_distinct_series(self):
        a = SimpleNamespace(event_ticker="KXSOLD-26SEP14")
        b = SimpleNamespace(event_ticker="KXBTCD-26SEP14")
        assert scanner._same_series(a, b) is False
        assert scanner._same_series(a, SimpleNamespace(event_ticker="KXSOLD-26SEP18")) is True

    def test_identical_wording_reads_all_three_strings(self):
        # Every one of title / subtitle / event title on its own is enough to
        # make two markets differently worded — which is what lets a genuine
        # cumulative pair through the time-series conjunct.
        base = {"title": "T", "subtitle": "S", "event_title": "E"}
        a = _mock_market(ticker="A1", event_ticker="KXA-1", **base)
        assert scanner._identical_wording(
            a, _mock_market(ticker="B1", event_ticker="KXB-1", **base)) is True
        for differing in ({"title": "T2"}, {"subtitle": "S2"}, {"event_title": "E2"}):
            other = _mock_market(ticker="B1", event_ticker="KXB-1", **{**base, **differing})
            assert scanner._identical_wording(a, other) is False, differing

    def test_the_skip_summary_is_silent_when_nothing_is_skipped(self, caplog):
        # Same idiom as the trading-inactive shard skip count: one summary line
        # when there is something to report, and no line at all at zero.
        mA = _mock_market(ticker="A1", event_ticker="KXFEDDEC-26", title="Q",
                          subtitle="Yes", event_title="E", yes_ask=0.40, no_ask=0.60)
        mB = _mock_market(ticker="B1", event_ticker="FEDCUTDEC-26", title="Q",
                          subtitle="Yes", event_title="E", yes_ask=0.30, no_ask=0.70)
        with caplog.at_level(logging.INFO):
            assert len(find_same_title_pairs([mA, mB])) == 1
        assert "one event series" not in caplog.text
        # M10's same-event count follows the same idiom.
        assert "same event ticker" not in caplog.text


class TestSameTitleCloseGap:
    """DR-74: identical wording on two DIFFERENT series is one question only
    when both markets close at the same moment.

    The one-series rule (DR-02/DR-54) is necessary but not sufficient. A men's
    and a women's college basketball game between the same two schools
    (KXNCAAMBGAME / KXNCAAWBGAME) share title, subtitle and event title on two
    different series; so do the Champions League and La Liga fixtures of one
    matchup. They close hours or days apart, and the 95% co-resolution prior
    is false for them: on the 365-day backtest 17 of 21 trades were such M/W
    pairs. scanner.closes_apart is the one definition of the gate, and
    find_same_title_pairs reads it through _closes_apart, right after the
    series test. Live close_time is the SCHEDULED close. Mirror:
    test_backtester.py::TestSameTitleCloseGapBacktest.
    """

    _WIU_TITLE = "Western Illinois at Eastern Illinois Winner?"
    _REFUSAL = (
        "Same-title candidates refused because the two markets close more than "
        "60 minutes apart (two different games or instants, not one question "
        "listed twice): "
    )

    @classmethod
    def _wiu_markets(cls, *, womens_close=None):
        # The real pair: KXNCAAMBGAME-26JAN13WIUEIU-WIU vs
        # KXNCAAWBGAME-26JAN13WIUEIU-WIU — identical title, subtitle and event
        # title on two series. Their SCHEDULED closes (what live reads) fall at
        # 01:30Z and 23:00Z, 2.5 h apart, as checked on archived payloads; the
        # calendar day here is the fixture's own.
        mens = _mock_market(
            ticker="KXNCAAMBGAME-26JAN13WIUEIU-WIU",
            event_ticker="KXNCAAMBGAME-26JAN13WIUEIU",
            title=cls._WIU_TITLE, subtitle="Western Illinois",
            event_title="Western Illinois at Eastern Illinois",
            yes_ask=0.40, no_ask=0.60,
            close_time=datetime(2026, 1, 28, 1, 30, tzinfo=UTC),
        )
        womens = _mock_market(
            ticker="KXNCAAWBGAME-26JAN13WIUEIU-WIU",
            event_ticker="KXNCAAWBGAME-26JAN13WIUEIU",
            title=cls._WIU_TITLE, subtitle="Western Illinois",
            event_title="Western Illinois at Eastern Illinois",
            yes_ask=0.25, no_ask=0.75,
            close_time=womens_close or datetime(2026, 1, 27, 23, 0, tzinfo=UTC),
        )
        return mens, womens

    @staticmethod
    def _refusals(caplog) -> list[str]:
        return [m for m in caplog.messages if "close more than" in m]

    def test_mens_and_womens_game_are_refused_and_counted_once(self, caplog):
        mens, womens = self._wiu_markets()
        # The fixture is the shape the gate must catch: identical wording,
        # two different series, so the one-series rule does NOT fire.
        assert scanner._identical_wording(mens, womens) is True
        assert scanner._same_series(mens, womens) is False
        with caplog.at_level(logging.INFO):
            assert find_same_title_pairs([mens, womens]) == []
        assert self._refusals(caplog) == [self._REFUSAL + "1"]

    def test_the_same_game_pair_at_one_close_instant_forms(self):
        # Positive control: move the women's game onto the men's close and
        # nothing else — the pair forms, so the refusal above is the gate.
        mens, womens = self._wiu_markets(womens_close=datetime(2026, 1, 28, 1, 30, tzinfo=UTC))
        [pair] = find_same_title_pairs([mens, womens])
        assert {pair.market_a.ticker, pair.market_b.ticker} == {mens.ticker, womens.ticker}

    def test_champions_league_and_la_liga_fixtures_are_refused(self):
        # Two competitions' fixtures of one matchup, ten days apart.
        title = "Atletico Madrid vs Barcelona Winner?"
        ucl = _mock_market(
            ticker="KXUCLGAME-26OCT21ATMBAR-BAR", event_ticker="KXUCLGAME-26OCT21ATMBAR",
            title=title, subtitle="Barcelona", event_title="Atletico Madrid vs Barcelona",
            yes_ask=0.55, no_ask=0.45, close_time=datetime(2026, 10, 21, 21, tzinfo=UTC),
        )
        liga = _mock_market(
            ticker="KXLALIGAGAME-26OCT31ATMBAR-BAR", event_ticker="KXLALIGAGAME-26OCT31ATMBAR",
            title=title, subtitle="Barcelona", event_title="Atletico Madrid vs Barcelona",
            yes_ask=0.40, no_ask=0.60, close_time=datetime(2026, 10, 31, 20, tzinfo=UTC),
        )
        assert scanner._same_series(ucl, liga) is False
        assert find_same_title_pairs([ucl, liga]) == []

    def test_weekly_and_monthly_listings_at_one_close_still_pair(self):
        # The shape the same-title strategy was built for, and what the
        # 365-day backtest's 3 surviving trades were: one Brent question listed
        # by a weekly and a monthly series, both closing at 21:00Z the same
        # day. (Series names here are stand-ins for the weekly and monthly
        # Brent series.)
        close = datetime(2026, 9, 25, 21, tzinfo=UTC)
        title = "Brent crude oil price on Sep 25, 2026?"
        weekly = _mock_market(
            ticker="KXBRENTW-26SEP25-T70", event_ticker="KXBRENTW-26SEP25",
            title=title, subtitle="$70 or above", event_title="Brent crude oil price",
            yes_ask=0.45, no_ask=0.55, close_time=close,
        )
        monthly = _mock_market(
            ticker="KXBRENTMON-26SEP-T70", event_ticker="KXBRENTMON-26SEP",
            title=title, subtitle="$70 or above", event_title="Brent crude oil price",
            yes_ask=0.35, no_ask=0.65, close_time=close,
        )
        assert scanner._closes_apart(weekly, monthly) is False
        assert len(find_same_title_pairs([weekly, monthly])) == 1

    @pytest.mark.parametrize("gap_seconds,pairs", [
        (0, 1),
        (3_599, 1),
        # Exactly the bound is NOT apart: the gate is a strict `>` (kills a
        # `>=` mutant).
        (3_600, 1),
        (3_601, 0),
    ])
    def test_the_bound_is_one_hour_inclusive(self, gap_seconds, pairs):
        assert config.SAME_TITLE_MAX_CLOSE_GAP_SECONDS == 3_600
        mens, womens = self._wiu_markets(
            womens_close=datetime(2026, 1, 28, 1, 30, tzinfo=UTC)
            - timedelta(seconds=gap_seconds),
        )
        assert len(find_same_title_pairs([mens, womens])) == pairs

    def test_the_gate_reads_the_scanner_binding(self, monkeypatch, caplog):
        # The finder reads scanner's by-value binding of the constant, so a
        # test narrows the bound by patching THAT name (the precedent
        # TIME_SERIES_SAME_EVENT_LADDERS set; patching config.* would be a
        # silent no-op here). A 60 s bound admits 60 s and refuses 61 s, and
        # the refusal line prints the patched bound (one minute — singular).
        monkeypatch.setattr(scanner, "SAME_TITLE_MAX_CLOSE_GAP_SECONDS", 60)
        base = datetime(2026, 1, 28, 1, 30, tzinfo=UTC)
        mens, at_60 = self._wiu_markets(womens_close=base - timedelta(seconds=60))
        assert len(find_same_title_pairs([mens, at_60])) == 1
        mens, at_61 = self._wiu_markets(womens_close=base - timedelta(seconds=61))
        with caplog.at_level(logging.INFO):
            assert find_same_title_pairs([mens, at_61]) == []
        assert self._refusals(caplog) == [
            self._REFUSAL.replace("60 minutes", "1 minute") + "1"
        ]

    def test_a_naive_and_an_aware_close_are_refused(self, caplog):
        # Fail CLOSED: the two cannot be subtracted, so the pair cannot be
        # shown to close at one moment — even though the wall-clock readings
        # coincide.
        mens, womens = self._wiu_markets(womens_close=datetime(2026, 1, 28, 1, 30))
        assert womens.close_time.utcoffset() is None
        with caplog.at_level(logging.INFO):
            assert find_same_title_pairs([mens, womens]) == []
        assert self._refusals(caplog) == [self._REFUSAL + "1"]

    def test_the_line_is_silent_at_zero(self, caplog):
        mens, womens = self._wiu_markets(womens_close=datetime(2026, 1, 28, 1, 30, tzinfo=UTC))
        with caplog.at_level(logging.INFO):
            assert len(find_same_title_pairs([mens, womens])) == 1
        assert self._refusals(caplog) == []

    def test_the_line_avoids_the_absence_pinned_substrings(self, caplog):
        # Other tests pin "one event series" and "same event ticker" as ABSENT
        # from runs that never skip on those rules, so the close-gap line must
        # not contain either — or it would fail them for the wrong reason.
        mens, womens = self._wiu_markets()
        with caplog.at_level(logging.INFO):
            find_same_title_pairs([mens, womens])
        assert self._refusals(caplog)
        assert "one event series" not in caplog.text
        assert "same event ticker" not in caplog.text

    def test_each_candidate_is_counted_once_on_one_line(self, caplog):
        # The gate sits AFTER the series test: a one-series candidate is
        # counted on the series line and never reaches the close gate, so the
        # two counts partition the refusals. Three markets, three candidates:
        # M/W (two series, apart) is a close-gap refusal; the two women's
        # listings of ONE series (apart too) are a series refusal; the men's
        # game vs the second women's listing is a close-gap refusal.
        mens, womens = self._wiu_markets()
        womens_2 = _mock_market(
            ticker="KXNCAAWBGAME-26FEB10WIUEIU-WIU",
            event_ticker="KXNCAAWBGAME-26FEB10WIUEIU",
            title=self._WIU_TITLE, subtitle="Western Illinois",
            event_title="Western Illinois at Eastern Illinois",
            yes_ask=0.30, no_ask=0.70, close_time=datetime(2026, 2, 24, 23, tzinfo=UTC),
        )
        with caplog.at_level(logging.INFO):
            assert find_same_title_pairs([mens, womens, womens_2]) == []
        assert self._refusals(caplog) == [self._REFUSAL + "2"]
        assert caplog.messages.count(
            "Same-title candidates skipped as two instances of one event series "
            "(identical wording, different fixture): 1"
        ) == 1

    def test_refusing_the_widest_candidate_promotes_the_runner_up(self):
        # One best pair per group: the widest divergence (A vs B, 0.40) closes
        # two days apart and is refused, so the group's slot goes to A vs C
        # (0.20), which close at one instant. A pairs table diff is therefore
        # not a pure deletion.
        close = datetime(2026, 9, 25, 21, tzinfo=UTC)
        common = {"title": "Q", "subtitle": "Yes", "event_title": "E"}
        mA = _mock_market(ticker="KXA-1-Y", event_ticker="KXA-1", yes_ask=0.60,
                          no_ask=0.40, close_time=close, **common)
        mB = _mock_market(ticker="KXB-1-Y", event_ticker="KXB-1", yes_ask=0.20,
                          no_ask=0.80, close_time=close + timedelta(days=2), **common)
        mC = _mock_market(ticker="KXC-1-Y", event_ticker="KXC-1", yes_ask=0.40,
                          no_ask=0.60, close_time=close, **common)
        [pair] = find_same_title_pairs([mA, mB, mC])
        assert (pair.market_a.ticker, pair.market_b.ticker) == ("KXA-1-Y", "KXC-1-Y")
        assert pair.pA - pair.pB == pytest.approx(0.20)
        # Control: with B on the shared close the widest candidate wins.
        mB_aligned = _mock_market(ticker="KXB-1-Y", event_ticker="KXB-1", yes_ask=0.20,
                                  no_ask=0.80, close_time=close, **common)
        [pair] = find_same_title_pairs([mA, mB_aligned, mC])
        assert (pair.market_a.ticker, pair.market_b.ticker) == ("KXA-1-Y", "KXB-1-Y")


class TestClosesApart:
    """Unit contract of scanner.closes_apart — the ONE definition of the
    same-title close gate — and its live reader _closes_apart (DR-74)."""

    _T = datetime(2026, 9, 25, 21, tzinfo=UTC)

    @pytest.mark.parametrize("a,b", [
        (None, _T),
        (_T, None),
        (None, None),
        ("2026-09-25T21:00:00Z", _T),
        (_T, "2026-09-25T21:00:00Z"),
        (MagicMock(), _T),
        (1_790_000_000, _T),
        (date(2026, 9, 25), _T),
    ])
    def test_anything_that_is_not_a_datetime_fails_closed(self, a, b):
        # A date is not a datetime instant (and date(..) is not an instance of
        # datetime), so it cannot demonstrate a same-moment close either.
        assert scanner.closes_apart(a, b) is True

    def test_a_naive_and_an_aware_close_fail_closed(self):
        naive = datetime(2026, 9, 25, 21)
        assert scanner.closes_apart(naive, self._T) is True
        assert scanner.closes_apart(self._T, naive) is True

    def test_two_naive_closes_compare(self):
        a = datetime(2026, 9, 25, 21)
        assert scanner.closes_apart(a, a) is False
        assert scanner.closes_apart(a, a + timedelta(hours=2)) is True

    def test_one_instant_in_two_offsets_is_not_apart(self):
        from datetime import timezone
        cest = self._T.astimezone(timezone(timedelta(hours=2)))
        assert cest.hour == 23
        assert scanner.closes_apart(self._T, cest) is False

    @pytest.mark.parametrize("seconds,apart", [
        (0, False), (3_600, False), (3_601, True), (-3_600, False), (-3_601, True),
    ])
    def test_the_bound_is_symmetric_and_inclusive(self, seconds, apart):
        assert scanner.closes_apart(self._T, self._T + timedelta(seconds=seconds)) is apart
        assert scanner.closes_apart(self._T + timedelta(seconds=seconds), self._T) is apart

    @pytest.mark.parametrize("seconds,text", [
        (3_600, "60 minutes"),
        (15 * 60, "15 minutes"),
        (60, "1 minute"),
        # Not a whole number of minutes: printed in seconds, never truncated
        # into a different bound ("1 minutes" for 90 s would be false).
        (90, "90 seconds"),
        (59, "59 seconds"),
        (1, "1 second"),
    ])
    def test_the_bound_text_never_misstates_the_bound(self, seconds, text):
        assert scanner.close_gap_bound_text(seconds) == text

    def test_the_bound_text_reads_the_gates_binding_at_call_time(self, monkeypatch):
        # No argument reads scanner's binding at CALL time — the one
        # closes_apart reads — so a patched bound reaches the refusal lines of
        # both paths, and never a def-time copy.
        assert scanner.close_gap_bound_text() == "60 minutes"
        monkeypatch.setattr(scanner, "SAME_TITLE_MAX_CLOSE_GAP_SECONDS", 90)
        assert scanner.close_gap_bound_text() == "90 seconds"

    def test_the_live_reader_reads_close_time_by_type(self):
        dated = SimpleNamespace(close_time=self._T)
        assert scanner._closes_apart(dated, SimpleNamespace(close_time=self._T)) is False
        # A MagicMock's auto-attribute is not a datetime; a missing attribute
        # reads as None. Both fail closed.
        assert scanner._closes_apart(dated, MagicMock()) is True
        assert scanner._closes_apart(dated, SimpleNamespace()) is True


class TestTimeSeriesGroupKey:
    """Unit-level contract of scanner.time_series_group_key()."""

    def test_one_outcome_at_two_deadlines_is_one_key(self):
        assert (
            time_series_group_key("Solana price on Sep 14, 2026?", "$180 or above")
            == time_series_group_key("Solana price on Sep 18, 2026?", "$180 or above")
        )

    def test_two_outcomes_under_one_title_are_two_keys(self):
        assert (
            time_series_group_key("Solana price on Sep 14, 2026?", "$180 or above")
            != time_series_group_key("Solana price on Sep 18, 2026?", "$190 or above")
        )

    def test_bare_four_digit_strike_is_not_stripped_as_a_year(self):
        # normalize_title() erases any bare 20xx token as a year; running it
        # over an outcome label would merge every strike in 2000-2099 into one
        # key, i.e. reintroduce DR-01 through the subtitle.
        assert (
            time_series_group_key("Ethereum price?", "2050 or above")
            != time_series_group_key("Ethereum price?", "2060 or above")
        )

    def test_trailing_punctuation_after_a_stripped_date_is_one_key(self):
        # Stripping the date leaves a space in FRONT of the period
        # ("$80,000 by June 30." -> "$80,000 by ."), so trimming whitespace
        # first and punctuation second left a trailing space in the key and
        # silently dropped a genuine same-strike, two-deadline pair.
        assert (
            time_series_group_key("Will BTC hit a new high", "$80,000 by June 30.")
            == time_series_group_key("Will BTC hit a new high", "$80,000 by July 31")
        )

    def test_empty_title_yields_the_empty_key(self):
        # The caller drops such markets; the subtitle must not resurrect them.
        assert time_series_group_key("", "$180 or above") == ""

    def test_none_subtitle_reads_as_absent(self):
        assert time_series_group_key("Will BTC exceed $80k", None) == normalize_title(
            "Will BTC exceed $80k"
        )

    def test_magicmock_subtitle_reads_as_absent(self):
        # Fail-safe by TYPE, not truthiness: a MagicMock stand-in answers any
        # attribute with a truthy child mock, and regex-substituting one would
        # raise. Same rule leg_sides and strategy._depth_levels follow.
        assert time_series_group_key("Will BTC exceed $80k", MagicMock()) == normalize_title(
            "Will BTC exceed $80k"
        )

    def test_empty_subtitle_leaves_the_title_key_unchanged(self):
        assert time_series_group_key("Will BTC exceed $80k", "") == normalize_title(
            "Will BTC exceed $80k"
        )


def _ingest_market(ticker, event_ticker, title, event_title, *, subtitle="",
                   close="2026-06-01T00:00:00Z", yes_ask="0.50", no_ask="0.50"):
    """Build a market exactly as the market list builds one."""
    return _market_from_dict({
        "ticker": ticker, "event_ticker": event_ticker, "title": title,
        "yes_sub_title": subtitle, "status": "active", "close_time": close,
        "yes_ask_dollars": yes_ask, "no_ask_dollars": no_ask,
        "yes_bid_dollars": str(round(1.0 - float(no_ask), 4)),
    }, event_title)


def _question_of(market):
    """The time-series finder's group key for one market."""
    return time_series_group_key(pair_key(market), market.subtitle)


def _question_label(market):
    """The question label of one market: its group key as ladders compare it."""
    return ("question", scanner._ladder_question(_question_of(market)))


class TestLadderKeys:
    """A ladder is one question asked at several deadlines. A market is on
    the ladder of its event and on the ladder of its question."""

    def test_each_label_is_tagged_with_its_kind(self):
        assert ladder_keys("KXSENATEREC-26MAY", "senate recount") == frozenset({
            ("event", "KXSENATEREC-26MAY"), ("question", "senate recount"),
        })

    def test_an_event_ticker_never_matches_a_question(self):
        # The same text as an event and as a question is two labels
        assert ladder_keys("SAME", "") & ladder_keys("", "SAME") == frozenset()

    @pytest.mark.parametrize("blank", ["", None, 7, MagicMock()])
    def test_blank_or_non_text_values_are_skipped(self, blank):
        assert ladder_keys(blank, blank) == frozenset()
        assert ladder_keys(blank, "q") == frozenset({("question", "q")})
        assert ladder_keys("E-1", blank) == frozenset({("event", "E-1")})

    def test_small_wording_differences_share_one_question_label(self):
        # One question listed twice by the exchange, once without "the"
        first = _ingest_market("KXVOTESAVEAMERICA-26-MAR20", "KXVOTESAVEAMERICA-26",
                               "Will the Senate vote on SAVE America Act?",
                               "When will the Senate vote on the SAVE America Act?",
                               subtitle="Before Mar 20, 2026")
        second = _ingest_market("KXVOTESAVEAMERICA-26MAR-MAR24", "KXVOTESAVEAMERICA-26MAR",
                                "Will the Senate vote on the SAVE America Act?",
                                "When will the Senate vote on the SAVE America Act?",
                                subtitle="Before Mar 24, 2026")
        assert _question_of(first) != _question_of(second)
        assert market_ladder_keys(first) & market_ladder_keys(second) == frozenset({
            _question_label(first),
        })

    @pytest.mark.parametrize("one, other", [
        ("Will BTC exceed \u2018$80k\u2019?", "will btc exceed '$80k'"),
        ("Who wins: Smith (R)?", "who wins smith r"),
        ("An upset in the final", "upset in final"),
    ])
    def test_case_quotes_punctuation_and_articles_are_ignored(self, one, other):
        assert ladder_keys("", one) == ladder_keys("", other)

    @pytest.mark.parametrize("one, other", [
        ("btc | $80,000 or above", "btc | $80,500 or above"),
        ("rate | 1.5%", "rate | 15%"),
        ("spacex starship 11th launch | will spacex launch another starship by",
         "spacex starship 12th launch | will spacex launch another starship by"),
        ("temperature | -5 or below", "temperature | 5 or below"),
    ])
    def test_different_questions_keep_different_labels(self, one, other):
        assert ladder_keys("", one) != ladder_keys("", other)

    def test_a_question_of_filler_alone_gives_no_label(self):
        assert ladder_keys("E-1", "The ?") == frozenset({("event", "E-1")})

    def test_a_mock_market_gets_no_event_label(self):
        # A mock market's event ticker is not text, so it names no event
        keys = market_ladder_keys(MagicMock())
        assert not any(kind == "event" for kind, _ in keys)

    def test_a_market_with_no_text_name_gets_no_question_label(self):
        market = SimpleNamespace(ticker=None, event_ticker="E-1", title=None, subtitle=None)
        assert market_ladder_keys(market) == frozenset({("event", "E-1")})

    def test_an_event_title_alone_still_gives_a_question_label(self):
        # The event title makes the name text even when the market has no title
        market = SimpleNamespace(ticker="T", event_ticker="E-1", title=None, subtitle=None,
                                 _event_title="Some Event")
        assert _question_of(market)
        assert market_ladder_keys(market) == frozenset({
            ("event", "E-1"), _question_label(market),
        })

    @pytest.mark.parametrize("market", [
        # One deadline of a question whose deadlines share one event
        _ingest_market("KXSTAR-14-SEP23", "KXSTAR-14",
                       "Will SpaceX launch another Starship before Sep 23, 2026?",
                       "SpaceX Starship launches"),
        # One deadline of a question listed as one event per deadline
        _ingest_market("KXBTCMAX-26MAR01-80K", "KXBTCMAX-26MAR01",
                       "Will BTC exceed $80k by March 1, 2026?", "Bitcoin record"),
        # One option of a multi-choice event, named only by its label
        _ingest_market("KXPRES-28-DJT", "KXPRES-28", "",
                       "2028 Presidential Election Winner", subtitle="Donald Trump"),
    ], ids=["same-event deadline", "cross-event deadline", "option label"])
    def test_the_question_label_is_the_finders_group_key(self, market):
        assert _question_of(market)
        assert market_ladder_keys(market) == frozenset({
            ("event", market.event_ticker), _question_label(market),
        })

    def test_two_deadlines_of_one_event_share_both_labels(self):
        early = _ingest_market("KXSTAR-14-SEP23", "KXSTAR-14",
                               "Will SpaceX launch another Starship before Sep 23, 2026?",
                               "SpaceX Starship launches")
        late = _ingest_market("KXSTAR-14-OCT16", "KXSTAR-14",
                              "Will SpaceX launch another Starship before Oct 16, 2026?",
                              "SpaceX Starship launches")
        assert market_ladder_keys(early) == market_ladder_keys(late)

    def test_two_options_of_one_event_share_only_the_event_label(self):
        trump = _ingest_market("KXPRES-28-DJT", "KXPRES-28", "",
                               "2028 Presidential Election Winner", subtitle="Donald Trump")
        vance = _ingest_market("KXPRES-28-JDV", "KXPRES-28", "",
                               "2028 Presidential Election Winner", subtitle="JD Vance")
        assert market_ladder_keys(trump) & market_ladder_keys(vance) == frozenset({
            ("event", "KXPRES-28"),
        })

    @pytest.mark.usefixtures("pre_toggle_defaults")
    def test_a_pair_the_finder_forms_shares_one_question_label(self):
        # Two deadlines of one question, listed as two events
        early = _ingest_market("KXBTCMAX-26MAR01-80K", "KXBTCMAX-26MAR01",
                               "Will BTC exceed $80k by March 1, 2026?", "Bitcoin record",
                               close="2026-03-01T00:00:00Z", yes_ask="0.30", no_ask="0.70")
        late = _ingest_market("KXBTCMAX-26MAR11-80K", "KXBTCMAX-26MAR11",
                              "Will BTC exceed $80k by March 11, 2026?", "Bitcoin record",
                              close="2026-03-11T00:00:00Z", yes_ask="0.50", no_ask="0.50")
        [pair] = find_time_series_pairs(MagicMock(), held_tickers=set(), markets=[early, late])
        assert pair_ladder_keys(pair) == frozenset({
            ("event", "KXBTCMAX-26MAR01"), ("event", "KXBTCMAX-26MAR11"),
            _question_label(early),
        })

    def test_a_pair_has_the_labels_of_both_its_markets(self):
        a = SimpleNamespace(ticker="A", event_ticker="E-A", title="Q one", subtitle="")
        b = SimpleNamespace(ticker="B", event_ticker="E-B", title="Q two", subtitle="")
        pair = SimpleNamespace(market_a=a, market_b=b)
        assert pair_ladder_keys(pair) == market_ladder_keys(a) | market_ladder_keys(b)
        assert len(pair_ladder_keys(pair)) == 4


# One row per _CUMULATIVE_DEADLINE_PATTERNS[0] alternative (phrase, expected
# span tuple), module-level so a later commit's own alternation-equivalence
# test can reuse these exact strings as its hermetic reference set instead of
# duplicating them (a duplicate copy can drift from this table with nothing
# to catch it). Kept as a plain tuple, not nested in a class, for the same
# reason. See TestCumulativeDeadlineRule.test_each_cumulative_entry_is_live.
_CUMULATIVE_ENTRY_CASES = (
    ("by 2027", ("by 2027",)),
    ("through Oct 1, 2026", ("through oct 1, 2026",)),
    ("until Oct 1, 2026", ("until oct 1, 2026",)),
    ("no later than Oct 1, 2026", ("no later than oct 1, 2026",)),
    ("on or before Oct 1, 2026", ("on or before oct 1, 2026",)),
    ("up to Oct 1, 2026", ("up to oct 1, 2026",)),
    ("prior to Oct 1, 2026", ("prior to oct 1, 2026",)),
    ("by year-end", ("by year-end",)),
    ("by EOY", ("by eoy",)),
    ("by Friday", ("by friday",)),
    ("by 2026-12-31", ("by 2026-12-31",)),
    ("by the end of June", ("by the end of june",)),
    ("by Q4 2026", ("by q4 2026",)),
    # The two SPANLESS cumulative markers establish the KIND of question
    # ("within N units", "at any time") without naming a comparable date —
    # TestCumulativeDeadlinePairPredicate.test_spanless_cumulative_leg_is_
    # refused is what makes that matter.
    ("within 30 days", ()),
    ("at any time", ()),
)


class TestCumulativeDeadlineRule:
    """deadline_phrasing() separates "will X happen BY <date>" (cumulative —
    nested, so a later deadline can only add probability) from "what is X ON
    <date>" (a snapshot, which does not nest at all).

    This is the premise the time-series bet rests on. Before this rule the
    scanner asserted it and never checked it: normalize_title strips a dated
    snapshot title just as readily as a dated deadline one, so
    "Bitcoin price on Sep 15, 2026?" and "... on Sep 18, 2026?" both normalize
    to "bitcoin price on ?" and landed in ONE time-series group.
    """

    @pytest.mark.parametrize("title, expected", [
        # ── cumulative: an event that may happen at any time up to a deadline
        ("Will BTC exceed $80k by March 2026?", scanner.DEADLINE_CUMULATIVE),
        ("Will BTC exceed $80k by March 01, 2026", scanner.DEADLINE_CUMULATIVE),
        ("Will Z happen by December 11, 2026?", scanner.DEADLINE_CUMULATIVE),
        ("Will X happen before Oct 1, 2026?", scanner.DEADLINE_CUMULATIVE),
        ("Will X happen prior to the end of the year?", scanner.DEADLINE_CUMULATIVE),
        ("Will X happen no later than Dec 31, 2026?", scanner.DEADLINE_CUMULATIVE),
        ("Will X resolve by Q1 2026?", scanner.DEADLINE_CUMULATIVE),
        ("Will X happen by 12/31/2026?", scanner.DEADLINE_CUMULATIVE),
        # ── snapshot: a state measured at one instant, or over a window that
        # does not nest
        ("Bitcoin price on Sep 15, 2026?", scanner.DEADLINE_SNAPSHOT),
        ("Solana price on Sep 14, 2026?", scanner.DEADLINE_SNAPSHOT),
        ("Highest temperature in NYC on Oct 1", scanner.DEADLINE_SNAPSHOT),
        ("Will X happen on June 30, 2026", scanner.DEADLINE_SNAPSHOT),
        ("S&P 500 at the close on Dec 31, 2026?", scanner.DEADLINE_SNAPSHOT),
        ("BTC price at 12:00 on Sep 15", scanner.DEADLINE_SNAPSHOT),
        # ── unknown: no deadline shape in the wording at all
        ("Fed cuts rates?", scanner.DEADLINE_UNKNOWN),
        ("Michal Olbrycht wins", scanner.DEADLINE_UNKNOWN),
    ])
    def test_title_classification(self, title, expected):
        assert scanner.deadline_phrasing("", title, "") == expected

    def test_in_a_month_is_a_window_not_a_deadline(self):
        # "top 10 in October" does not nest inside "top 10 in November" — the
        # two are disjoint windows, so P is not monotone in the date and there
        # is no in-between mass to dispute. The FIDE Oct/Nov markets are this
        # shape and used to pair.
        assert scanner.deadline_phrasing("", "Top 10 in October", "") == scanner.DEADLINE_SNAPSHOT
        assert scanner.deadline_phrasing(
            "", "Will the Fed cut rates in December 2026?", "",
        ) == scanner.DEADLINE_SNAPSHOT

    def test_after_a_date_inverts_the_monotonicity_and_is_refused(self):
        # "after June 30" carries LOWER probability the later the date, so the
        # earlier/later leg assignment means the opposite of what it says —
        # the monotonicity the trade rests on inverts.
        #
        # Two ways _DATE_PATTERNS lands such a market in a time-series group,
        # both measured against the real patterns: the year-less shape strips
        # its preposition along with the date, so a "by" title and an "after"
        # title collapse onto ONE key; and two "after" titles at different
        # dates collapse onto one key with each other. Only this table
        # separates them.
        assert normalize_title("Will X happen by March 1") == normalize_title(
            "Will X happen after March 1"
        )
        assert normalize_title("Will X happen after March 1, 2026") == normalize_title(
            "Will X happen after June 1, 2026"
        )
        assert scanner.deadline_phrasing(
            "", "Will X happen after March 1, 2026", "",
        ) == scanner.DEADLINE_SNAPSHOT

    def test_an_incidental_after_does_not_refuse_a_real_deadline(self):
        # "after" only rejects when it introduces a DATE. Without the date-token
        # requirement this title would be refused for naming the halving.
        assert scanner.deadline_phrasing(
            "", "Will BTC top $100k by Dec 31, 2026, after the halving?", "",
        ) == scanner.DEADLINE_CUMULATIVE
        assert scanner.deadline_phrasing(
            "", "Will the ETF named after Smith launch by June 2026?", "",
        ) == scanner.DEADLINE_CUMULATIVE

    @pytest.mark.parametrize("title", [
        "Will the Fed cut by 50 bps in December 2026?",
        "Will Team A win by 10 points?",
        "Will the bill pass by a 2/3 majority?",
    ])
    def test_by_a_quantity_is_not_a_deadline(self, title):
        # "by" must be followed by a DATE token. These are magnitudes, and the
        # first of them is a real Kalshi shape whose two months merge into one
        # normalized group.
        assert scanner.deadline_phrasing("", title, "") != scanner.DEADLINE_CUMULATIVE

    @pytest.mark.parametrize("args", [
        (None, None, None),
        (MagicMock(), MagicMock(), MagicMock()),
        (123, [], {}),
    ])
    def test_non_string_fields_read_as_absent_and_never_raise(self, args):
        # Same fail-safe-by-type rule leg_sides and strategy._depth_levels
        # follow. Real: historical._market_to_dict stores subtitle as
        # `subtitle or yes_sub_title`, which is None when both are absent.
        assert scanner.deadline_phrasing(*args) == scanner.DEADLINE_UNKNOWN

    def test_conflicting_markers_across_fields(self):
        # control — kills M04a, M04c. Cross-field precedence
        # (subtitle -> title -> event_title) is only half pinned by
        # TestSubContractDeadlineWins below (subtitle beats a snapshot EVENT
        # title). These three put a DIFFERENT marker in two fields at once,
        # so a field-order mutant is forced to pick the wrong one.
        #
        # The subtitle decides over a snapshot TITLE. Kills a title-checked-
        # before-subtitle reorder (M04a).
        assert scanner.deadline_phrasing(
            "", "Bitcoin price on Sep 15, 2026?", "$80,000 by June 30",
        ) == scanner.DEADLINE_CUMULATIVE
        # With no subtitle marker, the TITLE decides over the event title —
        # here the title is cumulative and the event title is a snapshot
        # window ("in 2026"). Kills an event-title-checked-before-title
        # reorder (M04c). This test calls deadline_phrasing directly, so it
        # cannot observe an arg swap INSIDE _market_deadline_profile (M23e) —
        # that one is pinned separately by
        # TestDeadlineProfileParity::test_dict_and_live_extraction_agree in
        # test_backtester.py, which calls _market_deadline_profile itself.
        assert scanner.deadline_phrasing(
            "Will X happen in 2026?", "Will X happen before Jan 1, 2027?", "",
        ) == scanner.DEADLINE_CUMULATIVE
        # Same field-order relation, the other direction: a snapshot TITLE
        # beats a cumulative event title (the real KXCITYCHAMPS/FIDE shape).
        assert scanner.deadline_phrasing(
            "Will X happen by Dec 31, 2026?", "Top 10 in October?", "",
        ) == scanner.DEADLINE_SNAPSHOT

    def test_snapshot_beats_cumulative_in_one_field(self):
        # control — kills M05. WITHIN one field, a snapshot marker must be
        # tested before a cumulative one, or a title naming both reads as
        # cumulative. M05 swaps _field_phrasing's snapshot-table and
        # cumulative-table tests (one alternation search each since DR-70).
        assert scanner.deadline_phrasing(
            "", "Will BTC be above $100k at the close before Oct 1, 2026?", "",
        ) == scanner.DEADLINE_SNAPSHOT
        assert scanner.deadline_phrasing(
            "", "Will BTC be above $100k in Q3 2026 by Sep 30, 2026?", "",
        ) == scanner.DEADLINE_SNAPSHOT

    @pytest.mark.parametrize("phrase", [
        "on June 30", "on Friday", "on 9/30", "on 2026-09-30",
        "in October", "in Q3", "in 2027", "during 2026", "for 2026",
        "at the close", "at 5pm ET", "at 17:00", "as of Sep 1", "end of day",
        "after March 1, 2026", "between 3 and 5",
    ])
    def test_each_snapshot_entry_is_live(self, phrase):
        # control — kills a dropped _SNAPSHOT_PATTERNS entry: one phrase per
        # table entry. A future edit that drops any single entry fails
        # exactly the rows that depended on it, instead of hiding behind the
        # others. This does not reach every branch WITHIN a multi-alternative
        # entry (e.g. narrowing "at the close|open|end" to "at the close"
        # alone still passes every row here) — that is a known residual, not
        # a claim this test makes.
        assert scanner.deadline_phrasing("", phrase, "") == scanner.DEADLINE_SNAPSHOT

    def test_capitalised_and_november(self):
        # control — kills a dropped re.IGNORECASE flag on either alternation
        # (_ANY_SNAPSHOT / _ANY_CUMULATIVE, which decide the verdict since
        # DR-70) or on _COMPILED_CUMULATIVE, whose index 0 _deadline_spans
        # still reads (the "Before Oct 1, 2026" span below), or "November"
        # dropped from the shared month list. A dropped flag on
        # _COMPILED_SNAPSHOT, which since DR-70 only the tests read, is killed
        # by TestAlternationEquivalence instead. All must stay — plain
        # member/flag drops a line-level diff would not otherwise catch.
        assert scanner.deadline_profile(
            "", "Before Oct 1, 2026", "",
        ) == (scanner.DEADLINE_CUMULATIVE, ("before oct 1, 2026",))
        assert scanner.deadline_phrasing(
            "", "On Nov 16, 2026", "",
        ) == scanner.DEADLINE_SNAPSHOT
        assert scanner.deadline_profile(
            "", "Will X happen by November 30, 2026?", "",
        ) == (scanner.DEADLINE_CUMULATIVE, ("by november 30, 2026",))

    @pytest.mark.parametrize("phrase, spans", _CUMULATIVE_ENTRY_CASES)
    def test_each_cumulative_entry_is_live(self, phrase, spans):
        # control — kills a dropped or narrowed _CUMULATIVE_DEADLINE_PATTERNS[0]
        # alternative (or, for the last two rows, a dropped spanless marker).
        # One phrase per alternative, asserting the EXACT span _deadline_spans
        # captures — a table entry that quietly stops matching, or starts
        # truncating its span, fails its OWN parametrized row rather than
        # hiding behind a nearby passing one (a shared for-loop would instead
        # stop at the first failure and leave every later row unchecked).
        assert scanner.deadline_profile("", f"Will X happen {phrase}?", "") == (
            scanner.DEADLINE_CUMULATIVE, spans,
        )


class TestSubContractDeadlineWins:
    """Across fields the precedence is subtitle -> title -> event_title: the
    SUB-CONTRACT is the thing that actually resolves, so its wording is the
    most authoritative statement of what the contract settles on.

    This is the half of the rule that keeps MVE markets eligible. It must not
    open a hole in the snapshot refusals, and the reason it does not is
    structural: a snapshot family's subtitle is a bare strike with no marker at
    all, so the verdict falls straight through to the title.
    """

    def test_sub_contract_deadline_overrides_a_snapshot_event_title(self):
        assert scanner.deadline_phrasing(
            "Bitcoin price on Sep 15, 2026?", "Bitcoin price?", "$80,000 by June 30",
        ) == scanner.DEADLINE_CUMULATIVE

    def test_deadline_in_the_sub_contract_alone_is_cumulative(self):
        assert scanner.deadline_phrasing(
            "BTC milestones", "Will BTC hit a new high", "$80,000 by June 30",
        ) == scanner.DEADLINE_CUMULATIVE

    def test_deadline_in_the_event_title_alone_is_cumulative(self):
        # The MVE shape: the option label carries no date, the parent event does.
        assert scanner.deadline_phrasing(
            "Presidential Election Winner by March 2026", "Trump", "Donald Trump",
        ) == scanner.DEADLINE_CUMULATIVE

    def test_bare_strike_sub_contract_falls_through_to_a_snapshot_title(self):
        # THE regression guard for the override: the KXBTCD/KXSOLD daily
        # families must still be refused.
        assert scanner.deadline_phrasing(
            "Bitcoin price on Sep 15, 2026 at 5pm EDT?",
            "Bitcoin price on Sep 15, 2026?",
            "$82,750 or above",
        ) == scanner.DEADLINE_SNAPSHOT

    def test_a_clock_time_is_not_a_date_token(self):
        # An intraday family must not fail OPEN through its own sub-contract
        # now that the subtitle leads. "by 5pm" names no date, so the verdict
        # falls through rather than reading as a cumulative deadline.
        assert scanner.deadline_phrasing(
            "", "Bitcoin price?", "$82,750 or above by 5pm",
        ) != scanner.DEADLINE_CUMULATIVE

    def test_a_dateless_combo_ticket_is_unknown(self):
        # A combo ticket's wording names its legs but never its date, so it
        # fails closed — correct, since such a pair never had the premise.
        assert scanner.deadline_phrasing(
            "Cross-category combo", "Parlay", "All legs hit",
        ) == scanner.DEADLINE_UNKNOWN


class TestDateTokenBoundaries:
    """DR-68: the phrasing tables' DATE tokens must name a date and nothing
    else.

    Before DR-68 the month abbreviations had no word boundary ("by Dec(ision)",
    "by Mar(vel) Studios" read as cumulative deadlines, "on Mar(s)" as a
    snapshot), any d/d fraction and any 20xx number read as a date ("by 2/3
    vote", "by 2000"), "ever" was a cumulative marker (it fires on names), a
    weekday truncated the date after it ("by Friday, Sep 19, 2026" and
    "by Friday, Sep 26, 2026" both read "by friday", so a genuine pair was
    refused as one deadline stated twice), "on THE <Month> <day>" was missed,
    and the dotted clock forms needed a following word character.

    Rows commented `regression` fail when DR-68 is reverted; rows commented
    `control` pass either way and each names the mutant it kills. The controls
    pin three things: the tightened tokens against over-tightening, the
    snapshot (reject) markers against narrowing for a genuine date ("after
    6/30 of this year", "on Sep30"), and the new "on the <Month> <day>"
    snapshot branch against over-widening ("on the May ballot").
    """

    _UNKNOWN = (scanner.DEADLINE_UNKNOWN, ())

    @pytest.mark.parametrize("title, expected", [
        # regression — a month abbreviation at the front of an ordinary word is not a
        # date (the \b after the month).
        ("Callum Connor wins by Decision?", _UNKNOWN),
        ("Will X be produced or distributed by Marvel Studios?", _UNKNOWN),
        ("Will X be endorsed by Marco Rubio?", _UNKNOWN),
        ("Will X be beaten by Novak Djokovic?", _UNKNOWN),
        ("Will X be hosted by Mayor Adams?", _UNKNOWN),
        # regression — a fraction, vote share or month/year is not a calendar m/d.
        ("Will the bill pass by 2/3 vote?", _UNKNOWN),
        ("Will X be ratified by 3/4 of states?", _UNKNOWN),
        ("Will X happen by 13/45?", _UNKNOWN),
        ("Will X happen by 3/2027?", _UNKNOWN),
        # regression — each calendar range on its own: "13/45" is invalid on BOTH
        # axes, so it cannot tell which check fired. A real day with month 13
        # kills widening the month group to \d{1,2}; a real month with day 45
        # kills widening the day group to \d{1,2}.
        ("Will X happen by 13/12?", _UNKNOWN),
        ("Will X happen by 3/45?", _UNKNOWN),
        # regression — "ever" is no longer a cumulative marker.
        ("Worst Neighbor Ever", _UNKNOWN),
        ("Chaco For Ever", _UNKNOWN),
        ("Will 2026 be the hottest year ever?", _UNKNOWN),
        # regression — each lookahead word after a fraction is pinned on its own row
        # (the article-free form of the "pass by a 2/3 majority" example,
        # which the article "a" already stops).
        ("Will the bill pass by 2/3 majority?", _UNKNOWN),
        ("Will the bill pass by 2/3 supermajority?", _UNKNOWN),
        ("Will the bill pass by 2/3 votes?", _UNKNOWN),
        # regression — a pre-2020 number is an amount, not a year; the real deadline
        # later in the title still decides.
        ("Will spending decrease by 2000 before 2027?",
         (scanner.DEADLINE_CUMULATIVE, ("before 2027",))),
        # regression — kills widening the bare-year floor to 2010 (20[1-9]\d), which
        # the "by 2000" row above cannot see.
        ("Will spending decrease by 2015 before 2027?",
         (scanner.DEADLINE_CUMULATIVE, ("before 2027",))),
        # regression — "on Mars" / "on declaring" are not snapshot dates, so the real
        # "before <date>" deadline decides instead of a false snapshot.
        ("Will SpaceX land on Mars before 2030?",
         (scanner.DEADLINE_CUMULATIVE, ("before 2030",))),
        ("Will Trump decide on declaring a national emergency before Oct 1, 2026?",
         (scanner.DEADLINE_CUMULATIVE, ("before oct 1, 2026",))),
        # regression — the "after" REJECT marker likewise no longer fires on a
        # month-prefixed NON-date word; kills dropping the letter lookahead from
        # its _SNAPSHOT_MONTH alternative ("after May(or)" would read snapshot).
        ("Will X happen after Mayor Adams resigns, by Dec 31, 2026?",
         (scanner.DEADLINE_CUMULATIVE, ("by dec 31, 2026",))),
        # regression — a weekday keeps the month-name date after it.
        ("Will X happen by Friday, Sep 19, 2026?",
         (scanner.DEADLINE_CUMULATIVE, ("by friday, sep 19, 2026",))),
        # control — kills a "\d{0,2}" (day made optional) mutant of the "on the"
        # branch: "on the May ballot" names no instant, so the verdict comes
        # from the real deadline.
        ("Will X be on the May ballot before Oct 1, 2026?",
         (scanner.DEADLINE_CUMULATIVE, ("before oct 1, 2026",))),
        # control — kills dropping the (?!\d) after the "on the" day: without it
        # the day eats "20" of the year and the ballot reads as a snapshot.
        ("Will X be on the November 2026 ballot by Aug 1, 2026?",
         (scanner.DEADLINE_CUMULATIVE, ("by aug 1, 2026",))),
        # control — kills dropping the 2-digit-year branch of the calendar m/d.
        ("Will X happen by 9/30/26?", (scanner.DEADLINE_CUMULATIVE, ("by 9/30/26",))),
        # control — a genuine deadline must keep its whole span through the
        # tightened tokens. Each row kills one mutant: dropping the m/d
        # 4-digit-year branch (span "by 9/30")...
        ("Will X happen by 9/30/2026?", (scanner.DEADLINE_CUMULATIVE, ("by 9/30/2026",))),
        # ...dropping the days-10-29 alternative ([12]\d) of the m/d day group
        # (every other m/d row uses day 30 or 31, so a genuine mid-month
        # deadline would silently read unknown)...
        ("Will X happen by 9/15/2026?", (scanner.DEADLINE_CUMULATIVE, ("by 9/15/2026",))),
        # ...making the m/d year mandatory (a year-less "by 9/30" reads unknown)...
        ("Will X happen by 9/30?", (scanner.DEADLINE_CUMULATIVE, ("by 9/30",))),
        # ...moving the month's \b after the optional period (span "by sept")...
        ("Will X happen by Sept. 30, 2026?",
         (scanner.DEADLINE_CUMULATIVE, ("by sept. 30, 2026",))),
        # ...widening the snapshot "on the <Month> <day>" branch to "on <any
        # words> <Month> <day>" (the verdict turns snapshot)...
        ("Will X happen on or before Oct 1, 2026?",
         (scanner.DEADLINE_CUMULATIVE, ("on or before oct 1, 2026",))),
        # ...moving the 2020-2099 year ahead of the ISO date (span "by 2026")...
        ("Will X happen by 2026-12-31?", (scanner.DEADLINE_CUMULATIVE, ("by 2026-12-31",))),
        # ...and rejecting season ranges (the verdict turns unknown).
        ("Will X retire before the 2027-28 NFL season?",
         (scanner.DEADLINE_CUMULATIVE, ("before the 2027",))),
        # control — the undotted clock forms keep their trailing \b; kills
        # dropping it, which would read "at 3 am(endments)" as a snapshot clock.
        ("Will the bill stand at 3 amendments by Oct 1, 2026?",
         (scanner.DEADLINE_CUMULATIVE, ("by oct 1, 2026",))),
    ])
    def test_profile(self, title, expected):
        assert scanner.deadline_profile("", title, "") == expected

    @pytest.mark.parametrize("title", [
        # regression — "on THE <Month> <day>" is the same snapshot as "on <Month>
        # <day>" (the CFP-rankings family used to read unknown).
        "Ohio St. to be a top 25 ranked team on the Dec 6 CFP rankings",
        # regression — the same branch with an UNSPACED day: its month ends at the
        # next letter (_SNAPSHOT_MONTH), not at a word boundary, so this also
        # kills a \b there, which would fail OPEN to the "before" deadline.
        # (A letter lookahead in this branch is an equivalent mutant: its
        # required day already excludes a letter after the month.)
        "Will X be ranked on the Dec6 CFP rankings before Oct 1, 2026?",
        # regression — a dotted clock form followed by a space ("5 p.m. ET").
        "Will BTC be above 100k at 5 p.m. ET?",
        # control — "after" is a REJECT marker and keeps its WIDE numeric
        # alternatives; kills narrowing it to the tightened date token, which
        # would fail OPEN (the 'of' lookahead and the 2020-2099 year).
        "Will X happen by Dec 31, 2026, after 6/30 of this year?",
        "Will X happen after 2019?",
        # control — an UNSPACED genuine date still fires both older reject
        # markers, as it did before DR-68. Kills a \b in the "on" entry's month
        # (the first row) and dropping _SNAPSHOT_MONTH from the "after" entry
        # (the second — the tightened token's \b-bounded month alone misses
        # "Sep30"). Both mutants fail OPEN (snapshot -> cumulative).
        "Will BTC be above $100k on Sep30, 2026, before Oct 1, 2026?",
        "Will X happen by Dec 31, 2026, after Sep30, 2026?",
    ])
    def test_snapshot(self, title):
        assert scanner.deadline_phrasing("", title, "") == scanner.DEADLINE_SNAPSHOT

    def test_weekday_dated_legs_seven_days_apart_pair(self):
        # regression — the live finder. Before DR-68 both legs' spans truncated to
        # "by friday", so cumulative_deadline_pair read one deadline stated
        # twice and refused a genuine pair 7 days apart. Mirror:
        # test_backtester.py::TestDateTokenBoundaries.
        mA = _mock_market(
            ticker="PA-1", event_ticker="EVA-1",
            title="Will X happen by Friday, Sep 19, 2026?",
            yes_ask=0.20, no_ask=0.80, close_time=datetime(2026, 9, 19, tzinfo=UTC),
        )
        mB = _mock_market(
            ticker="PB-1", event_ticker="EVB-1",
            title="Will X happen by Friday, Sep 26, 2026?",
            yes_ask=0.60, no_ask=0.38, close_time=datetime(2026, 9, 26, tzinfo=UTC),
        )
        assert normalize_title(pair_key(mA)) == normalize_title(pair_key(mB))
        assert scanner._market_deadline_profile(mA) == (
            scanner.DEADLINE_CUMULATIVE, ("by friday, sep 19, 2026",),
        )
        assert scanner._market_deadline_profile(mB) == (
            scanner.DEADLINE_CUMULATIVE, ("by friday, sep 26, 2026",),
        )
        [pair] = find_time_series_pairs(MagicMock(), held_tickers=set(), markets=[mA, mB])
        assert (pair.market_a.ticker, pair.market_b.ticker) == ("PA-1", "PB-1")


class TestCumulativeDeadlinePairPredicate:
    """cumulative_deadline_pair() requires BOTH legs cumulative AND two
    genuinely different stated deadlines."""

    @staticmethod
    def _pair(a, b):
        return scanner.cumulative_deadline_pair(
            scanner.deadline_profile(*a), scanner.deadline_profile(*b),
        )

    def test_two_deadlines_of_one_question_pair(self):
        assert self._pair(
            ("", "Will BTC exceed $80k by March 2026?", ""),
            ("", "Will BTC exceed $80k by June 2026?", ""),
        ) is True

    @pytest.mark.parametrize("early, late", [
        ("by Dec 31, 2026", "by Dec 20, 2026"),   # same month, different day
        ("by March 2026", "by March 2027"),       # same month, different year
        ("by Sep 14", "by Sep 18"),               # year-less daily family
    ])
    def test_deadlines_differing_only_in_their_day_or_year_still_pair(self, early, late):
        # The span capture has to take the WHOLE date. An earlier draft stopped
        # at the month, so "by March 2026" and "by March 2027" both read as
        # "by march 20" (the day group ate half the year) and were refused.
        assert self._pair(
            ("", f"Will X happen {early}?", ""), ("", f"Will X happen {late}?", ""),
        ) is True

    def test_one_deadline_stated_twice_is_not_a_two_deadline_pair(self):
        assert self._pair(
            ("", "Will X happen by Dec 31, 2026?", ""),
            ("", "Will X happen by Dec 31, 2026?", ""),
        ) is False

    def test_a_snapshot_leg_refuses_the_pair(self):
        assert self._pair(
            ("Solana price on Sep 14, 2026?", "Solana price on Sep 14, 2026?", "$180 or above"),
            ("Solana price on Sep 18, 2026?", "Solana price on Sep 18, 2026?", "$180 or above"),
        ) is False

    def test_a_cumulative_leg_naming_no_comparable_date_is_refused(self):
        # "within 30 days" establishes the KIND of question but names no date,
        # so the two deadlines cannot be shown to differ — fail closed.
        assert self._pair(
            ("", "Will X happen within 30 days?", ""),
            ("", "Will X happen within 60 days?", ""),
        ) is False

    def test_unknown_wording_fails_closed(self):
        assert self._pair(
            ("TT Elite Series", "Michal Olbrycht wins", "Yes"),
            ("TT Elite Series", "Michal Olbrycht wins", "Yes"),
        ) is False

    def test_order_independent(self):
        a = ("", "Will X happen by March 2026?", "")
        b = ("", "Will X happen by June 2026?", "")
        assert self._pair(a, b) == self._pair(b, a)

    def test_snapshot_leg_with_distinct_spans_is_refused(self):
        # control — kills M03full (the whole verdict gate deleted). A leg's
        # verdict deciding "cumulative" is required INDEPENDENTLY of whether
        # the two legs' spans differ: a snapshot verdict refuses the pair
        # even when both legs carry distinct dated spans, so a snapshot
        # family that happens to spell "before <date>" cannot slip through on
        # the span check alone. No test before this one ever put a span on a
        # snapshot leg, so deleting the whole verdict gate
        # (`if verdict_a != DEADLINE_CUMULATIVE or verdict_b != ...`) —
        # M03full — passed the suite: every snapshot fixture had empty or
        # identical spans and the span rule refused it on its own.
        snap_a = (scanner.DEADLINE_SNAPSHOT, ("before oct 1, 2026",))
        cum_b = (scanner.DEADLINE_CUMULATIVE, ("before oct 10, 2026",))
        snap_b = (scanner.DEADLINE_SNAPSHOT, ("before oct 10, 2026",))
        assert scanner.cumulative_deadline_pair(snap_a, cum_b) is False
        assert scanner.cumulative_deadline_pair(cum_b, snap_a) is False
        assert scanner.cumulative_deadline_pair(snap_a, snap_b) is False
        assert scanner.cumulative_deadline_pair(snap_b, snap_a) is False

    def test_spanless_cumulative_leg_is_refused(self):
        # control — kills M02, which deletes `if not spans_a or not spans_b:
        # return False`. Both legs must NAME a comparable deadline, not
        # merely classify cumulative: a dated leg paired with a spanless one
        # ("within 30 days", "at any time") can never be shown to state a
        # DIFFERENT deadline.
        #
        # M03a alone (the verdict gate narrowed to refuse only snapshot, not
        # unknown) is an EQUIVALENT mutant GLOBALLY, not just on this
        # fixture: an UNKNOWN verdict always carries empty spans (deadline_
        # phrasing returns unknown only when NO field matched any pattern in
        # EITHER table, including _COMPILED_CUMULATIVE[0], the only pattern
        # _deadline_spans reads — so if the verdict is unknown, that same
        # pattern found nothing to capture either), so the span check refuses
        # it unaided regardless of what the verdict gate does with an
        # unknown verdict. Measured: M03a alone survives the full suite. That
        # is exactly why TestDeadlineGuardFinders::
        # test_dated_leg_never_pairs_with_unknown_leg pairs a dated leg with
        # an UNKNOWN one instead of a spanless-cumulative one below — only
        # that shape needs BOTH guards gone (M02+M03a) to be admitted, so
        # only that shape can tell M02 and M03a apart.
        dated = (scanner.DEADLINE_CUMULATIVE, ("by march 1",))
        spanless = (scanner.DEADLINE_CUMULATIVE, ())
        assert scanner.cumulative_deadline_pair(dated, spanless) is False
        assert scanner.cumulative_deadline_pair(spanless, dated) is False


class TestDeadlinePairRefusal:
    """deadline_pair_refusal()'s precedence is total and order-independent
    (DR-72): exactly one of its four outcomes — REFUSED_SNAPSHOT,
    REFUSED_NO_STATED_DEADLINE, REFUSED_SAME_DEADLINE, or None — always
    applies, and swapping the two profiles never changes which one.
    cumulative_deadline_pair() is defined as
    `deadline_pair_refusal(...) is None` (DR-72), so every case here also
    pins that the two can never disagree about which pairs are eligible —
    a regression here catches a future edit to either function that drifts
    the boolean verdict away from the reason.
    """

    @pytest.mark.parametrize("profile_a, profile_b, expected", [
        # snapshot + unknown -> snapshot: the snapshot verdict alone disproves
        # the premise, regardless of what the other leg's field says.
        (
            (scanner.DEADLINE_SNAPSHOT, ("on june 1, 2026",)),
            (scanner.DEADLINE_UNKNOWN, ()),
            scanner.REFUSED_SNAPSHOT,
        ),
        # snapshot + spanless-cumulative -> snapshot, not no-stated-deadline:
        # the snapshot check runs FIRST regardless of whether the other leg
        # would independently have failed the span check too.
        (
            (scanner.DEADLINE_SNAPSHOT, ("on june 1, 2026",)),
            (scanner.DEADLINE_CUMULATIVE, ()),
            scanner.REFUSED_SNAPSHOT,
        ),
        # unknown + dated cumulative -> no-stated-deadline: one leg names no
        # deadline shape at all, so there is nothing to compare it against.
        (
            (scanner.DEADLINE_UNKNOWN, ()),
            (scanner.DEADLINE_CUMULATIVE, ("by june 1, 2026",)),
            scanner.REFUSED_NO_STATED_DEADLINE,
        ),
        # equal spans -> same-deadline: both cumulative, both name a date,
        # but it's the SAME date stated twice.
        (
            (scanner.DEADLINE_CUMULATIVE, ("by june 1, 2026",)),
            (scanner.DEADLINE_CUMULATIVE, ("by june 1, 2026",)),
            scanner.REFUSED_SAME_DEADLINE,
        ),
        # Genuinely eligible: both cumulative, spans differ -> None.
        (
            (scanner.DEADLINE_CUMULATIVE, ("by june 1, 2026",)),
            (scanner.DEADLINE_CUMULATIVE, ("by june 10, 2026",)),
            None,
        ),
    ])
    def test_precedence_is_order_independent(self, profile_a, profile_b, expected):
        assert scanner.deadline_pair_refusal(profile_a, profile_b) == expected
        assert scanner.deadline_pair_refusal(profile_b, profile_a) == expected
        assert scanner.cumulative_deadline_pair(profile_a, profile_b) == (expected is None)
        assert scanner.cumulative_deadline_pair(profile_b, profile_a) == (expected is None)


class TestStatedDeadline:
    """DR-73a: reading a rung's wording as the LAST CALENDAR DAY its deadline
    includes.

    Every row drives the PUBLIC helper end to end — deadline_profile() over a
    real title, then stated_deadline() over the same three fields — so the
    rows pin the span table, the preposition split and the field cross-check
    together, exactly as a finder will call them.

    The refusals are as load-bearing as the acceptances and each one is
    measured, not defensive: a year-less rung cannot be placed at all, an
    article with no "end of" is a truncated season range, "until"/"up to" do
    not say whether the named day counts, and a bare year behind "through" is
    the recorded "2018-19 through 2025-26" false-cumulative residual. See
    scanner._span_deadline.
    """

    @staticmethod
    def _read(event_title="", title="", subtitle=""):
        """Classify and read one market exactly as a finder does."""
        profile = scanner.deadline_profile(event_title, title, subtitle)
        return scanner.stated_deadline(profile, event_title, title, subtitle)

    @pytest.mark.parametrize("span, expected", [
        # Inclusive prepositions: the named day is the last one covered.
        ("by Sep 23, 2026", date(2026, 9, 23)),
        ("no later than Sep 23, 2026", date(2026, 9, 23)),
        ("on or before Sep 23, 2026", date(2026, 9, 23)),
        # Exclusive prepositions: the deadline stops the day BEFORE.
        ("before Sep 23, 2026", date(2026, 9, 22)),
        ("prior to Sep 23, 2026", date(2026, 9, 22)),
        # Month granularity: inclusive covers the whole month, exclusive stops
        # at the end of the previous one. The leap-year row is why the last
        # day comes from calendar.monthrange rather than a fixed table.
        ("by March 2026", date(2026, 3, 31)),
        ("before March 2026", date(2026, 2, 28)),
        ("before March 2024", date(2024, 2, 29)),
        # Year granularity, the same two directions.
        ("by 2027", date(2027, 12, 31)),
        ("before 2026", date(2025, 12, 31)),
        # ISO, INCLUSIVE only — a bare ISO date and one cut out of a full
        # timestamp are indistinguishable, and they agree on the inclusive
        # side. The exclusive side is in the refusal table below.
        ("by 2026-03-15", date(2026, 3, 15)),
        # A weekday in front of the date is noise; the date behind it names
        # the day (DR-68 keeps the whole span rather than truncating it).
        ("by Friday, Sep 19, 2026", date(2026, 9, 19)),
        # "end of", month and year granularity, behind an inclusive
        # preposition only. The article is allowed HERE and nowhere else.
        ("by the end of 2027", date(2027, 12, 31)),
        ("by end of 2027", date(2027, 12, 31)),
        ("by the end of June 2027", date(2027, 6, 30)),
        # "through" reads inclusive with a month-name date; it occurs live.
        ("through June 30, 2027", date(2027, 6, 30)),
    ])
    def test_reads_the_last_included_day(self, span, expected):
        assert self._read(title=f"Will the thing happen {span}?") == expected

    @pytest.mark.parametrize("span", [
        # A bare year behind "through" is a season range, not a deadline:
        # CLAUDE.md's DR-67 residuals record "2018-19 through 2025-26"
        # spanning as "through 2025". Accepting it would promote a known
        # false-cumulative into a usable calendar date.
        "through 2025",
        # An article with no "end of" is what "Before the 2027-28 season"
        # truncates to (105 such rungs in the live ladder population).
        "before the 2027",
        # ... but "end of" is exactly what makes an article meaningful, so
        # the guard is `article and not end_of` and not a bare article test.
        # "before the end of 2027" refuses for the OTHER reason: "end of"
        # behind an exclusive preposition names the day before an unstated
        # last day.
        "before the end of 2027",
        # Inclusivity not stated.
        "until Dec 31, 2026",
        "up to Dec 31, 2026",
        # Year-less: nothing anchors the year, and guessing mis-orders a
        # ladder crossing New Year (KXAPCALLSENATE-26AUG20 lists "Before
        # Nov 4" (2026) beside "Before Jan 5" (2027)).
        "before Nov 4",
        "by Dec 31",
        "before October",
        "by Friday",
        # Quarter and end-of-period nouns name a period this reader does not
        # place; the phrasing table accepts them as deadline WORDING, which is
        # why they have to be refused here rather than assumed absent.
        "by Q1 2026",
        "by end of Q4 2026",
        "by year-end",
        "by EOY",
        # An article IMMEDIATELY in front of a bare year is the same season
        # range as "before the 2027", reached past that guard because
        # "end of" disarms it: KXCANADACUP-30's title truncates "the 2030-31
        # season" to this span. The "by the end of 2027" row in the
        # acceptance table above is the control that keeps this narrow.
        "by the end of the 2030",
        "by end of the 2030",
        # A bare ISO behind an EXCLUSIVE preposition. _COMPILED_CUMULATIVE[0]
        # cuts one out of a full timestamp, whose last included day is the
        # NAMED one, so dating it a day early is this reader's only fail-OPEN
        # direction — KXDIAZOUT-MDC lists five such rungs under one event
        # ticker in backtest_cache/archive_days.
        "before 2026-03-15",
        "prior to 2026-03-15",
        # A NUMERIC calendar date. _DEADLINE_DATE_TOKEN's m/d alternative
        # spans one, so the phrase reaches this reader, but no accepted token
        # is numeric-m/d and the shape occurs zero times in either candidate
        # population.
        "by 12/31/2026",
        "before 12/31/2026",
        # "end of" at DAY granularity names no period.
        "by the end of Sep 23, 2026",
        # An impossible calendar date states no day.
        "by Feb 30, 2026",
        # Below the 2020 floor the bare-year token shares with the phrasing
        # table: "decrease by 2019" is an amount, not a deadline.
        "by 2019",
    ])
    def test_refuses_wording_it_cannot_place(self, span):
        assert self._read(title=f"Will the thing happen {span}?") is None

    def test_the_bare_year_floor_matches_the_phrasing_table(self):
        # _span_deadline shares _DEADLINE_DATE_TOKEN's 2020-2099 bound, so
        # "decrease by 2019" is an amount rather than a deadline. On the
        # finder path the phrasing table refuses "by 2019" first, so this row
        # drives _span_deadline directly with a bare SPAN — not raw wording,
        # which the anchored _STATED_SPAN rejects — to pin the floor itself.
        assert scanner._span_deadline("by 2019") is None
        assert scanner._span_deadline("by 2020") == date(2020, 12, 31)
        assert scanner._span_deadline("by 2099") == date(2099, 12, 31)
        assert scanner._span_deadline("by 2100") is None

    def test_the_reader_is_not_a_wording_parser(self):
        # _STATED_SPAN is anchored at both ends, so the string must begin with
        # the preposition and end with the date token. Pinned because the
        # docstring used to invite raw wording and C2/C3 feed this reader from
        # market fields: a silent None on every market is indistinguishable
        # from "this corpus has no dated ladders" (the DR-66 shape).
        assert scanner._span_deadline("Will X happen by Dec 31, 2026?") is None
        assert scanner._span_deadline("by Dec 31, 2026?") is None
        assert scanner._span_deadline("resolved by Dec 31, 2026") is None
        # Case and whitespace ARE normalized, which is all the docstring claims.
        assert scanner._span_deadline("  BY   DEC 31,  2026  ") == date(2026, 12, 31)

    def test_every_month_spelling_the_phrasing_table_matches_has_a_number(self):
        # _STATED_MONTH_NUMBERS is a hand-written literal (it must not be read
        # off the locale-sensitive calendar module — see its comment), so the
        # anti-drift guarantee lives here: a spelling _DEADLINE_MONTH matches
        # but this table lacks reads as None, silently removing that family
        # from the ladder population.
        alternation = scanner._DEADLINE_MONTH.removeprefix("(?:").removesuffix(")")
        spellings = set()
        for alternative in alternation.split("|"):
            if alternative.endswith("?"):
                spellings.add(alternative[:-1])  # "Sept?" -> "Sept"
                spellings.add(alternative[:-2])  # "Sept?" -> "Sep"
            else:
                spellings.add(alternative)
        assert len(spellings) >= 24
        missing = sorted(s for s in spellings if s.lower() not in scanner._STATED_MONTH_NUMBERS)
        assert missing == [], missing

    def test_the_reader_covers_every_preposition_the_phrasing_table_can_span(self):
        # Two separately-maintained preposition lists that must agree:
        # _CUMULATIVE_DEADLINE_PATTERNS[0] decides which prepositions can
        # APPEAR in a span, _STATED_SPAN plus the four tables decide which can
        # be READ. A preposition added to the phrasing table alone makes every
        # span carrying it read as None — the same silent family loss
        # TestPhrasingTableInvariants exists to prevent on the tables above.
        inner = scanner._CUMULATIVE_DEADLINE_PATTERNS[0].split("(?:", 1)[1].split(")", 1)[0]
        spannable = {a.replace(r"\s+", " ") for a in inner.split("|")}
        assert "no later than" in spannable and "prior to" in spannable
        readable = (
            scanner._STATED_INCLUSIVE | scanner._STATED_EXCLUSIVE
            | scanner._STATED_THROUGH | scanner._STATED_AMBIGUOUS
        )
        assert spannable == readable, spannable ^ readable
        # ... and _STATED_SPAN's own alternation is the same set again, or a
        # table entry would be unreachable.
        head = scanner._STATED_SPAN.pattern.split("^(", 1)[1].split(")", 1)[0]
        assert set(head.split("|")) == readable

    def test_a_non_str_span_reads_as_absent(self):
        # Same fail-safe-by-type rule leg_sides and _depth_levels follow: a
        # MagicMock auto-attribute must not raise out of the parser.
        assert scanner._span_deadline(MagicMock()) is None
        assert scanner._span_deadline(None) is None
        assert scanner._span_deadline("") is None

    def test_a_non_cumulative_verdict_has_no_stated_deadline(self):
        # Fails closed exactly as cumulative_deadline_pair does: a snapshot
        # or unknown leg has no nesting premise, so it has no ladder rung.
        assert scanner.stated_deadline(
            (scanner.DEADLINE_SNAPSHOT, ("by sep 23, 2026",)), "", "", ""
        ) is None
        assert scanner.stated_deadline((scanner.DEADLINE_UNKNOWN, ()), "", "", "") is None
        assert scanner.stated_deadline(
            (scanner.DEADLINE_CUMULATIVE, ()), "", "", ""
        ) is None

    def test_two_different_days_in_the_deciding_field_refuse(self):
        # Ambiguity in the one field that decides is refused, not resolved:
        # there is no more authoritative field to break the tie.
        assert self._read(
            title="Will it happen by Sep 23, 2026 or by Oct 16, 2026?"
        ) is None

    def test_one_day_stated_twice_in_the_deciding_field_still_reads(self):
        # The converse: two spans naming the SAME day are not ambiguity.
        assert self._read(
            title="Will it happen by Sep 23, 2026 — by Sep 23, 2026?"
        ) == date(2026, 9, 23)

    def test_by_and_before_the_same_named_day_are_one_spelling(self):
        # KXSPACEXSTARSHIP-14-26SEP23 on the 2026-09-22 live snapshot, and
        # Kalshi's standard ladder template: the title says "before Sep 23,
        # 2026" (last included day Sep 22) and the subtitle "By Sep 23, 2026"
        # (Sep 23). The market's close_time is 2026-09-23T03:59Z — 23:59 on
        # Sep 22 in New York — so those are ONE deadline named two ways, not a
        # one-day disagreement, and the day-SET cross-check must read it.
        # Comparing single last-included days instead refused all five rungs
        # of the widest live ladder. The subtitle decides under DR-69.
        assert self._read(
            event_title="SpaceX Starship 14th launch?",
            title="Will SpaceX launch another Starship before Sep 23, 2026?",
            subtitle="By Sep 23, 2026",
        ) == date(2026, 9, 23)

    def test_a_bare_year_against_a_named_day_still_refuses(self):
        # The day-SET check must not admit a field whose readings are
        # DISJOINT from the deciding field's. SCOTREF-27's rungs title a bare
        # year ("called before 2028" -> {2027-12-31, 2028-12-31}) against a
        # subtitle "By Jan 1, 2028" -> {2028-01-01}: no reading in common, so
        # the rung is refused. 30 of the 47 cross-field markets on the
        # 2026-09-22 snapshot are this shape.
        assert self._read(
            title="New Scottish referendum called before 2028?",
            subtitle="By Jan 1, 2028",
        ) is None

    def test_cross_field_conflict_refuses_starship_florida(self):
        # KXSTARSHIPFL-26JUN-26OCT01, same snapshot: a stale sub-contract
        # label ("Before 2026" -> 2025-12-31) against a title nine months
        # later. 47 actively-priced cumulative markets on that snapshot state
        # two different days across their own fields.
        assert self._read(
            event_title="When will SpaceX's Starship launch from Florida?",
            title="Will SpaceX's Starship launch from Florida before Oct 1, 2026",
            subtitle="Before 2026",
        ) is None

    def test_a_truncated_iso_timestamp_rung_is_refused(self):
        # KXDIAZOUT-MDC-26APR01 as backtest_cache/archive_days holds it: a
        # blank cached subtitle makes the TITLE decide, and the title's
        # deadline is a full timestamp that _COMPILED_CUMULATIVE[0] cuts down
        # to a bare ISO date. The market is open through 14:00 on Apr 1, so
        # the last included day is Apr 1, not Mar 31 — and reading it a day
        # early would turn a SAME_DAY refusal against a sibling worded "by
        # Apr 1, 2026" into a 1-day ladder. Five rungs of that one event are
        # in the cache; the reader refuses all five rather than guess.
        assert self._read(
            title="Will Miguel Díaz-Canel leave office before 2026-04-01T14:00:00.000Z?",
        ) is None

    def test_a_non_deciding_field_that_names_no_day_does_not_refuse(self):
        # The cross-check refuses on DISAGREEMENT, never on silence: a
        # year-less phrase elsewhere in the wording names no day to disagree
        # with. Refusing on it would throw away the rungs DR-73 exists for.
        assert self._read(
            event_title="Will it happen before Nov 4?",
            title="Will it happen by Sep 23, 2026?",
        ) == date(2026, 9, 23)

    def test_fields_that_agree_still_read(self):
        # Two fields spelling the SAME day (one inclusive, one exclusive by a
        # day) is agreement, not conflict.
        assert self._read(
            title="Will it happen before Apr 1, 2027?",
            subtitle="By Mar 31, 2027",
        ) == date(2027, 3, 31)

    # control — kills a "refuse everything" mutant, which every refusal row
    # above would pass. These are the SIX (preposition, token-shape) buckets
    # that actually occur in the live same-event ladder candidate population
    # (the 4,161 cumulative, span-carrying rungs of events holding >= 2 of
    # them, on the .git/dr67-scratch 2026-09-22 snapshot of 113,303 markets,
    # 3,951 of which read to a date), each with a real ticker and its measured
    # rung count — the six counts sum to exactly that 3,951. Four small
    # archive day slices add no shape these six do not already cover. If a
    # future tightening refuses one of them it removes a whole live family
    # from the strategy, silently.
    @pytest.mark.parametrize("ticker, span, expected", [
        # 2,834 rungs, e.g. KXXISUCCESSOR-45JAN01-DXUE
        ("KXXISUCCESSOR-45JAN01-DXUE", "before Jan 1, 2045", date(2044, 12, 31)),
        # 525 rungs
        ("KXMILLENNIUMNEXT-45-BSD", "before 2045", date(2044, 12, 31)),
        # 444 rungs
        ("KXFEDHIKE-2-26DEC31", "by Dec 31, 2026", date(2026, 12, 31)),
        # 127 rungs
        ("KXTVSEASONRELEASETHELASTOFUS-26-OCT", "before Oct 2026", date(2026, 9, 30)),
        # 19 rungs — the only live "through" family
        ("KXNYCSTAT-HOME27-A275", "through June 30, 2027", date(2027, 6, 30)),
        # 2 rungs
        ("USCLIMATE-2025", "by 2025", date(2025, 12, 31)),
    ])
    def test_control_every_live_shape_still_reads(self, ticker, span, expected):
        assert self._read(title=f"Will the thing happen {span}?") == expected, ticker


class TestSameEventLadder:
    """DR-73a: ordering two rungs of one event's ladder and measuring the gap.

    Pure arithmetic over two already-read deadlines, so the live finder and
    the backtester can share one definition. The three outcomes are kept
    apart deliberately — a readable ladder, two rungs naming ONE day, and an
    unreadable rung have different remedies and the callers count them
    separately.
    """

    def test_orders_two_rungs_and_measures_the_gap(self):
        assert scanner.same_event_ladder(date(2026, 9, 23), date(2026, 10, 16)) == (
            False, 23,
        )

    def test_swap_is_true_when_the_second_argument_is_earlier(self):
        # swap says "market_a must become market_b": the gap is
        # order-independent, the ordering is not.
        assert scanner.same_event_ladder(date(2026, 10, 16), date(2026, 9, 23)) == (
            True, 23,
        )

    def test_the_same_day_is_its_own_outcome(self):
        assert scanner.same_event_ladder(
            date(2026, 9, 23), date(2026, 9, 23)
        ) == scanner.SAME_DAY

    def test_one_deadline_spelled_two_ways_is_the_same_day(self):
        # "by Mar 31, 2027" and "before Apr 1, 2027" name the identical last
        # included day. Without SAME_DAY this is a one-rung "ladder" with a
        # gap of zero, which the tier arithmetic would happily price.
        inclusive = scanner._span_deadline("by mar 31, 2027")
        exclusive = scanner._span_deadline("before apr 1, 2027")
        assert inclusive == exclusive == date(2027, 3, 31)
        assert scanner.same_event_ladder(inclusive, exclusive) == scanner.SAME_DAY

    def test_mixed_prepositions_one_day_apart_are_a_real_ladder(self):
        # SCOTREF-27 is the one live event (of the 492 holding >= 2 dated
        # cumulative rungs on the 2026-09-22 snapshot) that mixes an
        # inclusive and an exclusive preposition, so the distinction is not
        # theoretical. This row is its arithmetic: the two wordings below name
        # days one apart, not the same day.
        later = scanner._span_deadline("by jan 1, 2028")
        earlier = scanner._span_deadline("before jan 1, 2028")
        assert (later, earlier) == (date(2028, 1, 1), date(2027, 12, 31))
        assert scanner.same_event_ladder(later, earlier) == (True, 1)

    @pytest.mark.parametrize("bad", [
        None,
        "2026-09-23",
        # datetime is a date SUBCLASS, so isinstance would admit it and the
        # date - datetime subtraction would raise TypeError.
        datetime(2026, 9, 23, tzinfo=UTC),
    ])
    def test_an_unreadable_deadline_is_none_on_either_side(self, bad):
        assert scanner.same_event_ladder(bad, date(2026, 9, 23)) is None
        assert scanner.same_event_ladder(date(2026, 9, 23), bad) is None

    def test_a_magicmock_deadline_does_not_raise(self):
        assert scanner.same_event_ladder(MagicMock(), MagicMock()) is None


_LADDER_TITLE = "Will SpaceX launch another Starship %s?"


def _ladder_rung(ticker, deadline_text, *, event="KXSTARSHIP-14", yes_ask, no_ask,
                 close, event_title="", subtitle=""):
    """One rung of a same-event cumulative deadline ladder (DR-73).

    Every rung of one ladder shares an event_ticker and a title that differs
    ONLY in its deadline, so normalize_title collapses them onto one
    time_series_group_key — the shape KXSPACEXSTARSHIP-14 has live.
    """
    return _mock_market(
        ticker=ticker, event_ticker=event, event_title=event_title,
        subtitle=subtitle, title=_LADDER_TITLE % deadline_text,
        yes_ask=yes_ask, no_ask=no_ask, close_time=close,
    )


def _assert_one_ladder_group(mA, mB):
    """Every ladder fixture must actually BE a ladder before its rule is tested.

    Two rungs the group key separates, or one the wording screen does not call
    cumulative, would make a test pass for a reason that has nothing to do
    with DR-73 — the way test_same_event_ticker_never_pairs went vacuous.
    """
    assert time_series_group_key(pair_key(mA), mA.subtitle) == \
        time_series_group_key(pair_key(mB), mB.subtitle)
    assert scanner._market_deadline_profile(mA)[0] == scanner.DEADLINE_CUMULATIVE
    assert scanner._market_deadline_profile(mB)[0] == scanner.DEADLINE_CUMULATIVE
    assert mA.event_ticker == mB.event_ticker


class TestSameEventDeadlineLadders:
    """DR-73: two rungs of ONE event's cumulative deadline ladder are a
    time-series pair, behind config.TIME_SERIES_SAME_EVENT_LADDERS.

    Kalshi lists a question's several deadlines as separate markets inside a
    single event, and every previous version of this finder refused them as
    "multi-choice options". The branch orders and tiers such a pair on its two
    STATED deadlines, never on close_time — a settled or single-instant event
    closes every rung together. _scan sets the switch EXPLICITLY, on or off, so
    no test here depends on the value the switch ships with, and every test
    here has a control.
    """

    def _scan(self, markets, monkeypatch, *, on=True):
        # Both ways: an "off" row that merely left the switch alone would test
        # its shipped value, and change meaning the day that value flips.
        monkeypatch.setattr(scanner, "TIME_SERIES_SAME_EVENT_LADDERS", on)
        return find_time_series_pairs(MagicMock(), held_tickers=set(), markets=markets)

    def _two_rungs(self, *, pA=0.20, pB=0.60, nB=0.40,
                   early="by March 1, 2026", late="by March 20, 2026",
                   close_a=datetime(2026, 3, 1, tzinfo=UTC),
                   close_b=datetime(2026, 3, 20, tzinfo=UTC),
                   event_title=""):
        mA = _ladder_rung("RUNG-EARLY", early, yes_ask=pA, no_ask=round(1 - pA, 4),
                          close=close_a, event_title=event_title)
        mB = _ladder_rung("RUNG-LATE", late, yes_ask=pB, no_ask=nB,
                          close=close_b, event_title=event_title)
        _assert_one_ladder_group(mA, mB)
        return mA, mB

    # ── the admitted case ────────────────────────────────────────────────────

    def test_two_dated_rungs_of_one_event_pair(self, monkeypatch):
        mA, mB = self._two_rungs()
        pairs = self._scan([mA, mB], monkeypatch)
        assert len(pairs) == 1
        pair = pairs[0]
        assert (pair.market_a.ticker, pair.market_b.ticker) == ("RUNG-EARLY", "RUNG-LATE")
        # 2026-03-01 -> 2026-03-20 is 19 days, which is the LONG tier: the
        # 0.40 spread clears 0.30. Carried out on the pair so nothing
        # downstream has to re-derive it from close_time.
        assert pair.stated_gap_days == 19
        assert pair.tradeable

    def test_the_same_fixture_is_refused_with_the_switch_off(self, monkeypatch):
        # control: the ONLY thing standing between this fixture and a pair is
        # the switch, so the test above is not passing for some other reason.
        mA, mB = self._two_rungs()
        assert self._scan([mA, mB], monkeypatch, on=False) == []

    def test_the_disabled_skip_is_counted_and_reported(self, monkeypatch, caplog):
        mA, mB = self._two_rungs()
        with caplog.at_level(logging.INFO):
            self._scan([mA, mB], monkeypatch, on=False)
        assert "same-event deadline ladders are disabled" in caplog.text
        # DR-66: a switch that produces nothing must be distinguishable from a
        # broken rule, so the count is reported rather than silently dropped.
        assert caplog.text.rstrip().endswith("ladders are disabled "
                                             "(config.TIME_SERIES_SAME_EVENT_LADDERS): 1") or \
            "(config.TIME_SERIES_SAME_EVENT_LADDERS): 1" in caplog.text

    def test_the_emitted_ladder_count_is_always_logged(self, monkeypatch, caplog):
        mA, mB = self._two_rungs()
        with caplog.at_level(logging.INFO):
            self._scan([mA, mB], monkeypatch)
        assert "Same-event ladder pairs among the time-series pairs: 1" in caplog.text

    # ── ordering: the stated deadline, never close_time ─────────────────────

    def test_legs_are_ordered_by_stated_deadline_not_close_time(self, monkeypatch):
        # The Mar 1 rung closes a MONTH after the Mar 20 one, so the
        # close_time sort hands this candidate to the loop the wrong way round
        # and only the stated-deadline swap can fix it. market_a must be the
        # earlier DEADLINE, because every leg-side, settlement and reporting
        # contract downstream reads that position (leg_sides, _ordered_legs,
        # _settlement_receipt).
        mA, mB = self._two_rungs(
            close_a=datetime(2026, 4, 1, tzinfo=UTC),
            close_b=datetime(2026, 3, 5, tzinfo=UTC),
        )
        pairs = self._scan([mA, mB], monkeypatch)
        assert len(pairs) == 1
        assert pairs[0].market_a.ticker == "RUNG-EARLY"
        assert pairs[0].stated_gap_days == 19
        # control: the gap is the STATED one (19), not the realized close gap
        # (27 days), so the pair is priced on the tier its deadlines choose.
        assert deadline_gap_days(mA, mB) == 27

    def test_a_swap_does_not_leak_into_later_candidates(self, monkeypatch):
        # mA/mB are per-candidate locals, not the loop variables: swapping the
        # OUTER one in place would re-order every later candidate of that
        # iteration. RUNG-EARLY is the outer member here (it closes first) and
        # its Apr 10 sibling must still pair with it in the right order after
        # the Mar 20 candidate has swapped.
        early = _ladder_rung("RUNG-EARLY", "by April 5, 2026", yes_ask=0.20,
                             no_ask=0.80, close=datetime(2026, 3, 1, tzinfo=UTC))
        swapper = _ladder_rung("RUNG-SWAP", "by March 20, 2026", yes_ask=0.60,
                               no_ask=0.40, close=datetime(2026, 3, 20, tzinfo=UTC))
        later = _ladder_rung("RUNG-LATER", "by April 25, 2026", yes_ask=0.70,
                             no_ask=0.30, close=datetime(2026, 4, 25, tzinfo=UTC))
        _assert_one_ladder_group(early, swapper)
        _assert_one_ladder_group(early, later)
        pairs = self._scan([early, swapper, later], monkeypatch)
        assert len(pairs) == 1
        # The best pair of the group is EARLY(Apr 5) x LATER(Apr 25): spread
        # 0.50, stated gap 20. If the Mar 20 candidate's swap had rewritten
        # the outer variable, this pair would have been built from RUNG-SWAP.
        assert (pairs[0].market_a.ticker, pairs[0].market_b.ticker) == (
            "RUNG-EARLY", "RUNG-LATER")
        assert pairs[0].stated_gap_days == 20

    def test_rungs_that_close_at_one_instant_still_pair_on_the_stated_gap(self, monkeypatch):
        # A settled event closes every rung at once — 681 of 1,821 dated
        # same-event pairs in the archive have a close gap of ZERO days — so
        # close_time cannot tier this pair at all.
        one_instant = datetime(2026, 3, 20, tzinfo=UTC)
        mA, mB = self._two_rungs(close_a=one_instant, close_b=one_instant)
        assert deadline_gap_days(mA, mB) == 0
        pairs = self._scan([mA, mB], monkeypatch)
        assert len(pairs) == 1
        assert pairs[0].stated_gap_days == 19

    @pytest.mark.usefixtures("pre_toggle_defaults")
    def test_the_stated_gap_chooses_the_tier_a_zero_close_gap_would_not(self, monkeypatch):
        # control for the row above, and the reason the stated gap must travel
        # with the pair: a 0.20 spread clears the SHORT tier (0.15) that a
        # close gap of 0 days would select, and fails the LONG tier (0.30) the
        # 19-day stated gap actually demands.
        one_instant = datetime(2026, 3, 20, tzinfo=UTC)
        mA, mB = self._two_rungs(pA=0.20, pB=0.40, nB=0.60,
                                 close_a=one_instant, close_b=one_instant)
        assert self._scan([mA, mB], monkeypatch) == []

    # ── the gap cap, measured on the stated gap ─────────────────────────────

    def test_a_31_day_stated_gap_is_refused_although_the_closes_are_30_apart(self, monkeypatch):
        mA, mB = self._two_rungs(
            early="by March 1, 2026", late="by April 1, 2026",
            close_a=datetime(2026, 3, 1, tzinfo=UTC),
            close_b=datetime(2026, 3, 31, tzinfo=UTC),
        )
        assert deadline_gap_days(mA, mB) == 30  # within the cap on close_time
        assert self._scan([mA, mB], monkeypatch) == []

    def test_a_30_day_stated_gap_is_admitted(self, monkeypatch):
        # control for the row above: one day narrower and the same fixture pairs.
        mA, mB = self._two_rungs(
            early="by March 1, 2026", late="by March 31, 2026",
            close_a=datetime(2026, 3, 1, tzinfo=UTC),
            close_b=datetime(2026, 3, 31, tzinfo=UTC),
        )
        pairs = self._scan([mA, mB], monkeypatch)
        assert len(pairs) == 1
        assert pairs[0].stated_gap_days == MAX_DEADLINE_GAP_DAYS

    # ── the refusals, each with its control ─────────────────────────────────

    def test_a_rung_with_no_readable_year_is_refused(self, monkeypatch):
        # Year-less wording states no placeable day, and guessing the year
        # mis-orders a ladder crossing New Year. Both rungs are year-less here
        # because normalize_title strips the PREPOSITION along with a
        # year-less date ("by November 4" -> ""), so a year-less rung and a
        # dated one never land in one group to begin with.
        mA, mB = self._two_rungs(early="by November 4", late="by December 4")
        assert scanner.stated_deadline(
            scanner._market_deadline_profile(mA), "", mA.title, "") is None
        assert self._scan([mA, mB], monkeypatch) == []

    def test_the_same_rungs_dated_are_admitted(self, monkeypatch):
        # control for the row above: the refusal is the missing YEAR, not the
        # wording shape.
        mA, mB = self._two_rungs(early="by November 4, 2026", late="by December 4, 2026",
                                 close_a=datetime(2026, 11, 4, tzinfo=UTC),
                                 close_b=datetime(2026, 12, 4, tzinfo=UTC))
        # 2026-11-04 -> 2026-12-04 is 30 days, inside the cap, and the 0.40
        # spread clears the LONG tier the gap selects.
        pairs = self._scan([mA, mB], monkeypatch)
        assert len(pairs) == 1 and pairs[0].stated_gap_days == 30

    def test_two_rungs_naming_one_calendar_day_are_refused(self, monkeypatch):
        # "by December 2026" and "by December 31, 2026" are ONE deadline
        # spelled two ways (the month form's last included day IS the 31st),
        # not a two-rung ladder — the SAME_DAY outcome. Their spans differ, so
        # cumulative_deadline_pair admits them and only the date reader can
        # tell.
        mA, mB = self._two_rungs(early="by December 2026", late="by December 31, 2026",
                                 close_a=datetime(2026, 12, 1, tzinfo=UTC),
                                 close_b=datetime(2026, 12, 31, tzinfo=UTC))
        assert scanner.cumulative_deadline_pair(
            scanner._market_deadline_profile(mA), scanner._market_deadline_profile(mB))
        assert self._scan([mA, mB], monkeypatch) == []

    def test_a_rung_whose_own_fields_disagree_is_refused(self, monkeypatch):
        # Both rungs sit under a stale event title naming an irreconcilable
        # deadline ("before 2026" against titles in October and November
        # 2026) — the KXSTARSHIPFL shape. stated_deadline's cross-check
        # refuses the rung rather than trusting the deciding field alone.
        mA, mB = self._two_rungs(
            early="by October 1, 2026", late="by October 21, 2026",
            close_a=datetime(2026, 10, 1, tzinfo=UTC),
            close_b=datetime(2026, 10, 21, tzinfo=UTC),
            event_title="Starship flights before 2026",
        )
        # 20 days apart deliberately: a wider fixture would be refused by the
        # gap cap too, and would pass with the cross-check bypassed.
        assert scanner.stated_deadline(
            scanner._market_deadline_profile(mA),
            mA._event_title, mA.title, mA.subtitle) is None
        assert scanner.stated_deadline(
            scanner._market_deadline_profile(mA), "", "", "") == date(2026, 10, 1)
        assert self._scan([mA, mB], monkeypatch) == []

    def test_the_same_rungs_under_a_dateless_event_title_are_admitted(self, monkeypatch):
        # control: the ONLY difference from the row above is the stale event
        # title, so that row is pinning the cross-check and not the gap cap.
        mA, mB = self._two_rungs(
            early="by October 1, 2026", late="by October 21, 2026",
            close_a=datetime(2026, 10, 1, tzinfo=UTC),
            close_b=datetime(2026, 10, 21, tzinfo=UTC),
            event_title="Starship flights",
        )
        pairs = self._scan([mA, mB], monkeypatch)
        assert len(pairs) == 1 and pairs[0].stated_gap_days == 20

    def test_identical_wording_in_one_event_never_pairs(self, monkeypatch, caplog):
        # Same event AND the same wording: the deadline is not in the wording,
        # so there is nothing to order two rungs by (DR-02's reasoning one
        # level in).
        same = "by March 1, 2026"
        mA = _ladder_rung("R1", same, yes_ask=0.20, no_ask=0.80,
                          close=datetime(2026, 3, 1, tzinfo=UTC))
        mB = _ladder_rung("R2", same, yes_ask=0.60, no_ask=0.40,
                          close=datetime(2026, 3, 20, tzinfo=UTC))
        _assert_one_ladder_group(mA, mB)
        assert scanner._identical_wording(mA, mB)
        with caplog.at_level(logging.INFO):
            assert self._scan([mA, mB], monkeypatch) == []
        # Refused HERE, not downstream. cumulative_deadline_pair would also
        # refuse it — identical wording states identical spans — but as
        # "the two rungs state the same deadline", which is a different
        # finding and would make that counter mean two things at once
        # (DR-72's whole point).
        assert "state the same deadline" not in caplog.text
        # And COUNTED here, on its own silent-at-zero line. It is the largest
        # single ladder refusal on the real snapshot (502 of 3,354), so a
        # bare `continue` would drop 15% of the branch's input out of the
        # funnel with nothing in the log to reconstruct it from.
        assert (
            "refused because the two rungs' wording is identical "
            "(the deadline is not in the wording): 1"
        ) in caplog.text

    def test_an_empty_shared_event_ticker_never_pairs(self, monkeypatch, caplog):
        # Two markets sharing an EMPTY event ticker share no event at all, so
        # nothing identifies the ladder they would belong to — fails closed,
        # and on its OWN line rather than being attributed to whichever check
        # happens to follow.
        mA = _ladder_rung("E1", "by March 1, 2026", yes_ask=0.20, no_ask=0.80,
                          close=datetime(2026, 3, 1, tzinfo=UTC), event="")
        mB = _ladder_rung("E2", "by March 20, 2026", yes_ask=0.60, no_ask=0.40,
                          close=datetime(2026, 3, 20, tzinfo=UTC), event="")
        _assert_one_ladder_group(mA, mB)
        with caplog.at_level(logging.INFO):
            assert self._scan([mA, mB], monkeypatch) == []
        assert (
            "refused because the shared event ticker is empty: 1"
        ) in caplog.text
        # Control: the identical fixture with a real shared event ticker is
        # admitted, so the refusal above is the empty ticker and nothing else.
        gA = _ladder_rung("E1", "by March 1, 2026", yes_ask=0.20, no_ask=0.80,
                          close=datetime(2026, 3, 1, tzinfo=UTC))
        gB = _ladder_rung("E2", "by March 20, 2026", yes_ask=0.60, no_ask=0.40,
                          close=datetime(2026, 3, 20, tzinfo=UTC))
        assert len(self._scan([gA, gB], monkeypatch)) == 1

    def test_a_snapshot_rung_in_one_event_never_pairs(self, monkeypatch, caplog):
        # The wording screen is unchanged inside the branch: "Starship count
        # ON <date>" does not nest, whether or not the two markets share an
        # event.
        mA = _mock_market(ticker="S1", event_ticker="KXSNAP-1",
                          title="Starship flights on March 1, 2026",
                          yes_ask=0.20, no_ask=0.80,
                          close_time=datetime(2026, 3, 1, tzinfo=UTC))
        mB = _mock_market(ticker="S2", event_ticker="KXSNAP-1",
                          title="Starship flights on March 20, 2026",
                          yes_ask=0.60, no_ask=0.40,
                          close_time=datetime(2026, 3, 20, tzinfo=UTC))
        assert time_series_group_key(pair_key(mA), "") == \
            time_series_group_key(pair_key(mB), "")
        assert scanner._market_deadline_profile(mA)[0] == scanner.DEADLINE_SNAPSHOT
        with caplog.at_level(logging.INFO):
            assert self._scan([mA, mB], monkeypatch) == []
        # And refused BY the wording screen, on its own counter. Without that
        # screen the pair still fails — stated_deadline reads nothing off a
        # snapshot verdict — but it would be reported as a rung whose deadline
        # states no placeable day, which is a parser problem, not a
        # wrong-kind-of-market one.
        assert "deciding field is snapshot wording" in caplog.text
        assert "states no placeable calendar day" not in caplog.text

    # ── the mixed group ─────────────────────────────────────────────────────

    def _mixed_group(self):
        """One group holding a cross-event pair AND a wider same-event ladder."""
        m1 = _mock_market(ticker="M1", event_ticker="EVA-1",
                          title=_LADDER_TITLE % "by March 1, 2026",
                          yes_ask=0.20, no_ask=0.80,
                          close_time=datetime(2026, 3, 1, tzinfo=UTC))
        m2 = _mock_market(ticker="M2", event_ticker="EVB-1",
                          title=_LADDER_TITLE % "by March 20, 2026",
                          yes_ask=0.60, no_ask=0.40,
                          close_time=datetime(2026, 3, 20, tzinfo=UTC))
        m3 = _mock_market(ticker="M3", event_ticker="EVA-1",
                          title=_LADDER_TITLE % "by March 15, 2026",
                          yes_ask=0.95, no_ask=0.05,
                          close_time=datetime(2026, 3, 15, tzinfo=UTC))
        return m1, m2, m3

    @pytest.mark.usefixtures("pre_toggle_defaults")
    def test_a_wider_ladder_wins_the_groups_one_best_slot(self, monkeypatch):
        # (pre_toggle_defaults: the shipped 0.5 ceiling would refuse the 0.75 ladder)
        pairs = self._scan(list(self._mixed_group()), monkeypatch)
        assert len(pairs) == 1
        # M1 x M3 (one event, spread 0.75) beats the cross-event M1 x M2
        # (spread 0.40). Skipping is not only subtractive and neither is
        # admitting: turning ladders on can DISPLACE a cross-event pair.
        assert (pairs[0].market_a.ticker, pairs[0].market_b.ticker) == ("M1", "M3")
        assert pairs[0].stated_gap_days == 14

    def test_with_the_switch_off_the_cross_event_pair_is_returned_unchanged(self, monkeypatch):
        # control, and the cross-event invariant: nothing about this pair —
        # its legs, its prices, or its (absent) stated gap — moves.
        pairs = self._scan(list(self._mixed_group()), monkeypatch, on=False)
        assert len(pairs) == 1
        assert (pairs[0].market_a.ticker, pairs[0].market_b.ticker) == ("M1", "M2")
        assert pairs[0].stated_gap_days is None
        assert (pairs[0].pA, pairs[0].pB, pairs[0].nB) == (0.20, 0.60, 0.40)


class TestPairGapDays:
    """DR-73: one reader for the gap every downstream tier is measured on.

    A ladder carries its STATED gap; everything else is tiered on close_time.
    Read by TYPE, never truthiness — a genuine 0-day stated gap is falsy and a
    bool is not an int here.
    """

    def _pair(self, stated, *, close_gap=0):
        mA = _mock_market(ticker="A", event_ticker="E1",
                          close_time=datetime(2026, 3, 1, tzinfo=UTC))
        mB = _mock_market(ticker="B", event_ticker="E1",
                          close_time=datetime(2026, 3, 1, tzinfo=UTC) + timedelta(days=close_gap))
        return CandidatePair(
            market_a=mA, market_b=mB, pA=0.20, pB=0.60, nA=0.80, tradeable=True,
            canonical_title="t", pair_type="time_series", nB=0.40,
            stated_gap_days=stated,
        )

    def test_a_ladder_reads_its_stated_gap(self):
        assert scanner.pair_gap_days(self._pair(19, close_gap=0)) == 19

    def test_a_non_ladder_reads_the_close_time_gap(self):
        assert scanner.pair_gap_days(self._pair(None, close_gap=9)) == 9

    def test_a_zero_day_stated_gap_is_not_falsy_back_to_close_time(self):
        # control for a truthiness read: `stated or deadline_gap_days(...)`
        # would return 9 here.
        assert scanner.pair_gap_days(self._pair(0, close_gap=9)) == 0

    @pytest.mark.parametrize("bad", [True, 19.0, "19", MagicMock()])
    def test_anything_that_is_not_an_int_falls_back(self, bad):
        # bool is an int SUBCLASS, so isinstance would read True as a one-day
        # ladder — the same fail-safe-by-type rule leg_sides follows.
        assert scanner.pair_gap_days(self._pair(bad, close_gap=9)) == 9

    @pytest.mark.usefixtures("pre_toggle_defaults")
    def test_the_ceiling_is_tiered_on_the_stated_gap(self):
        # _pair_max_sum is the downstream re-derivation that matters: a ladder
        # whose rungs closed at one instant must keep the LONG tier its stated
        # gap chose, not drop to the short one a 0-day close gap implies.
        # config.py's settings, tier floors on (pre_toggle_defaults).
        settings = config.live_settings()
        assert scanner._pair_max_sum(self._pair(19, close_gap=0), settings) == pytest.approx(0.70)
        assert scanner._pair_max_sum(self._pair(None, close_gap=0), settings) == pytest.approx(0.85)


class TestDeadlineGuardFinders:
    """Finder-level pins for the cumulative-deadline guard: the verdict gate,
    the span-presence check, and running the screen before best-pair
    selection.

    The predicate-level tests above prove the RULE; these prove
    find_time_series_pairs actually APPLIES it end to end, on fixtures where
    nothing else — the price tiers, the deadline-gap cap, the one-series rule
    — would independently have refused the pair. Every test whose fixture
    depends on its legs grouping together — which is every test that asserts
    [], and test_screen_runs_before_best_pair_selection, which asserts a
    specific surviving pair — also asserts, inside the test, that those legs
    share a group key (so a fixture that silently stops grouping together —
    e.g. a normalize_title drift — fails loudly instead of passing for the
    wrong reason). Every test that asserts [] additionally
    carries an in-test positive control: the same fixture, changed only in
    the property under test, that returns exactly one pair.
    """

    def test_level_at_instant_legs_are_refused(self):
        # control — kills M03full. "at the close on <date>," is a SNAPSHOT
        # marker inside an otherwise cumulative-looking title. Both legs
        # carry distinct dated spans, so only the verdict gate — not the span
        # check — refuses this pair.
        t1 = "Will BTC be above $100k at the close on Sep 30, 2026, before Oct 1, 2026?"
        t2 = "Will BTC be above $100k at the close on Oct 9, 2026, before Oct 10, 2026?"
        mA = _mock_market(
            ticker="PA-1", event_ticker="EVA-1", title=t1,
            yes_ask=0.10, no_ask=0.90, close_time=datetime(2026, 9, 30, tzinfo=UTC),
        )
        mB = _mock_market(
            ticker="PB-1", event_ticker="EVB-1", title=t2,
            yes_ask=0.30, no_ask=0.70, close_time=datetime(2026, 10, 9, tzinfo=UTC),
        )
        assert scanner.deadline_gap_days(mA, mB) == 9  # inside the short tier
        assert normalize_title(pair_key(mA)) == normalize_title(pair_key(mB))
        assert scanner._market_deadline_profile(mA) == (
            scanner.DEADLINE_SNAPSHOT, ("before oct 1, 2026",),
        )
        assert scanner._market_deadline_profile(mB) == (
            scanner.DEADLINE_SNAPSHOT, ("before oct 10, 2026",),
        )
        assert find_time_series_pairs(
            MagicMock(), held_tickers=set(), markets=[mA, mB],
        ) == []

        # Control: the same gap and prices with "at the close on <date>,"
        # removed — both legs read cumulative and the pair forms, proving the
        # refusal above comes from the marker, not the fixture's price or gap.
        cA = _mock_market(
            ticker="PA-1", event_ticker="EVA-1",
            title="Will BTC be above $100k before Oct 1, 2026?",
            yes_ask=0.10, no_ask=0.90, close_time=datetime(2026, 9, 30, tzinfo=UTC),
        )
        cB = _mock_market(
            ticker="PB-1", event_ticker="EVB-1",
            title="Will BTC be above $100k before Oct 10, 2026?",
            yes_ask=0.30, no_ask=0.70, close_time=datetime(2026, 10, 9, tzinfo=UTC),
        )
        assert len(find_time_series_pairs(
            MagicMock(), held_tickers=set(), markets=[cA, cB],
        )) == 1

    def test_dated_leg_never_pairs_with_spanless_cumulative_leg(self):
        # control — kills M02 at the finder level (TestCumulativeDeadlinePair
        # Predicate::test_spanless_cumulative_leg_is_refused kills it at the
        # predicate level already; this proves find_time_series_pairs really
        # applies that guard end to end). A shared event title carries a
        # SPANLESS cumulative marker ("at any time"); A's own title also
        # names a dated deadline, B's names none at all. Both legs still
        # classify cumulative (title falls through to the event title for
        # B), but B's spans are empty.
        evt = "Will X happen at any time?"
        mA = _mock_market(
            ticker="PA-1", event_ticker="EVA-1", title="Will X happen by March 1?",
            event_title=evt, yes_ask=0.20, no_ask=0.80,
            close_time=datetime(2026, 3, 1, tzinfo=UTC),
        )
        mB = _mock_market(
            ticker="PB-1", event_ticker="EVB-1", title="Will X happen Mar 9?",
            event_title=evt, yes_ask=0.60, no_ask=0.40,
            close_time=datetime(2026, 3, 9, tzinfo=UTC),
        )
        assert normalize_title(pair_key(mA)) == normalize_title(pair_key(mB))
        assert scanner._market_deadline_profile(mA) == (
            scanner.DEADLINE_CUMULATIVE, ("by march 1",),
        )
        assert scanner._market_deadline_profile(mB) == (scanner.DEADLINE_CUMULATIVE, ())
        assert find_time_series_pairs(
            MagicMock(), held_tickers=set(), markets=[mA, mB],
        ) == []

        # Control: B titled "by March 9?" instead — now it names its own
        # deadline and the pair forms.
        mB2 = _mock_market(
            ticker="PB-1", event_ticker="EVB-1", title="Will X happen by March 9?",
            event_title=evt, yes_ask=0.60, no_ask=0.40,
            close_time=datetime(2026, 3, 9, tzinfo=UTC),
        )
        assert len(find_time_series_pairs(
            MagicMock(), held_tickers=set(), markets=[mA, mB2],
        )) == 1

    def test_dated_leg_never_pairs_with_unknown_leg(self):
        # control — kills M02+M03a together (M03a alone is an equivalent
        # mutant everywhere — see test_spanless_cumulative_leg_is_refused's
        # comment — so only this shape, where the span check's own "both
        # legs must name a span" arm is what M03a leaves undefended, can tell
        # M02 and M03a apart). Same shape as the spanless-cumulative test
        # above, without the shared event-title marker: B's title alone
        # names no deadline at all, so B classifies UNKNOWN rather than
        # spanless-cumulative. Reaching [] here needs BOTH guards — the
        # verdict gate AND the span check.
        mA = _mock_market(
            ticker="PA-1", event_ticker="EVA-1", title="Will X happen by March 1?",
            yes_ask=0.20, no_ask=0.80, close_time=datetime(2026, 3, 1, tzinfo=UTC),
        )
        mB = _mock_market(
            ticker="PB-1", event_ticker="EVB-1", title="Will X happen Mar 9?",
            yes_ask=0.60, no_ask=0.40, close_time=datetime(2026, 3, 9, tzinfo=UTC),
        )
        # The group-key assertion matters here specifically: with no shared
        # event_title, the ONLY reason mA and mB are ever compared at all is
        # that their titles' date tokens strip to the same normalized key —
        # a drift in _DATE_PATTERNS (e.g. losing the bare "Mon d" entry that
        # strips "Mar 9") would split them into different groups and this
        # test would read [] for a completely different reason, with the
        # guard under test never even reached. Reproduced: deleting that
        # _DATE_PATTERNS entry together with M02+M03a still returns [], but
        # this assertion catches it where the bare `== []` below would not.
        assert normalize_title(pair_key(mA)) == normalize_title(pair_key(mB))
        assert scanner._market_deadline_profile(mA) == (
            scanner.DEADLINE_CUMULATIVE, ("by march 1",),
        )
        assert scanner._market_deadline_profile(mB) == (scanner.DEADLINE_UNKNOWN, ())
        assert find_time_series_pairs(
            MagicMock(), held_tickers=set(), markets=[mA, mB],
        ) == []

        # Control: B titled "by March 9?" instead — now it names its own
        # deadline, both legs classify cumulative, and the pair forms.
        mB2 = _mock_market(
            ticker="PB-1", event_ticker="EVB-1", title="Will X happen by March 9?",
            yes_ask=0.60, no_ask=0.40, close_time=datetime(2026, 3, 9, tzinfo=UTC),
        )
        assert len(find_time_series_pairs(
            MagicMock(), held_tickers=set(), markets=[mA, mB2],
        )) == 1

    def test_screen_runs_before_best_pair_selection(self):
        # control — kills M26 (screening moved to run after selection). The
        # DR-67 screen must run BEFORE the one-best-pair-per-group selection,
        # or a refused candidate could still win the group's slot. Four
        # markets, one normalized group: the WIDEST raw price gap (T1 vs T2,
        # 0.90) pairs a cumulative leg with a snapshot ("after <date>") leg
        # and is refused outright; the group's returned pair is instead the
        # widest SURVIVING candidate (T1 vs T3, gap 0.30) — a genuinely
        # different pair, not merely a smaller version of the refused one.
        base = datetime(2026, 3, 1, tzinfo=UTC)
        m1 = _mock_market(
            ticker="T1", event_ticker="EVT1", title="Will X happen by March 1?",
            yes_ask=0.05, no_ask=0.95, close_time=base,
        )
        m2 = _mock_market(
            ticker="T2", event_ticker="EVT2", title="Will X happen after March 25?",
            yes_ask=0.95, no_ask=0.05, close_time=base + timedelta(days=24),
        )
        m3 = _mock_market(
            ticker="T3", event_ticker="EVT3", title="Will X happen by March 21?",
            yes_ask=0.35, no_ask=0.65, close_time=base + timedelta(days=20),
        )
        m4 = _mock_market(
            ticker="T4", event_ticker="EVT4", title="Will X happen by March 11?",
            yes_ask=0.15, no_ask=0.85, close_time=base + timedelta(days=10),
        )
        assert scanner._market_deadline_profile(m2)[0] == scanner.DEADLINE_SNAPSHOT
        # The fixture only exercises M26 while all four legs share ONE group:
        # if the refused leg fell out of the group, the asserted pair would
        # still be returned and the mutant would survive un-noticed. The
        # grouping of "after March 25?" rests on _DATE_PATTERNS, the list
        # CLAUDE.md flags as the most regression-prone in the codebase, so
        # pin it here rather than assume it.
        assert len({
            scanner.time_series_group_key(scanner.pair_key(m), m.subtitle)
            for m in (m1, m2, m3, m4)
        }) == 1
        [pair] = find_time_series_pairs(
            MagicMock(), held_tickers=set(), markets=[m1, m2, m3, m4],
        )
        assert (pair.market_a.ticker, pair.market_b.ticker) == ("T1", "T3")
        assert pair.pB - pair.pA == pytest.approx(0.30)

    def test_dated_identical_wording_is_same_title_only(self):
        # control — kills a mutant that drops the spans-differ requirement
        # (`return True` in place of `spans_a != spans_b`). Identical wording
        # that STATES a deadline is refused by the cumulative-deadline rule's
        # spans-differ conjunct (the two spans are equal) — NOT by the
        # one-series rule. The two legs sit on DIFFERENT series here, so
        # _same_series is False and would not refuse the pair on its own;
        # only the spans-differ check does. Both legs close at the SAME
        # instant, so the same-title close gate (DR-74) admits the same-title
        # copy — the one-pair, one-type exclusivity DR-67 guarantees. Mirror
        # of test_backtester.py::TestRunBacktestCrossTypeDedup::
        # test_dated_identical_wording_is_same_title_only.
        title = "Will X happen by Dec 31, 2026?"
        close = datetime(2026, 12, 1, tzinfo=UTC)
        mA = _mock_market(
            ticker="A1", event_ticker="EVA-1", title=title, event_title="EV",
            yes_ask=0.30, no_ask=0.70, close_time=close,
        )
        mB = _mock_market(
            ticker="B1", event_ticker="EVB-1", title=title, event_title="EV",
            yes_ask=0.60, no_ask=0.40, close_time=close,
        )
        assert normalize_title(pair_key(mA)) == normalize_title(pair_key(mB))
        assert scanner._market_deadline_profile(mA) == (
            scanner.DEADLINE_CUMULATIVE, ("by dec 31, 2026",),
        )
        assert scanner._market_deadline_profile(mB) == (
            scanner.DEADLINE_CUMULATIVE, ("by dec 31, 2026",),
        )
        assert scanner._identical_wording(mA, mB) is True
        assert scanner._same_series(mA, mB) is False
        assert scanner._closes_apart(mA, mB) is False
        assert len(find_same_title_pairs([mA, mB])) == 1
        assert find_time_series_pairs(
            MagicMock(), held_tickers=set(), markets=[mA, mB],
        ) == []

        # Positive control: B's wording states a DIFFERENT deadline ("Dec
        # 20" instead of "Dec 31") on the same two series, at the same prices
        # and the same close instant — the WORDING is the only thing that
        # changed (a 0-day close gap is a short-tier time-series pair, and
        # 0.60 - 0.30 clears its tier). The spans now differ, so the
        # time-series pair forms — proving the [] above comes from the
        # spans-differ conjunct, not from the price tier, the close times or
        # the two series being distinct.
        mB3 = _mock_market(
            ticker="B1", event_ticker="EVB-1", title="Will X happen by Dec 20, 2026?",
            event_title="EV", yes_ask=0.60, no_ask=0.40, close_time=close,
        )
        assert len(find_time_series_pairs(
            MagicMock(), held_tickers=set(), markets=[mA, mB3],
        )) == 1

    @pytest.mark.parametrize("title", ["Q", "Will X happen by Dec 31, 2026?"])
    def test_identical_wording_closing_apart_forms_no_pair_of_either_type(self, title):
        # regression (DR-74) — the RELABEL guard. Identical wording on two
        # DIFFERENT series whose markets close 19 days apart is two fixtures,
        # so the same-title close gate refuses it; and because the finders
        # are mutually exclusive on one ticker pair (DR-67: identical wording
        # states the same deadline spans, or none), the refused same-title
        # copy cannot come back as a time-series pair either — undated ("Q",
        # unknown wording) or dated (one deadline stated twice). So nothing
        # reaches main._dedup_pairs, and a same-title-only gate cannot
        # relabel the trade. Mirror of test_backtester.py::
        # TestRunBacktestCrossTypeDedup::
        # test_identical_wording_closing_apart_forms_no_pair_of_either_type.
        from kalshi_betting.main import _dedup_pairs

        mA = _mock_market(
            ticker="A1", event_ticker="EVA-1", title=title, event_title="EV",
            yes_ask=0.30, no_ask=0.70, close_time=datetime(2026, 12, 1, tzinfo=UTC),
        )
        mB = _mock_market(
            ticker="B1", event_ticker="EVB-1", title=title, event_title="EV",
            yes_ask=0.60, no_ask=0.40, close_time=datetime(2026, 12, 20, tzinfo=UTC),
        )
        assert normalize_title(pair_key(mA)) == normalize_title(pair_key(mB))
        assert scanner._identical_wording(mA, mB) is True
        assert scanner._same_series(mA, mB) is False
        assert scanner._closes_apart(mA, mB) is True
        same_title = find_same_title_pairs([mA, mB])
        time_series = find_time_series_pairs(
            MagicMock(), held_tickers=set(), markets=[mA, mB],
        )
        assert same_title == []
        assert time_series == []
        assert _dedup_pairs(same_title, time_series) == []

        # Positive control: B moved onto A's close instant, nothing else
        # changed — the same-title pair forms, so the [] above is the close
        # gate and not the price, the wording or the series.
        mB_aligned = _mock_market(
            ticker="B1", event_ticker="EVB-1", title=title, event_title="EV",
            yes_ask=0.60, no_ask=0.40, close_time=datetime(2026, 12, 1, tzinfo=UTC),
        )
        assert len(find_same_title_pairs([mA, mB_aligned])) == 1


class TestSpansFromDecidingField:
    """DR-69: a leg's deadline spans come from the field that DECIDED its
    verdict, and from no other field.

    Before DR-69 the verdict came from ONE field (subtitle, then title, then
    event title — the first carrying a marker) while the spans were gathered
    from ALL THREE. So a spanless deciding field ("at any time") could borrow a
    date from a field that decided nothing, and two legs whose deciding fields
    stated different deadlines could be refused because another field restated
    both. The change is not a pure tightening, so both directions are pinned
    here. Mirror: test_backtester.py::TestSpansFromDecidingField.
    """

    @staticmethod
    def _keys(*markets):
        return {time_series_group_key(pair_key(m), m.subtitle) for m in markets}

    def test_spanless_subtitle_cannot_borrow_event_title_spans(self):
        # regression — before DR-69 this pair FORMED. The subtitle "At any
        # time" decides "cumulative" but names no date; the event titles'
        # distinct "by <date>" phrases used to be lent to it, so the two legs
        # read as two different deadlines. The titles are snapshots ("on
        # <date>") and never get a say: the subtitle outranks them.
        mA = _mock_market(
            ticker="PA-1", event_ticker="EVA-1",
            title="Will SOL be above $180 on Sep 14, 2026?", subtitle="At any time",
            event_title="SOL above $180 by Sep 14, 2026?",
            yes_ask=0.20, no_ask=0.80, close_time=datetime(2026, 9, 14, tzinfo=UTC),
        )
        mB = _mock_market(
            ticker="PB-1", event_ticker="EVB-1",
            title="Will SOL be above $180 on Sep 18, 2026?", subtitle="At any time",
            event_title="SOL above $180 by Sep 18, 2026?",
            yes_ask=0.60, no_ask=0.38, close_time=datetime(2026, 9, 18, tzinfo=UTC),
        )
        assert len(self._keys(mA, mB)) == 1
        # The event titles really do carry two distinct dates — the fixture
        # exercises borrowing, not a pair with no date anywhere.
        assert scanner.deadline_profile(mA._event_title, "", "") == (
            scanner.DEADLINE_CUMULATIVE, ("by sep 14, 2026",),
        )
        assert scanner.deadline_profile(mB._event_title, "", "") == (
            scanner.DEADLINE_CUMULATIVE, ("by sep 18, 2026",),
        )
        assert scanner._market_deadline_profile(mA) == (scanner.DEADLINE_CUMULATIVE, ())
        assert scanner._market_deadline_profile(mB) == (scanner.DEADLINE_CUMULATIVE, ())
        assert find_time_series_pairs(
            MagicMock(), held_tickers=set(), markets=[mA, mB],
        ) == []

        # Control: the deciding field itself names the two deadlines — same
        # prices, gap and series — and the pair forms.
        cA = _mock_market(
            ticker="PA-1", event_ticker="EVA-1",
            title="Will SOL be above $180 on Sep 14, 2026?", subtitle="By Sep 14, 2026",
            event_title="SOL above $180 by Sep 14, 2026?",
            yes_ask=0.20, no_ask=0.80, close_time=datetime(2026, 9, 14, tzinfo=UTC),
        )
        cB = _mock_market(
            ticker="PB-1", event_ticker="EVB-1",
            title="Will SOL be above $180 on Sep 18, 2026?", subtitle="By Sep 18, 2026",
            event_title="SOL above $180 by Sep 18, 2026?",
            yes_ask=0.60, no_ask=0.38, close_time=datetime(2026, 9, 18, tzinfo=UTC),
        )
        assert len(self._keys(cA, cB)) == 1
        assert len(find_time_series_pairs(
            MagicMock(), held_tickers=set(), markets=[cA, cB],
        )) == 1

    def test_deciding_field_spans_can_admit_a_pair(self):
        # regression — the OTHER direction: before DR-69 this pair was
        # REFUSED. Each title decides and states its own deadline, but each
        # event title names the OTHER leg's date, so the all-field span unions
        # were equal and the pair read as one deadline stated twice.
        mA = _mock_market(
            ticker="PA-1", event_ticker="EVA-1", title="Will X cut by June 1, 2026?",
            event_title="Will X cut by June 20, 2026?",
            yes_ask=0.20, no_ask=0.80, close_time=datetime(2026, 6, 1, tzinfo=UTC),
        )
        mB = _mock_market(
            ticker="PB-1", event_ticker="EVB-1", title="Will X cut by June 20, 2026?",
            event_title="Will X cut by June 1, 2026?",
            yes_ask=0.60, no_ask=0.38, close_time=datetime(2026, 6, 20, tzinfo=UTC),
        )
        assert len(self._keys(mA, mB)) == 1
        # The union over all fields is identical on both legs — the old rule's
        # reason for refusing.
        def union(m):
            return set(scanner.deadline_profile("", m.title, "")[1]) | set(
                scanner.deadline_profile(m._event_title, "", "")[1]
            )

        assert union(mA) == union(mB) == {"by june 1, 2026", "by june 20, 2026"}
        assert scanner._market_deadline_profile(mA) == (
            scanner.DEADLINE_CUMULATIVE, ("by june 1, 2026",),
        )
        assert scanner._market_deadline_profile(mB) == (
            scanner.DEADLINE_CUMULATIVE, ("by june 20, 2026",),
        )
        [pair] = find_time_series_pairs(MagicMock(), held_tickers=set(), markets=[mA, mB])
        assert (pair.market_a.ticker, pair.market_b.ticker) == ("PA-1", "PB-1")

    def test_mve_event_title_route_still_pairs(self):
        # control — kills a mutant that stops _deciding_field falling through
        # to the event title (`for text in (subtitle, title):`). The MVE
        # shape: an option label with no date and no title; the deadline lives
        # in the parent EVENT title, which is then the deciding field and
        # supplies the spans itself, so DR-69 keeps this route open.
        mA = _mock_market(
            ticker="PA-1", event_ticker="EVA-1", title="", subtitle="Trump",
            event_title="Presidential Election Winner by March 1, 2026",
            yes_ask=0.20, no_ask=0.80, close_time=datetime(2026, 3, 1, tzinfo=UTC),
        )
        mB = _mock_market(
            ticker="PB-1", event_ticker="EVB-1", title="", subtitle="Trump",
            event_title="Presidential Election Winner by March 20, 2026",
            yes_ask=0.60, no_ask=0.38, close_time=datetime(2026, 3, 20, tzinfo=UTC),
        )
        assert len(self._keys(mA, mB)) == 1
        assert scanner._market_deadline_profile(mA) == (
            scanner.DEADLINE_CUMULATIVE, ("by march 1, 2026",),
        )
        assert scanner._market_deadline_profile(mB) == (
            scanner.DEADLINE_CUMULATIVE, ("by march 20, 2026",),
        )
        assert len(find_time_series_pairs(
            MagicMock(), held_tickers=set(), markets=[mA, mB],
        )) == 1

    def test_span_normalization_folds_case_and_whitespace(self):
        # control — kills M21 (the span's .lower() dropped) and M22 (its
        # whitespace collapse dropped). One deadline spelled with a doubled
        # space and one in capitals must still read as ONE deadline, or the
        # pair would be taken as two deadlines when it is one stated twice.
        spaced = scanner.deadline_profile("", "Will X happen by  June 30, 2026?", "")
        shouted = scanner.deadline_profile("", "Will X happen BY JUNE 30, 2026?", "")
        assert spaced == shouted == (scanner.DEADLINE_CUMULATIVE, ("by june 30, 2026",))
        assert scanner.cumulative_deadline_pair(spaced, shouted) is False

    def test_single_field_overlapping_spans_still_differ(self):
        # control — kills a mutant that refuses any pair whose span sets
        # share a phrase (`return set(spans_a).isdisjoint(spans_b)` in place
        # of `spans_a != spans_b`). One field naming two deadlines, one of
        # them shared with the other leg, still states a different SET.
        a = scanner.deadline_profile("", "Will X happen by June 30 or by Dec 31, 2026?", "")
        b = scanner.deadline_profile("", "Will X happen by June 30 or by Mar 31, 2027?", "")
        assert a == (scanner.DEADLINE_CUMULATIVE, ("by dec 31, 2026", "by june 30"))
        assert b == (scanner.DEADLINE_CUMULATIVE, ("by june 30", "by mar 31, 2027"))
        assert scanner.cumulative_deadline_pair(a, b) is True

    def test_cross_field_overlap_is_refused_by_design(self):
        # regression — this pair outcome CHANGED with DR-69, deliberately
        # (the phrasing verdict did not: both legs read cumulative before
        # and after). Both
        # legs' sub-contract reads "$1 by 2027": the subtitle decides, and it
        # states ONE deadline, the same on both legs. The titles' differing
        # "by June 30" / "by July 31" belong to a field that decided nothing,
        # so they can no longer make the two legs look like two deadlines.
        # Before DR-69 the all-field unions differed and this returned True.
        a = scanner.deadline_profile("", "Y by June 30", "$1 by 2027")
        b = scanner.deadline_profile("", "Y by July 31", "$1 by 2027")
        assert a == b == (scanner.DEADLINE_CUMULATIVE, ("by 2027",))
        assert scanner.cumulative_deadline_pair(a, b) is False


class TestClassifyOncePerMarket:
    """The classifier must run exactly ONCE PER MARKET, never once per
    candidate PAIR — a group of N members produces O(N^2) candidate pairs, so
    per-pair classification re-runs the regex tables an order of magnitude
    more often (CLAUDE.md's "classifier is computed ONCE PER MARKET"
    paragraph, and TestExtractPairsPerformanceSmoke's 50,000-member budget).
    """

    def test_live_finder_classifies_each_market_once(self, monkeypatch):
        # control — kills M11a (901 calls instead of 31 — one per candidate
        # pair rather than one per market).
        base = datetime(2026, 1, 1, tzinfo=UTC)
        markets = []
        for i in range(29):
            d = base + timedelta(days=i)
            yes_ask = round(0.02 + i * 0.03, 2)
            markets.append(_mock_market(
                ticker=f"T{i}", event_ticker=f"EVT{i}",
                title=f"Will X happen by {d:%B %d, %Y}?",
                yes_ask=yes_ask, no_ask=round(1 - yes_ask, 2), close_time=d,
            ))
        # A duplicate deadline (same stated span as T5) forces a REAL phrasing
        # refusal inside the group, so classify-once is exercised on the
        # refusal path too, not only the accepted one — this keeps the pin
        # alive even if a future change adds a classification call at the
        # per-refusal site (rather than only at the up-front, once-per-market
        # site this test currently observes through the monkeypatched
        # deadline_profile).
        d5 = base + timedelta(days=5)
        markets.append(_mock_market(
            ticker="T5DUP", event_ticker="EVT5DUP",
            title=f"Will X happen by {d5:%B %d, %Y}?",
            yes_ask=0.80, no_ask=0.20, close_time=d5 + timedelta(hours=1),
        ))
        # Deviation from the plan's C1 row: the plan's fixture puts an "after
        # March 5, 2026" leg INSIDE the 30-market group, as the candidate the
        # phrasing screen refuses. That title normalizes to
        # "will x happen after ?" (the year strips), while the dated by-
        # titles above normalize to "will x happen by ?" — so it lands in its
        # OWN normalized group instead and is never compared against
        # anything. The refused-candidate role the plan wanted is played by
        # T5DUP above (a genuine duplicate-deadline refusal) instead. This
        # leg is kept anyway — never compared, but still profiled once, since
        # the live finder classifies every actively priced market up front,
        # before grouping — so `calls["n"] == len(active)` still requires it
        # to be seen exactly once.
        markets.append(_mock_market(
            ticker="EXTRA", event_ticker="EVTX",
            title="Will X happen after March 5, 2026?",
            yes_ask=0.50, no_ask=0.50, close_time=base + timedelta(days=5),
        ))

        calls = {"n": 0}
        original = scanner.deadline_profile

        def counting(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(scanner, "deadline_profile", counting)
        active = _filter_active_markets(markets, None)
        pairs = find_time_series_pairs(MagicMock(), held_tickers=set(), markets=markets)
        assert len(pairs) >= 1
        assert calls["n"] == len(active)


class TestPhrasingCensusLine:
    """The scanner half of the per-run phrasing census (the backtester
    census, the dashboard rendering, and the skip-count summaries are pinned
    separately). It is the ONE signal that separates "the exchange lists no
    cumulative families right now" from "the phrasing tables are broken" (the
    DR-66 lesson, applied to this rule). This is a CONTROL — it pins that the
    census reports the true bucket counts, not merely that some line fires."""

    def _markets(self, n_cumulative, n_snapshot, n_unknown):
        markets = []
        for i in range(n_cumulative):
            markets.append(_mock_market(
                ticker=f"C{i}", event_ticker=f"EVC{i}",
                title=f"Will X happen by June {i + 1}, 2026?",
            ))
        for i in range(n_snapshot):
            markets.append(_mock_market(
                ticker=f"S{i}", event_ticker=f"EVS{i}",
                title=f"Bitcoin price on Sep {15 + i}, 2026?",
            ))
        for i in range(n_unknown):
            markets.append(_mock_market(
                ticker=f"U{i}", event_ticker=f"EVU{i}", title=f"Fed cuts rates {i}?",
            ))
        return markets

    @staticmethod
    def _census_lines(caplog):
        return [
            r.getMessage() for r in caplog.records
            if r.getMessage().startswith("Deadline phrasing of actively priced markets")
        ]

    def test_census_line_reports_each_bucket(self, caplog):
        # control — kills the census line being deleted, demoted to
        # logging.debug, or its snapshot/unknown bucket args swapped.
        with caplog.at_level(logging.INFO):
            find_time_series_pairs(
                MagicMock(), held_tickers=set(), markets=self._markets(3, 2, 5),
            )
        assert self._census_lines(caplog) == [
            "Deadline phrasing of actively priced markets: 3 cumulative, 2 snapshot, 5 unknown",
        ]

    def test_all_unknown_reads_zero_cumulative(self, caplog):
        # control — kills the census line being deleted (an all-zero-
        # cumulative run must still say so explicitly, per the DR-66 lesson).
        with caplog.at_level(logging.INFO):
            find_time_series_pairs(
                MagicMock(), held_tickers=set(), markets=self._markets(0, 0, 5),
            )
        assert self._census_lines(caplog) == [
            "Deadline phrasing of actively priced markets: 0 cumulative, 0 snapshot, 5 unknown",
        ]


class TestPhrasingSkipCounts:
    """DR-72: the single "not a cumulative-deadline pair" skip counter is
    split into three honest, separately-reported reasons. Each is exercised
    with its own two-member group (so that group's one candidate pair has an
    unambiguous, predictable refusal reason) on a distinct series pair
    (EVA-x / EVB-x) so the DR-02 one-series conjunct — which runs BEFORE the
    deadline check — never fires ahead of the check under test.
    """

    @staticmethod
    def _stub_profiles(overrides: dict):
        def fake(market):
            return overrides[market.ticker]
        return fake

    @staticmethod
    def _refusal_lines(caplog):
        return [
            r.getMessage() for r in caplog.records
            if r.getMessage().startswith("Time-series candidates refused because")
        ]

    def test_each_reason_is_reported_once(self, monkeypatch, caplog):
        # regression — fails on revert to the single folded counter, which
        # reported one line ("not a cumulative-deadline pair...") instead of
        # three, so none of the three exact-prefix assertions below would
        # ever have matched.
        #
        # Each reason gets a DIFFERENT candidate count (1, 2, 3) — not just a
        # different fixture — so a mutant that swaps which counter a reason
        # increments (e.g. REFUSED_SNAPSHOT bumping same_deadline_skips)
        # cannot pass by coincidence: with every reason at count 1, such a
        # swap still emits three "...: 1" lines and this test could not tell.
        snap_a = _mock_market(ticker="SNAP-A", event_ticker="EVA-1", title="Will Group Snap happen?")
        snap_b = _mock_market(ticker="SNAP-B", event_ticker="EVB-1", title="Will Group Snap happen?")
        nod_a = _mock_market(ticker="NOD-A", event_ticker="EVA-2", title="Will Group Nodate happen?")
        nod_b = _mock_market(ticker="NOD-B", event_ticker="EVB-2", title="Will Group Nodate happen?")
        nod2_a = _mock_market(ticker="NOD2-A", event_ticker="EVA-4", title="Will Group Nodate2 happen?")
        nod2_b = _mock_market(ticker="NOD2-B", event_ticker="EVB-4", title="Will Group Nodate2 happen?")
        same_a = _mock_market(ticker="SAME-A", event_ticker="EVA-3", title="Will Group Same happen?")
        same_b = _mock_market(ticker="SAME-B", event_ticker="EVB-3", title="Will Group Same happen?")
        same2_a = _mock_market(ticker="SAME2-A", event_ticker="EVA-5", title="Will Group Same2 happen?")
        same2_b = _mock_market(ticker="SAME2-B", event_ticker="EVB-5", title="Will Group Same2 happen?")
        same3_a = _mock_market(ticker="SAME3-A", event_ticker="EVA-6", title="Will Group Same3 happen?")
        same3_b = _mock_market(ticker="SAME3-B", event_ticker="EVB-6", title="Will Group Same3 happen?")
        markets = [
            snap_a, snap_b,
            nod_a, nod_b, nod2_a, nod2_b,
            same_a, same_b, same2_a, same2_b, same3_a, same3_b,
        ]

        overrides = {
            "SNAP-A": (scanner.DEADLINE_SNAPSHOT, ("on june 1, 2026",)),
            "SNAP-B": (scanner.DEADLINE_CUMULATIVE, ("by june 10, 2026",)),
            "NOD-A": (scanner.DEADLINE_CUMULATIVE, ("by june 1, 2026",)),
            "NOD-B": (scanner.DEADLINE_UNKNOWN, ()),
            "NOD2-A": (scanner.DEADLINE_CUMULATIVE, ("by june 1, 2026",)),
            "NOD2-B": (scanner.DEADLINE_UNKNOWN, ()),
            "SAME-A": (scanner.DEADLINE_CUMULATIVE, ("by june 1, 2026",)),
            "SAME-B": (scanner.DEADLINE_CUMULATIVE, ("by june 1, 2026",)),
            "SAME2-A": (scanner.DEADLINE_CUMULATIVE, ("by june 1, 2026",)),
            "SAME2-B": (scanner.DEADLINE_CUMULATIVE, ("by june 1, 2026",)),
            "SAME3-A": (scanner.DEADLINE_CUMULATIVE, ("by june 1, 2026",)),
            "SAME3-B": (scanner.DEADLINE_CUMULATIVE, ("by june 1, 2026",)),
        }
        monkeypatch.setattr(
            scanner, "_market_deadline_profile", self._stub_profiles(overrides)
        )

        with caplog.at_level(logging.INFO):
            pairs = find_time_series_pairs(
                MagicMock(), held_tickers=set(), markets=markets,
            )
        assert pairs == []

        lines = self._refusal_lines(caplog)
        assert len(lines) == 3  # exactly one per reason, none folded together
        assert any(
            line.startswith(
                "Time-series candidates refused because a leg's deciding "
                "field is snapshot wording"
            ) and line.endswith(": 1")
            for line in lines
        )
        assert any(
            line.startswith(
                "Time-series candidates refused because a leg's deciding "
                "field carries no recognised deadline wording or no "
                "comparable date"
            ) and line.endswith(": 2")
            for line in lines
        )
        assert any(
            line.startswith(
                "Time-series candidates refused because the two deciding "
                "fields state the same deadline, or truncate to one"
            ) and line.endswith(": 3")
            for line in lines
        )
        # None of the six refused pairs ever reached the gap check.
        assert not any("gap cap" in m for m in (r.getMessage() for r in caplog.records))

    def test_silent_at_zero(self, caplog):
        # control — kills a mutant that logs a refusal line unconditionally
        # (dropping the `if snapshot_skips:` / etc. guards). A fixture where
        # the one candidate pair in the group is genuinely eligible must
        # produce none of the three lines.
        a = _mock_market(
            ticker="OK-A", event_ticker="EVA-1",
            title="Will Group OK happen by June 1, 2026?",
            yes_ask=0.10, no_ask=0.90, close_time=datetime(2026, 6, 1, tzinfo=UTC),
        )
        b = _mock_market(
            ticker="OK-B", event_ticker="EVB-1",
            title="Will Group OK happen by June 10, 2026?",
            yes_ask=0.30, no_ask=0.70, close_time=datetime(2026, 6, 10, tzinfo=UTC),
        )
        assert normalize_title(pair_key(a)) == normalize_title(pair_key(b))
        with caplog.at_level(logging.INFO):
            pairs = find_time_series_pairs(
                MagicMock(), held_tickers=set(), markets=[a, b],
            )
        assert len(pairs) == 1
        assert self._refusal_lines(caplog) == []
        # ... nor the spread rule's four count lines, which a mutant dropping
        # their `if ...:` guards logs at 0
        spread_lines = [
            r.getMessage() for r in caplog.records
            if "refused below the entry floor" in r.getMessage()
            or "refused above the spread band's" in r.getMessage()
        ]
        assert spread_lines == []


class TestGapCapSkipLine:
    """DR-72: a candidate WORDED as two different cumulative deadlines, but
    refused only for sitting more than MAX_DEADLINE_GAP_DAYS apart, gets its
    own line — distinct from the three wording-refusal reasons above, which
    all mean the wording itself never established a genuine two-deadline
    pair in the first place.
    """

    def test_two_cumulative_legs_45_days_apart(self, caplog):
        # regression — fails on revert: before DR-72 this candidate was
        # dropped by the bare `if gap_days > MAX_DEADLINE_GAP_DAYS: continue`
        # with no counter or log line at all, so no "gap cap" line could ever
        # appear.
        mA = _mock_market(
            ticker="GAP-A", event_ticker="EVA-1",
            title="Will X happen by March 1, 2026?",
            yes_ask=0.20, no_ask=0.80, close_time=datetime(2026, 3, 1, tzinfo=UTC),
        )
        mB = _mock_market(
            ticker="GAP-B", event_ticker="EVB-1",
            title="Will X happen by April 15, 2026?",
            yes_ask=0.60, no_ask=0.40, close_time=datetime(2026, 4, 15, tzinfo=UTC),
        )
        assert normalize_title(pair_key(mA)) == normalize_title(pair_key(mB))
        assert scanner.deadline_gap_days(mA, mB) == 45
        assert scanner.cumulative_deadline_pair(
            scanner._market_deadline_profile(mA),
            scanner._market_deadline_profile(mB),
        ) is True

        with caplog.at_level(logging.INFO):
            pairs = find_time_series_pairs(
                MagicMock(), held_tickers=set(), markets=[mA, mB],
            )
        assert pairs == []

        msgs = [r.getMessage() for r in caplog.records]
        gap_lines = [m for m in msgs if "gap cap" in m]
        assert gap_lines == [
            "Time-series candidates worded as two different cumulative "
            f"deadlines, refused at the {MAX_DEADLINE_GAP_DAYS}-day gap cap "
            "(tier and price not evaluated): 1",
        ]
        # The wording was fine — none of the three refusal-reason lines fire.
        assert not any(
            m.startswith("Time-series candidates refused because") for m in msgs
        )


# ── DR-70: the phrasing classifier is cheap, and its output does not move ─────


def _parametrize_values(test_func) -> list:
    """The argvalues of test_func's single @pytest.mark.parametrize.

    Read off the function object so the DR-70 equivalence test reuses the
    DR-67/DR-68 tables' EXACT strings as its reference set instead of a second
    copy that could drift from them (the reason _CUMULATIVE_ENTRY_CASES is
    module-level). Raises ValueError if the test stops being parametrized, so
    a refactor cannot silently empty the reference set.
    """
    [mark] = [m for m in getattr(test_func, "pytestmark", []) if m.name == "parametrize"]
    return list(mark.args[1])


def _phrasing_reference_strings() -> list:
    """Every wording string the classifier tests above already pin, plus the
    capitalisation cases, as one hermetic reference set for DR-70."""
    strings = []
    for phrase, _spans in _CUMULATIVE_ENTRY_CASES:
        strings += [phrase, f"Will X happen {phrase}?"]
    strings += _parametrize_values(TestCumulativeDeadlineRule.test_each_snapshot_entry_is_live)
    strings += [t for t, _ in _parametrize_values(TestCumulativeDeadlineRule.test_title_classification)]
    strings += _parametrize_values(TestCumulativeDeadlineRule.test_by_a_quantity_is_not_a_deadline)
    strings += [t for t, _ in _parametrize_values(TestDateTokenBoundaries.test_profile)]
    strings += _parametrize_values(TestDateTokenBoundaries.test_snapshot)
    # Not parametrized upstream, so named here: the capitalised rows of
    # TestCumulativeDeadlineRule.test_capitalised_and_november, and lower-case
    # months. Every table string above spells a month capitalised and a
    # preposition lower-case, exactly as the patterns do, so without these an
    # alternation compiled WITHOUT re.IGNORECASE would agree with the
    # reference on every row.
    strings += [
        "Before Oct 1, 2026", "On Nov 16, 2026", "BY DEC 31, 2026",
        "Will X happen by november 30, 2026?", "bitcoin price on sep 15, 2026?",
        "AT THE CLOSE", "Will X happen AFTER March 1, 2026?", "WITHIN 30 DAYS",
    ]
    # De-duplicated in order: several tables pin the same string, and one row
    # per distinct string keeps the parametrized ids readable.
    return list(dict.fromkeys(strings))


_PHRASING_REFERENCE = _phrasing_reference_strings()


def _reference_field_phrasing(text: str):
    """_field_phrasing as it was before DR-70: every entry of each table
    searched on its own, snapshot table first."""
    if any(pat.search(text) for pat in scanner._COMPILED_SNAPSHOT):
        return scanner.DEADLINE_SNAPSHOT
    if any(pat.search(text) for pat in scanner._COMPILED_CUMULATIVE):
        return scanner.DEADLINE_CUMULATIVE
    return None


class TestPhrasingTableInvariants:
    """DR-70 tests each phrasing table with ONE precompiled alternation.
    "Some entry matches somewhere" equals "the alternation matches somewhere"
    only while every entry is alternation-safe. These are CONTROLS: they pin
    the invariant the equivalence rests on, so a future table entry that
    breaks it fails here rather than silently changing verdicts."""

    @pytest.mark.parametrize(
        "pattern", [*scanner._CUMULATIVE_DEADLINE_PATTERNS, *scanner._SNAPSHOT_PATTERNS],
    )
    def test_entry_has_no_group_and_no_inline_flag(self, pattern):
        # control — kills a table entry that adds a capturing group (it would
        # renumber every backreference after it in the alternation; zero
        # groups also rules out backreferences, named groups and
        # conditionals by construction) or an inline flag. A global one is
        # legal only at position 0 of a pattern and every entry is wrapped in
        # (?:...), so inside the alternation it fails to compile (Python
        # 3.11+, the project floor) and the module would not import; scoped
        # ones — positive "(?x:" or negative "(?-i:" — are forbidden too, to
        # keep the rule one line.
        assert re.compile(pattern).groups == 0
        assert not re.search(r"\(\?[-aiLmsux]", pattern)

    def test_alternations_are_group_free_and_case_insensitive(self):
        # control — kills an alternation compiled without re.IGNORECASE (the
        # per-pattern lists carry it, so the two would disagree on "Before
        # Oct 1, 2026").
        for alternation in (scanner._ANY_SNAPSHOT, scanner._ANY_CUMULATIVE):
            assert alternation.groups == 0
            assert alternation.flags & re.IGNORECASE


class TestAlternationIsUsed:
    """DR-70: _field_phrasing must read the precompiled alternations, not
    loop over the per-pattern lists."""

    def test_field_phrasing_reads_the_alternations(self, monkeypatch):
        # regression — with both per-pattern lists emptied, the pre-DR-70
        # loops (and a skip-only implementation) find nothing and return None.
        monkeypatch.setattr(scanner, "_COMPILED_SNAPSHOT", [])
        monkeypatch.setattr(scanner, "_COMPILED_CUMULATIVE", [])
        assert scanner._field_phrasing("on June 30") == scanner.DEADLINE_SNAPSHOT
        assert scanner._field_phrasing("Before Oct 1, 2026") == scanner.DEADLINE_CUMULATIVE


class TestAlternationEquivalence:
    """DR-70 must change no verdict. Hermetic: the reference set is the
    strings the DR-67/DR-68 classifier tests above already pin."""

    def test_reference_set_is_populated(self):
        # control — keeps the parametrized test below from passing vacuously
        # if a refactor ever empties or narrows the reference set.
        assert len(_PHRASING_REFERENCE) >= 100
        assert {_reference_field_phrasing(t) for t in _PHRASING_REFERENCE} == {
            scanner.DEADLINE_SNAPSHOT, scanner.DEADLINE_CUMULATIVE, None,
        }

    @pytest.mark.parametrize("text", _PHRASING_REFERENCE)
    def test_alternation_agrees_with_every_entry(self, text):
        # control — kills an alternation compiled without re.IGNORECASE or
        # joined from the wrong table. Each alternation is checked on its own
        # as well as through _field_phrasing, so a cumulative-side divergence
        # cannot hide behind a snapshot verdict.
        assert bool(scanner._ANY_SNAPSHOT.search(text)) == any(
            pat.search(text) for pat in scanner._COMPILED_SNAPSHOT
        )
        assert bool(scanner._ANY_CUMULATIVE.search(text)) == any(
            pat.search(text) for pat in scanner._COMPILED_CUMULATIVE
        )
        assert scanner._field_phrasing(text) == _reference_field_phrasing(text)


class _RaisingEq:
    """A wording field whose == raises: proves _deciding_field's
    duplicate-field skip never calls a non-str field's __eq__, directly or
    reflected from a str comparison."""

    __hash__ = None

    def __eq__(self, other):
        raise TypeError("a non-str field's __eq__ must never be called")


class TestDuplicateFieldSkip:
    """DR-70: _deciding_field skips a field identical to the one scanned just
    before it. The skipped field could only have returned the same None."""

    def test_identical_fields_are_scanned_once(self, monkeypatch):
        # regression — the pre-DR-70 walk (and an alternation-only
        # implementation) scans "x" three times.
        calls = []
        original = scanner._field_phrasing

        def counting(text):
            calls.append(text)
            return original(text)

        monkeypatch.setattr(scanner, "_field_phrasing", counting)
        assert scanner.deadline_phrasing("x", "x", "x") == scanner.DEADLINE_UNKNOWN
        assert calls == ["x"]
        # In-test positive control: a DIFFERENT event title after two
        # identical fields is still scanned, and still decides.
        calls.clear()
        assert scanner.deadline_phrasing(
            "Will X happen by June 1, 2026?", "x", "x",
        ) == scanner.DEADLINE_CUMULATIVE
        assert calls == ["x", "Will X happen by June 1, 2026?"]

    @pytest.mark.parametrize("fields", [
        # control — kills dropping the isinstance guard from the skip test
        # (`if text == previous`): the raising title is compared with the str
        # subtitle scanned before it.
        pytest.param(lambda: (_RaisingEq(), _RaisingEq(), "Trump"), id="raising-title"),
        # control — kills keeping a non-str field as `previous`: the str
        # title's comparison with it falls back to the reflected
        # _RaisingEq.__eq__.
        pytest.param(lambda: ("Trump", "Trump", _RaisingEq()), id="raising-subtitle"),
        pytest.param(lambda: (float("nan"),) * 3, id="nan"),
    ])
    def test_non_str_fields_are_never_compared(self, fields):
        assert scanner.deadline_phrasing(*fields()) == scanner.DEADLINE_UNKNOWN


def _ts_pair_markets(*, gap_days: int, pA: float, pB: float, nB: float | None = None):
    """Build an earlier/later mock market pair gap_days apart, one question at
    two deadlines.

    The earlier market carries YES ask pA (NO ask 1-pA) and the later one YES
    ask pB with NO ask nB (default 1-pB, a tight book), so pB - pA is the
    directional price gap seen by find_time_series_pairs and (pA, nB) are
    the two LEG prices (YES on EARLY, NO on LATE).

    Each title names its own market's close date, so the two titles DIFFER by
    the deadline and normalize_title collapses both to "will btc exceed $80k
    by" — the shape a real cumulative-deadline family has, and the shape the
    one-series rule (DR-02, DR-54) is built to leave alone. The two event
    tickers deliberately stay in ONE series (EVT), because a daily family's
    two deadline events are two events of one series: this fixture is
    therefore also the positive control that scanner._identical_wording, not
    the series, is what the time-series conjunct turns on.
    """
    from datetime import UTC, datetime, timedelta
    early_close = datetime(2026, 3, 1, tzinfo=UTC)
    late_close = early_close + timedelta(days=gap_days)
    mA = _mock_market(
        ticker="EARLY", event_ticker="EVT-A",
        title=f"Will BTC exceed $80k by {early_close:%B %d, %Y}",
        yes_ask=pA, no_ask=round(1.0 - pA, 4),
        close_time=early_close,
    )
    mB = _mock_market(
        ticker="LATE", event_ticker="EVT-B",
        title=f"Will BTC exceed $80k by {late_close:%B %d, %Y}",
        yes_ask=pB, no_ask=round(1.0 - pB, 4) if nB is None else nB,
        close_time=late_close,
    )
    return mA, mB


@pytest.mark.usefixtures("pre_toggle_defaults")
class TestTimeSeriesTieredThreshold:
    """The minimum price gap (later YES ask minus earlier YES ask) is tiered
    by deadline gap: 15% for gaps <= 15 days, 30% for 16-30 days, and gaps
    > 30 days are never candidates. Every fixture has the LATER contract
    pricier (pB > pA) except the direction test. The finder reads config.py's
    toggles, pinned by pre_toggle_defaults to tier floors on, no band."""

    def _scan(self, gap_days, pA, pB):
        mA, mB = _ts_pair_markets(gap_days=gap_days, pA=pA, pB=pB)
        return find_time_series_pairs(MagicMock(), held_tickers=set(), markets=[mA, mB])

    def test_short_gap_20pct_price_gap_accepted(self):
        # 10-day deadline gap → 15% tier; a 20% price gap qualifies
        pairs = self._scan(10, pA=0.30, pB=0.50)
        assert len(pairs) == 1
        assert pairs[0].market_a.ticker == "EARLY"
        assert pairs[0].market_b.ticker == "LATE"

    def test_candidate_carries_leg_price_nB_and_reporting_nA(self):
        # nB (LATE's NO ask) is the NO leg's price; nA (EARLY's NO ask) is still
        # read so the prod log's "nA (NO ask)" column stays meaningful.
        [pair] = self._scan(10, pA=0.30, pB=0.50)
        assert pair.pA == pytest.approx(0.30)
        assert pair.pB == pytest.approx(0.50)
        assert pair.nB == pytest.approx(0.50)
        assert pair.nA == pytest.approx(0.70)
        # Tight book: pA + nB = 0.80 leaves 0.20 above fees → tradeable
        assert pair.tradeable is True
        assert leg_prices(pair) == (pytest.approx(0.30), pytest.approx(0.50))

    def test_short_gap_10pct_price_gap_rejected(self):
        # 10-day deadline gap → 15% tier; a 10% price gap is below it
        assert self._scan(10, pA=0.30, pB=0.40) == []

    def test_long_gap_18pct_price_gap_rejected(self):
        # 20-day deadline gap → 30% tier; 18% would have passed the old flat
        # 15% threshold but must now be rejected
        assert self._scan(20, pA=0.30, pB=0.48) == []

    def test_long_gap_35pct_price_gap_accepted(self):
        # 20-day deadline gap → 30% tier; a 35% price gap clears it
        pairs = self._scan(20, pA=0.30, pB=0.65)
        assert len(pairs) == 1

    def test_boundary_15_day_gap_uses_short_tier(self):
        # Exactly 15 days is inclusive in the 15% tier — 20% qualifies
        pairs = self._scan(15, pA=0.30, pB=0.50)
        assert len(pairs) == 1

    def test_boundary_16_day_gap_uses_long_tier(self):
        # 16 days falls into the 30% tier — the same 20% gap now fails
        assert self._scan(16, pA=0.30, pB=0.50) == []

    def test_boundary_30_day_gap_still_allowed(self):
        # 30 days is the maximum allowed deadline gap; 35% clears the 30% tier
        pairs = self._scan(30, pA=0.30, pB=0.65)
        assert len(pairs) == 1

    def test_over_max_gap_rejected_regardless_of_price(self):
        # 35 days exceeds MAX_DEADLINE_GAP_DAYS — even a 40% price gap is out
        assert self._scan(35, pA=0.30, pB=0.70) == []

    def test_direction_still_rules_out_pricier_earlier_contract(self):
        # 10-day gap, EARLIER contract pricier by 35%: magnitude alone never
        # qualifies — the filter is directional (the later leg must be pricier)
        assert self._scan(10, pA=0.65, pB=0.30) == []

    def test_wide_later_book_is_not_a_candidate_at_all(self):
        # A 30% YES-ask gap clears the tier, but with LATE's NO ask at 0.75 the
        # legs cost pA + nB = 1.05. A win pays only $1, so a pair whose leg ASK
        # prices already sum to $1 or more cannot profit in any cell — nesting
        # only makes the FAIR VALUE of YES-A + NO-B at most $1, not the quoted
        # asks — so this is not a candidate at all.
        #
        # RE-PINNED: it used to be carried as an untradeable row. That was not
        # free — both finders keep ONE pair per group, so a pair no trade can
        # ever come from could win the slot and block a sound runner-up. The
        # explicit skip in find_time_series_pairs is what removes it; the
        # tradeable flag alone never did.
        mA, mB = _ts_pair_markets(gap_days=10, pA=0.30, pB=0.60, nB=0.75)
        assert find_time_series_pairs(
            MagicMock(), held_tickers=set(), markets=[mA, mB],
        ) == []

    def _same_event_markets(self):
        """Two DATED cumulative rungs sharing one event ticker.

        RE-PINNED (DR-73): this fixture used to carry the undated wording
        "Will BTC exceed $80k" on both markets, which DR-67 refuses for naming
        no deadline and DR-02 refuses for being identical — so the test passed
        with the same-event guard DELETED and pinned nothing. Dated, differing
        titles clear both of those rules, leaving the same-event guard as the
        only thing that can refuse the pair.
        """
        from datetime import UTC, datetime
        mA = _mock_market(
            ticker="OPT-A", event_ticker="MVE-1",
            title="Will BTC exceed $80k by March 1, 2026",
            yes_ask=0.30, no_ask=0.70,
            close_time=datetime(2026, 3, 1, tzinfo=UTC),
        )
        mB = _mock_market(
            ticker="OPT-B", event_ticker="MVE-1",
            title="Will BTC exceed $80k by March 11, 2026",
            yes_ask=0.60, no_ask=0.40,
            close_time=datetime(2026, 3, 11, tzinfo=UTC),
        )
        return mA, mB

    def test_same_event_ticker_never_pairs(self, monkeypatch):
        # Two markets inside one event must not form a time-series pair with
        # the ladder switch off — scanner's own by-value binding, set here, so
        # this row holds whatever value the switch ships with.
        monkeypatch.setattr(scanner, "TIME_SERIES_SAME_EVENT_LADDERS", False)
        mA, mB = self._same_event_markets()
        assert find_time_series_pairs(MagicMock(), held_tickers=set(), markets=[mA, mB]) == []

    def test_the_same_event_fixture_is_not_vacuous(self, monkeypatch):
        # control: every OTHER rule admits this fixture, so the row above is
        # pinning the same-event guard and nothing else.
        monkeypatch.setattr(scanner, "TIME_SERIES_SAME_EVENT_LADDERS", True)
        pairs = find_time_series_pairs(MagicMock(), held_tickers=set(), markets=list(
            self._same_event_markets()))
        assert len(pairs) == 1 and pairs[0].stated_gap_days == 10

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

    def test_same_title_pairs_ignore_the_deadline_tier_but_not_the_close_gap(self):
        # Same-title pairs keep the flat 5% threshold — a 6% divergence, far
        # under either time-series tier, is still a candidate when the two
        # markets close at one instant (the deadline-gap tiers apply only to
        # time-series pairs).
        #
        # INVERTED (DR-74): this test used to pin that a pair closing 20 DAYS
        # apart "is still a candidate" — exactly the shape DR-74 refuses.
        # Identical wording on two series closing 20 days apart is two
        # fixtures (two games, two instants), not one question, so the
        # co-resolution prior does not apply; the same-title close gate now
        # refuses it. RE-PINNED earlier (DR-02/DR-54): the two event tickers
        # were EVT-A and EVT-B, one series; they name two DIFFERENT series, so
        # only the close gate decides the first assertion below.
        from datetime import UTC, datetime
        mA = _mock_market(
            ticker="A1", event_ticker="EVA-1",
            title="Republicans control Senate after 2026", event_title="2026 Senate Control",
            yes_ask=0.36, no_ask=0.64,
            close_time=datetime(2026, 3, 1, tzinfo=UTC),
        )
        mB = _mock_market(
            ticker="B1", event_ticker="EVB-1",
            title="Republicans control Senate after 2026", event_title="2026 Senate Control",
            yes_ask=0.30, no_ask=0.70,
            close_time=datetime(2026, 3, 21, tzinfo=UTC),
        )
        assert find_same_title_pairs([mA, mB]) == []

        # Within the gap: B at A's close instant — the 6% divergence pairs.
        mB_aligned = _mock_market(
            ticker="B1", event_ticker="EVB-1",
            title="Republicans control Senate after 2026", event_title="2026 Senate Control",
            yes_ask=0.30, no_ask=0.70,
            close_time=datetime(2026, 3, 1, tzinfo=UTC),
        )
        pairs = find_same_title_pairs([mA, mB_aligned])
        assert len(pairs) == 1
        assert pairs[0].pA - pairs[0].pB == pytest.approx(0.06)


def _raw_book_response(ob: dict) -> SimpleNamespace:
    """Wrap one orderbook_fp side dict as a raw *_without_preload_content response.

    Responses use the raw orderbook_fp JSON wire format (the SDK's modeled
    orderbook response can't deserialize live payloads anymore).
    """
    payload = {"orderbook_fp": ob}
    return SimpleNamespace(status=200, data=json.dumps(payload).encode("utf-8"))


# The fake time-series books' default YES bid on EARLY: at EARLY's best YES
# ask, so its book has no width and its midpoint is that ask
_AT_THE_ASK = object()


def _ts_orderbook_client(
    *, pA_fill: float, nB_fill: float, qty: int = 100, pB_ref: float | None = None,
    a_yes_bid: object = _AT_THE_ASK,
):
    """Mock KalshiClient serving TIME-SERIES-shaped depth at exactly one level.

    A time-series pair buys YES on EARLY and NO on LATE. Buying YES on EARLY
    consumes EARLY's NO bids (YES ask = 1 - NO bid) and buying NO on LATE
    consumes LATE's YES bids (NO ask = 1 - YES bid) — so a NO bid of
    (1 - pA_fill) on EARLY and a YES bid of (1 - nB_fill) on LATE yield
    qualifying depth priced at exactly pA_fill + nB_fill. LATE's NO side is
    left empty unless pB_ref is given.

    EARLY also rests a YES bid at a_yes_bid, its own YES ask pA_fill by
    default, so its book has no width and its midpoint, which the mid spread
    reads, is pA_fill; None leaves EARLY's YES side empty (a bid of 0).

    pB_ref, when given, additionally rests a NO bid of (1 - pB_ref) on LATE so
    LATE's best YES ask is exactly pB_ref — the reference quote
    _reference_yes_ask reads. It is NOT a leg side for this pair type, so it
    changes no fill price. Below LATE's YES bid (1 - nB_fill) the book is
    CROSSED; left None, a time-series pair has no fresh reference and fails
    closed, so a test that keeps its pair tradeable passes an uncrossed pB_ref.
    """
    a_bid = pA_fill if a_yes_bid is _AT_THE_ASK else a_yes_bid

    def fake_orderbook(ticker):
        if ticker == "EARLY":  # market A — NO bids become YES ask levels
            ob = {"yes_dollars": [] if a_bid is None else [[str(round(a_bid, 4)), str(qty)]],
                  "no_dollars": [[str(round(1.0 - pA_fill, 4)), str(qty)]]}
        else:                  # market B — YES bids become NO ask levels
            ob = {"yes_dollars": [[str(round(1.0 - nB_fill, 4)), str(qty)]],
                  "no_dollars": (
                      [] if pB_ref is None
                      else [[str(round(1.0 - pB_ref, 4)), str(qty)]]
                  )}
        return _raw_book_response(ob)

    client = MagicMock()
    client.get_market_orderbook_without_preload_content = MagicMock(side_effect=fake_orderbook)
    return client


def _st_orderbook_client(
    *, nA_fill: float, pB_fill: float, qty: int = 100, ticker_a: str = "A1",
    pA_ref: float | None = None,
):
    """Mock KalshiClient serving SAME-TITLE-shaped depth at exactly one level.

    A same-title pair buys NO on A and YES on B. Buying NO on A consumes A's
    YES bids (NO ask = 1 - YES bid) and buying YES on B consumes B's NO bids
    (YES ask = 1 - NO bid) — so a YES bid of (1 - nA_fill) on ticker_a and a
    NO bid of (1 - pB_fill) on every other ticker yield qualifying depth
    priced at exactly nA_fill + pB_fill. This is the book shape the scanner
    read for EVERY pair before the 2026-09 time-series inversion.

    pA_ref mirrors _ts_orderbook_client's pB_ref: it rests a NO bid of
    (1 - pA_ref) on ticker_a so A's best YES ask is exactly pA_ref — the
    reference quote for THIS pair type. Left None, A's NO side stays empty and
    the pair keeps its scan-time pA.
    """
    def fake_orderbook(ticker):
        if ticker == ticker_a:  # market A — YES bids become NO ask levels
            ob = {"yes_dollars": [[str(round(1.0 - nA_fill, 4)), str(qty)]],
                  "no_dollars": (
                      [] if pA_ref is None
                      else [[str(round(1.0 - pA_ref, 4)), str(qty)]]
                  )}
        else:                   # market B — NO bids become YES ask levels
            ob = {"yes_dollars": [],
                  "no_dollars": [[str(round(1.0 - pB_fill, 4)), str(qty)]]}
        return _raw_book_response(ob)

    client = MagicMock()
    client.get_market_orderbook_without_preload_content = MagicMock(side_effect=fake_orderbook)
    return client


def _ts_multilevel_client(levels: list[tuple[float, float, int]], *, pB_ref: float = 0.62,
                          a_yes_bid: object = _AT_THE_ASK):
    """Mock KalshiClient serving TIME-SERIES depth at SEVERAL price levels.

    levels is [(pA_fill, nB_fill, qty), ...]. Same side mapping as
    _ts_orderbook_client (EARLY's NO bids -> YES asks, LATE's YES bids -> NO
    asks), just with more than one rung, so the affordability bound has
    somewhere worse to reach when the budget is large.

    The two legs are SEPARATE books that _pair_orderbooks merges with a
    two-pointer sweep, so each column must ascend on its own for the rungs here
    to pair up 1:1 with the slices that sweep emits — a column that dips gets
    re-sorted by _bids_to_ask_levels and the quantities no longer line up.

    LATE's YES ask (the reference quote) is pB_ref; the default 0.62 sits above
    every LATE YES bid of the fixtures relying on it, so their books are uncrossed.
    EARLY rests a YES bid at a_yes_bid, its best YES ask by default (a top
    with no width); None leaves EARLY's YES side empty.
    """
    a_bid = min(pa for pa, _, _ in levels) if a_yes_bid is _AT_THE_ASK else a_yes_bid

    def fake_orderbook(ticker):
        if ticker == "EARLY":
            ob = {"yes_dollars": [] if a_bid is None else [[str(round(a_bid, 4)), "1000"]],
                  "no_dollars": [[str(round(1.0 - pa, 4)), str(q)]
                                 for pa, _, q in levels]}
        else:
            ob = {"yes_dollars": [[str(round(1.0 - nb, 4)), str(q)]
                                  for _, nb, q in levels],
                  "no_dollars": [[str(round(1.0 - pB_ref, 4)), "1000"]]}
        return _raw_book_response(ob)

    client = MagicMock()
    client.get_market_orderbook_without_preload_content = MagicMock(side_effect=fake_orderbook)
    return client


def _ts_candidate(
    *, gap_days: int, pA: float, pB: float, nB: float, nA: float | None = None,
) -> CandidatePair:
    """Build a time_series CandidatePair whose legs close gap_days apart.

    (pA, nB) are the leg prices (YES on EARLY, NO on LATE); nA defaults to the
    tight complement 1 - pA and is reporting-only for this pair type.
    """
    mA, mB = _ts_pair_markets(gap_days=gap_days, pA=pA, pB=pB, nB=nB)
    return CandidatePair(
        market_a=mA, market_b=mB,
        pA=pA, pB=pB, nA=round(1.0 - pA, 4) if nA is None else nA,
        tradeable=True,
        # What time_series_group_key(pair_key(m), "") yields for the titles
        # _ts_pair_markets builds — the trailing "by" survives date stripping.
        canonical_title="will btc exceed $80k by",
        pair_type="time_series",
        nB=nB,
    )


class TestPrefixFillPrices:
    """prefix_fill_prices is the shared definition of "what would n contract
    pairs actually cost". Enrichment and strategy.compute_trade both read the
    book through it, so the price a pair is gated on and the price it is sized
    on can never be computed two different ways."""

    # (price_a, price_b, qty), market order, ascending by combined price —
    # the shape CandidatePair.depth_levels carries.
    BOOK = ((0.40, 0.45, 10.0), (0.42, 0.46, 20.0), (0.50, 0.48, 70.0))

    def test_single_contract_is_the_best_level(self):
        assert prefix_fill_prices(self.BOOK, 1) == (0.40, 0.45)

    def test_prefix_within_one_level_does_not_reach_the_next(self):
        assert prefix_fill_prices(self.BOOK, 10) == (0.40, 0.45)

    def test_partial_level_is_weighted_by_the_quantity_taken(self):
        # 10 @ 0.40 then 5 @ 0.42 -> (10*0.40 + 5*0.42) / 15
        avg_a, avg_b = prefix_fill_prices(self.BOOK, 15)
        assert avg_a == pytest.approx((10 * 0.40 + 5 * 0.42) / 15)
        assert avg_b == pytest.approx((10 * 0.45 + 5 * 0.46) / 15)

    def test_full_depth_equals_the_whole_book_average(self):
        # The pre-change behaviour is the n == total-depth special case, so the
        # old number is still reachable — it is just no longer what we price on.
        total = sum(q for _, _, q in self.BOOK)
        avg_a, avg_b = prefix_fill_prices(self.BOOK, int(total))
        assert avg_a == pytest.approx(
            sum(a * q for a, _, q in self.BOOK) / total
        )
        assert avg_b == pytest.approx(
            sum(b * q for _, b, q in self.BOOK) / total
        )

    def test_price_is_non_decreasing_in_n(self):
        # The property the fixed-point descent in compute_trade relies on:
        # buying more can only reach further down the book into worse levels.
        sums = [sum(prefix_fill_prices(self.BOOK, n)) for n in range(1, 101)]
        # Compared with a tolerance, not exactly: within one level every prefix
        # average is the same price, but sum_a/n reintroduces binary float noise
        # (0.40 * 3.0 / 3 == 0.4000000000000001), which is not a real increase.
        assert all(sums[i + 1] >= sums[i] - 1e-12 for i in range(len(sums) - 1))

    def test_insufficient_depth_returns_none(self):
        assert prefix_fill_prices(self.BOOK, 101) is None

    def test_exact_depth_is_not_insufficient(self):
        # Boundary: the last take is exactly `remaining`, so the float
        # accumulator lands on 0.0 and needs no epsilon.
        assert prefix_fill_prices(self.BOOK, 100) is not None

    def test_zero_or_negative_n_returns_none(self):
        assert prefix_fill_prices(self.BOOK, 0) is None
        assert prefix_fill_prices(self.BOOK, -1) is None

    def test_empty_book_returns_none(self):
        assert prefix_fill_prices((), 1) is None

    def test_fractional_quantities_accumulate_exactly(self):
        # Order-book quantities arrive as floats (_bids_to_ask_levels parses
        # them with float()), so a book of fractional levels must still resolve
        # rather than tripping the insufficient-depth arm.
        book = ((0.30, 0.40, 0.5), (0.31, 0.41, 0.5), (0.32, 0.42, 4.0))
        avg_a, avg_b = prefix_fill_prices(book, 1)
        assert avg_a == pytest.approx((0.5 * 0.30 + 0.5 * 0.31) / 1)
        assert avg_b == pytest.approx((0.5 * 0.40 + 0.5 * 0.41) / 1)


def _st_candidate(*, pA: float, pB: float, nA: float, nB: float = 0.70) -> CandidatePair:
    """Build a same_title CandidatePair on tickers A1/B1.

    (nA, pB) are the leg prices (NO on A, YES on B); pA is the reference quote
    and nB is reporting-only for this pair type.
    """
    mA = _mock_market(ticker="A1", event_ticker="EVT-A", title="Q", yes_ask=pA, no_ask=nA)
    mB = _mock_market(ticker="B1", event_ticker="EVT-B", title="Q", yes_ask=pB, no_ask=nB)
    return CandidatePair(
        market_a=mA, market_b=mB,
        pA=pA, pB=pB, nA=nA,
        tradeable=True,
        canonical_title="Q",
        pair_type="same_title",
        nB=nB,
    )


@pytest.mark.usefixtures("pre_toggle_defaults")
class TestEnrichmentBoundsDepthByAffordability:
    """Enrichment must average only the depth this account could actually buy.

    One pair's budget is its capped Kelly fraction of the portfolio value, never
    more than the cash on hand, so averaging a liquid market's full book priced
    every pair against levels no single trade can reach — inflating the fill
    price and killing pairs at the profitability gate on contracts we would
    never have bought.

    Worked under pre_toggle_defaults (tier floors on, a 20% cap); the shipped
    rule is pinned by test_config.py's TestShippedLiveToggles.
    """

    # Each column ascends on its own (see _ts_multilevel_client), so the sweep
    # emits these rungs 1:1 at combined prices 0.75, 0.85 and 0.97. The 10-day
    # short-tier ceiling is 0.85 and the filter is inclusive, so the first two
    # qualify and the deep 500-lot is trimmed — leaving 10 cheap contracts and
    # 90 dearer ones for the affordability bound to choose between.
    LEVELS = [(0.30, 0.45, 10), (0.36, 0.49, 90), (0.45, 0.52, 500)]

    def _enrich(self, balance_cents):
        pair = _ts_candidate(gap_days=10, pA=0.30, pB=0.62, nB=0.45)
        client = _ts_multilevel_client(self.LEVELS)
        [enriched] = enrich_with_orderbook_prices(client, [pair], balance_cents)
        return enriched

    def test_small_balance_prices_at_the_best_level_only(self):
        # $40 * 20% = $8.00, and the best rung costs 0.75 a pair -> 10 pairs,
        # exactly what rests there, so the fill is the best level outright.
        enriched = self._enrich(4_000)
        assert enriched.tradeable is True
        assert enriched.pA == pytest.approx(0.30)
        assert enriched.nB == pytest.approx(0.45)
        assert enriched.max_contracts == 10

    def test_larger_balance_reaches_further_down_the_book(self):
        # $1,000 * 20% = $200 at 0.75 a pair -> 266 pairs, more than the 100
        # qualifying, so the whole qualifying book is averaged.
        enriched = self._enrich(100_000)
        assert enriched.max_contracts == 100
        assert enriched.pA == pytest.approx((10 * 0.30 + 90 * 0.36) / 100)
        assert enriched.nB == pytest.approx((10 * 0.45 + 90 * 0.49) / 100)

    def test_price_is_never_better_for_a_bigger_balance(self):
        # The direction that matters: a bigger budget can only reach worse
        # levels, so its fill price is never better than a smaller budget's.
        small = self._enrich(4_000)
        large = self._enrich(100_000)
        assert small.pA + small.nB <= large.pA + large.nB

    def test_whole_book_average_would_have_killed_the_pair(self):
        # The regression this change exists for. Averaged over ALL 100
        # qualifying contracts the pair still trades here, but the same book
        # priced at the top rung is strictly cheaper — so the gate sees the
        # price a real trade would pay, not one it could never get.
        small, large = self._enrich(4_000), self._enrich(100_000)
        assert small.pA < large.pA
        assert small.nB < large.nB

    def test_budget_too_small_for_one_contract_is_not_tradeable(self):
        # 1 cent of balance affords nothing. The pair must be DROPPED, never
        # written with max_contracts=0 — compute_trade reads that as UNCAPPED.
        enriched = self._enrich(1)
        assert enriched.tradeable is False
        assert enriched.max_contracts == 0

    def test_unaffordable_pair_logs_its_own_reason(self, caplog):
        with caplog.at_level(logging.INFO, logger=""):
            self._enrich(1)
        [line] = [r.getMessage() for r in caplog.records
                  if "No affordable contract pairs" in r.getMessage()]
        # Depth and budget are named SEPARATELY: they are different faults with
        # different fixes (the book is too thin vs. add funds), and a message
        # that printed only the binding minimum misattributed one as the other.
        assert "100.00 contract(s) rest at the gap" in line, line
        assert "budget affords 0" in line, line

    def test_thin_book_and_poor_budget_are_reported_distinctly(self, caplog):
        # Ample balance, so the BOOK is what binds — the message must say so
        # rather than blaming the budget.
        pair = _ts_candidate(gap_days=10, pA=0.30, pB=0.62, nB=0.45)
        client = _ts_multilevel_client([(0.30, 0.45, 0.4)])
        with caplog.at_level(logging.INFO, logger=""):
            enrich_with_orderbook_prices(client, [pair], _AMPLE_BALANCE_CENTS)
        [line] = [r.getMessage() for r in caplog.records
                  if "No affordable contract pairs" in r.getMessage()]
        assert "0.40 contract(s) rest at the gap" in line, line
        assert "budget affords 0" not in line, line

    def test_the_cash_bounds_the_depth_averaged(self):
        # $1,000 x 20% = $200 would reach all 100 qualifying contracts, but
        # $8.00 of cash buys only the 10 at the best 0.75 level — the same
        # count a $40 portfolio value affords
        pair = _ts_candidate(gap_days=10, pA=0.30, pB=0.62, nB=0.45)
        client = _ts_multilevel_client(self.LEVELS)
        [enriched] = enrich_with_orderbook_prices(client, [pair], 100_000, cash_cents=800)
        assert enriched.tradeable is True
        assert enriched.max_contracts == 10
        assert enriched.pA == pytest.approx(0.30)
        assert enriched.nB == pytest.approx(0.45)
        # Cash above the Kelly share changes nothing
        [ample] = enrich_with_orderbook_prices(
            _ts_multilevel_client(self.LEVELS), [pair], 100_000, cash_cents=1_000_000)
        assert ample.max_contracts == self._enrich(100_000).max_contracts == 100

    def test_the_unaffordable_line_says_when_the_cash_binds(self, caplog):
        pair = _ts_candidate(gap_days=10, pA=0.30, pB=0.62, nB=0.45)
        with caplog.at_level(logging.INFO, logger=""):
            [enriched] = enrich_with_orderbook_prices(
                _ts_multilevel_client(self.LEVELS), [pair], 100_000, cash_cents=50)
        assert enriched.tradeable is False
        [line] = [r.getMessage() for r in caplog.records
                  if "No affordable contract pairs" in r.getMessage()]
        assert "budget affords 0; the $0.50 of cash binds" in line, line
        # A budget the portfolio value's share limits says nothing about cash
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=""):
            enrich_with_orderbook_prices(
                _ts_multilevel_client(self.LEVELS), [pair], 100, cash_cents=1_000_000)
        [line] = [r.getMessage() for r in caplog.records
                  if "No affordable contract pairs" in r.getMessage()]
        assert "cash binds" not in line, line
        # Nor does a thin book, even when the cash is below the share: $100 of
        # cash affords 133 pairs, and only the 0.4 contracts on the book stop it
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=""):
            enrich_with_orderbook_prices(
                _ts_multilevel_client([(0.30, 0.45, 0.4)]), [pair], 100_000_000,
                cash_cents=10_000)
        [line] = [r.getMessage() for r in caplog.records
                  if "No affordable contract pairs" in r.getMessage()]
        assert "budget affords 133" in line, line
        assert "cash binds" not in line, line

    def test_max_contracts_is_what_the_written_price_covers(self):
        # The invariant compute_trade's depth clamp relies on: the price written
        # back is the average over exactly max_contracts contracts.
        for balance in (4_000, 5_000, 20_000, 100_000):
            enriched = self._enrich(balance)
            expected = prefix_fill_prices(enriched.depth_levels, enriched.max_contracts)
            assert leg_prices(enriched) == pytest.approx(expected)


class TestEnrichmentStoresOrientedDepthLevels:
    """depth_levels must be in MARKET order — (market_a's leg price, market_b's
    leg price, qty) — so strategy.compute_trade reads them exactly the way
    leg_prices reads the pair's scalars, with no pair-type logic of its own."""

    def test_time_series_levels_are_pA_then_nB(self):
        pair = _ts_candidate(gap_days=10, pA=0.30, pB=0.62, nB=0.50)
        client = _ts_orderbook_client(pA_fill=0.32, nB_fill=0.42, qty=40, pB_ref=0.62)
        [enriched] = enrich_with_orderbook_prices(client, [pair], _AMPLE_BALANCE_CENTS)
        [level] = enriched.depth_levels
        assert level == pytest.approx((0.32, 0.42, 40.0))
        # market_a's entry is the YES leg (pA), market_b's the NO leg (nB)
        assert leg_prices(enriched) == pytest.approx((0.32, 0.42))

    def test_same_title_levels_are_nA_then_pB(self):
        pair = _st_candidate(pA=0.60, pB=0.31, nA=0.44)
        client = _st_orderbook_client(nA_fill=0.44, pB_fill=0.31, qty=100)
        [enriched] = enrich_with_orderbook_prices(client, [pair], _AMPLE_BALANCE_CENTS)
        [level] = enriched.depth_levels
        assert level == pytest.approx((0.44, 0.31, 100.0))
        # market_a's entry is the NO leg (nA), market_b's the YES leg (pB)
        assert leg_prices(enriched) == pytest.approx((0.44, 0.31))

    def test_levels_match_leg_prices_ordering_for_both_types(self):
        # The orientation contract stated once: depth_levels[i][0] always
        # belongs to market_a and [1] to market_b, whichever side each buys.
        ts = _ts_candidate(gap_days=10, pA=0.30, pB=0.62, nB=0.50)
        [ts_e] = enrich_with_orderbook_prices(
            _ts_orderbook_client(pA_fill=0.32, nB_fill=0.42, pB_ref=0.62), [ts], _AMPLE_BALANCE_CENTS,
        )
        st = _st_candidate(pA=0.60, pB=0.31, nA=0.44)
        [st_e] = enrich_with_orderbook_prices(
            _st_orderbook_client(nA_fill=0.44, pB_fill=0.31), [st], _AMPLE_BALANCE_CENTS,
        )
        for enriched in (ts_e, st_e):
            a, b, _ = enriched.depth_levels[0]
            assert (a, b) == pytest.approx(leg_prices(enriched))

    def test_unenriched_pair_has_empty_levels(self):
        pair = _ts_candidate(gap_days=10, pA=0.30, pB=0.62, nB=0.50)
        assert pair.depth_levels == ()


@pytest.mark.usefixtures("pre_toggle_defaults")
class TestOrderbookCeilingTieredByDeadlineGap:
    """enrich_with_orderbook_prices and validate_pair_price must apply the
    deadline-gap-tiered LEG-price-sum ceiling (0.85 for gaps <= 15 days, 0.70
    for 16-30 days), not the old flat 1 - 15% = 0.85, with the tier floors on
    (pre_toggle_defaults; off, the ceiling is 1 - the band floor alone).

    The fixtures use a deliberately WIDE later book: with a tight nB = 1 - pB
    the leg sum is exactly 1 - (pB - pA), which is always <= the ceiling once
    the pair has passed the gap filter, so the ceiling could never bind."""

    def test_long_gap_depth_at_075_sum_marked_untradeable(self):
        # 20-day gap → ceiling 0.70. Leg depth priced at pA 0.30 + nB 0.45 =
        # 0.75 would have passed the old flat 0.85 ceiling but must disqualify.
        pair = _ts_candidate(gap_days=20, pA=0.30, pB=0.65, nB=0.45)
        client = _ts_orderbook_client(pA_fill=0.30, nB_fill=0.45, pB_ref=0.65)
        [enriched] = enrich_with_orderbook_prices(client, [pair], _AMPLE_BALANCE_CENTS)
        assert enriched.tradeable is False

    def test_short_gap_depth_at_080_sum_qualifies(self):
        # 10-day gap → ceiling 0.85. Leg depth priced at 0.30 + 0.50 = 0.80
        # qualifies and the pair picks up the depth-weighted fill prices in
        # the LEG fields (pA/nB). pB is not a leg price but IS the reference
        # quote the spread rule tests and the mid spread reads, so it is
        # refreshed from LATE's NO bids in the same pass (0.62 here, against a
        # scan-time 0.60); nA is reporting-only for this pair type and stays
        # put.
        pair = _ts_candidate(gap_days=10, pA=0.30, pB=0.60, nB=0.50)
        client = _ts_orderbook_client(pA_fill=0.30, nB_fill=0.50, pB_ref=0.62)
        [enriched] = enrich_with_orderbook_prices(client, [pair], _AMPLE_BALANCE_CENTS)
        assert enriched.tradeable is True
        assert enriched.max_contracts == 100
        assert enriched.pA == pytest.approx(0.30)
        assert enriched.nB == pytest.approx(0.50)
        assert enriched.nA == pair.nA
        assert enriched.pB == pytest.approx(0.62)

    def test_validate_pair_price_rejects_long_gap_at_old_ceiling(self):
        # Pre-execution re-check applies the same tiered ceiling: a 20-day-gap
        # pair whose remaining leg depth sums to 0.80 no longer qualifies.
        pair = _ts_candidate(gap_days=20, pA=0.30, pB=0.65, nB=0.50)
        spec = SimpleNamespace(pair=pair, x=10)
        client = _ts_orderbook_client(pA_fill=0.30, nB_fill=0.50, pB_ref=0.65)
        assert validate_pair_price(client, spec) is False

    def test_validate_pair_price_accepts_short_gap_at_same_depth(self):
        # Identical depth passes for a 10-day-gap pair (ceiling 0.85)
        pair = _ts_candidate(gap_days=10, pA=0.30, pB=0.60, nB=0.50)
        spec = SimpleNamespace(pair=pair, x=10)
        client = _ts_orderbook_client(pA_fill=0.30, nB_fill=0.50, pB_ref=0.60)
        assert validate_pair_price(client, spec) is True

    def test_validate_pair_price_logs_gap_rejection_at_warning(self, caplog):
        # Same rejecting fixture as test_validate_pair_price_rejects_long_gap_at_old_ceiling
        # (0.30 + 0.50 = 0.80 exceeds the 20-day-gap ceiling of 0.70, so this hits
        # the "gap no longer qualifies" branch, not the depth branch). The drop must
        # be logged exactly once, at WARNING, with "; dropping" appended — this is
        # the one log line for the drop; pre_execution_check must not log a second.
        pair = _ts_candidate(gap_days=20, pA=0.30, pB=0.65, nB=0.50)
        spec = SimpleNamespace(pair=pair, x=10)
        client = _ts_orderbook_client(pA_fill=0.30, nB_fill=0.50, pB_ref=0.65)
        with caplog.at_level(logging.INFO):
            assert validate_pair_price(client, spec) is False

        matching = [
            r for r in caplog.records
            if "gap no longer qualifies; dropping" in r.getMessage()
        ]
        assert len(matching) == 1
        assert matching[0].levelno == logging.WARNING

        info_drops = [
            r for r in caplog.records
            if r.levelno == logging.INFO and "gap no longer qualifies" in r.getMessage()
        ]
        assert info_drops == []


class TestTimeSeriesEnrichmentSides:
    """A time-series pair buys YES on EARLY (consuming EARLY's NO bids) and NO
    on LATE (consuming LATE's YES bids); enrichment must read those sides and
    write the fills back to pA/nB — the fields leg_prices() reads."""

    def test_leg_ask_levels_time_series_reads_b_yes_bids_and_a_no_bids(self):
        pair = _ts_candidate(gap_days=10, pA=0.30, pB=0.60, nB=0.40)
        ob_a = {"yes": [["0.29", "7"]], "no": [["0.68", "5"]]}   # NO bid 0.68 → YES ask 0.32
        ob_b = {"yes": [["0.58", "9"]], "no": [["0.39", "3"]]}   # YES bid 0.58 → NO ask 0.42
        no_levels, yes_levels = _leg_ask_levels(pair, ob_a, ob_b)
        assert no_levels == [(pytest.approx(0.42), 9.0)]
        assert yes_levels == [(pytest.approx(0.32), 5.0)]

    def test_leg_ask_levels_same_title_reads_a_yes_bids_and_b_no_bids(self):
        pair = SimpleNamespace(pair_type="same_title")
        ob_a = {"yes": [["0.55", "7"]], "no": [["0.40", "5"]]}   # YES bid 0.55 → NO ask 0.45
        ob_b = {"yes": [["0.20", "9"]], "no": [["0.69", "3"]]}   # NO bid 0.69 → YES ask 0.31
        no_levels, yes_levels = _leg_ask_levels(pair, ob_a, ob_b)
        assert no_levels == [(pytest.approx(0.45), 7.0)]
        assert yes_levels == [(pytest.approx(0.31), 3.0)]

    def test_leg_ask_levels_unknown_pair_type_uses_same_title_sides(self):
        # Fail-safe like leg_sides: None / a bogus type / a bare namespace with
        # no pair_type all read the same-title sides
        ob_a = {"yes": [["0.55", "7"]], "no": []}
        ob_b = {"yes": [], "no": [["0.69", "3"]]}
        for pair in (SimpleNamespace(), SimpleNamespace(pair_type=None), SimpleNamespace(pair_type="bogus")):
            no_levels, yes_levels = _leg_ask_levels(pair, ob_a, ob_b)
            assert no_levels == [(pytest.approx(0.45), 7.0)]
            assert yes_levels == [(pytest.approx(0.31), 3.0)]

    def test_enrichment_writes_fills_to_pA_nB_and_refreshes_pB_leaving_nA(self):
        pair = _ts_candidate(gap_days=10, pA=0.30, pB=0.60, nB=0.40)
        client = _ts_orderbook_client(pA_fill=0.32, nB_fill=0.42, qty=40, pB_ref=0.58)
        [enriched] = enrich_with_orderbook_prices(client, [pair], _AMPLE_BALANCE_CENTS)
        assert enriched.tradeable is True
        assert enriched.max_contracts == 40
        assert enriched.pA == pytest.approx(0.32)
        assert enriched.nB == pytest.approx(0.42)
        # pB is the reference quote the spread rule tests and the mid spread
        # reads, so it comes from this same snapshot (LATE's NO bids) rather
        # than the scan
        assert enriched.pB == pytest.approx(0.58)
        # nA is reporting-only for this pair type — byte-identical to the input
        assert enriched.nA == pair.nA
        assert leg_prices(enriched) == (pytest.approx(0.32), pytest.approx(0.42))

    def test_same_title_shaped_books_yield_no_depth_for_time_series(self):
        # The pre-inversion book shape (EARLY YES bids, LATE NO bids) is the
        # wrong side for both time-series legs — enrichment must find no depth
        # rather than pricing the legs off the wrong side of each book.
        pair = _ts_candidate(gap_days=10, pA=0.30, pB=0.60, nB=0.40)
        client = _st_orderbook_client(nA_fill=0.70, pB_fill=0.60, ticker_a="EARLY")
        [enriched] = enrich_with_orderbook_prices(client, [pair], _AMPLE_BALANCE_CENTS)
        assert enriched.tradeable is False
        assert enriched.max_contracts == 0
        # Same for the pre-execution re-check
        spec = SimpleNamespace(pair=pair, x=1)
        assert validate_pair_price(client, spec) is False

    def test_depth_short_of_spec_count_fails_validate(self):
        pair = _ts_candidate(gap_days=10, pA=0.30, pB=0.60, nB=0.40)
        client = _ts_orderbook_client(pA_fill=0.30, nB_fill=0.40, qty=9, pB_ref=0.60)
        assert validate_pair_price(client, SimpleNamespace(pair=pair, x=10)) is False
        assert validate_pair_price(client, SimpleNamespace(pair=pair, x=9)) is True


class TestSameTitleEnrichmentByteIdentity:
    """Same-title behaviour must not change with the time-series inversion:
    NO on A still consumes A's YES bids, YES on B still consumes B's NO bids,
    and the fills still land in nA/pB with nB untouched. pA is refreshed from
    A's NO bids (TS-34) — the reporting-only reference quote — but same-title
    is deliberately NOT direction-guarded: its model is the fixed co-resolution
    prior, so nothing sizes on pA and there is no clamp to protect."""

    @staticmethod
    def _pair() -> CandidatePair:
        return _st_candidate(pA=0.55, pB=0.30, nA=0.45, nB=0.70)

    def test_enrichment_writes_fills_to_nA_pB_and_refreshes_pA_leaving_nB(self):
        pair = self._pair()
        client = _st_orderbook_client(nA_fill=0.44, pB_fill=0.31, qty=100, pA_ref=0.57)
        [enriched] = enrich_with_orderbook_prices(client, [pair], _AMPLE_BALANCE_CENTS)
        assert enriched.tradeable is True
        assert enriched.max_contracts == 100
        assert enriched.nA == pytest.approx(0.44)
        assert enriched.pB == pytest.approx(0.31)
        # pA is same_title's reference quote — refreshed from A's NO bids in the
        # same snapshot as the fills (scan-time was 0.55)
        assert enriched.pA == pytest.approx(0.57)
        # nB is reporting-only for this pair type — byte-identical to the input
        assert enriched.nB == pair.nB
        assert leg_prices(enriched) == (pytest.approx(0.44), pytest.approx(0.31))

    def test_ceiling_is_flat_five_percent(self):
        # 0.50 + 0.46 = 0.96 > 0.95 — the same-title ceiling, no deadline tiering
        pair = self._pair()
        client = _st_orderbook_client(nA_fill=0.50, pB_fill=0.46)
        [enriched] = enrich_with_orderbook_prices(client, [pair], _AMPLE_BALANCE_CENTS)
        assert enriched.tradeable is False
        client_ok = _st_orderbook_client(nA_fill=0.50, pB_fill=0.45)
        [enriched_ok] = enrich_with_orderbook_prices(client_ok, [pair], _AMPLE_BALANCE_CENTS)
        assert enriched_ok.tradeable is True

    def test_validate_pair_price_same_title(self):
        pair = self._pair()
        client = _st_orderbook_client(nA_fill=0.44, pB_fill=0.31, qty=100)
        assert validate_pair_price(client, SimpleNamespace(pair=pair, x=100)) is True
        assert validate_pair_price(client, SimpleNamespace(pair=pair, x=101)) is False

    def test_time_series_shaped_books_yield_no_depth_for_same_title(self):
        # The inverse of the time-series wrong-shape test: a same-title pair
        # served time-series-shaped books (A NO bids, B YES bids) finds nothing.
        pair = self._pair()

        def fake_orderbook(ticker):
            if ticker == "A1":
                ob = {"yes_dollars": [], "no_dollars": [["0.56", "100"]]}
            else:
                ob = {"yes_dollars": [["0.69", "100"]], "no_dollars": []}
            return _raw_book_response(ob)

        client = MagicMock()
        client.get_market_orderbook_without_preload_content = MagicMock(side_effect=fake_orderbook)
        [enriched] = enrich_with_orderbook_prices(client, [pair], _AMPLE_BALANCE_CENTS)
        assert enriched.tradeable is False


class TestEnrichmentRefreshesReferenceQuote:
    """enrich_with_orderbook_prices must refresh the pair's REFERENCE YES ask —
    the non-leg market's YES ask (pB for time_series, pA for same_title) — from
    the book it already fetched, and drop a time-series pair with no fresh
    reference (fail closed), a crossed book or a refused spread.

    Left stale, that quote would test the spread rule against a book that has
    moved and, for a time-series pair, put a stale later YES ask into the mid
    spread the forecast reads. A spread at or below zero is what
    config.time_series_profit_prob's max(0, spread) clamp would read as
    p = 1.0 — a riskless model on a directional bet (TS-34, DR-78)."""

    def test_enrichment_refreshes_the_time_series_reference_ask(self):
        # LATE's YES ask has moved to 0.65 since the scan captured 0.60; the
        # refreshed value is what lands in pB, from LATE's NO bids
        pair = _ts_candidate(gap_days=10, pA=0.30, pB=0.60, nB=0.50)
        client = _ts_orderbook_client(pA_fill=0.30, nB_fill=0.50, pB_ref=0.65)
        [enriched] = enrich_with_orderbook_prices(client, [pair], _AMPLE_BALANCE_CENTS)
        assert enriched.tradeable is True
        assert enriched.pB == pytest.approx(0.65)
        assert enriched.pB != pytest.approx(pair.pB)
        # The leg prices and the reporting-only nA are unaffected by the refresh
        assert enriched.pA == pytest.approx(0.30)
        assert enriched.nB == pytest.approx(0.50)
        assert enriched.nA == pair.nA

    def test_enrichment_refreshes_the_same_title_reference_ask(self):
        # Mirror: for a same-title pair the YES leg is on B, so the reference
        # is A's YES ask (pA), read from A's NO bids
        mA = _mock_market(ticker="A1", event_ticker="EVT-A", title="Q", yes_ask=0.55, no_ask=0.45)
        mB = _mock_market(ticker="B1", event_ticker="EVT-B", title="Q", yes_ask=0.30, no_ask=0.70)
        pair = CandidatePair(
            market_a=mA, market_b=mB,
            pA=0.55, pB=0.30, nA=0.45,
            tradeable=True,
            canonical_title="Q",
            pair_type="same_title",
            nB=0.70,
        )
        client = _st_orderbook_client(nA_fill=0.44, pB_fill=0.31, pA_ref=0.60)
        [enriched] = enrich_with_orderbook_prices(client, [pair], _AMPLE_BALANCE_CENTS)
        assert enriched.tradeable is True
        assert enriched.pA == pytest.approx(0.60)
        assert enriched.pA != pytest.approx(pair.pA)
        assert enriched.nA == pytest.approx(0.44)
        assert enriched.pB == pytest.approx(0.31)
        assert enriched.nB == pair.nB

    @staticmethod
    def _inverted_pair_and_client():
        """A time-series pair whose depth-weighted pA lands ABOVE LATE's fresh pB.

        LATE's book is deliberately CROSSED (YES bid 0.70 with a NO bid of
        0.55, summing to 1.25): once the reference is refreshed from the same
        snapshot, the qualifying ceiling avg_yes + avg_no <= 1 - floor makes the
        inversion arithmetically impossible on an uncrossed book, so a crossed
        book is the only shape that can still produce it; the crossed-book guard drops it.

        Leg fills: pA 0.54 (EARLY NO bid 0.46) + nB 0.30 (LATE YES bid 0.70) =
        0.84, inside config.py's price-sum ceiling of 1.0 and profitable after fees.
        The refreshed reference is LATE's YES ask of 0.45 — below the 0.54 fill.
        """
        pair = _ts_candidate(gap_days=10, pA=0.30, pB=0.50, nB=0.30)
        client = _ts_orderbook_client(pA_fill=0.54, nB_fill=0.30, pB_ref=0.45)
        return pair, client

    def test_inverted_pair_after_enrichment_is_dropped(self, caplog):
        pair, client = self._inverted_pair_and_client()
        with caplog.at_level(logging.INFO):
            [enriched] = enrich_with_orderbook_prices(client, [pair], _AMPLE_BALANCE_CENTS)

        assert enriched.tradeable is False

        # The WARNING names a crossed book (YES ask 0.45 under its bid 0.70), not a spread
        direction_drops = [
            r for r in caplog.records
            if "sits below its own YES bid" in r.getMessage()
        ]
        assert len(direction_drops) == 1
        assert direction_drops[0].levelno == logging.WARNING
        assert "0.4500" in direction_drops[0].getMessage()
        assert "0.7000" in direction_drops[0].getMessage()

        # The pair IS profitable at those fills — it must not also be reported
        # as unprofitable, which would misattribute the drop
        assert [
            r for r in caplog.records
            if "unprofitable after depth adjustment" in r.getMessage()
        ] == []

    def test_refresh_makes_the_riskless_clamp_unreachable(self):
        # The consequence, not just the flag. Import locally so this scanner
        # test file does not take a module-level dependency on strategy.
        from kalshi_betting.strategy import _kelly_p, compute_trade

        pair, client = self._inverted_pair_and_client()

        # The clamp's shape: a spread at or below zero, which
        # time_series_profit_prob reads as RISKLESS. The sizer never prices
        # on one: pair_mid_spread reads it as no forecast, and compute_trade
        # refuses the pair
        assert config.time_series_profit_prob(-0.04) == 1.0
        clamped = dataclasses.replace(pair, pA=0.54, mid_spread=-0.04)
        assert _kelly_p(clamped, config.live_settings()) is None
        assert compute_trade(clamped, _AMPLE_BALANCE_CENTS) is None

        # Enrichment never marks this inverted pair tradeable: the refreshed
        # reference sits below the later book's own YES bid, so the
        # crossed-book guard drops it before _kelly_p is ever consulted.
        [enriched] = enrich_with_orderbook_prices(client, [pair], _AMPLE_BALANCE_CENTS)
        assert enriched.tradeable is False

    def test_reference_ask_falls_back_to_scan_time_when_side_is_empty(self, caplog):
        # The fallback is SAME-TITLE only: a time-series pair with no later YES
        # ask fails closed, its scan-time pB left as it was (never None).
        pair = _ts_candidate(gap_days=10, pA=0.30, pB=0.60, nB=0.50)
        client = _ts_orderbook_client(pA_fill=0.30, nB_fill=0.50)
        with caplog.at_level(logging.INFO):
            [enriched] = enrich_with_orderbook_prices(client, [pair], _AMPLE_BALANCE_CENTS)
        assert enriched.tradeable is False
        assert enriched.pB == pair.pB
        assert enriched.pA == pytest.approx(0.30)
        drops = [r for r in caplog.records
                 if "the later contract has no YES ask on its book" in r.getMessage()]
        assert len(drops) == 1
        assert drops[0].levelno == logging.WARNING

        # A same-title pair with A's NO side empty keeps its scan-time pA: its
        # model is the fixed prior, so pA is a reporting-only reference.
        st = _st_candidate(pA=0.60, pB=0.31, nA=0.44)
        [st_e] = enrich_with_orderbook_prices(
            _st_orderbook_client(nA_fill=0.44, pB_fill=0.31), [st], _AMPLE_BALANCE_CENTS,
        )
        assert st_e.tradeable is True
        assert st_e.pA == st.pA

    @pytest.mark.usefixtures("pre_toggle_defaults")
    def test_ceiling_tier_is_the_stated_gap_for_a_ladder(self, caplog):
        # DR-73, pinned BY VALUE because the AST pin beside it cannot see
        # this: test_ast_pair_ceiling_reads_the_pair_gap only asserts that a
        # pair_gap_days call is present and a deadline_gap_days call absent,
        # which a shadowing close_time gap after the real one also satisfies.
        #
        # A ladder whose rungs close at ONE instant (the shape a settled or
        # single-instant event produces) with a STATED gap of 19 days must be
        # held to the 0.30 long tier's price-sum ceiling (0.70), not the 0.85 a
        # 0-day close gap implies. Its uncrossed book fills pA 0.30 + nB 0.45 =
        # 0.75, between the two, so only the stated gap drops it.
        ladder = dataclasses.replace(
            _ts_candidate(gap_days=0, pA=0.30, pB=0.60, nB=0.45),
            stated_gap_days=19,
        )
        client = _ts_orderbook_client(pA_fill=0.30, nB_fill=0.45, pB_ref=0.60)
        with caplog.at_level(logging.INFO):
            [enriched] = enrich_with_orderbook_prices(client, [ladder], _AMPLE_BALANCE_CENTS)

        assert enriched.tradeable is False
        drops = [
            r for r in caplog.records
            if "No qualifying contract pairs" in r.getMessage()
        ]
        assert len(drops) == 1

        # Control: the SAME book and prices with no stated gap — the pair is a
        # cross-event one closing 0 days apart, so the 0.85 ceiling applies and
        # the 0.75 level qualifies. Proves the drop above comes from the stated
        # gap, not from the fixture's prices.
        plain = _ts_candidate(gap_days=0, pA=0.30, pB=0.60, nB=0.45)
        assert plain.stated_gap_days is None
        [kept] = enrich_with_orderbook_prices(
            _ts_orderbook_client(pA_fill=0.30, nB_fill=0.45, pB_ref=0.60), [plain],
            _AMPLE_BALANCE_CENTS,
        )
        assert kept.tradeable is True

    def test_reference_refresh_costs_no_extra_orderbook_fetch(self):
        # The reference comes off an array _fetch_orderbook already returned,
        # so the fetch count stays one per DISTINCT ticker (get_ob's cache),
        # i.e. two for a pair and still two for two pairs on the same markets
        pair = _ts_candidate(gap_days=10, pA=0.30, pB=0.60, nB=0.50)
        client = _ts_orderbook_client(pA_fill=0.30, nB_fill=0.50, pB_ref=0.65)
        enrich_with_orderbook_prices(client, [pair], _AMPLE_BALANCE_CENTS)
        assert client.get_market_orderbook_without_preload_content.call_count == 2

        client_two = _ts_orderbook_client(pA_fill=0.30, nB_fill=0.50, pB_ref=0.65)
        enrich_with_orderbook_prices(client_two, [pair, dataclasses.replace(pair)], _AMPLE_BALANCE_CENTS)
        assert client_two.get_market_orderbook_without_preload_content.call_count == 2


def _live(tier_floors=True, spread_band=(0.0, 1.0), interval_discount=0.75, size_cap=0.20):
    """An explicit LiveSettings of its four required fields (the rest default)."""
    return config.LiveSettings(tier_floors=tier_floors, spread_band=spread_band,
                               interval_discount=interval_discount, size_cap=size_cap)


# Tier floors off, band 0-0.5: pB - pA strictly positive and at most 0.5
_TIERS_OFF_HALF = _live(tier_floors=False, spread_band=(0.0, 0.5))


class TestLiveSpreadRule:
    """find_time_series_pairs applies config.time_series_spread_refusal: the
    entry floor, and the band's ceiling BEFORE the group's one-best contest."""

    @staticmethod
    def _scan(markets, settings, caplog=None):
        return find_time_series_pairs(MagicMock(), held_tickers=set(), markets=markets,
                                      settings=settings)

    @staticmethod
    def _lines(caplog, text):
        return [r.getMessage() for r in caplog.records if text in r.getMessage()]

    def test_tiers_off_admits_a_spread_under_the_tier(self):
        # 0.10 at 10 days: under the 0.15 tier, admitted only with the tiers off
        mA, mB = _ts_pair_markets(gap_days=10, pA=0.30, pB=0.40)
        assert self._scan([mA, mB], _live()) == []
        [pair] = self._scan([mA, mB], _TIERS_OFF_HALF)
        assert pair.pB - pair.pA == pytest.approx(0.10)
        assert pair.tradeable is True

    def test_a_spread_above_the_ceiling_is_refused_and_counted(self, caplog):
        mA, mB = _ts_pair_markets(gap_days=10, pA=0.30, pB=0.85)
        with caplog.at_level(logging.INFO):
            assert self._scan([mA, mB], _TIERS_OFF_HALF) == []
        [line] = self._lines(caplog, "refused above the spread band's")
        assert line == ("Time-series candidates refused above the spread band's 0.5 ceiling "
                        "(pB - pA; before the one-best-per-group contest): 1")
        # control: no band, same pair
        assert len(self._scan([mA, mB], _live(tier_floors=False))) == 1

    def test_an_equal_price_pair_is_refused_and_not_counted(self, caplog):
        # pB == pA: refused by the direction filter, which counts nothing
        mA, mB = _ts_pair_markets(gap_days=10, pA=0.40, pB=0.40)
        with caplog.at_level(logging.INFO):
            assert self._scan([mA, mB], _TIERS_OFF_HALF) == []
        assert self._lines(caplog, "entry floor") == []
        assert self._lines(caplog, "spread band's") == []

    def test_a_spread_under_the_floor_is_counted(self, caplog):
        mA, mB = _ts_pair_markets(gap_days=10, pA=0.30, pB=0.40)
        with caplog.at_level(logging.INFO):
            assert self._scan([mA, mB], _live()) == []
        [line] = self._lines(caplog, "refused below the entry floor")
        assert line.startswith("Time-series candidates refused below the entry floor "
                               "(tier floors on (")
        assert line.endswith("no spread band): 1")

    def test_the_ceiling_acts_before_the_one_best_contest(self):
        # The widest spread, EARLY->LATE (0.60), is over the 0.5 ceiling, so the
        # in-band runner-up EARLY->MID (0.40) represents the group
        close = datetime(2026, 3, 1, tzinfo=UTC)
        early = _mock_market(ticker="EARLY", event_ticker="EVA-1",
                             title="Will BTC exceed $80k by March 01, 2026",
                             yes_ask=0.20, no_ask=0.80, close_time=close)
        mid = _mock_market(ticker="MID", event_ticker="EVM-1",
                           title="Will BTC exceed $80k by March 06, 2026",
                           yes_ask=0.60, no_ask=0.40, close_time=close + timedelta(days=5))
        late = _mock_market(ticker="LATE", event_ticker="EVB-1",
                            title="Will BTC exceed $80k by March 11, 2026",
                            yes_ask=0.80, no_ask=0.20, close_time=close + timedelta(days=10))
        [pair] = self._scan([early, mid, late], _TIERS_OFF_HALF)
        assert (pair.market_a.ticker, pair.market_b.ticker) == ("EARLY", "MID")
        assert pair.pB - pair.pA == pytest.approx(0.40)
        # control: with no band the widest spread wins
        [widest] = self._scan([early, mid, late], _live(tier_floors=False))
        assert (widest.market_a.ticker, widest.market_b.ticker) == ("EARLY", "LATE")

    def test_the_price_sum_guard_still_counts_a_pair_refused_by_both(self, caplog):
        # Over $1 AND over the ceiling: the $1 guard runs first and alone counts it
        mA, mB = _ts_pair_markets(gap_days=10, pA=0.30, pB=0.90, nB=0.75)
        with caplog.at_level(logging.INFO):
            assert self._scan([mA, mB], _TIERS_OFF_HALF) == []
        assert len(self._lines(caplog, "leg price sum at or above $1: 1")) == 1
        assert self._lines(caplog, "spread band's") == []

    def test_ladder_refusals_are_counted_apart(self, monkeypatch, caplog):
        monkeypatch.setattr(scanner, "TIME_SERIES_SAME_EVENT_LADDERS", True)
        early_close = datetime(2026, 3, 1, tzinfo=UTC)
        late_close = datetime(2026, 3, 20, tzinfo=UTC)
        # Above the band's ceiling: 0.55 against 0.5
        wide = [_ladder_rung("W-EARLY", "by March 1, 2026", event="KXWIDE-1",
                             yes_ask=0.20, no_ask=0.80, close=early_close),
                _ladder_rung("W-LATE", "by March 20, 2026", event="KXWIDE-1",
                             yes_ask=0.75, no_ask=0.25, close=late_close)]
        _assert_one_ladder_group(*wide)
        with caplog.at_level(logging.INFO):
            assert self._scan(wide, _TIERS_OFF_HALF) == []
        assert self._lines(caplog, "Same-event ladder candidates refused above the "
                                   "spread band's 0.5 ceiling: 1")
        assert self._lines(caplog, "Time-series candidates refused above") == []
        caplog.clear()
        # Under the 0.30 tier its 19-day STATED gap chooses: 0.20
        narrow = [_ladder_rung("N-EARLY", "by March 1, 2026", event="KXNARROW-1",
                               yes_ask=0.20, no_ask=0.80, close=early_close),
                  _ladder_rung("N-LATE", "by March 20, 2026", event="KXNARROW-1",
                               yes_ask=0.40, no_ask=0.60, close=late_close)]
        with caplog.at_level(logging.INFO):
            assert self._scan(narrow, _live()) == []
        [line] = self._lines(caplog, "Same-event ladder candidates refused below the entry floor")
        assert line.endswith(": 1")
        assert self._lines(caplog, "Time-series candidates refused below") == []

    def test_the_rule_is_always_logged(self, caplog):
        with caplog.at_level(logging.INFO):
            self._scan([], _TIERS_OFF_HALF)
        [line] = self._lines(caplog, "Time-series entry rule:")
        assert line == ("Time-series entry rule: tier floors off (pB - pA must still be "
                        "positive), spread band 0-0.5 on pB - pA")
        caplog.clear()
        with caplog.at_level(logging.INFO):
            self._scan([], _live())
        [line] = self._lines(caplog, "Time-series entry rule:")
        assert line == ("Time-series entry rule: tier floors on (≥15% up to 15 days apart, "
                        "≥30% for 16-30), no spread band")

    def test_no_settings_reads_config_once(self, monkeypatch):
        calls = []

        def counting():
            calls.append(1)
            return config.live_settings()

        monkeypatch.setattr(scanner, "live_settings", counting)
        mA, mB = _ts_pair_markets(gap_days=10, pA=0.30, pB=0.50)
        find_time_series_pairs(MagicMock(), held_tickers=set(), markets=[mA, mB])
        assert calls == [1]
        # ... and never when handed settings
        find_time_series_pairs(MagicMock(), held_tickers=set(), markets=[mA, mB],
                               settings=_live())
        assert calls == [1]


class TestEnrichmentSpreadRule:
    """enrich_with_orderbook_prices on the fresh book: fail closed with no later
    YES ask, the crossed-book guard, the top-of-book ceiling, the fee cut, and
    the config.max_kelly_fraction bound."""

    @staticmethod
    def _lines(caplog, text):
        return [r for r in caplog.records if text in r.getMessage()]

    def test_no_reference_fails_closed_for_time_series_only(self, caplog):
        ts = _ts_candidate(gap_days=10, pA=0.30, pB=0.60, nB=0.50)
        with caplog.at_level(logging.INFO):
            [ts_e] = enrich_with_orderbook_prices(
                _ts_orderbook_client(pA_fill=0.30, nB_fill=0.50), [ts], _AMPLE_BALANCE_CENTS,
                settings=_live())
        assert ts_e.tradeable is False
        [line] = self._lines(caplog, "no YES ask on its book (no resting NO bids)")
        assert line.levelno == logging.WARNING
        # same-title still falls back to its scan-time reference
        st = _st_candidate(pA=0.60, pB=0.31, nA=0.44)
        [st_e] = enrich_with_orderbook_prices(
            _st_orderbook_client(nA_fill=0.44, pB_fill=0.31), [st], _AMPLE_BALANCE_CENTS,
            settings=_live())
        assert st_e.tradeable is True and st_e.pA == st.pA

    def test_a_crossed_later_book_is_dropped(self, caplog):
        # LATE: YES bid 0.60, YES ask 0.55 — crossed; the 0.25 spread clears
        # the 0.15 tier, so only the guard can drop it
        pair = _ts_candidate(gap_days=10, pA=0.30, pB=0.60, nB=0.40)
        with caplog.at_level(logging.INFO):
            [enriched] = enrich_with_orderbook_prices(
                _ts_orderbook_client(pA_fill=0.30, nB_fill=0.40, pB_ref=0.55), [pair],
                _AMPLE_BALANCE_CENTS, settings=_live())
        assert enriched.tradeable is False
        [line] = self._lines(caplog, "sits below its own YES bid")
        assert line.levelno == logging.WARNING
        assert "0.5500" in line.getMessage() and "0.6000" in line.getMessage()

    def test_a_crossed_earlier_book_is_dropped(self, caplog):
        # EARLY: YES ask 0.30 under its own YES bid 0.35 — crossed, so its
        # midpoint sits above its YES ask. LATE is uncrossed and the 0.30
        # spread clears the 0.15 tier, so only the earlier book's guard can drop it
        pair = _ts_candidate(gap_days=10, pA=0.30, pB=0.60, nB=0.40)
        with caplog.at_level(logging.INFO):
            [enriched] = enrich_with_orderbook_prices(
                _ts_orderbook_client(pA_fill=0.30, nB_fill=0.40, pB_ref=0.60, a_yes_bid=0.35),
                [pair], _AMPLE_BALANCE_CENTS, settings=_live())
        assert enriched.tradeable is False
        [line] = self._lines(caplog, "sits below its own YES bid")
        assert line.levelno == logging.WARNING
        assert line.getMessage() == (
            "Pair 'will btc exceed $80k by' dropped: the earlier contract's YES ask "
            "0.3000 sits below its own YES bid 0.3500 — a crossed book")
        # control: the same books with EARLY's YES bid at its YES ask are kept
        [kept] = enrich_with_orderbook_prices(
            _ts_orderbook_client(pA_fill=0.30, nB_fill=0.40, pB_ref=0.60, a_yes_bid=0.30),
            [pair], _AMPLE_BALANCE_CENTS, settings=_live())
        assert kept.tradeable is True

    def test_the_earlier_guard_reads_its_best_yes_bid(self, caplog):
        # EARLY rests YES bids 0.35 and 0.25 under its 0.30 YES ask: crossed
        # only at the TOP, so a guard reading any level but the best one passes
        # it. LATE is uncrossed (YES ask 0.64, YES bid 0.52)
        pair = _ts_candidate(gap_days=10, pA=0.30, pB=0.64, nB=0.48)

        def enrich(a_yes_bids):
            client = _books_client(_mid_books(a_yes_bids=a_yes_bids))
            return scanner._enrich_pair(
                pair, _fetch_orderbook(client, "EARLY"), _fetch_orderbook(client, "LATE"),
                _AMPLE_BALANCE_CENTS, settings=_live(), cash_cents=None)

        with caplog.at_level(logging.INFO):
            priced, refused = enrich(((0.25, 100), (0.35, 100)))
        assert (priced.tradeable, refused) == (False, scanner.ENRICH_CROSSED)
        [line] = self._lines(caplog, "sits below its own YES bid")
        assert "0.3000" in line.getMessage() and "0.3500" in line.getMessage()
        # control: with its best YES bid at the ask the pair is kept
        priced, refused = enrich(((0.25, 100), (0.30, 100)))
        assert (priced.tradeable, refused) == (True, None)

    @pytest.mark.parametrize(("cent", "pB_ref"), [
        (0.07, 0.60), (0.32, 0.60), (0.33, 0.60), (0.34, 0.60),
        (0.66, 0.80), (0.67, 0.80), (0.68, 0.80), (0.93, 0.97)])
    def test_an_earlier_book_one_float_step_short_is_not_crossed(self, caplog, cent, pB_ref):
        # EARLY's YES ask and YES bid both at `cent`: a book with no width.
        # The book reader builds each ask as 1 - the other side's bid, and on
        # these eight whole cents the two come to one float step under 1.0.
        # The guard tolerates PRICE_EPSILON, so the pair is kept. LATE has no
        # width either, and its two add up to 1.0 exactly
        nB = round(1.0 - pB_ref, 2)
        pair = _ts_candidate(gap_days=10, pA=cent, pB=pB_ref, nB=nB)
        client = _ts_orderbook_client(pA_fill=cent, nB_fill=nB, pB_ref=pB_ref)
        ob_a, ob_b = _fetch_orderbook(client, "EARLY"), _fetch_orderbook(client, "LATE")
        yes_ask_a = _bids_to_ask_levels(ob_a["no"])[0][0]
        no_ask_a = _bids_to_ask_levels(ob_a["yes"])[0][0]
        assert yes_ask_a + no_ask_a < 1.0
        assert (_reference_yes_ask(pair, ob_a, ob_b)
                + _bids_to_ask_levels(ob_b["yes"])[0][0]) >= 1.0
        with caplog.at_level(logging.INFO):
            priced, refused = scanner._enrich_pair(
                pair, ob_a, ob_b, _AMPLE_BALANCE_CENTS, settings=_live(tier_floors=False),
                cash_cents=None)
        assert (priced.tradeable, refused) == (True, None)
        assert self._lines(caplog, "sits below its own YES bid") == []

    def test_the_guard_reads_the_later_books_best_yes_bid(self, caplog):
        # LATE rests YES bids 0.60 and 0.55 and a YES ask of 0.58: crossed only at the
        # TOP (0.60 > 0.58), so a guard reading any level but the best one passes it.
        # The spread and price-sum ceiling pass too, so only the guard drops it
        pair = _ts_candidate(gap_days=10, pA=0.20, pB=0.58, nB=0.40)
        levels = [(0.20, 0.40, 10), (0.21, 0.45, 10)]
        with caplog.at_level(logging.INFO):
            [enriched] = enrich_with_orderbook_prices(
                _ts_multilevel_client(levels, pB_ref=0.58), [pair], _AMPLE_BALANCE_CENTS,
                settings=_live())
        assert enriched.tradeable is False
        [line] = self._lines(caplog, "sits below its own YES bid")
        assert line.levelno == logging.WARNING
        assert "0.5800" in line.getMessage() and "0.6000" in line.getMessage()
        # control: the same book with its YES ask AT the best YES bid is kept
        [kept] = enrich_with_orderbook_prices(
            _ts_multilevel_client(levels, pB_ref=0.60), [pair], _AMPLE_BALANCE_CENTS,
            settings=_live())
        assert kept.tradeable is True

    def test_the_guard_is_silent_on_an_uncrossed_book(self, caplog):
        # The YES ask exactly AT the YES bid (0.60 + 0.40 = 1.0) is uncrossed
        pair = _ts_candidate(gap_days=10, pA=0.30, pB=0.60, nB=0.40)
        with caplog.at_level(logging.INFO):
            [enriched] = enrich_with_orderbook_prices(
                _ts_orderbook_client(pA_fill=0.30, nB_fill=0.40, pB_ref=0.60), [pair],
                _AMPLE_BALANCE_CENTS, settings=_live())
        assert enriched.tradeable is True
        assert self._lines(caplog, "sits below its own YES bid") == []
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    @pytest.mark.parametrize("pB_ref", [0.0199, 0.01])
    def test_the_guard_drops_a_book_crossed_by_one_tick(self, caplog, pB_ref):
        # A sub-cent book at the shipped rule, LATE's YES bid 0.02: a reference
        # one tick (0.0001) or a cent under it is crossed and must drop — the
        # guard tolerates PRICE_EPSILON, never a tick
        settings = _live(tier_floors=False, spread_band=(0.0, 0.5),
                         interval_discount=0.80, size_cap=1.0)
        pair = _ts_candidate(gap_days=5, pA=0.005, pB=0.02, nB=0.98)
        with caplog.at_level(logging.INFO):
            [enriched] = enrich_with_orderbook_prices(
                _ts_orderbook_client(pA_fill=0.005, nB_fill=0.98, pB_ref=pB_ref), [pair],
                _AMPLE_BALANCE_CENTS, settings=settings)
        assert enriched.tradeable is False
        [line] = self._lines(caplog, "sits below its own YES bid")
        assert line.levelno == logging.WARNING

    def test_a_reference_at_the_yes_bid_on_a_sub_cent_book_is_kept_under_one_minus_k(self):
        # Control for the rows above: a YES ask AT the 0.02 bid is uncrossed, is
        # kept, and sizes under 1 - k = 0.20 with no per-trade cap
        from kalshi_betting.strategy import compute_trade

        settings = _live(tier_floors=False, spread_band=(0.0, 0.5),
                         interval_discount=0.80, size_cap=1.0)
        pair = _ts_candidate(gap_days=5, pA=0.005, pB=0.02, nB=0.98)
        [kept] = enrich_with_orderbook_prices(
            _ts_orderbook_client(pA_fill=0.005, nB_fill=0.98, pB_ref=0.02), [pair],
            _AMPLE_BALANCE_CENTS, settings=settings)
        assert kept.tradeable is True
        spec = compute_trade(kept, _AMPLE_BALANCE_CENTS, settings=settings)
        assert spec is not None
        assert 0.0 < spec.kelly_fraction < 0.20

    def test_the_ceiling_is_tested_on_the_top_of_the_book(self, caplog):
        # YES fills 0.20 x10 / 0.30 x90, reference 0.72 (uncrossed): the average
        # fill's 0.43 spread is inside the 0.5 ceiling, the TOP's 0.52 is not
        pair = _ts_candidate(gap_days=10, pA=0.20, pB=0.72, nB=0.28)
        client = _ts_multilevel_client([(0.20, 0.28, 10), (0.30, 0.28, 90)], pB_ref=0.72)
        with caplog.at_level(logging.INFO):
            [enriched] = enrich_with_orderbook_prices(
                client, [pair], _AMPLE_BALANCE_CENTS, settings=_TIERS_OFF_HALF)
        assert enriched.pA == pytest.approx(0.29)
        assert enriched.tradeable is False
        [line] = self._lines(caplog, "at the top of the book exceeds the spread band's 0.5")
        assert line.levelno == logging.WARNING
        assert "0.5200" in line.getMessage()
        # control: no band, same book
        [kept] = enrich_with_orderbook_prices(
            _ts_multilevel_client([(0.20, 0.28, 10), (0.30, 0.28, 90)], pB_ref=0.72),
            [pair], _AMPLE_BALANCE_CENTS, settings=_live(tier_floors=False))
        assert kept.tradeable is True

    @staticmethod
    def _adversary_client():
        # A YES asks 0.30 x100 / 0.53 x2000 and YES bid 0.30 (its top has no
        # width); B NO ask 0.45 x2100 and YES ask 0.56 (above its 0.55 bid:
        # uncrossed)
        def fake_orderbook(ticker):
            if ticker == "EARLY":
                ob = {"yes_dollars": [["0.30", "100"]],
                      "no_dollars": [["0.70", "100"], ["0.47", "2000"]]}
            else:
                ob = {"yes_dollars": [["0.55", "2100"]], "no_dollars": [["0.44", "1000"]]}
            return _raw_book_response(ob)

        client = MagicMock()
        client.get_market_orderbook_without_preload_content = MagicMock(
            side_effect=fake_orderbook)
        return client

    def test_levels_are_cut_at_the_first_with_no_edge_after_the_fee(self, caplog):
        # Tiers off (1.0 ceiling): the second level (0.98, a 0.02 edge under its
        # ~0.035 fee) qualifies; averaged in, it drops the pair (#51); cut, the
        # sizer buys exactly the 100 contracts with a 0.25 edge
        from kalshi_betting.strategy import compute_trade

        pair = _ts_candidate(gap_days=10, pA=0.30, pB=0.56, nB=0.45)
        with caplog.at_level(logging.INFO):
            [enriched] = enrich_with_orderbook_prices(
                self._adversary_client(), [pair], 1_000_000,
                settings=_live(tier_floors=False))
        assert enriched.tradeable is True
        assert enriched.depth_levels == (pytest.approx((0.30, 0.45, 100.0)),)
        assert enriched.max_contracts == 100
        assert (enriched.pA, enriched.nB, enriched.pB) == pytest.approx((0.30, 0.45, 0.56))
        assert self._lines(caplog, "unprofitable after depth adjustment") == []
        spec = compute_trade(enriched, 1_000_000)
        assert spec is not None and spec.x == 100

    def test_the_cut_is_inert_at_a_floor_of_015(self):
        # Tiers on: the 0.85 ceiling leaves every level more edge than any fee,
        # so the level right on it (0.36 + 0.49) is kept
        pair = _ts_candidate(gap_days=10, pA=0.30, pB=0.62, nB=0.45)
        client = _ts_multilevel_client([(0.30, 0.45, 10), (0.36, 0.49, 90), (0.45, 0.52, 500)])
        [enriched] = enrich_with_orderbook_prices(client, [pair], _AMPLE_BALANCE_CENTS,
                                                  settings=_live())
        assert len(enriched.depth_levels) == 2
        assert enriched.max_contracts == 100

    def test_a_book_with_no_edge_after_the_fee_is_named(self, caplog):
        # One level, 0.53 + 0.45 = 0.98: inside the 1.0 ceiling, its edge under the fee
        pair = _ts_candidate(gap_days=10, pA=0.53, pB=0.56, nB=0.45)
        with caplog.at_level(logging.INFO):
            [enriched] = enrich_with_orderbook_prices(
                _ts_orderbook_client(pA_fill=0.53, nB_fill=0.45, pB_ref=0.56), [pair],
                _AMPLE_BALANCE_CENTS, settings=_live(tier_floors=False))
        assert enriched.tradeable is False
        [line] = self._lines(caplog, "keeps an edge after the fee")
        assert line.getMessage() == (
            f"No contract pair for '{pair.canonical_title}' keeps an edge after the fee "
            "— skipping")
        assert self._lines(caplog, "No qualifying contract pairs") == []

    def test_k_of_one_names_the_zero_bound(self, caplog):
        pair = _ts_candidate(gap_days=10, pA=0.30, pB=0.62, nB=0.50)
        with caplog.at_level(logging.INFO):
            [enriched] = enrich_with_orderbook_prices(
                _ts_orderbook_client(pA_fill=0.30, nB_fill=0.50, pB_ref=0.62), [pair],
                _AMPLE_BALANCE_CENTS, settings=_live(interval_discount=1.0, size_cap=1.0))
        assert enriched.tradeable is False
        [line] = self._lines(caplog, "No affordable contract pairs")
        assert ("the budget affords 0; the per-trade bound is 0 (k = 1.00: time-series "
                "Kelly cannot be positive); skipping") in line.getMessage()
        # control: a same-title pair under the same settings is not bounded at 0
        st = _st_candidate(pA=0.60, pB=0.31, nA=0.44)
        [st_e] = enrich_with_orderbook_prices(
            _st_orderbook_client(nA_fill=0.44, pB_fill=0.31, pA_ref=0.60), [st],
            _AMPLE_BALANCE_CENTS, settings=_live(interval_discount=1.0, size_cap=1.0))
        assert st_e.tradeable is True

    def test_the_bound_is_max_kelly_fraction_exactly(self):
        # $10,000 at a best level of 0.80: 0.20 affords 2500 pairs, the unrounded
        # 1 - 0.8 (0.19999999999999996) only 2499
        ts = _ts_candidate(gap_days=10, pA=0.30, pB=0.62, nB=0.50)
        [ts_e] = enrich_with_orderbook_prices(
            _ts_orderbook_client(pA_fill=0.30, nB_fill=0.50, qty=5000, pB_ref=0.62), [ts],
            1_000_000, settings=_live(interval_discount=0.8, size_cap=1.0))
        assert ts_e.max_contracts == 2500
        # Same-title at no cap: the 0.95 co-resolution prior, over 0.75
        st = _st_candidate(pA=0.60, pB=0.31, nA=0.44)
        [st_e] = enrich_with_orderbook_prices(
            _st_orderbook_client(nA_fill=0.44, pB_fill=0.31, qty=20000, pA_ref=0.60), [st],
            1_000_000, settings=_live(size_cap=1.0))
        assert st_e.max_contracts == int(10_000 * 0.95 / 0.75)
        # _live()'s 20% cap bounds it at 0.20 too
        [ts_t] = enrich_with_orderbook_prices(
            _ts_orderbook_client(pA_fill=0.30, nB_fill=0.50, qty=5000, pB_ref=0.62), [ts],
            1_000_000, settings=_live())
        assert ts_t.max_contracts == 2500

    def test_no_settings_reads_config_once(self, monkeypatch):
        calls = []

        def counting():
            calls.append(1)
            return config.live_settings()

        monkeypatch.setattr(scanner, "live_settings", counting)
        pairs = [_ts_candidate(gap_days=10, pA=0.30, pB=0.62, nB=0.50) for _ in range(3)]
        enrich_with_orderbook_prices(
            _ts_orderbook_client(pA_fill=0.30, nB_fill=0.50, pB_ref=0.62), pairs,
            _AMPLE_BALANCE_CENTS)
        assert calls == [1]


# enrich_with_orderbook_prices as it was before its per-pair loop moved into
# scanner._enrich_pair, kept as the oracle the split must match: the same
# pairs, the same log lines, the same fetches. It is that code word for word
# (only renamed), but for the mid spread's lines, which are the live code's:
# the earlier market's best NO ask and YES ask read in the time-series block,
# and mid_spread written beside the refreshed pB.
def _old_enrich_with_orderbook_prices(
    client: Any, pairs: list, portfolio_value_cents: int, *,
    settings: LiveSettings | None = None, cash_cents: int | None = None,
) -> list:
    """
    Check each pair against its live order books and price it at what the account could really pay.

    For each tradeable pair, fetches both order books, matches the two legs'
    asks level by level, keeps the levels whose combined price leaves the
    required gap, and stops at the first level with no edge left after the fee.
    The leg prices become the average fill price over the contracts this trade
    could afford: at most what the largest possible Kelly share of the
    portfolio value buys at the book's best price, and never more than the
    cash. The kept levels are stored on the pair (depth_levels) so
    compute_trade can price any count, and max_contracts is the number of
    contracts the prices are for. A time-series pair also fails when the later
    market shows no YES ask, its YES ask sits below its own YES bid, or the
    spread rule refuses it. A pair that fails a check is marked tradeable=False.

    Args:
        client (Any): Kalshi client used to fetch order books (each fetched once per call).
        pairs (list): CandidatePairs; one already marked tradeable=False is passed through unchanged.
        portfolio_value_cents (int): Cash plus open positions' value, in cents; required, as it limits how much of the book is averaged.
        settings (LiveSettings | None): Keyword-only. The run's settings; None reads config.py's (tests only).
        cash_cents (int | None): Keyword-only. The cash on hand in cents; None means it is all cash. Hand compute_trade the same value.

    Returns:
        list: One CandidatePair per input, in order, with prices, tradeable, max_contracts and depth_levels set from the books.
    """
    # Resolved once, so every pair below is judged under one rule
    settings = live_settings() if settings is None else settings

    # Cache order books by ticker to avoid fetching the same book twice
    # when the same market appears in multiple pairs
    ob_cache: dict[str, dict | None] = {}

    def get_ob(ticker: str) -> dict | None:
        if ticker not in ob_cache:
            # Fetch the order book from the Kalshi API and cache the result
            ob_cache[ticker] = _fetch_orderbook(client, ticker)
        return ob_cache[ticker]

    enriched = []
    for pair in pairs:
        # Non-tradeable pairs (failed best-ask check) are passed through unchanged —
        # they still appear in the dev simulation Excel sheet for transparency
        if not pair.tradeable:
            enriched.append(pair)
            continue

        ob_a = get_ob(pair.market_a.ticker)
        ob_b = get_ob(pair.market_b.ticker)

        if ob_a is None or ob_b is None:
            # If either order book is unavailable we cannot validate depth, so
            # mark the pair non-tradeable. Passing best-ask prices through with
            # tradeable=True would let strategy.compute_trade size against unbounded
            # depth (max_contracts=0 is treated as "no cap" downstream) — the
            # opposite of what pre_execution_check is meant to catch.
            logging.warning(
                "Orderbook unavailable for '%s' — marking non-tradeable",
                pair.canonical_title,
            )
            enriched.append(dc_replace(pair, tradeable=False))
            continue

        # Derive the ask levels each leg would consume — each buy fills against
        # the OPPOSITE-side bids of ITS OWN market, and which market carries the
        # NO leg depends on the pair type (see _leg_ask_levels)
        no_levels, yes_levels = _leg_ask_levels(pair, ob_a, ob_b)

        # Merge-pair the NO and YES depth levels into (yes_price, no_price, qty) tuples
        paired     = _pair_orderbooks(no_levels, yes_levels)

        # Keep only contract pairs where the combined fill price leaves the required
        # gap: same_title requires >= SAME_TITLE_MIN_PRICE_DIFF; time_series requires
        # the run's entry floor — see _pair_max_sum for the exact ceilings.
        max_sum    = _pair_max_sum(pair, settings)
        qualifying = [
            (yp, np_, qty)
            for yp, np_, qty in paired
            if yp + np_ <= max_sum + PRICE_EPSILON
        ]

        # Keep the prefix that still has an edge after the fee, as validate_pair_price does
        before = len(qualifying)
        qualifying = _levels_with_edge_after_fee(qualifying)
        if before and not qualifying:
            logging.info(
                "No contract pair for '%s' keeps an edge after the fee — skipping",
                pair.canonical_title)
            enriched.append(dc_replace(pair, tradeable=False))
            continue

        if not qualifying:
            # No depth available at the required gap — mark untradeable to skip execution
            logging.info(
                "No qualifying contract pairs for '%s' after price gap filter — skipping",
                pair.canonical_title,
            )
            enriched.append(dc_replace(pair, tradeable=False))
            continue

        # Orient the levels into MARKET order — (market_a's leg price,
        # market_b's leg price, qty) — so everything downstream reads them the
        # way leg_prices reads the pair's scalars. leg_sides is the only source
        # of truth for which market buys which side.
        side_a, _side_b = leg_sides(pair.pair_type)
        a_is_no = side_a == "no"
        depth_levels = tuple(
            (np_, yp, q) if a_is_no else (yp, np_, q) for yp, np_, q in qualifying
        )

        total_qty = sum(qty for _, _, qty in qualifying)
        # Average only over the most contracts the largest possible Kelly share
        # could buy at the book's best price (never more than the cash), not the
        # whole book. For time-series pairs this limit holds only because of the
        # checks below.
        best_a, best_b, _ = depth_levels[0]
        bound = max_kelly_fraction(pair.pair_type, settings)
        affordable = max_affordable_pairs(portfolio_value_cents, best_a + best_b, bound,
                                          cash_cents=cash_cents)
        cap = min(int(total_qty), affordable)
        fills = prefix_fill_prices(depth_levels, cap)

        if fills is None:
            # cap < 1: the budget cannot afford one contract pair, or the book
            # holds under one contract of qualifying depth. Drop the pair rather
            # than write max_contracts=0, which compute_trade reads as UNCAPPED
            # — the sub-one-contract hole that overloaded sentinel used to have.
            # Both figures are named because they are different faults with
            # different fixes (add funds vs. the book is too thin), and the
            # binding one is whichever is smaller. A bound of 0 gets its own wording: only
            # a time-series 1 - k rounding to 0 (k = 1) makes one (no cap can be 0).
            why = ""
            if bound == 0 and pair.pair_type == "time_series":
                why = (
                    f"; the per-trade bound is 0 (k = {settings.interval_discount:.2f}: "
                    "time-series Kelly cannot be positive)"
                )
            # If the cash, not the portfolio share, limited the budget, say so
            if (affordable < 1 and cash_cents is not None
                    and kelly_budget(portfolio_value_cents / 100.0, bound)
                    > cash_cents / 100.0):
                why += f"; the ${cash_cents / 100:.2f} of cash binds"
            logging.info(
                "No affordable contract pairs for '%s' — %.2f contract(s) rest at "
                "the gap and the budget affords %d%s; skipping",
                pair.canonical_title, total_qty, affordable, why,
            )
            enriched.append(dc_replace(pair, tradeable=False))
            continue

        # Back to SIDE order ("the YES leg"/"the NO leg") for the code below
        avg_yes, avg_no = (fills[1], fills[0]) if a_is_no else (fills[0], fills[1])

        # The REFERENCE quote — the non-leg market's YES ask — refreshed from the
        # book already in hand, so the spread rule below and the mid spread
        # read a quote of this snapshot, never a scan-time one.
        ref_yes = _reference_yes_ask(pair, ob_a, ob_b)

        # Time-series only: same_title's model is the fixed co-resolution prior,
        # so its pA is not a model input and there is no clamp to protect
        is_time_series = leg_sides(pair.pair_type) == TIME_SERIES_LEG_SIDES
        direction_ok = True
        if is_time_series:
            # pair_gap_days: a same-event ladder is tiered on its STATED gap (DR-73)
            gap = pair_gap_days(pair)
            # The earlier market's best NO ask (1 - its best YES bid), 1.0 with
            # no YES bid, and its best YES ask: the mid spread's A side
            a_no_asks = _bids_to_ask_levels(ob_a["yes"], _pair_ticker(pair, "market_a"))
            no_ask_a = a_no_asks[0][0] if a_no_asks else 1.0
            yes_ask_a = yes_levels[0][0]
            if ref_yes is None:
                # Nothing prices the in-between mass now: fail CLOSED. A stale pB below the
                # book's YES bid would let Kelly exceed 1 - k, which a cap above 1 - k (the
                # 100% default) does not stop. backtester._find_entry does not mirror this
                # (CLAUDE.md: "Known residual of the live spread rule").
                direction_ok = False
                logging.warning(
                    "Pair '%s' dropped: the later contract has no YES ask on its book "
                    "(no resting NO bids), so nothing prices its in-between mass now",
                    pair.canonical_title)
            elif ref_yes + no_levels[0][0] < 1.0 - PRICE_EPSILON:
                # no_levels[0][0] is the later market's best NO ask, 1 - its best YES bid; a
                # YES ask below that bid is a crossed book, where Kelly could exceed 1 - k
                direction_ok = False
                logging.warning(
                    "Pair '%s' dropped: the later contract's YES ask %.4f sits below "
                    "its own YES bid %.4f — a crossed book",
                    pair.canonical_title, ref_yes, 1.0 - no_levels[0][0])
            elif yes_ask_a + no_ask_a < 1.0 - PRICE_EPSILON:
                # The earlier book crossed: its YES ask below its own YES bid
                direction_ok = False
                logging.warning(
                    "Pair '%s' dropped: the earlier contract's YES ask %.4f sits below "
                    "its own YES bid %.4f — a crossed book",
                    pair.canonical_title, yes_ask_a, 1.0 - no_ask_a)
            else:
                # Positivity and the floor are tested on the YES-ask gap at the
                # YES leg's fill (the prefix average the order pays), the
                # ceiling on the TOP of the book. Past the guards above, the
                # price-sum ceiling implies the floor to within 2 * PRICE_EPSILON
                # and the edge cut leaves a spread above the fee, so only the
                # ceiling is live here; the other two stay as defence (TS-34).
                basis = f"fresh reference ask {ref_yes:.4f}"
                refusal = time_series_spread_refusal(ref_yes - avg_yes, gap, settings)
                if refusal is None:
                    refusal = time_series_spread_refusal(
                        ref_yes - qualifying[0][0], gap, settings)
                direction_ok = refusal is None
                if refusal == SPREAD_NOT_POSITIVE:
                    logging.warning(
                        "Pair '%s' dropped: the later contract no longer prices above "
                        "the YES leg fill %.4f — %s", pair.canonical_title, avg_yes, basis)
                elif refusal == SPREAD_BELOW_FLOOR:
                    logging.warning(
                        "Pair '%s' dropped: pB - pA %.4f at the YES leg fill is under the "
                        "%.2f entry floor — %s", pair.canonical_title, ref_yes - avg_yes,
                        live_time_series_floor(gap, settings), basis)
                elif refusal == SPREAD_ABOVE_CEILING:
                    logging.warning(
                        "Pair '%s' dropped: pB - pA %.4f at the top of the book exceeds "
                        "the spread band's %g ceiling — %s", pair.canonical_title,
                        ref_yes - qualifying[0][0], settings.spread_band[1], basis)

        # Re-validate tradeability at the depth-weighted prices (the pair may still be
        # unprofitable if all qualifying contracts are at the edge of the gap threshold).
        profitable = (1.0 - avg_no - avg_yes) > fee_per_pair_approx(avg_no, avg_yes)

        if not profitable:
            # Kept as its own arm so this line only ever reports the profitability
            # verdict — a direction drop has already logged its own WARNING above
            logging.info(
                "Pair '%s' unprofitable after depth adjustment: avg_no=%.3f avg_yes=%.3f",
                pair.canonical_title, avg_no, avg_yes,
            )

        new_tradeable = profitable and direction_ok

        # Replace the LEG prices and contract count with depth-accurate values,
        # writing back to whichever fields are the leg prices for this pair type
        # (nA/pB for same_title, pA/nB for time_series — the fields
        # leg_prices() reads); strategy.py sizes the final Kelly trade on them
        if leg_sides(pair.pair_type) == TIME_SERIES_LEG_SIDES:
            leg_updates = {"pA": avg_yes, "nB": avg_no}
            # pB is the reference quote, not a leg price — written fresh from
            # the book. Left alone when None, which has dropped the pair above.
            if ref_yes is not None:
                leg_updates["pB"] = ref_yes
                # The mid spread at the tops of both books
                leg_updates["mid_spread"] = time_series_mid_spread(
                    yes_ask_a, no_ask_a, ref_yes, no_levels[0][0])
        else:
            leg_updates = {"nA": avg_no, "pB": avg_yes}
            # Mirror: pA is same_title's reference quote. Nothing sizes on it
            # (_kelly_p uses the fixed co-resolution prior here) and it is NOT
            # guarded above, but it is the operand of the reported pA - pB
            # "Price Diff", which otherwise subtracts a scan-time quote from a
            # depth-weighted fill and can print a negative gap for a pair that
            # passed the finder's directional filter. Refreshed for coherence.
            if ref_yes is not None:
                leg_updates["pA"] = ref_yes
        enriched.append(dc_replace(
            pair,
            tradeable=new_tradeable,
            # How many contracts the prices just written are valid for — the
            # bound compute_trade's own depth clamp then honours
            max_contracts=cap,
            depth_levels=depth_levels,
            **leg_updates,
        ))

    tradeable_after = sum(1 for p in enriched if p.tradeable)
    logging.info(
        "Orderbook depth check: %d/%d pairs remain tradeable after price gap filter",
        tradeable_after,
        sum(1 for p in pairs if p.tradeable),
    )
    return enriched


def _books_client(books: dict):
    """
    Mock KalshiClient serving orderbook_fp books by ticker.

    Args:
        books (dict): ticker -> {"yes_dollars": [...], "no_dollars": [...]}.
            A ticker not in it answers with no orderbook key, so the fetch
            logs a WARNING and returns None (a missing book).

    Returns:
        MagicMock: The client; its get_market_orderbook_without_preload_content
            mock records each fetch.
    """
    def fake_orderbook(ticker):
        payload = {"orderbook_fp": books[ticker]} if ticker in books else {"unknown": {}}
        return SimpleNamespace(status=200, data=json.dumps(payload).encode("utf-8"))

    client = MagicMock()
    client.get_market_orderbook_without_preload_content = MagicMock(side_effect=fake_orderbook)
    return client


def _bids(*levels: tuple) -> list:
    """Bid levels [[price, qty], ...] as dollar strings, from (price, qty) pairs."""
    return [[str(round(price, 4)), str(qty)] for price, qty in levels]


def _ts_books(levels, *, pB_ref=None, late="LATE", ref_qty=1000,
              a_yes_bid=_AT_THE_ASK) -> dict:
    """
    Time-series books on EARLY and `late`: YES on EARLY at each pA, NO on `late` at each nB.

    Args:
        levels (list): (pA, nB, qty) fills; EARLY's NO bids and the later
            market's YES bids are their complements.
        pB_ref (float | None): The later market's YES ask (a NO bid at
            1 - pB_ref); None leaves its NO side empty.
        late (str): The later market's ticker.
        ref_qty (float): Contracts resting at the reference ask.
        a_yes_bid (float | None): EARLY's YES bid; by default its best YES
            ask (a top with no width), None for no YES bid.

    Returns:
        dict: ticker -> orderbook_fp side dict.
    """
    a_bid = min(a for a, _, _ in levels) if a_yes_bid is _AT_THE_ASK else a_yes_bid
    return {
        "EARLY": {"yes_dollars": [] if a_bid is None else _bids((a_bid, ref_qty)),
                  "no_dollars": _bids(*((1 - a, q) for a, _, q in levels))},
        late: {"yes_dollars": _bids(*((1 - b, q) for _, b, q in levels)),
               "no_dollars": [] if pB_ref is None else _bids((1 - pB_ref, ref_qty))},
    }


def _st_books(nA, pB, *, qty=100, pA_ref=None) -> dict:
    """Same-title books: NO on A1 at nA, YES on B1 at pB; A1's YES ask is pA_ref."""
    return {
        "A1": {"yes_dollars": _bids((1 - nA, qty)),
               "no_dollars": [] if pA_ref is None else _bids((1 - pA_ref, qty))},
        "B1": {"yes_dollars": [], "no_dollars": _bids((1 - pB, qty))},
    }


def _mid_books(*, a_yes_bids=((0.27, 100),), a_no_bids=((0.70, 100),),
               b_yes_bids=((0.52, 100),), b_no_bids=((0.36, 100),)) -> dict:
    """
    Two-sided time-series books on EARLY and LATE, each side's bids as (price, qty).

    EARLY's NO bids become the YES leg's asks and its YES bids its NO asks;
    LATE's YES bids become the NO leg's asks and its NO bids its YES ask (the
    reference). The defaults: EARLY YES ask 0.30, NO ask 0.73 (YES bid 0.27,
    midpoint 0.285); LATE YES ask 0.64, NO ask 0.48 (YES bid 0.52, midpoint
    0.58). The mid spread is 0.295 against an ask gap of 0.34.

    Returns:
        dict: ticker -> orderbook_fp side dict, for _books_client.
    """
    return {"EARLY": {"yes_dollars": _bids(*a_yes_bids), "no_dollars": _bids(*a_no_bids)},
            "LATE": {"yes_dollars": _bids(*b_yes_bids), "no_dollars": _bids(*b_no_bids)}}


# The default _mid_books tops, each ask computed as the book reader computes
# it (1 - the bid), so the expected mid spread is the code's to the last bit
_MID_TOPS = (1.0 - 0.70, 1.0 - 0.27, 1.0 - 0.36, 1.0 - 0.52)


def _asymmetric_fee(price_a, price_b):
    """A fake fee: none when the first price is the lower, $1 otherwise.

    The edge cut asks it (YES price, NO price) and the profitability check
    (NO price, YES price), so a pair whose YES leg is the cheaper one keeps
    its levels and then fails profitability, a path no real fee reaches.
    """
    return 0.0 if price_a <= price_b else 1.0


def _ts_ladder():
    """A ladder closing at one instant whose STATED gap is 19 days (the 0.70 ceiling)."""
    return dataclasses.replace(_ts_candidate(gap_days=0, pA=0.30, pB=0.60, nB=0.45),
                               stated_gap_days=19)


def _late2(pair):
    """The same time-series pair with its later market renamed LATE2."""
    return dataclasses.replace(
        pair, market_b=SimpleNamespace(**{**vars(pair.market_b), "ticker": "LATE2"}))


# One case: id -> (pairs, books, portfolio value in cents, settings, cash in
# cents, {scanner name: stand-in}, the refusal _enrich_pair names for a
# single tradeable pair, or None)
def _enrichment_cases() -> dict:
    ts = _ts_candidate(gap_days=10, pA=0.30, pB=0.60, nB=0.50)
    st = _st_candidate(pA=0.60, pB=0.31, nA=0.44)
    tiers_off = _live(tier_floors=False)
    k_one = _live(interval_discount=1.0, size_cap=1.0)
    big = _AMPLE_BALANCE_CENTS
    fee = {"fee_per_pair_approx": _asymmetric_fee}

    def spread(code):
        return {"time_series_spread_refusal": lambda spread, gap, settings: code}

    return {
        "ts-tradeable": ([ts], _ts_books([(0.30, 0.50, 100)], pB_ref=0.65),
                         big, _live(), None, {}, None),
        # EARLY has YES bids too, so the mid spread reads both of its sides;
        # every scan-time quote of the candidate differs from its book's top,
        # so a mid spread read off the candidate would not match the oracle's
        "ts-two-sided-books": ([_ts_candidate(gap_days=10, pA=0.28, pB=0.66, nB=0.46)],
                               _mid_books(), big, _live(), None, {}, None),
        "ts-no-edge-after-fee": (
            [_ts_candidate(gap_days=10, pA=0.53, pB=0.56, nB=0.45)],
            _ts_books([(0.53, 0.45, 100)], pB_ref=0.56), big, tiers_off, None, {},
            scanner.ENRICH_NO_EDGE_AFTER_FEE),
        "ts-ladder-no-qualifying": (
            [_ts_ladder()], _ts_books([(0.30, 0.45, 100)], pB_ref=0.60), big, _live(),
            None, {}, scanner.ENRICH_NO_QUALIFYING),
        "ts-empty-book": (
            [ts], {"EARLY": {"yes_dollars": [], "no_dollars": []},
                   "LATE": {"yes_dollars": _bids((0.50, 100)), "no_dollars": []}},
            big, _live(), None, {}, scanner.ENRICH_NO_QUALIFYING),
        "ts-k-of-one": ([ts], _ts_books([(0.30, 0.50, 100)], pB_ref=0.62), big, k_one,
                        None, {}, scanner.ENRICH_UNAFFORDABLE),
        "ts-cash-binds": ([ts], _ts_books([(0.30, 0.50, 100)], pB_ref=0.62), 1_000_000,
                          _live(), 10, {}, scanner.ENRICH_UNAFFORDABLE),
        "ts-small-budget": ([ts], _ts_books([(0.30, 0.50, 100)], pB_ref=0.62), 50,
                            _live(), None, {}, scanner.ENRICH_UNAFFORDABLE),
        "ts-under-one-contract": ([ts], _ts_books([(0.30, 0.50, 0.5)], pB_ref=0.62), big,
                                  _live(), None, {}, scanner.ENRICH_THIN_BOOK),
        # Thin and no cash: the budget is named first
        "ts-thin-and-no-cash": ([ts], _ts_books([(0.30, 0.50, 0.5)], pB_ref=0.62), big,
                                _live(), 0, {}, scanner.ENRICH_UNAFFORDABLE),
        "ts-no-reference": ([ts], _ts_books([(0.30, 0.50, 100)]), big, _live(), None, {},
                            scanner.ENRICH_NO_REFERENCE),
        "ts-crossed": ([_ts_candidate(gap_days=10, pA=0.30, pB=0.60, nB=0.40)],
                       _ts_books([(0.30, 0.40, 100)], pB_ref=0.55), big, _live(), None, {},
                       scanner.ENRICH_CROSSED),
        # The earlier book crossed (YES ask 0.30 under its 0.35 YES bid), the
        # later one not
        "ts-earlier-crossed": ([_ts_candidate(gap_days=10, pA=0.30, pB=0.64, nB=0.48)],
                               _mid_books(a_yes_bids=((0.35, 100),)), big, _live(), None,
                               {}, scanner.ENRICH_CROSSED),
        "ts-above-ceiling": (
            [_ts_candidate(gap_days=10, pA=0.20, pB=0.72, nB=0.28)],
            _ts_books([(0.20, 0.28, 10), (0.30, 0.28, 90)], pB_ref=0.72), big,
            _TIERS_OFF_HALF, None, {}, config.SPREAD_ABOVE_CEILING),
        "ts-not-positive": ([ts], _ts_books([(0.30, 0.50, 100)], pB_ref=0.65), big, _live(),
                            None, spread(config.SPREAD_NOT_POSITIVE),
                            config.SPREAD_NOT_POSITIVE),
        "ts-below-floor": ([ts], _ts_books([(0.30, 0.50, 100)], pB_ref=0.65), big, _live(),
                           None, spread(config.SPREAD_BELOW_FLOOR), config.SPREAD_BELOW_FLOOR),
        "ts-unprofitable": ([ts], _ts_books([(0.30, 0.50, 100)], pB_ref=0.65), big, _live(),
                            None, fee, scanner.ENRICH_UNPROFITABLE),
        # Both checks fail: the first one names the refusal, and both lines are logged
        "ts-crossed-and-unprofitable": (
            [_ts_candidate(gap_days=10, pA=0.30, pB=0.60, nB=0.40)],
            _ts_books([(0.30, 0.40, 100)], pB_ref=0.55), big, _live(), None, fee,
            scanner.ENRICH_CROSSED),
        "ts-cut-at-the-fee": (
            [_ts_candidate(gap_days=10, pA=0.30, pB=0.56, nB=0.45)],
            {"EARLY": {"yes_dollars": [], "no_dollars": _bids((0.70, 100), (0.47, 2000))},
             "LATE": {"yes_dollars": _bids((0.55, 2100)), "no_dollars": _bids((0.44, 1000))}},
            1_000_000, tiers_off, None, {}, None),
        "st-tradeable": ([st], _st_books(0.44, 0.31, pA_ref=0.60), big, _live(), None, {},
                         None),
        "st-no-reference-kept": ([st], _st_books(0.44, 0.31), big, _live(), None, {}, None),
        "st-no-qualifying": ([_st_candidate(pA=0.60, pB=0.40, nA=0.60)],
                             _st_books(0.60, 0.40, pA_ref=0.60), big, _live(), None, {},
                             scanner.ENRICH_NO_QUALIFYING),
        "st-no-cash": ([st], _st_books(0.44, 0.31, pA_ref=0.60), big, _live(), 0, {},
                       scanner.ENRICH_UNAFFORDABLE),
        "st-unprofitable": ([st], _st_books(0.44, 0.31, pA_ref=0.60), big, _live(), None,
                            fee, scanner.ENRICH_UNPROFITABLE),
        # EARLY's book serves both time-series pairs, fetched once
        "shared-book": (
            [ts, _late2(ts), st],
            {**_ts_books([(0.30, 0.50, 100)], pB_ref=0.65),
             **_ts_books([(0.30, 0.50, 100)], pB_ref=0.85, late="LATE2"),
             **_st_books(0.44, 0.31, pA_ref=0.60)},
            big, _TIERS_OFF_HALF, None, {}, None),
        # LATE2 has no book; the pair before it is passed through untouched
        "missing-book": (
            [dataclasses.replace(st, tradeable=False), _late2(ts), ts],
            _ts_books([(0.30, 0.50, 100)], pB_ref=0.65), big, _live(), None, {}, None),
    }


_ENRICHMENT_CASES = _enrichment_cases()


class TestEnrichPairMatchesTheOldLoop:
    """scanner._enrich_pair is enrich_with_orderbook_prices' old per-pair loop,
    moved: the live function returns the same pairs, logs the same lines at the
    same levels and fetches each book once, on every refusal path; and
    _enrich_pair names each refusal and sends its own lines to `log`."""

    @staticmethod
    def _patch(monkeypatch, stand_ins: dict) -> None:
        """Replace names in scanner and in this module, where the oracle reads them."""
        module = sys.modules[__name__]
        for name, value in stand_ins.items():
            monkeypatch.setattr(scanner, name, value)
            monkeypatch.setattr(module, name, value)

    @staticmethod
    def _run(caplog, enrich, case):
        """Run one enrichment function on a fresh client; (pairs, records, fetches)."""
        pairs, books, value, settings, cash, _stand_ins, _code = case
        client = _books_client(books)
        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            out = enrich(client, pairs, value, settings=settings, cash_cents=cash)
        records = [(r.levelno, r.getMessage()) for r in caplog.records]
        fetches = [c.kwargs["ticker"] for c in
                   client.get_market_orderbook_without_preload_content.call_args_list]
        return out, records, fetches

    @pytest.mark.parametrize("name", sorted(_ENRICHMENT_CASES))
    def test_the_live_function_matches_the_old_one(self, caplog, monkeypatch, name):
        case = _ENRICHMENT_CASES[name]
        self._patch(monkeypatch, case[5])
        old, old_records, old_fetches = self._run(caplog, _old_enrich_with_orderbook_prices,
                                                  case)
        new, new_records, new_fetches = self._run(caplog, enrich_with_orderbook_prices, case)
        assert len(new) == len(old) == len(case[0])
        for before, after, given in zip(old, new, case[0], strict=True):
            for f in dataclasses.fields(CandidatePair):
                a, b = getattr(before, f.name), getattr(after, f.name)
                assert type(a) is type(b) and a == b, (name, f.name, a, b)
            # A pair passed through is the same object in both
            assert (before is given) == (after is given)
        assert new_records == old_records
        assert new_fetches == old_fetches
        # Each book is fetched once
        assert len(new_fetches) == len(set(new_fetches))

    @pytest.mark.parametrize("name", sorted(n for n, c in _ENRICHMENT_CASES.items()
                                            if len(c[0]) == 1))
    def test_enrich_pair_names_the_refusal_and_logs_only_to_log(self, caplog, monkeypatch,
                                                                name):
        case = _ENRICHMENT_CASES[name]
        pairs, books, value, settings, cash, stand_ins, code = case
        self._patch(monkeypatch, stand_ins)
        [pair] = pairs
        [expected], old_records, _fetches = self._run(
            caplog, _old_enrich_with_orderbook_prices, case)
        client = _books_client(books)
        ob_a = _fetch_orderbook(client, pair.market_a.ticker)
        ob_b = _fetch_orderbook(client, pair.market_b.ticker)
        lines = []

        def log(level, msg, *args):
            lines.append((level, msg % args))

        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            priced, refusal = scanner._enrich_pair(pair, ob_a, ob_b, value, settings=settings,
                                                   cash_cents=cash, log=log)
        # These books have no unusable level, so nothing reaches the logging
        # module when log is given
        assert caplog.records == []
        assert priced == expected
        assert refusal == code
        assert (refusal is None) == priced.tradeable
        # log got every line the old loop logged for this pair, in order: all
        # but the closing summary
        assert lines == old_records[:-1]
        assert old_records[-1][1].startswith("Orderbook depth check:")

    def test_every_refusal_code_is_reached(self):
        codes = {c[6] for c in _ENRICHMENT_CASES.values() if len(c[0]) == 1}
        assert codes == {
            None, scanner.ENRICH_NO_EDGE_AFTER_FEE, scanner.ENRICH_NO_QUALIFYING,
            scanner.ENRICH_UNAFFORDABLE, scanner.ENRICH_THIN_BOOK, scanner.ENRICH_NO_REFERENCE,
            scanner.ENRICH_CROSSED, scanner.ENRICH_UNPROFITABLE, config.SPREAD_NOT_POSITIVE,
            config.SPREAD_BELOW_FLOOR, config.SPREAD_ABOVE_CEILING}
        # Each code is its own short string
        enrich_codes = [scanner.ENRICH_NO_EDGE_AFTER_FEE, scanner.ENRICH_NO_QUALIFYING,
                        scanner.ENRICH_UNAFFORDABLE, scanner.ENRICH_THIN_BOOK,
                        scanner.ENRICH_NO_REFERENCE, scanner.ENRICH_CROSSED,
                        scanner.ENRICH_UNPROFITABLE]
        spread_codes = [config.SPREAD_NOT_POSITIVE, config.SPREAD_BELOW_FLOOR,
                        config.SPREAD_ABOVE_CEILING]
        assert len(set(enrich_codes + spread_codes)) == 10

    def test_an_unusable_book_level_is_still_logged_by_the_book_reader(self, caplog):
        # log gets _enrich_pair's own lines; the book reader's WARNING about a
        # level it drops (here a NO bid of zero contracts) goes to logging
        pair = _ts_candidate(gap_days=10, pA=0.30, pB=0.60, nB=0.50)
        books = _ts_books([(0.30, 0.50, 100)])
        books["EARLY"]["no_dollars"].append(["0.6000", "0"])
        client = _books_client(books)
        ob_a = _fetch_orderbook(client, "EARLY")
        ob_b = _fetch_orderbook(client, "LATE")
        lines = []

        def log(level, msg, *args):
            lines.append((level, msg % args))

        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            _priced, refusal = scanner._enrich_pair(
                pair, ob_a, ob_b, _AMPLE_BALANCE_CENTS, settings=_live(), cash_cents=None,
                log=log)
        assert refusal == scanner.ENRICH_NO_REFERENCE
        [record] = caplog.records
        assert record.levelno == logging.WARNING
        assert record.getMessage().startswith("Orderbook for EARLY: dropped 1 of 2 bid levels")
        [(level, line)] = lines
        assert level == logging.WARNING
        assert "the later contract has no YES ask on its book" in line

    def test_enrich_pair_logs_through_logging_by_default(self, caplog):
        # With no log handed in, the lines go to the logging module, as live
        pair = _ts_candidate(gap_days=10, pA=0.30, pB=0.60, nB=0.50)
        client = _books_client(_ts_books([(0.30, 0.50, 100)]))
        ob_a = _fetch_orderbook(client, "EARLY")
        ob_b = _fetch_orderbook(client, "LATE")
        with caplog.at_level(logging.INFO):
            _priced, refusal = scanner._enrich_pair(
                pair, ob_a, ob_b, _AMPLE_BALANCE_CENTS, settings=_live(), cash_cents=None)
        assert refusal == scanner.ENRICH_NO_REFERENCE
        [record] = caplog.records
        assert record.levelno == logging.WARNING
        assert "the later contract has no YES ask on its book" in record.getMessage()


class TestPairMidSpread:
    """pair_mid_spread reads CandidatePair.mid_spread by type: only a finite
    int or float above PRICE_EPSILON counts, and anything else reads as None
    (no mid spread to use)."""

    def test_a_real_spread_is_returned_as_a_float(self):
        assert pair_mid_spread(SimpleNamespace(mid_spread=0.31)) == 0.31
        one = pair_mid_spread(SimpleNamespace(mid_spread=1))
        assert one == 1.0 and type(one) is float
        # Just above the tolerance counts
        assert pair_mid_spread(SimpleNamespace(mid_spread=2 * PRICE_EPSILON)) == 2 * PRICE_EPSILON

    @pytest.mark.parametrize("value", [
        None, True, False, "0.31", Decimal("0.31"), MagicMock(),
        float("nan"), float("inf"), float("-inf"),
        0.0, PRICE_EPSILON, -0.05,
    ], ids=["none", "true", "false", "str", "decimal", "mock", "nan", "inf", "-inf",
            "zero", "epsilon", "negative"])
    def test_anything_else_reads_as_none(self, value):
        assert pair_mid_spread(SimpleNamespace(mid_spread=value)) is None

    def test_a_pair_never_priced_off_its_books_has_none(self):
        # The field's default, what the finders leave; an object without the
        # field; and a MagicMock pair, whose truthy auto-attribute is no number
        mA, mB = _ts_pair_markets(gap_days=10, pA=0.30, pB=0.60)
        [found] = find_time_series_pairs(MagicMock(), held_tickers=set(), markets=[mA, mB],
                                         settings=_live())
        assert found.mid_spread is None and pair_mid_spread(found) is None
        assert pair_mid_spread(SimpleNamespace()) is None
        assert pair_mid_spread(MagicMock()) is None


class TestEnrichmentWritesTheMidSpread:
    """Enrichment writes a time-series pair's mid spread from the tops of the
    two books it fetched (config.time_series_mid_spread): the earlier
    market's YES ask and NO ask (its YES-bid side, one of the two sides
    neither leg buys from, which the earlier-book crossed check reads too)
    and the later market's. It reads the tops, never the averaged
    fills nor the pair's own scan-time quotes; a missing YES bid counts as a
    bid of 0; a same-title pair, and a pair refused before it is priced or
    with no later YES ask, carry none. Nothing else about the pair changes."""

    @staticmethod
    def _pair():
        return _ts_candidate(gap_days=10, pA=0.30, pB=0.64, nB=0.48)

    def test_the_pair_s_own_quotes_are_not_read(self):
        # Every scan-time quote of this candidate (YES asks 0.28 and 0.66, NO
        # asks 0.72 and 0.46) differs from its book's top (0.30 and 0.64,
        # 0.73 and 0.48): the mid spread comes from the books alone
        stale = _ts_candidate(gap_days=10, pA=0.28, pB=0.66, nB=0.46)
        assert (stale.pA, stale.nA, stale.pB, stale.nB) == (0.28, 0.72, 0.66, 0.46)
        [enriched] = enrich_with_orderbook_prices(_books_client(_mid_books()), [stale],
                                                  _AMPLE_BALANCE_CENTS, settings=_live())
        assert enriched.tradeable is True
        assert enriched.mid_spread == time_series_mid_spread(*_MID_TOPS)
        assert enriched.mid_spread == pytest.approx(0.295, abs=1e-9)

    def test_the_mid_spread_comes_from_the_tops_of_both_books(self):
        client = _books_client(_mid_books())
        [enriched] = enrich_with_orderbook_prices(client, [self._pair()], _AMPLE_BALANCE_CENTS,
                                                  settings=_live())
        assert enriched.tradeable is True
        assert enriched.mid_spread == time_series_mid_spread(*_MID_TOPS)
        # mid B 0.58 - mid A 0.285, not the ask gap 0.64 - 0.30
        assert enriched.mid_spread == pytest.approx(0.295)
        assert enriched.mid_spread == pytest.approx((0.64 + 0.52) / 2 - (0.30 + 0.27) / 2)
        assert enriched.pB - enriched.pA == pytest.approx(0.34)
        assert pair_mid_spread(enriched) == enriched.mid_spread
        # Read off the two books already fetched: no extra request
        assert client.get_market_orderbook_without_preload_content.call_count == 2

    def test_no_yes_bid_on_the_earlier_book_counts_as_a_bid_of_zero(self):
        # EARLY has no YES bids: its NO ask reads 1.0 and its midpoint is its
        # YES ask / 2 = 0.15, so the spread is 0.58 - 0.15
        client = _books_client(_mid_books(a_yes_bids=()))
        [enriched] = enrich_with_orderbook_prices(client, [self._pair()], _AMPLE_BALANCE_CENTS,
                                                  settings=_live())
        assert enriched.tradeable is True
        assert enriched.mid_spread == time_series_mid_spread(_MID_TOPS[0], 1.0, *_MID_TOPS[2:])
        assert enriched.mid_spread == pytest.approx(0.43)

    def test_deep_books_are_read_at_their_tops(self):
        # Three YES-leg levels and two NO-leg levels, all qualifying, and an
        # ample budget: the leg prices average down the book, the mid spread
        # still reads each side's best level. EARLY's YES bids arrive out of
        # order; its best is 0.27
        books = _mid_books(a_yes_bids=((0.25, 50), (0.27, 20), (0.20, 30)),
                           a_no_bids=((0.70, 10), (0.68, 40), (0.66, 50)),
                           b_yes_bids=((0.52, 10), (0.50, 90)),
                           b_no_bids=((0.36, 5), (0.34, 100)))
        [enriched] = enrich_with_orderbook_prices(_books_client(books), [self._pair()],
                                                  _AMPLE_BALANCE_CENTS, settings=_live())
        assert enriched.tradeable is True
        # (10 x 0.30 + 40 x 0.32 + 50 x 0.34) / 100 and (10 x 0.48 + 90 x 0.50) / 100
        assert enriched.pA == pytest.approx(0.328)
        assert enriched.nB == pytest.approx(0.498)
        assert enriched.mid_spread == time_series_mid_spread(*_MID_TOPS)

    def test_a_same_title_pair_carries_none(self):
        st = _st_candidate(pA=0.60, pB=0.31, nA=0.44)
        [enriched] = enrich_with_orderbook_prices(
            _books_client(_st_books(0.44, 0.31, pA_ref=0.60)), [st], _AMPLE_BALANCE_CENTS,
            settings=_live())
        assert enriched.tradeable is True
        assert enriched.mid_spread is None

    @pytest.mark.parametrize("books, cash, code", [
        # 0.45 + 0.48 sits over the 0.85 price-sum ceiling: nothing qualifies
        (_mid_books(a_no_bids=((0.55, 100),)), None, scanner.ENRICH_NO_QUALIFYING),
        # No cash: not one contract pair is affordable
        (_mid_books(), 0, scanner.ENRICH_UNAFFORDABLE),
        # No YES ask on the later book: nothing to read its midpoint from
        (_mid_books(b_no_bids=()), None, scanner.ENRICH_NO_REFERENCE),
    ], ids=["no-qualifying", "unaffordable", "no-later-yes-ask"])
    def test_a_refused_pair_with_nothing_to_read_carries_none(self, books, cash, code):
        client = _books_client(books)
        ob_a, ob_b = _fetch_orderbook(client, "EARLY"), _fetch_orderbook(client, "LATE")
        priced, refusal = scanner._enrich_pair(self._pair(), ob_a, ob_b, _AMPLE_BALANCE_CENTS,
                                               settings=_live(), cash_cents=cash)
        assert refusal == code
        assert priced.tradeable is False
        assert priced.mid_spread is None

    def test_the_earlier_yes_bids_change_nothing_but_the_mid_spread(self, caplog):
        # The same pair on the same books with and without EARLY's YES bids:
        # every other field, and every log line, is the same
        runs = []
        for a_yes_bids in (((0.27, 100),), ()):
            caplog.clear()
            with caplog.at_level(logging.DEBUG):
                [enriched] = enrich_with_orderbook_prices(
                    _books_client(_mid_books(a_yes_bids=a_yes_bids)), [self._pair()],
                    _AMPLE_BALANCE_CENTS, settings=_live())
            runs.append((enriched, [(r.levelno, r.getMessage()) for r in caplog.records]))
        (with_bids, lines), (without, lines_without) = runs
        assert with_bids.mid_spread != without.mid_spread
        assert dc_replace(with_bids, mid_spread=None) == dc_replace(without, mid_spread=None)
        assert lines == lines_without

    def test_an_unusable_yes_bid_on_the_earlier_book_is_named_by_the_book_reader(self, caplog):
        # A YES bid of zero contracts on EARLY is dropped with the book
        # reader's one summary WARNING; the mid spread reads the level left
        books = _mid_books(a_yes_bids=((0.27, 100), (0.40, 0)))
        with caplog.at_level(logging.INFO):
            [enriched] = enrich_with_orderbook_prices(_books_client(books), [self._pair()],
                                                      _AMPLE_BALANCE_CENTS, settings=_live())
        assert enriched.mid_spread == time_series_mid_spread(*_MID_TOPS)
        [warning] = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning.getMessage().startswith("Orderbook for EARLY: dropped 1 of 2 bid levels")


class TestValidatePairPriceSpreadRule:
    """validate_pair_price on a FRESH book: fail closed with no later YES ask,
    the top-of-book ceiling, and the fee cut before any depth is counted."""

    def test_a_fresh_spread_over_the_ceiling_is_dropped(self, caplog):
        # Fill 0.30, fresh reference 0.85: 0.55 at the top of the book, over 0.5
        pair = _ts_candidate(gap_days=10, pA=0.30, pB=0.60, nB=0.30)
        spec = SimpleNamespace(pair=pair, x=10)
        client = _ts_orderbook_client(pA_fill=0.30, nB_fill=0.30, pB_ref=0.85)
        with caplog.at_level(logging.INFO):
            assert validate_pair_price(client, spec, settings=_TIERS_OFF_HALF) is False
        [line] = [r for r in caplog.records if "exceeds the spread band's 0.5 ceiling" in
                  r.getMessage()]
        assert line.levelno == logging.WARNING
        assert "0.5500" in line.getMessage()
        # Nothing changes at a 1.0 ceiling: same book, no band
        assert validate_pair_price(client, spec, settings=_live(tier_floors=False)) is True
        assert validate_pair_price(client, spec, settings=_live()) is True

    def test_the_ceiling_is_tested_on_the_fresh_top_of_the_book(self, caplog):
        # Sized at a YES fill of 0.35; the fresh book's top (0.20 under a 0.75
        # reference) spreads 0.55, over 0.5, while the spec's fill and the
        # deeper level spread 0.40. The fresh top decides.
        pair = _ts_candidate(gap_days=10, pA=0.35, pB=0.75, nB=0.31)
        spec = SimpleNamespace(pair=pair, x=10)
        levels = [(0.20, 0.30, 10), (0.35, 0.31, 90)]
        with caplog.at_level(logging.INFO):
            assert validate_pair_price(_ts_multilevel_client(levels, pB_ref=0.75), spec,
                                       settings=_TIERS_OFF_HALF) is False
        [line] = [r for r in caplog.records if "exceeds the spread band's 0.5 ceiling" in
                  r.getMessage()]
        assert line.levelno == logging.WARNING
        assert "0.5500" in line.getMessage()
        # control: the same fresh book passes at a 1.0 ceiling
        assert validate_pair_price(_ts_multilevel_client(levels, pB_ref=0.75), spec,
                                   settings=_live(tier_floors=False)) is True

    def test_no_later_yes_ask_fails_closed(self, caplog):
        pair = _ts_candidate(gap_days=10, pA=0.30, pB=0.60, nB=0.40)
        spec = SimpleNamespace(pair=pair, x=10)
        client = _ts_orderbook_client(pA_fill=0.30, nB_fill=0.40)
        with caplog.at_level(logging.INFO):
            assert validate_pair_price(client, spec, settings=_live()) is False
        [line] = [r for r in caplog.records if "has no YES ask on its book now" in
                  r.getMessage()]
        assert line.levelno == logging.WARNING
        # control: the same book with a reference passes
        assert validate_pair_price(
            _ts_orderbook_client(pA_fill=0.30, nB_fill=0.40, pB_ref=0.60), spec,
            settings=_live()) is True

    def test_a_fresh_book_with_no_edge_after_the_fee_is_dropped(self, caplog):
        # A thin-edge spec sized at 0.01 / 0.97, tiers off; the book moves a tick
        # to 0.02 / 0.98, inside the 1.0 ceiling and both FoK caps, so only the
        # fee cut drops it (filled there, every settlement cell loses)
        pair = _ts_candidate(gap_days=10, pA=0.01, pB=0.03, nB=0.97)
        spec = SimpleNamespace(pair=pair, x=748)
        moved = _ts_orderbook_client(pA_fill=0.02, nB_fill=0.98, qty=5000, pB_ref=0.03)
        with caplog.at_level(logging.INFO):
            assert validate_pair_price(moved, spec, settings=_TIERS_OFF_HALF) is False
        [line] = [r for r in caplog.records if "keeps an edge after the fee" in r.getMessage()]
        assert line.levelno == logging.WARNING
        assert line.getMessage() == (
            f"Pre-execution check failed for '{pair.canonical_title}' — no contract pair "
            "on the book now keeps an edge after the fee; dropping")
        assert [r for r in caplog.records if "gap no longer qualifies" in r.getMessage()] == []
        # control: the book it was sized on (a 0.02 edge, ~0.003 fee) passes
        unmoved = _ts_orderbook_client(pA_fill=0.01, nB_fill=0.97, qty=5000, pB_ref=0.03)
        assert validate_pair_price(unmoved, spec, settings=_TIERS_OFF_HALF) is True
        # control: with the tiers on the 0.85 ceiling refuses the moved book itself
        caplog.clear()
        with caplog.at_level(logging.INFO):
            assert validate_pair_price(moved, spec, settings=_live()) is False
        assert [r for r in caplog.records if "gap no longer qualifies" in r.getMessage()]

    def test_depth_is_counted_only_over_levels_with_an_edge(self, caplog):
        # 100 contracts at 0.30 + 0.45 and 900 no-edge ones at 0.53 + 0.46 (0.99,
        # inside the 1.0 ceiling): a spec of 150 needs the no-edge ones and drops
        pair = _ts_candidate(gap_days=10, pA=0.52, pB=0.60, nB=0.45)
        levels = [(0.30, 0.45, 100), (0.53, 0.46, 900)]
        with caplog.at_level(logging.INFO):
            assert validate_pair_price(_ts_multilevel_client(levels, pB_ref=0.60),
                                       SimpleNamespace(pair=pair, x=150),
                                       settings=_live(tier_floors=False)) is False
        [line] = [r for r in caplog.records if "reachable at the FoK limit" in r.getMessage()]
        assert "only 100.0 contracts" in line.getMessage()
        assert validate_pair_price(_ts_multilevel_client(levels, pB_ref=0.60),
                                   SimpleNamespace(pair=pair, x=100),
                                   settings=_live(tier_floors=False)) is True

    def test_the_cut_is_inert_under_a_tier_on_ceiling(self):
        # Tiers on: every level under the 0.85 ceiling keeps more edge than any
        # fee, so a spec reaching the one on it (0.36 + 0.49) counts all 100
        pair = _ts_candidate(gap_days=10, pA=0.35, pB=0.62, nB=0.48)
        levels = [(0.30, 0.45, 10), (0.36, 0.49, 90)]
        assert validate_pair_price(_ts_multilevel_client(levels, pB_ref=0.62),
                                   SimpleNamespace(pair=pair, x=100), settings=_live()) is True

    def test_same_title_reads_no_reference(self):
        # Same-title reads no reference: with A's NO side empty it still passes
        pair = _st_candidate(pA=0.55, pB=0.30, nA=0.45, nB=0.70)
        client = _st_orderbook_client(nA_fill=0.44, pB_fill=0.31, qty=100)
        assert validate_pair_price(client, SimpleNamespace(pair=pair, x=100),
                                   settings=_TIERS_OFF_HALF) is True


class TestTimeSeriesBestPairPerGroup:
    @pytest.mark.usefixtures("pre_toggle_defaults")
    def test_group_of_three_keeps_largest_later_minus_earlier_gap(self):
        # (pre_toggle_defaults: with the tier floors off MID -> LATE qualifies too)
        # Three contracts on one normalized title, all within the short tier:
        # EARLY→MID gap 0.20 and EARLY→LATE gap 0.30 both qualify, MID→LATE
        # (0.10) does not. One pair per group survives — the largest pB - pA.
        # Each title names its own deadline (they normalize to one key), so
        # the one-series conjunct added for DR-02/DR-54 never fires here.
        early_close = datetime(2026, 3, 1, tzinfo=UTC)
        markets = [
            _mock_market(ticker="EARLY", event_ticker="EVT-A",
                         title="Will BTC exceed $80k by March 01, 2026",
                         yes_ask=0.30, no_ask=0.70, close_time=early_close),
            _mock_market(ticker="MID", event_ticker="EVT-M",
                         title="Will BTC exceed $80k by March 06, 2026",
                         yes_ask=0.50, no_ask=0.50, close_time=early_close + timedelta(days=5)),
            _mock_market(ticker="LATE", event_ticker="EVT-B",
                         title="Will BTC exceed $80k by March 11, 2026",
                         yes_ask=0.60, no_ask=0.40, close_time=early_close + timedelta(days=10)),
        ]
        pairs = find_time_series_pairs(MagicMock(), held_tickers=set(), markets=markets)
        assert len(pairs) == 1
        [pair] = pairs
        assert (pair.market_a.ticker, pair.market_b.ticker) == ("EARLY", "LATE")
        assert pair.pB - pair.pA == pytest.approx(0.30)
        assert pair.nB == pytest.approx(0.40)
        assert pair.tradeable is True

    def test_tradeable_pair_outranks_wider_untradeable_gap(self):
        # A wide-book later contract (NO ask 0.80) has the bigger YES-ask gap
        # but no win scenario covers pA + nB = 1.10; the narrower tradeable
        # pair wins the group's slot.
        early_close = datetime(2026, 3, 1, tzinfo=UTC)
        markets = [
            _mock_market(ticker="EARLY", event_ticker="EVT-A",
                         title="Will BTC exceed $80k by March 01, 2026",
                         yes_ask=0.30, no_ask=0.70, close_time=early_close),
            _mock_market(ticker="MID", event_ticker="EVT-M",
                         title="Will BTC exceed $80k by March 06, 2026",
                         yes_ask=0.50, no_ask=0.50, close_time=early_close + timedelta(days=5)),
            _mock_market(ticker="WIDE", event_ticker="EVT-W",
                         title="Will BTC exceed $80k by March 11, 2026",
                         yes_ask=0.70, no_ask=0.80, close_time=early_close + timedelta(days=10)),
        ]
        [pair] = find_time_series_pairs(MagicMock(), held_tickers=set(), markets=markets)
        assert (pair.market_a.ticker, pair.market_b.ticker) == ("EARLY", "MID")
        assert pair.tradeable is True


def _btc_market(ticker, event_ticker, day, *, strike="$80k", yes_ask, no_ask):
    """One Bitcoin question at a March deadline, one event per deadline."""
    return _ingest_market(ticker, event_ticker,
                          f"Will BTC exceed {strike} by March {day}, 2026?", "Bitcoin record",
                          close=f"2026-03-{day:02d}T00:00:00Z", yes_ask=yes_ask, no_ask=no_ask)


_HELD_LINE = ("Time-series candidates refused because one of their markets is on a "
              "ladder we already hold (counted before every other check): ")


class TestFinderRefusesHeldLadders:
    """No new time-series pair may use a market on a ladder we already hold.
    Each test scans with the held market left out of the list, as a live run
    does."""

    @staticmethod
    def _scan(markets, **kwargs):
        # Explicit settings, so no test depends on the toggles config.py ships
        return find_time_series_pairs(MagicMock(), held_tickers=set(), markets=markets,
                                      settings=_live(), **kwargs)

    def _btc_family(self):
        """Three deadlines of one question, each in its own event."""
        early = _btc_market("BTC-MAR01", "KXBTCMAX-26MAR01", 1, yes_ask="0.20", no_ask="0.80")
        mid = _btc_market("BTC-MAR06", "KXBTCMAX-26MAR06", 6, yes_ask="0.35", no_ask="0.65")
        late = _btc_market("BTC-MAR11", "KXBTCMAX-26MAR11", 11, yes_ask="0.70", no_ask="0.30")
        return early, mid, late

    def test_a_held_rung_refuses_every_pair_of_its_ladder(self, monkeypatch, caplog):
        monkeypatch.setattr(scanner, "TIME_SERIES_SAME_EVENT_LADDERS", True)
        held = _ladder_rung("RUNG-HELD", "by March 10, 2026", yes_ask=0.40, no_ask=0.60,
                            close=datetime(2026, 3, 10, tzinfo=UTC))
        rungs = [
            _ladder_rung("RUNG-EARLY", "by March 1, 2026", yes_ask=0.20, no_ask=0.80,
                         close=datetime(2026, 3, 1, tzinfo=UTC)),
            _ladder_rung("RUNG-MID", "by March 15, 2026", yes_ask=0.45, no_ask=0.55,
                         close=datetime(2026, 3, 15, tzinfo=UTC)),
            _ladder_rung("RUNG-LATE", "by March 20, 2026", yes_ask=0.60, no_ask=0.40,
                         close=datetime(2026, 3, 20, tzinfo=UTC)),
        ]
        _assert_one_ladder_group(held, rungs[0])
        # control: the rungs we do not hold pair on their own
        assert len(self._scan(rungs)) == 1
        with caplog.at_level(logging.INFO):
            assert self._scan(rungs, held_ladders=market_ladder_keys(held)) == []
        # All three candidates of the ladder are refused and counted
        assert _HELD_LINE + "3" in caplog.text

    def test_a_held_question_blocks_a_relisting_worded_slightly_differently(
            self, monkeypatch, caplog):
        monkeypatch.setattr(scanner, "TIME_SERIES_SAME_EVENT_LADDERS", True)
        # The exchange listed one question twice; the first listing drops "the"
        held = _ingest_market("KXVOTESAVEAMERICA-26-MAR20", "KXVOTESAVEAMERICA-26",
                              "Will the Senate vote on SAVE America Act?",
                              "When will the Senate vote on the SAVE America Act?",
                              subtitle="Before Mar 20, 2026", close="2026-03-20T00:00:00Z")
        rungs = [_ingest_market(f"KXVOTESAVEAMERICA-26MAR-{day}", "KXVOTESAVEAMERICA-26MAR",
                                "Will the Senate vote on the SAVE America Act?",
                                "When will the Senate vote on the SAVE America Act?",
                                subtitle=f"Before {label}, 2026", close=close,
                                yes_ask=yes, no_ask=f"{1 - float(yes):.2f}")
                 for day, label, close, yes in (
                     ("MAR24", "Mar 24", "2026-03-24T00:00:00Z", "0.20"),
                     ("APR01", "Apr 1", "2026-04-01T00:00:00Z", "0.45"))]
        # control: the relisting's rungs pair on their own
        assert len(self._scan(rungs)) == 1
        with caplog.at_level(logging.INFO):
            assert self._scan(rungs, held_ladders=market_ladder_keys(held)) == []
        assert _HELD_LINE + "1" in caplog.text

    def test_a_held_question_refuses_a_family_listed_as_one_event_per_deadline(self, caplog):
        early, mid, late = self._btc_family()
        # The same question at a later deadline, in an event not in this run's list
        held = _btc_market("BTC-MAR21", "KXBTCMAX-26MAR21", 21, yes_ask="0.80", no_ask="0.20")
        assert market_ladder_keys(held) & market_ladder_keys(early) == frozenset({
            _question_label(early),
        })
        with caplog.at_level(logging.INFO):
            assert self._scan([early, mid, late], held_ladders=market_ladder_keys(held)) == []
        assert _HELD_LINE + "3" in caplog.text

    def test_a_held_event_blocks_its_own_markets_and_promotes_the_runner_up(self, caplog):
        early, mid, late = self._btc_family()
        # control: the widest pair uses the Mar 1 market
        [best] = self._scan([early, mid, late])
        assert (best.market_a.ticker, best.market_b.ticker) == ("BTC-MAR01", "BTC-MAR11")
        # We hold a different question in the Mar 1 event
        held = _btc_market("BTC-90K-MAR01", "KXBTCMAX-26MAR01", 1, strike="$90k",
                           yes_ask="0.10", no_ask="0.90")
        assert _question_of(held) != _question_of(early)
        with caplog.at_level(logging.INFO):
            [pair] = self._scan([early, mid, late], held_ladders=market_ladder_keys(held))
        # The two candidates on the Mar 1 market are refused, and the next best wins
        assert (pair.market_a.ticker, pair.market_b.ticker) == ("BTC-MAR06", "BTC-MAR11")
        assert _HELD_LINE + "2" in caplog.text

    def test_a_held_event_of_the_later_market_blocks_it_too(self, caplog):
        early, mid, late = self._btc_family()
        # We hold a different question in the Mar 11 event, the later market's
        held = _btc_market("BTC-90K-MAR11", "KXBTCMAX-26MAR11", 11, strike="$90k",
                           yes_ask="0.10", no_ask="0.90")
        assert _question_of(held) != _question_of(late)
        with caplog.at_level(logging.INFO):
            [pair] = self._scan([early, mid, late], held_ladders=market_ladder_keys(held))
        # Both candidates using the Mar 11 market are refused; the Mar 1 / Mar 6 pair wins
        assert (pair.market_a.ticker, pair.market_b.ticker) == ("BTC-MAR01", "BTC-MAR06")
        assert _HELD_LINE + "2" in caplog.text

    def test_an_empty_held_set_changes_nothing(self, caplog):
        markets = list(self._btc_family())
        with caplog.at_level(logging.INFO):
            plain = self._scan(markets)
            plain_log = caplog.text
            caplog.clear()
            explicit = self._scan(markets, held_ladders=frozenset())
        assert explicit == plain and plain
        assert caplog.text == plain_log
        # Silent at zero
        assert "on a ladder we already hold" not in plain_log

    def test_an_unrelated_held_ladder_refuses_nothing_and_logs_nothing(self, caplog):
        markets = list(self._btc_family())
        unrelated = _ingest_market("RAIN-1", "KXRAIN-1", "Will it rain in NYC by March 1, 2026?",
                                   "NYC weather")
        with caplog.at_level(logging.INFO):
            pairs = self._scan(markets, held_ladders=market_ladder_keys(unrelated))
        assert pairs == self._scan(markets)
        assert "on a ladder we already hold" not in caplog.text

    def test_the_check_runs_before_every_other_check(self, monkeypatch, caplog):
        # With the same-event switch off these rungs would be counted as held
        # back by the switch; on a held ladder they are counted here instead
        monkeypatch.setattr(scanner, "TIME_SERIES_SAME_EVENT_LADDERS", False)
        early = _ladder_rung("RUNG-EARLY", "by March 1, 2026", yes_ask=0.20, no_ask=0.80,
                             close=datetime(2026, 3, 1, tzinfo=UTC))
        late = _ladder_rung("RUNG-LATE", "by March 20, 2026", yes_ask=0.60, no_ask=0.40,
                            close=datetime(2026, 3, 20, tzinfo=UTC))
        # A third rung of the same ladder, the one we hold
        held = _ladder_rung("RUNG-HELD", "by March 10, 2026", yes_ask=0.40, no_ask=0.60,
                            close=datetime(2026, 3, 10, tzinfo=UTC))
        with caplog.at_level(logging.INFO):
            assert self._scan([early, late], held_ladders=market_ladder_keys(held)) == []
        assert _HELD_LINE + "1" in caplog.text
        assert "same-event deadline ladders are disabled" not in caplog.text


_SIDE_LINE = ("Time-series candidates refused for buying a held pair's sides the wrong "
              "way round: ")
_ADD_ON_LINE = "Time-series pairs that add to a held pair: "


def _add_on(*sides: tuple, count: float = 30.0) -> HeldPair:
    """A held pair the run may add to, holding `sides` ((ticker, side) each)."""
    return HeldPair(sides=tuple(sorted(sides)), count=count, cost_dollars=18.9,
                    value_dollars=18.0, fees_dollars=0.9)


class TestFinderAddsToHeldPairs:
    """find_time_series_pairs lets through exactly the held pair a run may
    add to (add_on_pairs): the same two tickers, buying the side already held
    on each, judged after a ladder's legs are ordered. Every other candidate
    on the held ladder is still refused, so the add-on can never stack a new
    rung beside the held pair."""

    @staticmethod
    def _scan(markets, **kwargs):
        # Explicit settings, so no test depends on the toggles config.py ships
        return find_time_series_pairs(MagicMock(), held_tickers=set(), markets=markets,
                                      settings=_live(), **kwargs)

    @staticmethod
    def _rungs(monkeypatch, *, early_close=datetime(2026, 3, 1, tzinfo=UTC),
               late_close=datetime(2026, 3, 20, tzinfo=UTC)):
        """Three rungs of one ladder; the account holds YES on the first and NO on the last."""
        monkeypatch.setattr(scanner, "TIME_SERIES_SAME_EVENT_LADDERS", True)
        early = _ladder_rung("RUNG-A", "by March 1, 2026", yes_ask=0.20, no_ask=0.80,
                             close=early_close)
        mid = _ladder_rung("RUNG-B", "by March 10, 2026", yes_ask=0.40, no_ask=0.60,
                           close=datetime(2026, 3, 10, tzinfo=UTC))
        late = _ladder_rung("RUNG-C", "by March 20, 2026", yes_ask=0.60, no_ask=0.40,
                            close=late_close)
        _assert_one_ladder_group(early, late)
        _assert_one_ladder_group(early, mid)
        return early, mid, late

    @staticmethod
    def _held(early, late):
        """The held ladders and the one exact held pair, YES early / NO late."""
        held = market_ladder_keys(early) | market_ladder_keys(late)
        pair = _add_on((early.ticker, "yes"), (late.ticker, "no"))
        return held, {frozenset({early.ticker, late.ticker}): pair}

    def test_the_exact_held_pair_is_emitted_and_the_rest_of_its_ladder_refused(
            self, monkeypatch, caplog):
        early, mid, late = self._rungs(monkeypatch)
        held, add_ons = self._held(early, late)
        with caplog.at_level(logging.INFO):
            [pair] = self._scan([early, mid, late], held_ladders=held, add_on_pairs=add_ons)
        assert (pair.market_a.ticker, pair.market_b.ticker) == ("RUNG-A", "RUNG-C")
        [add_on] = add_ons.values()
        assert pair.held is add_on and pair_held(pair) is add_on
        # The two candidates using the unheld middle rung are refused and counted
        assert _HELD_LINE + "2" in caplog.text
        assert _ADD_ON_LINE + "1" in caplog.text

    def test_without_add_ons_every_candidate_on_the_ladder_is_refused(
            self, monkeypatch, caplog):
        early, mid, late = self._rungs(monkeypatch)
        held, _ = self._held(early, late)
        with caplog.at_level(logging.INFO):
            assert self._scan([early, mid, late], held_ladders=held) == []
        assert _HELD_LINE + "3" in caplog.text
        assert _ADD_ON_LINE not in caplog.text

    def test_a_held_pair_the_other_way_round_is_refused(self, monkeypatch, caplog):
        early, mid, late = self._rungs(monkeypatch)
        held = market_ladder_keys(early) | market_ladder_keys(late)
        # The account holds NO on the earlier rung and YES on the later one
        add_ons = {frozenset({early.ticker, late.ticker}):
                   _add_on((early.ticker, "no"), (late.ticker, "yes"))}
        with caplog.at_level(logging.INFO):
            assert self._scan([early, mid, late], held_ladders=held,
                              add_on_pairs=add_ons) == []
        assert _SIDE_LINE + "1" in caplog.text
        assert _HELD_LINE + "2" in caplog.text
        assert _ADD_ON_LINE + "0" in caplog.text

    @pytest.mark.parametrize("with_held_ladders", [False, True])
    def test_an_empty_add_on_set_changes_nothing(self, monkeypatch, caplog,
                                                 with_held_ladders):
        early, mid, late = self._rungs(monkeypatch)
        held = (market_ladder_keys(early) | market_ladder_keys(late)
                if with_held_ladders else frozenset())
        markets = [early, mid, late]
        with caplog.at_level(logging.INFO):
            plain = self._scan(markets, held_ladders=held)
            plain_log = caplog.text
            caplog.clear()
            explicit = self._scan(markets, held_ladders=held, add_on_pairs={})
        assert explicit == plain
        assert caplog.text == plain_log
        assert "held pair" not in plain_log

    def test_a_held_pairs_market_pairs_only_with_its_partner_without_held_ladders(
            self, monkeypatch, caplog):
        # The finder refuses on its own any other candidate touching a market
        # of a held pair, even when handed no held ladders: here the pair is
        # held the other way round, so it is refused on its sides, and the
        # unheld middle rung must not pair with either held rung instead
        early, mid, late = self._rungs(monkeypatch)
        add_ons = {frozenset({early.ticker, late.ticker}):
                   _add_on((early.ticker, "no"), (late.ticker, "yes"))}
        with caplog.at_level(logging.INFO):
            assert self._scan([early, mid, late], add_on_pairs=add_ons) == []
        # RUNG-A/RUNG-B and RUNG-B/RUNG-C, counted with the held-ladder refusals
        assert _HELD_LINE + "2" in caplog.text
        assert _SIDE_LINE + "1" in caplog.text
        # Non-vacuous: with no held pair at all, the rungs pair among themselves
        assert self._scan([early, mid, late])

    def test_the_held_pair_survives_enrichment(self):
        # Enrichment rewrites the pair through dc_replace, which must carry
        # the held pair on to the sizer
        pair = _ts_candidate(gap_days=10, pA=0.30, pB=0.62, nB=0.50)
        held = _add_on((pair.market_a.ticker, "yes"), (pair.market_b.ticker, "no"))
        pair.held = held
        client = _ts_orderbook_client(pA_fill=0.32, nB_fill=0.42, qty=40, pB_ref=0.62)
        [enriched] = enrich_with_orderbook_prices(client, [pair], _AMPLE_BALANCE_CENTS)
        assert enriched.tradeable
        assert enriched.held is held and pair_held(enriched) is held

    def test_the_sides_are_checked_after_a_ladder_swaps_its_legs(self, monkeypatch):
        # The earlier deadline closes last, so the close-time sort puts the
        # later rung first and the ladder branch swaps them back
        early, mid, late = self._rungs(monkeypatch,
                                       early_close=datetime(2026, 3, 25, tzinfo=UTC),
                                       late_close=datetime(2026, 3, 5, tzinfo=UTC))
        assert early.close_time > late.close_time
        held, add_ons = self._held(early, late)
        [pair] = self._scan([early, mid, late], held_ladders=held, add_on_pairs=add_ons)
        assert (pair.market_a.ticker, pair.market_b.ticker) == ("RUNG-A", "RUNG-C")
        assert pair.stated_gap_days == 19
        assert pair_held(pair) is not None


def _lone(ticker: str, side: str, labels: frozenset, count: float = 30.0) -> HeldPair:
    """A lone held leg (its partner paid out) the run may add to."""
    return HeldPair(sides=((ticker, side),), count=count, cost_dollars=12.5,
                    value_dollars=13.5, fees_dollars=0.5, labels=labels)


class TestFinderAddsToALoneLeg:
    """find_time_series_pairs lets a new pair add to a lone held leg (its
    partner paid out): the held market bought on its held side, beside a
    market the account does not hold that sits on no held ladder but the
    leg's own. Every other candidate on the leg's ladder is still refused."""

    _scan = staticmethod(TestFinderAddsToHeldPairs._scan)

    @staticmethod
    def _rungs(monkeypatch, *, mid_event="KXSTARSHIP-14"):
        """Three rungs of one question; the account holds a lone leg on the last."""
        monkeypatch.setattr(scanner, "TIME_SERIES_SAME_EVENT_LADDERS", True)
        early = _ladder_rung("RUNG-A", "by March 1, 2026", yes_ask=0.20, no_ask=0.80,
                             close=datetime(2026, 3, 1, tzinfo=UTC))
        mid = _ladder_rung("RUNG-B", "by March 10, 2026", event=mid_event,
                           yes_ask=0.40, no_ask=0.60, close=datetime(2026, 3, 10, tzinfo=UTC))
        late = _ladder_rung("RUNG-C", "by March 20, 2026", yes_ask=0.60, no_ask=0.40,
                            close=datetime(2026, 3, 20, tzinfo=UTC))
        _assert_one_ladder_group(early, late)
        # One question, though the middle rung may be listed under another event
        assert time_series_group_key(pair_key(mid), mid.subtitle) == \
            time_series_group_key(pair_key(late), late.subtitle)
        return early, mid, late

    @staticmethod
    def _lone_late(late, side="no"):
        """The held ladders and the lone leg on the last rung."""
        labels = market_ladder_keys(late)
        return labels, {frozenset({late.ticker}): _lone(late.ticker, side, labels)}

    def test_a_new_rung_adds_to_the_lone_leg(self, monkeypatch, caplog):
        # The first rung's partner paid out; the run pairs the middle rung,
        # not held, with the held NO on the last
        _early, mid, late = self._rungs(monkeypatch)
        held, add_ons = self._lone_late(late)
        with caplog.at_level(logging.INFO):
            [pair] = self._scan([mid, late], held_ladders=held, add_on_pairs=add_ons)
        assert (pair.market_a.ticker, pair.market_b.ticker) == ("RUNG-B", "RUNG-C")
        assert pair.held is add_ons[frozenset({"RUNG-C"})]
        assert _ADD_ON_LINE + "1" in caplog.text

    def test_two_unheld_rungs_still_never_pair_on_the_held_ladder(self, monkeypatch, caplog):
        # Both other rungs may pair with the lone leg; the pair of the two
        # unheld rungs is refused, and the widest add-on wins the group
        early, mid, late = self._rungs(monkeypatch)
        held, add_ons = self._lone_late(late)
        with caplog.at_level(logging.INFO):
            [pair] = self._scan([early, mid, late], held_ladders=held, add_on_pairs=add_ons)
        assert (pair.market_a.ticker, pair.market_b.ticker) == ("RUNG-A", "RUNG-C")
        assert pair_held(pair) is not None
        assert _HELD_LINE + "1" in caplog.text

    def test_a_lone_leg_held_on_the_other_side_is_refused(self, monkeypatch, caplog):
        # YES held on the last rung: every candidate would buy NO there
        early, mid, late = self._rungs(monkeypatch)
        held, add_ons = self._lone_late(late, side="yes")
        with caplog.at_level(logging.INFO):
            assert self._scan([early, mid, late], held_ladders=held,
                              add_on_pairs=add_ons) == []
        assert _SIDE_LINE + "2" in caplog.text
        assert _HELD_LINE + "1" in caplog.text

    def test_a_new_market_on_another_held_ladder_is_refused(self, monkeypatch, caplog):
        # The middle rung is listed under another event, which another held
        # market sits on: adding there would stack a trade beside that one
        _early, mid, late = self._rungs(monkeypatch, mid_event="KXOTHER-14")
        held, add_ons = self._lone_late(late)
        other_held = held | {("event", "KXOTHER-14")}
        assert ("event", "KXOTHER-14") in market_ladder_keys(mid)
        with caplog.at_level(logging.INFO):
            assert self._scan([mid, late], held_ladders=other_held, add_on_pairs=add_ons) == []
        assert _HELD_LINE + "1" in caplog.text
        # CONTROL: with only the lone leg's own ladders held, the pair forms
        [pair] = self._scan([mid, late], held_ladders=held, add_on_pairs=add_ons)
        assert pair_held(pair) is not None


class TestAddOnFor:
    """_add_on_for finds what a candidate adds to: an exact pair by its two
    tickers, a lone leg by its one, never a candidate touching two held
    markets that are not one exact pair."""

    _PAIR = HeldPair(sides=(("P-A", "yes"), ("P-B", "no")), count=5.0, cost_dollars=3.0,
                     value_dollars=3.0, fees_dollars=0.1)
    _LONE = _lone("L-1", "no", frozenset({("event", "L")}))
    _ADD_ONS = {frozenset({"P-A", "P-B"}): _PAIR, frozenset({"L-1"}): _LONE,
                frozenset({"M-1"}): _lone("M-1", "yes", frozenset({("event", "M")}))}
    _TICKERS = {"P-A", "P-B", "L-1", "M-1"}

    def _find(self, a, b):
        return scanner._add_on_for(self._ADD_ONS, self._TICKERS, a, b)

    def test_an_exact_pair_by_its_two_tickers(self):
        assert self._find("P-B", "P-A") is self._PAIR

    def test_a_lone_leg_beside_a_new_market(self):
        assert self._find("NEW-1", "L-1") is self._LONE
        assert self._find("L-1", "NEW-1") is self._LONE

    @pytest.mark.parametrize("a, b", [
        ("L-1", "M-1"),     # two lone legs
        ("L-1", "P-A"),     # a lone leg and a pair's market
        ("P-A", "NEW-1"),   # a pair's market beside a new one
        ("L-1", "L-1"),     # one ticker named twice
        ("NEW-1", "NEW-2"),
    ])
    def test_anything_else_adds_to_nothing(self, a, b):
        assert self._find(a, b) is None


_ST_HELD_LINE = ("Same-title candidates refused because they pair a held pair's market "
                 "with another market: ")
_ST_SIDE_LINE = ("Same-title candidates refused for buying a held pair's sides the wrong "
                 "way round: ")


def _st_market(ticker: str, event_ticker: str, yes_ask: float, no_ask: float):
    """One listing of a question four series ask, each closing at one instant."""
    return _mock_market(ticker=ticker, event_ticker=event_ticker,
                        title="Will the Fed cut rates in March?", subtitle="Yes",
                        event_title="Fed decision", yes_ask=yes_ask, no_ask=no_ask)


class TestSameTitleAddsToHeldPairs:
    """find_same_title_pairs lets through exactly the held pair a run may add
    to, buying NO on the market held NO and YES on the one held YES; any other
    candidate touching one of its markets is refused before the group contest,
    so an unheld pair of the same group can still win."""

    # The account holds NO on X and YES on Y
    _ADD_ONS = {frozenset({"X-1", "Y-1"}): _add_on(("X-1", "no"), ("Y-1", "yes"), count=10.0)}

    @staticmethod
    def _held_pair(x_yes: float = 0.60, y_yes: float = 0.40):
        """The held pair's two markets, X pricier by default."""
        return (_st_market("X-1", "KXA-1", x_yes, round(1.02 - x_yes, 2)),
                _st_market("Y-1", "KXB-1", y_yes, round(1.02 - y_yes, 2)))

    @staticmethod
    def _other_pair(z_yes: float, w_yes: float):
        """An unheld pair asking the same question on two more series."""
        return (_st_market("Z-1", "KXC-1", z_yes, round(1.02 - z_yes, 2)),
                _st_market("W-1", "KXD-1", w_yes, round(1.02 - w_yes, 2)))

    def test_the_exact_held_pair_is_emitted(self):
        [pair] = find_same_title_pairs(list(self._held_pair()), add_on_pairs=self._ADD_ONS)
        assert (pair.market_a.ticker, pair.market_b.ticker) == ("X-1", "Y-1")
        assert pair.held is self._ADD_ONS[frozenset({"X-1", "Y-1"})]

    def test_an_unheld_pair_with_the_wider_gap_takes_the_groups_slot(self, caplog):
        markets = [*self._held_pair(), *self._other_pair(0.75, 0.45)]
        with caplog.at_level(logging.INFO):
            [pair] = find_same_title_pairs(markets, add_on_pairs=self._ADD_ONS)
        assert (pair.market_a.ticker, pair.market_b.ticker) == ("Z-1", "W-1")
        assert pair_held(pair) is None
        # X-Z, X-W, Y-Z and Y-W are refused before the contest
        assert _ST_HELD_LINE + "4" in caplog.text

    def test_the_held_pair_with_the_wider_gap_takes_the_groups_slot(self, caplog):
        markets = [*self._held_pair(), *self._other_pair(0.55, 0.45)]
        with caplog.at_level(logging.INFO):
            [pair] = find_same_title_pairs(markets, add_on_pairs=self._ADD_ONS)
        assert (pair.market_a.ticker, pair.market_b.ticker) == ("X-1", "Y-1")
        assert pair_held(pair) is not None
        assert _ST_HELD_LINE + "4" in caplog.text

    def test_crossed_prices_would_buy_the_held_pair_the_other_way_round(self, caplog):
        # Y is the pricier market now, so the pair would buy NO on Y
        markets = [*self._held_pair(x_yes=0.40, y_yes=0.60), *self._other_pair(0.55, 0.45)]
        with caplog.at_level(logging.INFO):
            [pair] = find_same_title_pairs(markets, add_on_pairs=self._ADD_ONS)
        assert (pair.market_a.ticker, pair.market_b.ticker) == ("Z-1", "W-1")
        assert _ST_SIDE_LINE + "1" in caplog.text

    def test_a_lone_leg_pairs_with_a_market_not_held(self):
        # X paid out; YES is still held on Y, which pairs with Z (NO on the
        # pricier Z, YES on Y)
        lone = {frozenset({"Y-1"}): _lone("Y-1", "yes", frozenset(), count=10.0)}
        _x, y = self._held_pair()
        z, _w = self._other_pair(0.60, 0.45)
        [pair] = find_same_title_pairs([y, z], add_on_pairs=lone)
        assert (pair.market_a.ticker, pair.market_b.ticker) == ("Z-1", "Y-1")
        assert pair.held is lone[frozenset({"Y-1"})]

    def test_a_lone_leg_bought_on_the_other_side_is_refused(self, caplog):
        # Y is the pricier market now, so the pair would buy NO on Y
        lone = {frozenset({"Y-1"}): _lone("Y-1", "yes", frozenset(), count=10.0)}
        _x, y = self._held_pair(y_yes=0.60)
        z, _w = self._other_pair(0.40, 0.45)
        with caplog.at_level(logging.INFO):
            assert find_same_title_pairs([y, z], add_on_pairs=lone) == []
        assert _ST_SIDE_LINE + "1" in caplog.text

    def test_an_empty_add_on_set_changes_nothing(self, caplog):
        markets = [*self._held_pair(), *self._other_pair(0.75, 0.45)]
        with caplog.at_level(logging.INFO):
            plain = find_same_title_pairs(markets)
            plain_log = caplog.text
            caplog.clear()
            explicit = find_same_title_pairs(markets, add_on_pairs={})
        assert explicit == plain
        assert caplog.text == plain_log
        assert "held pair" not in plain_log


class TestLegHelpers:
    """leg_sides / leg_prices / deadline_gap_days are the cross-module contract
    for which side each leg buys and what it costs."""

    def test_leg_sides_time_series(self):
        assert leg_sides("time_series") == TIME_SERIES_LEG_SIDES == ("yes", "no")

    def test_leg_sides_same_title(self):
        assert leg_sides("same_title") == SAME_TITLE_LEG_SIDES == ("no", "yes")

    def test_leg_sides_fail_safe_to_same_title(self):
        # None, a bogus string, and a MagicMock auto-attribute all resolve to
        # the same-title sides — an unknown pair type must never be traded as
        # the directional time-series bet
        assert leg_sides(None) == SAME_TITLE_LEG_SIDES
        assert leg_sides("bogus") == SAME_TITLE_LEG_SIDES
        assert leg_sides(MagicMock().pair_type) == SAME_TITLE_LEG_SIDES

    def test_leg_prices_time_series_real_pair(self):
        pair = _ts_candidate(gap_days=10, pA=0.30, pB=0.60, nB=0.40)
        assert leg_prices(pair) == (0.30, 0.40)

    def test_leg_prices_same_title_real_pair(self):
        mA = _mock_market(ticker="A1", event_ticker="EVT-A", title="Q", yes_ask=0.55, no_ask=0.45)
        mB = _mock_market(ticker="B1", event_ticker="EVT-B", title="Q", yes_ask=0.30, no_ask=0.70)
        pair = CandidatePair(
            market_a=mA, market_b=mB, pA=0.55, pB=0.30, nA=0.45,
            tradeable=True, canonical_title="Q", pair_type="same_title", nB=0.70,
        )
        assert leg_prices(pair) == (0.45, 0.30)

    def test_leg_prices_simple_namespace(self):
        ts = SimpleNamespace(pair_type="time_series", pA=0.30, pB=0.60, nA=0.70, nB=0.40)
        st = SimpleNamespace(pair_type="same_title", pA=0.30, pB=0.60, nA=0.70, nB=0.40)
        untyped = SimpleNamespace(pA=0.30, pB=0.60, nA=0.70, nB=0.40)
        assert leg_prices(ts) == (0.30, 0.40)
        assert leg_prices(st) == (0.70, 0.60)
        assert leg_prices(untyped) == (0.70, 0.60)

    def test_leg_prices_reads_nB_directly(self):
        # No getattr default: a time-series stub without nB must fail loudly
        # rather than price the NO leg at a placeholder
        stub = SimpleNamespace(pair_type="time_series", pA=0.30, pB=0.60, nA=0.70)
        with pytest.raises(AttributeError):
            leg_prices(stub)

    def test_deadline_gap_days_is_symmetric(self):
        mA = SimpleNamespace(close_time=datetime(2026, 3, 1, tzinfo=UTC))
        mB = SimpleNamespace(close_time=datetime(2026, 3, 21, tzinfo=UTC))
        assert deadline_gap_days(mA, mB) == 20
        assert deadline_gap_days(mB, mA) == 20

    def test_deadline_gap_days_whole_days_23h_to_01h(self):
        # 2026-03-01 23:00Z → 2026-03-17 01:00Z is 15 days 2 hours: .days == 15,
        # so this pair sits in the short tier (inclusive boundary) either way round
        mA = SimpleNamespace(close_time=datetime(2026, 3, 1, 23, 0, tzinfo=UTC))
        mB = SimpleNamespace(close_time=datetime(2026, 3, 17, 1, 0, tzinfo=UTC))
        assert deadline_gap_days(mA, mB) == 15
        assert deadline_gap_days(mB, mA) == 15

    def test_deadline_gap_days_zero_for_same_close(self):
        m = SimpleNamespace(close_time=datetime(2026, 3, 1, tzinfo=UTC))
        assert deadline_gap_days(m, m) == 0

    @pytest.mark.usefixtures("pre_toggle_defaults")
    def test_scanner_and_ceiling_share_the_gap(self):
        # _pair_max_sum tiers off the same order-independent gap the finder
        # used, so a 16-day pair gets the long-tier ceiling from either side
        pair = _ts_candidate(gap_days=16, pA=0.30, pB=0.65, nB=0.40)
        settings = config.live_settings()
        assert scanner._pair_max_sum(pair, settings) == pytest.approx(0.70)
        swapped = dataclasses.replace(pair, market_a=pair.market_b, market_b=pair.market_a)
        assert scanner._pair_max_sum(swapped, settings) == pytest.approx(0.70)


def _orderbook_payload_client(payload: dict):
    """Mock KalshiClient whose orderbook endpoint returns `payload` (serialized to
    raw JSON bytes) for any ticker — for exercising _fetch_orderbook key handling
    directly with arbitrary response shapes."""
    client = MagicMock()
    client.get_market_orderbook_without_preload_content = MagicMock(
        side_effect=lambda ticker: SimpleNamespace(
            status=200, data=json.dumps(payload).encode("utf-8")
        )
    )
    return client


class TestFetchOrderbookKeyMapping:
    """_fetch_orderbook must couple the container key to its OWN candidate side-key
    sets: orderbook_fp -> yes_dollars/no_dollars (dollars); orderbook ->
    yes_dollars/no_dollars (dollars) or the SDK-aliased legacy true/false arrays
    (integer cents, converted to dollars before parsing). A null/empty container is
    an empty book; a non-empty container matching none of its own sets is a
    potential API-shape mismatch — logged and treated as unavailable (None), never
    cross-read against the other generation."""

    def test_current_format_parses(self):
        # orderbook_fp container with dollar-string side arrays (the live shape)
        payload = {"orderbook_fp": {"yes_dollars": [["0.50", "10"]],
                                    "no_dollars": [["0.40", "5"]]}}
        ob = _fetch_orderbook(_orderbook_payload_client(payload), "MKT-CURRENT")
        assert ob == {"yes": [["0.50", "10"]], "no": [["0.40", "5"]]}

    def test_orderbook_container_dollar_keys_parse(self):
        # The SDK's Orderbook model REQUIRES yes_dollars/no_dollars even under the
        # plain `orderbook` container — that shape must parse identically.
        payload = {"orderbook": {"yes_dollars": [["0.50", "10"]],
                                 "no_dollars": [["0.40", "5"]]}}
        ob = _fetch_orderbook(_orderbook_payload_client(payload), "MKT-OB-DOLLARS")
        assert ob == {"yes": [["0.50", "10"]], "no": [["0.40", "5"]]}

    def test_orderbook_container_legacy_cent_keys_convert_to_dollars(self, caplog):
        # The legacy integer-cent arrays are aliased "true"/"false" by the SDK.
        # A 45c YES bid must become a 0.45 dollar bid -> NO ask at 0.55, and
        # NOTHING may be silently dropped (cents through the dollars parser would
        # yield 1-45 = -44 and be discarded with no warning at all).
        payload = {"orderbook": {"true": [[45, 100], [40, 50]],
                                 "false": [[30, 20]]}}
        with caplog.at_level(logging.WARNING):
            ob = _fetch_orderbook(_orderbook_payload_client(payload), "MKT-CENTS")
        assert ob == {"yes": [[0.45, 100], [0.40, 50]], "no": [[0.30, 20]]}
        assert caplog.text == ""
        # Downstream complement: YES bid 0.45 -> NO ask 0.55 at the same qty
        assert _bids_to_ask_levels(ob["yes"]) == [(0.55, 100.0), (0.60, 50.0)]

    def test_orderbook_container_legacy_cent_string_prices_convert(self):
        # Cents may arrive as strings ("45"); still integral cents, still converted
        payload = {"orderbook": {"true": [["45", "100"]], "false": []}}
        ob = _fetch_orderbook(_orderbook_payload_client(payload), "MKT-CENTSTR")
        assert ob == {"yes": [[0.45, "100"]], "no": []}

    def test_malformed_cent_level_is_dropped_with_warning(self, caplog):
        # 4500 is not a whole cent in [1, 99] — the level is dropped LOUDLY,
        # naming the ticker and the raw value; valid levels still come through.
        payload = {"orderbook": {"true": [[45, 100], [4500, 7]], "false": []}}
        with caplog.at_level(logging.WARNING):
            ob = _fetch_orderbook(_orderbook_payload_client(payload), "MKT-BADCENTS")
        assert ob == {"yes": [[0.45, 100]], "no": []}
        assert "legacy cents array" in caplog.text
        assert "MKT-BADCENTS" in caplog.text
        assert "4500" in caplog.text

    def test_orderbook_container_with_removed_yes_no_keys_is_a_mismatch(self, caplog):
        # Regression for BS-03: `orderbook` + yes/no was never a real SDK shape
        # (the model aliases the legacy arrays as true/false). It is no longer a
        # recognized generation, so it must fail closed rather than parse.
        payload = {"orderbook": {"yes": [["0.50", "10"]], "no": [["0.40", "5"]]}}
        with caplog.at_level(logging.WARNING):
            ob = _fetch_orderbook(_orderbook_payload_client(payload), "MKT-LEGACY")
        assert ob is None
        assert "Potential orderbook key mismatch" in caplog.text
        assert "MKT-LEGACY" in caplog.text

    def test_mixed_generation_keys_return_none_and_warn(self, caplog):
        # orderbook_fp container but with the OTHER generation's side keys —
        # must not cross-read; returns None and logs a key-mismatch warning.
        payload = {"orderbook_fp": {"true": [[50, 10]], "false": [[40, 5]]}}
        with caplog.at_level(logging.WARNING):
            ob = _fetch_orderbook(_orderbook_payload_client(payload), "MKT-MIXED")
        assert ob is None
        assert "Potential orderbook key mismatch" in caplog.text
        assert "MKT-MIXED" in caplog.text

    def test_unrecognized_side_keys_return_none_and_warn(self, caplog):
        # A non-empty dict container matching NEITHER candidate set fails closed
        payload = {"orderbook": {"bids_yes": [["0.50", "10"]]}}
        with caplog.at_level(logging.WARNING):
            ob = _fetch_orderbook(_orderbook_payload_client(payload), "MKT-NOSET")
        assert ob is None
        assert "Potential orderbook key mismatch" in caplog.text

    def test_non_dict_container_returns_none_and_warns(self, caplog):
        # A container that is present and non-empty but is not a dict at all
        # (here a bare list of levels) can't be probed for side keys. It must
        # fail closed exactly like an unrecognized key set, and the warning
        # names the TYPE rather than a key list — the branch that logs
        # type(ob).__name__ instead of sorted(ob.keys()).
        payload = {"orderbook_fp": [["0.50", "10"]]}
        with caplog.at_level(logging.WARNING):
            ob = _fetch_orderbook(_orderbook_payload_client(payload), "MKT-NOTDICT")
        assert ob is None
        assert "Potential orderbook key mismatch" in caplog.text
        assert "MKT-NOTDICT" in caplog.text
        assert "list" in caplog.text

    def test_empty_and_null_sides_are_not_a_mismatch(self, caplog):
        # Side keys present but empty ([]) or null map to empty sides, NOT a
        # mismatch — a market with no resting bids on a side is normal.
        payload = {"orderbook_fp": {"yes_dollars": [], "no_dollars": None}}
        with caplog.at_level(logging.WARNING):
            ob = _fetch_orderbook(_orderbook_payload_client(payload), "MKT-EMPTY")
        assert ob == {"yes": [], "no": []}
        assert "key mismatch" not in caplog.text

    def test_null_container_is_an_empty_book(self, caplog):
        # BS-28: container key present but null — that is a book with no resting
        # bids at all, not an API shape change. Empty book, no warning.
        payload = {"orderbook_fp": None}
        with caplog.at_level(logging.WARNING):
            ob = _fetch_orderbook(_orderbook_payload_client(payload), "MKT-NULLC")
        assert ob == {"yes": [], "no": []}
        assert caplog.text == ""

    def test_empty_dict_container_is_an_empty_book(self, caplog):
        # Same for an empty-dict container, under either generation
        payload = {"orderbook": {}}
        with caplog.at_level(logging.WARNING):
            ob = _fetch_orderbook(_orderbook_payload_client(payload), "MKT-EMPTYC")
        assert ob == {"yes": [], "no": []}
        assert caplog.text == ""

    def test_unknown_container_key_returns_none_and_warns(self, caplog):
        # No recognized container key at all — log the actual response keys so a
        # future rename is diagnosable rather than a silent 0-trade run.
        payload = {"orderbook_v2": {"yes_dollars": [["0.50", "10"]]}}
        with caplog.at_level(logging.WARNING):
            ob = _fetch_orderbook(_orderbook_payload_client(payload), "MKT-UNKNOWN")
        assert ob is None
        assert "No usable orderbook" in caplog.text
        assert "orderbook_v2" in caplog.text

    def test_mismatch_marks_pair_untradeable(self):
        # End-to-end: a mismatched orderbook makes enrich_with_orderbook_prices
        # mark the pair non-tradeable (both legs resolve to None depth).
        pair = _ts_candidate(gap_days=10, pA=0.30, pB=0.60, nB=0.40)
        payload = {"orderbook_fp": {"yes": [["0.55", "100"]], "no": [["0.65", "100"]]}}
        [enriched] = enrich_with_orderbook_prices(_orderbook_payload_client(payload), [pair], _AMPLE_BALANCE_CENTS)
        assert enriched.tradeable is False


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


class TestFetchOrderbookFailureIsOneLine:
    """A failed order-book read is logged as one line: the HTTP status and
    reason, then Kalshi's error code and message."""

    def test_a_rejected_read_logs_one_line(self, caplog):
        client = MagicMock()
        client.get_market_orderbook_without_preload_content = MagicMock(
            return_value=SimpleNamespace(
                status=400, reason="Bad Request",
                data=b'{"error":{"code":"bad_request","message":"bad request"}}',
                getheaders=lambda: {"Via": "1.1 x.cloudfront.net (CloudFront)"},
            )
        )
        with caplog.at_level(logging.WARNING):
            assert _fetch_orderbook(client, "MKT-400") is None
        assert [r.getMessage() for r in caplog.records] == [
            "Orderbook fetch failed for MKT-400: HTTP 400 Bad Request — bad_request: bad request"
        ]


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

    @pytest.mark.skipif(not INCLUDE_MVE_MARKETS, reason="MVE scanning disabled in config")
    def test_mve_pull_bails_after_consecutive_pages_with_no_active_markets(self, caplog):
        # Regression (sandbox, live 2026-08): the bail-out counter used to
        # reset on RAW nested-market count regardless of status. The MVE
        # listing can serve thousands of pages whose nested markets are all
        # non-active (closed/settled) — those pages must NOT reset the
        # counter, or the bail-out never fires (observed live: 4,600+ pages
        # flat at the same appended-market count with no bound). An endless
        # supply of pages that each carry several non-active nested markets
        # must still bail after exactly MVE_MAX_EMPTY_PAGES pages.
        from itertools import count

        from kalshi_betting.config import MVE_MAX_EMPTY_PAGES

        def endless_inactive_mve_pages(**kwargs):
            n = next(counter)
            return _raw_page(
                [{"title": f"MVE {n}", "markets": [
                    _raw_market(f"MVE-SET-{n}-A", "Option A", status="settled"),
                    _raw_market(f"MVE-CLS-{n}-B", "Option B", status="closed"),
                ]}],
                cursor=f"CUR-{n}",
            )
        counter = count()

        client = MagicMock()
        client.get_events_without_preload_content = MagicMock(return_value=_raw_page([]))
        client.get_multivariate_events_without_preload_content = MagicMock(
            side_effect=endless_inactive_mve_pages
        )

        with caplog.at_level(logging.WARNING):
            markets = fetch_open_events_with_markets(client)

        assert markets == []
        calls = client.get_multivariate_events_without_preload_content.call_count
        assert calls == MVE_MAX_EMPTY_PAGES, f"expected bail-out at {MVE_MAX_EMPTY_PAGES} pages, made {calls} calls"
        assert "no active nested" in caplog.text

    @pytest.mark.skipif(not INCLUDE_MVE_MARKETS, reason="MVE scanning disabled in config")
    def test_mve_page_with_one_active_market_resets_bailout_counter(self):
        # Mixed case: a page containing one ACTIVE market among otherwise
        # inactive ones must reset the empty-page counter, so pagination
        # continues past it rather than counting toward the bail-out.
        from itertools import count

        from kalshi_betting.config import MVE_MAX_EMPTY_PAGES

        RESET_PAGE_INDEX = 3  # 0-indexed page that contains the one active market

        def mve_pages_with_one_active(**kwargs):
            n = next(counter)
            if n == RESET_PAGE_INDEX:
                mkts = [
                    _raw_market(f"MVE-SET-{n}-A", "Option A", status="settled"),
                    _raw_market(f"MVE-ACT-{n}-B", "Option B", status="active"),
                ]
            else:
                mkts = [
                    _raw_market(f"MVE-SET-{n}-A", "Option A", status="settled"),
                ]
            return _raw_page([{"title": f"MVE {n}", "markets": mkts}], cursor=f"CUR-{n}")
        counter = count()

        client = MagicMock()
        client.get_events_without_preload_content = MagicMock(return_value=_raw_page([]))
        client.get_multivariate_events_without_preload_content = MagicMock(
            side_effect=mve_pages_with_one_active
        )

        markets = fetch_open_events_with_markets(client)

        # The active market from the reset page must be present.
        assert {m.ticker for m in markets} == {f"MVE-ACT-{RESET_PAGE_INDEX}-B"}
        # Pagination must have continued past RESET_PAGE_INDEX: total calls
        # equal RESET_PAGE_INDEX + 1 (pages before/including the reset) plus
        # another full MVE_MAX_EMPTY_PAGES run of inactive-only pages after it
        # before the bail-out fires again.
        calls = client.get_multivariate_events_without_preload_content.call_count
        assert calls == RESET_PAGE_INDEX + 1 + MVE_MAX_EMPTY_PAGES, (
            f"expected the counter to reset at page {RESET_PAGE_INDEX}, "
            f"then bail after another {MVE_MAX_EMPTY_PAGES} pages; got {calls} calls"
        )

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

    def test_standard_events_progress_logged_at_cadence(self, caplog):
        # A silent multi-page fetch is indistinguishable from a hang (BS-13):
        # a live dev-mode run paged 125,538 markets in 13m27s with zero log
        # lines. A ~60-page fetch must emit progress lines at the configured
        # cadence, not just a final summary.
        from itertools import count

        from kalshi_betting.config import SCANNER_PROGRESS_LOG_EVERY_PAGES

        n_pages = 2 * SCANNER_PROGRESS_LOG_EVERY_PAGES + 10
        counter = count()

        def paged_events(**kwargs):
            n = next(counter)
            cursor = f"CUR-{n}" if n < n_pages - 1 else None
            return _raw_page(
                [{"title": f"E{n}", "markets": [_raw_market(f"T{n}", f"Q {n}")]}],
                cursor=cursor,
            )

        client = MagicMock()
        client.get_events_without_preload_content = MagicMock(side_effect=paged_events)
        client.get_multivariate_events_without_preload_content = MagicMock(
            return_value=_raw_page([])
        )

        with caplog.at_level(logging.INFO):
            markets = fetch_open_events_with_markets(client)

        assert len(markets) == n_pages
        progress_lines = [
            r.message for r in caplog.records
            if "Open-events fetch" in r.message and "pages" in r.message
        ]
        assert len(progress_lines) >= 2, f"expected >=2 progress lines, got {progress_lines}"
        assert all("markets" in line for line in progress_lines)

    def test_standard_events_stuck_cursor_stops_pagination(self, caplog):
        # The cursor is a keyset position — a page handing back the exact
        # cursor we just requested with already proves the server isn't
        # advancing. One repeat must stop the loop rather than spin forever.
        page1 = _raw_page(
            [{"title": "E1", "markets": [_raw_market("PAGE1", "Q one")]}], cursor="CUR-STUCK"
        )
        page2 = _raw_page(
            [{"title": "E2", "markets": [_raw_market("PAGE2", "Q two")]}], cursor="CUR-STUCK"
        )
        client = MagicMock()
        client.get_events_without_preload_content = MagicMock(side_effect=[page1, page2])
        client.get_multivariate_events_without_preload_content = MagicMock(
            return_value=_raw_page([])
        )

        with caplog.at_level(logging.WARNING):
            markets = fetch_open_events_with_markets(client)

        # Only the two pages fetched before the guard fired — a third call
        # would raise StopIteration against the side_effect list, which
        # would fail this test on its own.
        assert {m.ticker for m in markets} == {"PAGE1", "PAGE2"}
        assert client.get_events_without_preload_content.call_count == 2
        assert "cursor did not advance" in caplog.text
        assert "Open-events fetch" in caplog.text

    def test_get_held_tickers_pagination_unions_pages(self):
        # get_held_tickers previously had no multi-page test coverage at all.
        from kalshi_betting.scanner import get_held_tickers

        page1 = SimpleNamespace(
            status=200,
            data=json.dumps({
                "market_positions": [{"ticker": "HELD-1", "position_fp": "3"}],
                "cursor": "CUR-2",
            }).encode(),
        )
        page2 = SimpleNamespace(
            status=200,
            data=json.dumps({
                "market_positions": [{"ticker": "HELD-2", "position_fp": "-1"}],
                "cursor": None,
            }).encode(),
        )
        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(side_effect=[page1, page2])

        assert get_held_tickers(client) == {"HELD-1", "HELD-2"}
        assert client.get_positions_without_preload_content.call_count == 2

    def test_get_held_tickers_stuck_cursor_stops_and_warns(self, caplog):
        from kalshi_betting.scanner import get_held_tickers

        page1 = SimpleNamespace(
            status=200,
            data=json.dumps({
                "market_positions": [{"ticker": "HELD-1", "position_fp": "3"}],
                "cursor": "CUR-STUCK",
            }).encode(),
        )
        page2 = SimpleNamespace(
            status=200,
            data=json.dumps({
                "market_positions": [{"ticker": "HELD-2", "position_fp": "-1"}],
                "cursor": "CUR-STUCK",
            }).encode(),
        )
        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(side_effect=[page1, page2])

        with caplog.at_level(logging.WARNING):
            held = get_held_tickers(client)

        # Partial results from both fetched pages are kept, not discarded.
        assert held == {"HELD-1", "HELD-2"}
        assert client.get_positions_without_preload_content.call_count == 2
        assert "cursor did not advance" in caplog.text
        assert "Positions fetch" in caplog.text

    @pytest.mark.skipif(not INCLUDE_MVE_MARKETS, reason="MVE scanning disabled in config")
    def test_mve_stuck_cursor_stops_pagination(self, caplog):
        page1 = _raw_page(
            [{"title": "MVE E1", "markets": [_raw_market("MVE-1", "Q one")]}],
            cursor="CUR-STUCK",
        )
        page2 = _raw_page(
            [{"title": "MVE E2", "markets": [_raw_market("MVE-2", "Q two")]}],
            cursor="CUR-STUCK",
        )
        client = MagicMock()
        client.get_events_without_preload_content = MagicMock(return_value=_raw_page([]))
        client.get_multivariate_events_without_preload_content = MagicMock(
            side_effect=[page1, page2]
        )

        with caplog.at_level(logging.WARNING):
            markets = fetch_open_events_with_markets(client)

        assert {m.ticker for m in markets} == {"MVE-1", "MVE-2"}
        assert client.get_multivariate_events_without_preload_content.call_count == 2
        assert "cursor did not advance" in caplog.text
        assert "MVE events fetch" in caplog.text


def _positions_page(ticker: str, cursor: str | None) -> SimpleNamespace:
    """Raw-response stand-in for one /portfolio/positions page."""
    return SimpleNamespace(
        status=200,
        data=json.dumps({
            "market_positions": [{"ticker": ticker, "position_fp": "3"}],
            "cursor": cursor,
        }).encode(),
    )


class TestCursorLoopBounds:
    """TS-05: scanner.py's three cursor loops were the only unbounded scans
    left in the ingest path. The stuck-cursor guard proved only ONE failure
    shape (a cursor repeating consecutively); a keyset cycling with period > 1
    (A, B, A, B, ...) never repeats consecutively and paged forever. Each loop
    now remembers every cursor it has requested AND stops at SCANNER_MAX_PAGES,
    which bounds the walk against pathologies nobody enumerated."""

    def test_standard_events_cycling_cursor_stops_pagination(self, caplog):
        # A, B, A: the third page's cursor never equals the one just used, so
        # only the seen-set catches it.
        pages = [
            _raw_page([{"title": "E1", "markets": [_raw_market("PAGE1", "Q one")]}], cursor="A"),
            _raw_page([{"title": "E2", "markets": [_raw_market("PAGE2", "Q two")]}], cursor="B"),
            _raw_page([{"title": "E3", "markets": [_raw_market("PAGE3", "Q three")]}], cursor="A"),
        ]
        client = MagicMock()
        client.get_events_without_preload_content = MagicMock(side_effect=pages)
        client.get_multivariate_events_without_preload_content = MagicMock(
            return_value=_raw_page([])
        )

        with caplog.at_level(logging.WARNING):
            markets = fetch_open_events_with_markets(client)

        # A fourth call would raise StopIteration against the side_effect list.
        assert {m.ticker for m in markets} == {"PAGE1", "PAGE2", "PAGE3"}
        assert client.get_events_without_preload_content.call_count == 3
        assert "cursor did not advance" in caplog.text
        assert "Open-events fetch" in caplog.text

    def test_get_held_tickers_cycling_cursor_stops_and_warns(self, caplog):
        from kalshi_betting.scanner import get_held_tickers

        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(side_effect=[
            _positions_page("HELD-1", "A"),
            _positions_page("HELD-2", "B"),
            _positions_page("HELD-3", "A"),
        ])

        with caplog.at_level(logging.WARNING):
            held = get_held_tickers(client)

        assert held == {"HELD-1", "HELD-2", "HELD-3"}
        assert client.get_positions_without_preload_content.call_count == 3
        assert "cursor did not advance" in caplog.text
        assert "Positions fetch" in caplog.text

    @pytest.mark.skipif(not INCLUDE_MVE_MARKETS, reason="MVE scanning disabled in config")
    def test_mve_cycling_cursor_stops_pagination(self, caplog):
        # Every page carries an ACTIVE nested market, so MVE_MAX_EMPTY_PAGES
        # never fires — the seen-cursor set is the only thing that can stop it.
        pages = [
            _raw_page([{"title": "M1", "markets": [_raw_market("MVE-1", "Q one")]}], cursor="A"),
            _raw_page([{"title": "M2", "markets": [_raw_market("MVE-2", "Q two")]}], cursor="B"),
            _raw_page([{"title": "M3", "markets": [_raw_market("MVE-3", "Q three")]}], cursor="A"),
        ]
        client = MagicMock()
        client.get_events_without_preload_content = MagicMock(return_value=_raw_page([]))
        client.get_multivariate_events_without_preload_content = MagicMock(side_effect=pages)

        with caplog.at_level(logging.WARNING):
            markets = fetch_open_events_with_markets(client)

        assert {m.ticker for m in markets} == {"MVE-1", "MVE-2", "MVE-3"}
        assert client.get_multivariate_events_without_preload_content.call_count == 3
        assert "cursor did not advance" in caplog.text
        assert "MVE events fetch" in caplog.text

    def test_standard_events_page_cap_stops_pagination(self, caplog, monkeypatch):
        # A server handing back a FRESH cursor forever defeats both the
        # consecutive-repeat guard and the seen-set; only the page cap bounds
        # it. Patched to 5 so the test is instant rather than 5000 pages.
        from itertools import count

        monkeypatch.setattr(scanner, "SCANNER_MAX_PAGES", 5)
        counter = count()

        def endless(**kwargs):
            n = next(counter)
            return _raw_page(
                [{"title": f"E{n}", "markets": [_raw_market(f"T{n}", f"Q {n}")]}],
                cursor=f"CUR-{n}",
            )

        client = MagicMock()
        client.get_events_without_preload_content = MagicMock(side_effect=endless)
        client.get_multivariate_events_without_preload_content = MagicMock(
            return_value=_raw_page([])
        )

        with caplog.at_level(logging.WARNING):
            markets = fetch_open_events_with_markets(client)

        assert len(markets) == 5
        assert client.get_events_without_preload_content.call_count == 5
        assert "reached SCANNER_MAX_PAGES (5)" in caplog.text
        assert "Open-events fetch" in caplog.text

    def test_get_held_tickers_page_cap_stops_pagination(self, caplog, monkeypatch):
        from itertools import count

        from kalshi_betting.scanner import get_held_tickers

        monkeypatch.setattr(scanner, "SCANNER_MAX_PAGES", 5)
        counter = count()

        def endless(**kwargs):
            n = next(counter)
            return _positions_page(f"HELD-{n}", f"CUR-{n}")

        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(side_effect=endless)

        with caplog.at_level(logging.WARNING):
            held = get_held_tickers(client)

        assert len(held) == 5
        assert client.get_positions_without_preload_content.call_count == 5
        assert "reached SCANNER_MAX_PAGES (5)" in caplog.text
        assert "Positions fetch" in caplog.text

    @pytest.mark.skipif(not INCLUDE_MVE_MARKETS, reason="MVE scanning disabled in config")
    def test_mve_page_cap_stops_pagination(self, caplog, monkeypatch):
        # The cap is independent of MVE_MAX_EMPTY_PAGES (25): every page here
        # is productive, so the productivity bail-out can never fire.
        from itertools import count

        monkeypatch.setattr(scanner, "SCANNER_MAX_PAGES", 5)
        counter = count()

        def endless(**kwargs):
            n = next(counter)
            return _raw_page(
                [{"title": f"M{n}", "markets": [_raw_market(f"MVE-{n}", f"Q {n}")]}],
                cursor=f"CUR-{n}",
            )

        client = MagicMock()
        client.get_events_without_preload_content = MagicMock(return_value=_raw_page([]))
        client.get_multivariate_events_without_preload_content = MagicMock(side_effect=endless)

        with caplog.at_level(logging.WARNING):
            markets = fetch_open_events_with_markets(client)

        assert len(markets) == 5
        assert client.get_multivariate_events_without_preload_content.call_count == 5
        assert "reached SCANNER_MAX_PAGES (5)" in caplog.text
        assert "MVE events fetch" in caplog.text

    def test_complete_stream_ending_on_the_cap_does_not_warn(self, caplog, monkeypatch):
        # The cap check runs BEFORE `cursor = new_cursor` and before the
        # end-of-stream `if not cursor: break`, so a walk that COMPLETES on
        # page SCANNER_MAX_PAGES used to log a truncation warning for a stream
        # that was never truncated — a false alarm that reads, in a weekly
        # prod log, exactly like a real blind spot. Guarding the cap on
        # new_cursor fixes it: the final page carries a null cursor, so there
        # is provably nothing left to fetch.
        monkeypatch.setattr(scanner, "SCANNER_MAX_PAGES", 3)
        pages = [
            _raw_page([{"title": "E1", "markets": [_raw_market("T1", "Q one")]}], cursor="A"),
            _raw_page([{"title": "E2", "markets": [_raw_market("T2", "Q two")]}], cursor="B"),
            # Last page: end of stream, landing exactly on the cap.
            _raw_page([{"title": "E3", "markets": [_raw_market("T3", "Q three")]}], cursor=None),
        ]
        client = MagicMock()
        client.get_events_without_preload_content = MagicMock(side_effect=pages)
        client.get_multivariate_events_without_preload_content = MagicMock(
            return_value=_raw_page([])
        )

        with caplog.at_level(logging.WARNING):
            markets = fetch_open_events_with_markets(client)

        # Every page was ingested — the guard bounds the warning, not the walk.
        assert {m.ticker for m in markets} == {"T1", "T2", "T3"}
        assert client.get_events_without_preload_content.call_count == 3
        assert "SCANNER_MAX_PAGES" not in caplog.text
        assert "cursor did not advance" not in caplog.text


class TestShardIndex:
    """Unit coverage for the fail-safe shard *label* read itself. This never
    decides whether a market is kept — market data is cross-shard — so every
    unusable value must resolve to the default rather than raise."""

    def test_missing_exchange_index_is_default(self):
        assert _shard_index({"ticker": "T1"}) == 0

    def test_null_exchange_index_is_default(self):
        assert _shard_index({"exchange_index": None}) == 0

    def test_unparseable_exchange_index_is_default_fail_safe(self):
        # Garbage input must fail safe to the default rather than raise and
        # kill ingest on an unrelated API shape change.
        assert _shard_index({"exchange_index": "not-a-number"}) == 0

    def test_int_zero(self):
        assert _shard_index({"exchange_index": 0}) == 0

    def test_string_zero(self):
        assert _shard_index({"exchange_index": "0"}) == 0

    def test_int_one(self):
        assert _shard_index({"exchange_index": 1}) == 1

    def test_string_two(self):
        assert _shard_index({"exchange_index": "2"}) == 2

    def test_default_matches_config_constant(self):
        # Sanity check that the fallback is the config constant, not a
        # hardcoded 0 that would silently diverge from it.
        assert _shard_index({}) == DEFAULT_EXCHANGE_INDEX

    def test_explicit_zero_shard_index_not_conflated_with_missing(self, monkeypatch):
        # A declared exchange_index of 0 is a data statement, not an absent
        # field. With a falsy-conflating `or` fallback (int(m.get(...) or
        # DEFAULT_EXCHANGE_INDEX)), `0 or 9 == 9` would misreport a genuinely
        # shard-0 market as being on shard 9 the moment DEFAULT_EXCHANGE_INDEX
        # stops being 0 — exactly the conflation _shard_index's explicit
        # `is None` check exists to avoid.
        monkeypatch.setattr(scanner, "DEFAULT_EXCHANGE_INDEX", 9)
        assert _shard_index({"exchange_index": 0}) == 0
        assert _shard_index({}) == 9


class TestMarketFromDictTagsExchangeIndex:
    """_market_from_dict is the SOLE ApiMarket construction site — the shard
    tag must be attached there, so every market in the pipeline carries it."""

    def test_market_from_dict_tags_exchange_index(self):
        tagged = _market_from_dict(
            _raw_market("SHARD-2", "Rain tomorrow", exchange_index=2), "Weather Event"
        )
        assert tagged.exchange_index == 2

        untagged = _market_from_dict(_raw_market("SHARD-NONE", "Rain tomorrow"), "Weather Event")
        assert untagged.exchange_index == DEFAULT_EXCHANGE_INDEX


def _listing(rows: list, cursor: str | None = None) -> SimpleNamespace:
    """Raw-response stand-in for one /portfolio/positions page holding `rows`."""
    return SimpleNamespace(status=200, data=json.dumps(
        {"market_positions": rows, "cursor": cursor}).encode())


def _listing_client(*pages: SimpleNamespace) -> MagicMock:
    """A client whose positions listing serves `pages` in order."""
    client = MagicMock()
    client.get_positions_without_preload_content = MagicMock(side_effect=list(pages))
    return client


def _reference_held_set(rows: list) -> set:
    """Reference rule for which tickers count as held, written independently
    of get_held_positions: a non-zero count, or an unreadable count under a
    non-blank ticker."""
    held: set = set()
    for pos in rows:
        ticker = pos.get("ticker") or ""
        try:
            raw = pos.get("position_fp")
            if raw is None:
                raw = pos.get("position")
            if float(raw) != 0:
                held.add(ticker)
        except (ValueError, TypeError):
            if ticker:
                held.add(ticker)
    return held


class TestGetHeldPositions:
    """get_held_positions reads each held market's side, exposure and fees
    paid; get_held_tickers is its tickers. The held set must be exactly the
    tickers with a non-zero or unreadable count, because every held market is
    kept out of new trades, and a count or cost it cannot read must never be
    paired (held_pairs refuses a HeldPosition whose count is None)."""

    def test_side_exposure_and_fees_are_read(self):
        client = _listing_client(_listing([
            {"ticker": "KX-A", "position_fp": "30.00", "market_exposure_dollars": "6.3000",
             "fees_paid_dollars": "0.4200"},
            {"ticker": "KX-B", "position_fp": "-30.00", "market_exposure_dollars": "9.9000",
             "fees_paid_dollars": "0.5100"},
        ]))
        assert get_held_positions(client) == {
            "KX-A": HeldPosition("KX-A", 30.0, 6.3, 0.42),
            "KX-B": HeldPosition("KX-B", -30.0, 9.9, 0.51),
        }

    def test_zero_is_not_held_and_signed_or_fractional_counts_are(self):
        client = _listing_client(_listing([
            {"ticker": "HELD-NO", "position_fp": "-5"},
            {"ticker": "HELD-FRACTIONAL", "position_fp": "104.04"},
            {"ticker": "FLAT", "position_fp": "0"},
        ]))
        held = get_held_positions(client)
        assert set(held) == {"HELD-NO", "HELD-FRACTIONAL"}
        assert held["HELD-NO"].count == -5.0
        # No exposure or fees in the listing: held, with no cost to pair on
        assert held["HELD-NO"].exposure_dollars is None
        assert held["HELD-NO"].fees_dollars is None

    def test_an_unreadable_count_is_held_under_a_ticker_and_dropped_without(self):
        client = _listing_client(_listing([
            {"ticker": "KX-ODD", "position_fp": "abc", "market_exposure_dollars": "1.00"},
            {"ticker": "", "position_fp": "abc"},
        ]))
        assert get_held_positions(client) == {
            "KX-ODD": HeldPosition("KX-ODD", None, 1.0, None)}

    @pytest.mark.parametrize("raw", ["nan", "inf", "-inf", True])
    def test_a_count_that_is_no_number_is_held_with_an_unknown_count(self, raw):
        # "Not 0", so held even under a blank ticker, but never paired
        client = _listing_client(_listing([
            {"ticker": "KX-ODD", "position_fp": raw},
            {"ticker": "", "position_fp": raw},
        ]))
        held = get_held_positions(client)
        assert set(held) == {"KX-ODD", ""}
        assert held["KX-ODD"].count is None and held[""].count is None

    @pytest.mark.parametrize("exposure", [
        "missing", None, "-1.00", "nan", "inf", True, False, "abc", "1e400", [1],
    ])
    def test_an_exposure_or_fee_that_cannot_be_a_cost_reads_as_none(self, exposure):
        row = {"ticker": "KX-A", "position_fp": "10"}
        if exposure != "missing":
            row["market_exposure_dollars"] = exposure
            row["fees_paid_dollars"] = exposure
        position = get_held_positions(_listing_client(_listing([row])))["KX-A"]
        assert position.count == 10.0
        assert position.exposure_dollars is None
        assert position.fees_dollars is None

    @pytest.mark.parametrize("second", ["30", "-30", "0", "abc"])
    def test_a_ticker_listed_twice_is_held_but_never_paired(self, second):
        client = _listing_client(
            _listing([{"ticker": "KX-A", "position_fp": "30",
                       "market_exposure_dollars": "6.00", "fees_paid_dollars": "0.40"}],
                     cursor="C2"),
            _listing([{"ticker": "KX-A", "position_fp": second,
                       "market_exposure_dollars": "6.00", "fees_paid_dollars": "0.40"}]),
        )
        assert get_held_positions(client) == {
            "KX-A": HeldPosition("KX-A", None, None, None)}

    def test_a_flat_row_before_a_held_one_still_reads_as_listed_twice(self):
        # A flat row and a held row for one ticker cannot be read as one
        # position, so it stays held and is never paired
        client = _listing_client(_listing([
            {"ticker": "KX-A", "position_fp": "0"},
            {"ticker": "KX-A", "position_fp": "30", "market_exposure_dollars": "6.00",
             "fees_paid_dollars": "0.40"},
        ]))
        assert get_held_positions(client) == {
            "KX-A": HeldPosition("KX-A", None, None, None)}

    def test_the_held_set_matches_the_reference_rule(self):
        rows = [
            {"ticker": "YES", "position_fp": "3"},
            {"ticker": "NO", "position_fp": "-2.5"},
            {"ticker": "FLAT", "position_fp": "0"},
            {"ticker": "NEG-ZERO", "position_fp": "-0.0"},
            {"ticker": "LEGACY", "position": 4},
            {"ticker": "LEGACY-FLAT", "position": 0},
            {"ticker": "NONE", "position_fp": None},
            {"ticker": "ABC", "position_fp": "abc"},
            {"ticker": "LIST", "position_fp": [1]},
            {"ticker": "NAN", "position_fp": "nan"},
            {"ticker": "INF", "position_fp": "inf"},
            {"ticker": "TRUE", "position_fp": True},
            {"ticker": "FALSE", "position_fp": False},
            {"ticker": "", "position_fp": "abc"},
            {"ticker": "", "position_fp": "nan"},
            {"ticker": "TWICE", "position_fp": "1"},
            {"ticker": "TWICE", "position_fp": "0"},
            {"position_fp": "7"},
        ]
        expected = _reference_held_set(rows)
        # Non-vacuous: blank tickers and unreadable counts are both in the oracle
        assert "" in expected and "ABC" in expected and "FLAT" not in expected
        assert set(get_held_positions(_listing_client(_listing(rows)))) == expected
        assert scanner.get_held_tickers(_listing_client(_listing(rows))) == expected

    @pytest.mark.parametrize("ticker", [["KX-A"], {"t": "KX-A"}])
    def test_a_flat_row_whose_ticker_is_no_string_is_skipped(self, ticker):
        # A ticker sent as a JSON list or object names no market; its flat row
        # must be skipped, not end the run
        rows = [{"ticker": ticker, "position_fp": "0"},
                {"ticker": "KX-B", "position_fp": "3"}]
        assert _reference_held_set(rows) == {"KX-B"}
        assert scanner.get_held_tickers(_listing_client(_listing(rows))) == {"KX-B"}

    @pytest.mark.parametrize("count", ["3", "abc"])
    def test_a_held_row_whose_ticker_is_no_string_raises_as_the_reference_does(self, count):
        # Held under a ticker that cannot be a set member: the reference rule
        # raises TypeError too, so the run stops rather than guess the market
        rows = [{"ticker": ["KX-A"], "position_fp": count}]
        with pytest.raises(TypeError):
            _reference_held_set(rows)
        with pytest.raises(TypeError):
            get_held_positions(_listing_client(_listing(rows)))

    def test_complete_is_true_on_a_whole_listing(self):
        out: dict = {}
        client = _listing_client(_listing([{"ticker": "A", "position_fp": "1"}], cursor="C2"),
                                 _listing([{"ticker": "B", "position_fp": "-1"}]))
        get_held_positions(client, complete_out=out)
        assert out == {"complete": True}

    def test_complete_is_false_after_a_repeated_cursor(self):
        out: dict = {}
        client = _listing_client(_positions_page("A", "C1"), _positions_page("B", "C1"))
        held = get_held_positions(client, complete_out=out)
        assert set(held) == {"A", "B"}
        assert out == {"complete": False}

    def test_complete_is_false_after_the_page_cap(self, monkeypatch):
        from itertools import count

        monkeypatch.setattr(scanner, "SCANNER_MAX_PAGES", 5)
        counter = count()
        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(
            side_effect=lambda **kwargs: _positions_page(f"H-{next(counter)}", f"C-{next(counter)}"))
        out: dict = {}
        get_held_positions(client, complete_out=out)
        assert client.get_positions_without_preload_content.call_count == 5
        assert out == {"complete": False}

    def test_complete_is_false_when_the_walk_raises(self):
        out = {"complete": True}
        client = MagicMock()
        client.get_positions_without_preload_content = MagicMock(side_effect=[
            _positions_page("A", "C2"), ValueError("listing broke")])
        with pytest.raises(ValueError):
            get_held_positions(client, complete_out=out)
        # Set False on entry, so a caller that catches the error never reads True
        assert out == {"complete": False}

    @pytest.mark.parametrize("pages", [
        [[{"ticker": "HELD-NO", "position_fp": "-5"}, {"ticker": "FLAT", "position_fp": "0"}]],
        [[{"ticker": "HELD-1", "position_fp": "3"}], [{"ticker": "HELD-2", "position_fp": "-1"}]],
    ])
    def test_get_held_tickers_is_the_positions_tickers(self, pages):
        def client():
            return _listing_client(*[
                _listing(rows, cursor=f"C{i + 1}" if i + 1 < len(pages) else None)
                for i, rows in enumerate(pages)])
        assert scanner.get_held_tickers(client()) == set(get_held_positions(client()))


def _json_reply(payload) -> SimpleNamespace:
    """A successful raw reply carrying this JSON body."""
    return SimpleNamespace(status=200, data=json.dumps(payload).encode())


def _error_reply(status: int, reason: str) -> SimpleNamespace:
    """A failed raw reply whose headers must never reach the log."""
    return SimpleNamespace(
        status=status, reason=reason, data=b'{"error": "nope"}',
        getheaders=lambda: {"X-Header-Dump": "HEADER-DUMP"},
    )


_STAR_EVENT = "KXSTAR-14"
_STAR_TITLE = "SpaceX Starship launches"


def _star_raw(ticker: str, deadline: str, *, event_ticker: str = _STAR_EVENT) -> dict:
    """One deadline of the Starship question, as the market reply sends it."""
    return {
        "ticker": ticker, "event_ticker": event_ticker, "status": "closed",
        "title": f"Will SpaceX launch another Starship before {deadline}?",
        "close_time": "2026-10-16T00:00:00Z",
    }


def _btc_raw(ticker: str, event_ticker: str, deadline: str, *,
             strike: str = "$80,000 or above") -> dict:
    """One strike of a Bitcoin question, one event per deadline, as the market reply sends it."""
    return {
        "ticker": ticker, "event_ticker": event_ticker, "status": "closed",
        "title": f"Bitcoin high by {deadline}?", "yes_sub_title": strike,
        "close_time": "2026-03-11T00:00:00Z",
    }


def _titled_client(raws: dict, titles: dict) -> MagicMock:
    """A client that answers market lookups from `raws` and each event with its own title."""
    client = MagicMock()
    client.get_market_without_preload_content.side_effect = (
        lambda ticker: _json_reply({"market": raws[ticker]})
    )
    client.get_event_without_preload_content.side_effect = (
        lambda event_ticker: _json_reply(
            {"event": {"event_ticker": event_ticker, "title": titles[event_ticker]}})
    )
    return client


def _lookup_client(raws: dict, *, event_reply=None) -> MagicMock:
    """A client that answers market lookups from `raws` and one event reply."""
    client = MagicMock()
    client.get_market_without_preload_content.side_effect = (
        lambda ticker: _json_reply({"market": raws[ticker]})
    )
    client.get_event_without_preload_content.return_value = (
        event_reply if event_reply is not None
        else _json_reply({"event": {"event_ticker": _STAR_EVENT, "title": _STAR_TITLE}})
    )
    return client


class TestResolveHeldLadders:
    """The ladders we hold come from this run's market list, or from the
    exchange for a held market the list lacks. A held market that can't be
    identified gives None."""

    def _listed(self):
        """A deadline of the Starship question that is in this run's market list."""
        return _ingest_market("KXSTAR-14-SEP23", _STAR_EVENT,
                              "Will SpaceX launch another Starship before Sep 23, 2026?",
                              _STAR_TITLE)

    def test_a_held_market_in_the_list_needs_no_request(self, caplog):
        held = self._listed()
        client = MagicMock()
        with caplog.at_level(logging.INFO):
            keys = resolve_held_ladders(client, [held], {held.ticker})
        assert keys == market_ladder_keys(held)
        client.get_market_without_preload_content.assert_not_called()
        client.get_event_without_preload_content.assert_not_called()
        assert ("Open ladder exposure: 1 held market(s) in 1 event(s), asking 1 "
                "question(s) (0 looked up") in caplog.text

    def test_a_held_market_missing_from_the_list_is_looked_up(self, caplog):
        raw = _star_raw("KXSTAR-14-OCT16", "Oct 16, 2026")
        client = _lookup_client({raw["ticker"]: raw})
        with caplog.at_level(logging.INFO):
            keys = resolve_held_ladders(client, [self._listed()], {raw["ticker"]})
        # The looked-up market lands on the same ladder as the listed one
        assert keys == market_ladder_keys(self._listed())
        client.get_market_without_preload_content.assert_called_once_with(
            ticker=raw["ticker"])
        client.get_event_without_preload_content.assert_called_once_with(
            event_ticker=_STAR_EVENT)
        assert "(1 looked up" in caplog.text

    def test_the_event_title_is_what_makes_the_question_match(self):
        # Without its event's title, the market asks a different question
        raw = _star_raw("KXSTAR-14-OCT16", "Oct 16, 2026")
        untitled = _market_from_dict(raw, "")
        assert market_ladder_keys(untitled) != market_ladder_keys(self._listed())

    def test_two_held_markets_of_one_event_cost_one_event_request(self):
        raws = {t: _star_raw(t, d) for t, d in
                (("KXSTAR-14-OCT16", "Oct 16, 2026"), ("KXSTAR-14-NOV30", "Nov 30, 2026"))}
        client = _lookup_client(raws)
        keys = resolve_held_ladders(client, [], set(raws))
        assert keys == market_ladder_keys(self._listed())
        assert client.get_market_without_preload_content.call_count == 2
        client.get_event_without_preload_content.assert_called_once()

    def test_an_event_without_a_title_reads_as_untitled(self):
        # The market list reads a missing event title as "", and so does the lookup
        raw = _star_raw("KXSTAR-14-OCT16", "Oct 16, 2026")
        client = _lookup_client({raw["ticker"]: raw},
                                event_reply=_json_reply({"event": {"event_ticker": _STAR_EVENT}}))
        keys = resolve_held_ladders(client, [], {raw["ticker"]})
        assert keys == market_ladder_keys(_market_from_dict(raw, ""))

    def test_a_failed_market_lookup_gives_none_and_names_the_ticker(self, caplog):
        client = MagicMock()
        client.get_market_without_preload_content.return_value = _error_reply(404, "Not Found")
        with caplog.at_level(logging.INFO):
            keys = resolve_held_ladders(client, [], {"KXGONE-1"})
        assert keys is None
        errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1 and "KXGONE-1" in errors[0]
        assert "no time-series trade will be made this run" in errors[0]
        assert "HTTP 404 Not Found" in caplog.text
        # The failed reply's headers never reach the log
        assert "HEADER-DUMP" not in caplog.text
        assert "Open ladder exposure" not in caplog.text
        client.get_event_without_preload_content.assert_not_called()

    def test_a_failed_event_lookup_gives_none(self, caplog):
        raw = _star_raw("KXSTAR-14-OCT16", "Oct 16, 2026")
        client = _lookup_client({raw["ticker"]: raw},
                                event_reply=_error_reply(404, "Not Found"))
        with caplog.at_level(logging.INFO):
            keys = resolve_held_ladders(client, [], {raw["ticker"]})
        assert keys is None
        assert any(r.levelno == logging.ERROR and raw["ticker"] in r.getMessage()
                   for r in caplog.records)
        assert "HEADER-DUMP" not in caplog.text

    @pytest.mark.parametrize("body", [[], "ok", None, {}, {"market": None}, {"market": []}])
    def test_a_reply_without_a_market_gives_none(self, body):
        client = MagicMock()
        client.get_market_without_preload_content.return_value = _json_reply(body)
        assert resolve_held_ladders(client, [], {"KXODD-1"}) is None
        client.get_event_without_preload_content.assert_not_called()

    @pytest.mark.parametrize("body", [[], "ok", None, {}, {"event": None}, {"event": "x"}])
    def test_a_reply_without_an_event_gives_none(self, body):
        raw = _star_raw("KXSTAR-14-OCT16", "Oct 16, 2026")
        client = _lookup_client({raw["ticker"]: raw}, event_reply=_json_reply(body))
        assert resolve_held_ladders(client, [], {raw["ticker"]}) is None

    @pytest.mark.parametrize("event_ticker", ["", None, 5])
    def test_a_market_without_an_event_ticker_gives_none(self, event_ticker):
        raw = _star_raw("KXSTAR-14-OCT16", "Oct 16, 2026", event_ticker=event_ticker)
        client = _lookup_client({raw["ticker"]: raw})
        assert resolve_held_ladders(client, [], {raw["ticker"]}) is None
        client.get_event_without_preload_content.assert_not_called()

    def test_a_held_position_with_no_ticker_gives_none_without_a_request(self, caplog):
        client = MagicMock()
        # Not even a listed market with a blank ticker identifies it
        blank = _ingest_market("", _STAR_EVENT, "Will it rain by Oct 1, 2026?", _STAR_TITLE)
        with caplog.at_level(logging.ERROR):
            assert resolve_held_ladders(client, [self._listed(), blank], {""}) is None
        client.get_market_without_preload_content.assert_not_called()
        assert "Could not look up held market ''" in caplog.text

    def test_the_first_failure_stops_further_requests(self):
        client = MagicMock()
        client.get_market_without_preload_content.return_value = _error_reply(404, "Not Found")
        assert resolve_held_ladders(client, [], {"KXGONE-1", "KXGONE-2"}) is None
        client.get_market_without_preload_content.assert_called_once_with(ticker="KXGONE-1")

    def test_no_held_market_gives_no_labels_and_still_logs(self, caplog):
        client = MagicMock()
        with caplog.at_level(logging.INFO):
            assert resolve_held_ladders(client, [self._listed()], set()) == frozenset()
        client.get_market_without_preload_content.assert_not_called()
        assert ("Open ladder exposure: 0 held market(s) in 0 event(s), asking 0 "
                "question(s) (0 looked up") in caplog.text

    def test_each_looked_up_market_gets_its_own_events_title(self, caplog):
        # Two deadlines of one question in two events, and a market of a third event
        raws = {
            "KXBTCMAX-26MAR01-80K": _btc_raw("KXBTCMAX-26MAR01-80K", "KXBTCMAX-26MAR01",
                                             "March 1, 2026"),
            "KXBTCMAX-26MAR11-80K": _btc_raw("KXBTCMAX-26MAR11-80K", "KXBTCMAX-26MAR11",
                                             "March 11, 2026"),
            "KXSTAR-14-OCT16": _star_raw("KXSTAR-14-OCT16", "Oct 16, 2026"),
        }
        titles = {"KXBTCMAX-26MAR01": "Bitcoin record", "KXBTCMAX-26MAR11": "Bitcoin record",
                  _STAR_EVENT: _STAR_TITLE}
        client = _titled_client(raws, titles)
        with caplog.at_level(logging.INFO):
            keys = resolve_held_ladders(client, [], set(raws))
        assert keys == frozenset().union(*(
            market_ladder_keys(_market_from_dict(raw, titles[raw["event_ticker"]]))
            for raw in raws.values()))
        # One event request per event, each with that event's ticker
        assert sorted(c.kwargs["event_ticker"] for c in
                      client.get_event_without_preload_content.call_args_list) == sorted(titles)
        # The two Bitcoin markets ask one question, so three events hold two questions
        assert ("Open ladder exposure: 3 held market(s) in 3 event(s), asking 2 "
                "question(s) (3 looked up") in caplog.text

    def test_a_looked_up_market_keeps_its_outcome_label(self):
        # The held $80,000 market is on the $80,000 question, not the $90,000 one
        held = _btc_raw("KXBTCMAX-26MAR11-80K", "KXBTCMAX-26MAR11", "March 11, 2026")
        listed_80 = _ingest_market("KXBTCMAX-26MAR01-80K", "KXBTCMAX-26MAR01",
                                   "Bitcoin high by March 1, 2026?", "Bitcoin record",
                                   subtitle="$80,000 or above")
        listed_90 = _ingest_market("KXBTCMAX-26MAR01-90K", "KXBTCMAX-26MAR01",
                                   "Bitcoin high by March 1, 2026?", "Bitcoin record",
                                   subtitle="$90,000 or above")
        assert _question_of(listed_80) != _question_of(listed_90)
        client = _titled_client({held["ticker"]: held}, {"KXBTCMAX-26MAR11": "Bitcoin record"})
        keys = resolve_held_ladders(client, [listed_80, listed_90], {held["ticker"]})
        assert _question_label(listed_80) in keys
        assert _question_label(listed_90) not in keys

    def test_a_rate_limited_lookup_is_retried(self):
        raw = _star_raw("KXSTAR-14-OCT16", "Oct 16, 2026")
        client = MagicMock()
        client.get_market_without_preload_content.side_effect = [
            _error_reply(429, "Too Many Requests"), _json_reply({"market": raw})]
        client.get_event_without_preload_content.side_effect = [
            _error_reply(429, "Too Many Requests"),
            _json_reply({"event": {"event_ticker": _STAR_EVENT, "title": _STAR_TITLE}})]
        with patch.object(_http.time, "sleep"):
            keys = resolve_held_ladders(client, [self._listed()], {raw["ticker"]})
        assert keys == market_ladder_keys(self._listed())
        assert client.get_market_without_preload_content.call_count == 2
        assert client.get_event_without_preload_content.call_count == 2

    def test_a_connection_failure_gives_none_and_names_its_cause(self, caplog):
        client = MagicMock()
        client.get_market_without_preload_content.side_effect = ProtocolError(
            "Connection broken: IncompleteRead(0 bytes read)")
        with patch.object(_http.time, "sleep"), caplog.at_level(logging.WARNING):
            assert resolve_held_ladders(client, [], {"KXSTAR-14-OCT16"}) is None
        assert any(r.levelno == logging.ERROR and "KXSTAR-14-OCT16" in r.getMessage()
                   for r in caplog.records)
        assert ("Could not look up held market KXSTAR-14-OCT16: ProtocolError: "
                "Connection broken: IncompleteRead(0 bytes read)") in caplog.text

    def test_labels_out_names_every_identified_market(self):
        listed = self._listed()
        raw = _star_raw("KXSTAR-14-OCT16", "Oct 16, 2026")
        client = _lookup_client({raw["ticker"]: raw})
        labels: dict = {}
        keys = resolve_held_ladders(client, [listed], {listed.ticker, raw["ticker"]},
                                    labels_out=labels)
        assert labels == {
            listed.ticker: market_ladder_keys(listed),
            raw["ticker"]: market_ladder_keys(_market_from_dict(raw, _STAR_TITLE)),
        }
        # The return value does not depend on being asked for the labels
        again = resolve_held_ladders(_lookup_client({raw["ticker"]: raw}), [listed],
                                     {listed.ticker, raw["ticker"]})
        assert keys == again == frozenset().union(*labels.values())

    def test_labels_out_is_incomplete_when_a_market_is_unknown(self):
        client = MagicMock()
        client.get_market_without_preload_content.return_value = _error_reply(404, "Not Found")
        labels: dict = {}
        # "KXZZZ-1" sorts after the listed ticker, so the listed one is named first
        listed = self._listed()
        assert resolve_held_ladders(client, [listed], {listed.ticker, "KXZZZ-1"},
                                    labels_out=labels) is None
        assert labels == {listed.ticker: market_ladder_keys(listed)}


class TestMarketForLabels:
    """market_for_labels finds a market whose ladder labels live selling
    needs (a paid-out partner, say): from this run's list when it is there,
    else through the one held-market lookup."""

    def test_a_listed_market_needs_no_request(self):
        listed = _star_rung(_RUNG_EARLY, "Mar 1, 2026")
        client = MagicMock()
        with patch.object(scanner, "_fetch_held_market") as lookup:
            got = scanner.market_for_labels(client, listed.ticker, {listed.ticker: listed}, {})
        assert got is listed
        lookup.assert_not_called()
        client.get_market_without_preload_content.assert_not_called()

    def test_a_market_the_list_lacks_is_looked_up_once(self):
        raw = _star_raw("KXSTAR-14-OCT16", "Oct 16, 2026")
        client = _lookup_client({raw["ticker"]: raw})
        titles: dict = {}
        with patch.object(scanner, "_fetch_held_market",
                          wraps=scanner._fetch_held_market) as lookup:
            got = scanner.market_for_labels(client, raw["ticker"], {}, titles)
        # Its failure lines would say "market": a paid-out partner is not held
        lookup.assert_called_once_with(client, raw["ticker"], titles, noun="market")
        assert got == _market_from_dict(raw, _STAR_TITLE)
        # Its labels are its listed ladder-mate's: the same event and question
        assert market_ladder_keys(got) == market_ladder_keys(
            _star_rung(_RUNG_EARLY, "Mar 1, 2026"))
        assert titles == {_STAR_EVENT: _STAR_TITLE}

    def test_one_event_is_asked_about_once(self):
        raws = {t: _star_raw(t, d) for t, d in
                (("KXSTAR-14-OCT16", "Oct 16, 2026"), ("KXSTAR-14-NOV30", "Nov 30, 2026"))}
        client = _lookup_client(raws)
        titles: dict = {}
        for ticker in raws:
            assert scanner.market_for_labels(client, ticker, {}, titles) is not None
        assert client.get_market_without_preload_content.call_count == 2
        client.get_event_without_preload_content.assert_called_once()

    def test_a_failed_lookup_gives_none(self, caplog):
        client = MagicMock()
        client.get_market_without_preload_content.return_value = _error_reply(404, "Not Found")
        with caplog.at_level(logging.WARNING):
            assert scanner.market_for_labels(client, "KXGONE-1", {}, {}) is None
        assert "Could not look up market KXGONE-1: " in caplog.text
        assert "HTTP 404 Not Found" in caplog.text
        assert "HEADER-DUMP" not in caplog.text
        # A market looked up for its labels is not called held
        assert "held market" not in caplog.text

    @pytest.mark.parametrize("reply", [{"no": "market"}, {"market": {"ticker": "KXGONE-1"}}])
    def test_every_failure_line_calls_it_a_market(self, reply, caplog):
        client = MagicMock()
        client.get_market_without_preload_content.return_value = _json_reply(reply)
        with caplog.at_level(logging.WARNING):
            assert scanner.market_for_labels(client, "KXGONE-1", {}, {}) is None
        assert "Could not look up market KXGONE-1: " in caplog.text
        assert "held market" not in caplog.text

    @pytest.mark.parametrize("ticker", ["", None, 5, ["KX-A"]])
    def test_a_ticker_that_names_no_market_gives_none_without_a_request(self, ticker):
        client = MagicMock()
        assert scanner.market_for_labels(client, ticker, {}, {}) is None
        client.get_market_without_preload_content.assert_not_called()


# Five settlements as the live reply sent them (GET /portfolio/settlements, read
# 2026-10-08): newer fixed-point spellings, with revenue still in cents
_LIVE_SETTLEMENTS = [
    {"event_ticker": "KXKENNEDYREOPEN-28", "exchange_index": 0, "fee_cost": "0.030500",
     "market_result": "no", "no_count_fp": "0.00", "no_total_cost_dollars": "0.000000",
     "revenue": 0, "settled_time": "2026-10-08T04:35:27.118276Z",
     "ticker": "KXKENNEDYREOPEN-28-26OCT08", "value": 0, "yes_count_fp": "8.00",
     "yes_total_cost_dollars": "0.460000"},
    {"event_ticker": "KXTRUMPSAY-26OCT05", "exchange_index": 0, "fee_cost": "0.002800",
     "market_result": "no", "no_count_fp": "0.00", "no_total_cost_dollars": "0.000000",
     "revenue": 0, "settled_time": "2026-10-05T15:35:07.171663Z",
     "ticker": "KXTRUMPSAY-26OCT05-AUTO", "value": 0, "yes_count_fp": "4.00",
     "yes_total_cost_dollars": "0.040000"},
    {"event_ticker": "KXTRUMPAICZARWHEN-26", "exchange_index": 0, "fee_cost": "0.293700",
     "market_result": "yes", "no_count_fp": "23.00", "no_total_cost_dollars": "5.520000",
     "revenue": 0, "settled_time": "2026-10-04T18:14:17.116567Z",
     "ticker": "KXTRUMPAICZARWHEN-26-26OCT30", "value": 100, "yes_count_fp": "0.00",
     "yes_total_cost_dollars": "0.000000"},
    {"event_ticker": "KXNFLTOTAL-26OCT04INDWAS", "exchange_index": 0, "fee_cost": "0.000400",
     "market_result": "no", "no_count_fp": "0.01", "no_total_cost_dollars": "0.004200",
     "revenue": 0, "settled_time": "2026-10-04T16:45:27.12044Z",
     "ticker": "KXNFLTOTAL-26OCT04INDWAS-45", "value": 0, "yes_count_fp": "0.01",
     "yes_total_cost_dollars": "0.005900"},
    {"event_ticker": "KXTRUMPAICZARWHEN-26", "exchange_index": 0, "fee_cost": "0.331500",
     "market_result": "no", "no_count_fp": "0.00", "no_total_cost_dollars": "0.000000",
     "revenue": 0, "settled_time": "2026-10-02T14:45:33.963447Z",
     "ticker": "KXTRUMPAICZARWHEN-26-26OCT02", "value": 0, "yes_count_fp": "23.00",
     "yes_total_cost_dollars": "6.670000"},
]

# One settlement in the older spellings the pinned SDK documents: integer
# counts, costs and revenue in CENTS, fee_cost a dollar string
_LEGACY_SETTLEMENT = {
    "ticker": "KXOLD-26JAN-B1", "event_ticker": "KXOLD-26JAN", "market_result": "yes",
    "yes_count": 10, "yes_total_cost": 460, "no_count": 0, "no_total_cost": 0,
    "revenue": 1000, "fee_cost": "0.120000", "settled_time": "2026-01-02T03:04:05Z",
    "value": 100,
}


def _settlements_page(rows, cursor=None) -> SimpleNamespace:
    """Raw-response stand-in for one /portfolio/settlements page holding `rows`."""
    return _json_reply({"settlements": rows, "cursor": cursor})


def _settlements_client(*pages) -> MagicMock:
    """A client whose settlements listing serves `pages` in order."""
    client = MagicMock()
    client.get_settlements_without_preload_content = MagicMock(side_effect=list(pages))
    return client


class TestGetSettlements:
    """get_settlements reads every market the account held when it paid out.
    Live selling values a held market's paid-out partner from it, so it reads
    both spellings Kalshi has sent, turns cents into dollars exactly, and gives
    None (never a partial list) when a page cannot be read or the list is cut
    short."""

    def test_the_live_shape_is_read(self, caplog):
        client = _settlements_client(_settlements_page(_LIVE_SETTLEMENTS, cursor=""))
        with caplog.at_level(logging.INFO):
            got = scanner.get_settlements(client)
        assert got[0] == scanner.Settlement(
            "KXKENNEDYREOPEN-28-26OCT08", "KXKENNEDYREOPEN-28", "no", 8.0, 0.0, 0.46, 0.0,
            0.0305, 0.0, datetime(2026, 10, 8, 4, 35, 27, 118276, tzinfo=UTC))
        assert got[2] == scanner.Settlement(
            "KXTRUMPAICZARWHEN-26-26OCT30", "KXTRUMPAICZARWHEN-26", "yes", 0.0, 23.0, 0.0,
            5.52, 0.2937, 0.0, datetime(2026, 10, 4, 18, 14, 17, 116567, tzinfo=UTC))
        # A five-digit fraction of a second reads too
        assert got[3].settled_at == datetime(2026, 10, 4, 16, 45, 27, 120440, tzinfo=UTC)
        assert (got[3].yes_count, got[3].yes_cost_dollars) == (0.01, 0.0059)
        assert [s.ticker for s in got] == [r["ticker"] for r in _LIVE_SETTLEMENTS]
        # One page of the endpoint's largest size
        client.get_settlements_without_preload_content.assert_called_once_with(
            limit=config.SETTLEMENT_PAGE_SIZE)
        assert config.SETTLEMENT_PAGE_SIZE == 200
        assert "Settlements read: 5 (0 unreadable)" in caplog.text
        # Silent at zero
        assert not any(r.levelno >= logging.WARNING for r in caplog.records)

    def test_the_legacy_sdk_shape_is_read_in_dollars(self):
        got = scanner.get_settlements(_settlements_client(_settlements_page([_LEGACY_SETTLEMENT])))
        assert got == [scanner.Settlement(
            "KXOLD-26JAN-B1", "KXOLD-26JAN", "yes", 10.0, 0.0, 4.6, 0.0, 0.12, 10.0,
            datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC))]

    @pytest.mark.parametrize("cents, dollars", [
        (46, 0.46), (1, 0.01), (7, 0.07), (29, 0.29), (12345, 123.45), (0, 0.0),
    ])
    def test_cents_become_dollars_exactly(self, cents, dollars):
        row = dict(_LEGACY_SETTLEMENT, yes_total_cost=cents, revenue=cents)
        [got] = scanner.get_settlements(_settlements_client(_settlements_page([row])))
        assert got.yes_cost_dollars == dollars
        assert got.revenue_dollars == dollars

    @pytest.mark.parametrize("field", ["yes_total_cost", "no_total_cost", "revenue"])
    @pytest.mark.parametrize("value", ["46", 46.0, "23.000000", 0.46, "1e1000002",
                                       "1e999999999999999999", -1])
    def test_an_older_cents_field_must_be_a_whole_number(self, field, value, caplog):
        # A string or a decimal-point number under an older cents name may be a
        # dollar amount; read as cents it would be 100 times too small
        row = dict(_LEGACY_SETTLEMENT, **{field: value})
        client = _settlements_client(_settlements_page([_LEGACY_SETTLEMENT, row]))
        with caplog.at_level(logging.INFO):
            got = scanner.get_settlements(client)
        assert [s.ticker for s in got] == ["KXOLD-26JAN-B1"]
        assert "Settlements read: 1 (1 unreadable)" in caplog.text

    def test_a_value_beyond_the_decimal_context_is_counted_not_raised(self, caplog):
        # Turning cents into dollars is Decimal arithmetic; a value beyond the
        # context's limits leaves the record out rather than ending the run
        row = dict(_LEGACY_SETTLEMENT, ticker="KXBIG-1", revenue=10 ** 6)
        client = _settlements_client(_settlements_page([_LIVE_SETTLEMENTS[0], row]))
        with localcontext() as ctx, caplog.at_level(logging.INFO):
            ctx.Emax = 3
            got = scanner.get_settlements(client)
        assert [s.ticker for s in got] == ["KXKENNEDYREOPEN-28-26OCT08"]
        assert "Settlements read: 1 (1 unreadable)" in caplog.text

    def test_a_mixed_page_reads_each_record_by_its_own_spelling(self):
        both = dict(_LIVE_SETTLEMENTS[0], ticker="KXBOTH-1",
                    # The newer field wins whenever it is present
                    yes_count=3, yes_total_cost=999, revenue_dollars="1.00", revenue=0)
        got = scanner.get_settlements(_settlements_client(
            _settlements_page([_LIVE_SETTLEMENTS[0], _LEGACY_SETTLEMENT, both])))
        assert [s.ticker for s in got] == ["KXKENNEDYREOPEN-28-26OCT08", "KXOLD-26JAN-B1",
                                           "KXBOTH-1"]
        assert got[1].yes_cost_dollars == 4.6
        assert (got[2].yes_count, got[2].yes_cost_dollars, got[2].revenue_dollars) == (
            8.0, 0.46, 1.0)

    def test_presence_decides_never_truthiness(self):
        # A newer field of zero is a real zero, not a reason to read the older one
        row = dict(_LIVE_SETTLEMENTS[0], revenue_dollars="0", revenue=500,
                   yes_count_fp="0", yes_count=9)
        [got] = scanner.get_settlements(_settlements_client(_settlements_page([row])))
        assert (got.revenue_dollars, got.yes_count) == (0.0, 0.0)

    @pytest.mark.parametrize("field", ["yes_count_fp", "no_total_cost_dollars",
                                       "revenue_dollars"])
    def test_an_unreadable_newer_field_never_falls_back(self, field):
        row = dict(_LEGACY_SETTLEMENT, **{field: None})
        assert scanner.get_settlements(_settlements_client(_settlements_page([row]))) == []

    def test_two_pages_follow_the_cursor(self):
        client = _settlements_client(
            _settlements_page(_LIVE_SETTLEMENTS[:2], cursor="C1"),
            _settlements_page(_LIVE_SETTLEMENTS[2:], cursor=""))
        got = scanner.get_settlements(client)
        assert [s.ticker for s in got] == [r["ticker"] for r in _LIVE_SETTLEMENTS]
        assert [c.kwargs for c in client.get_settlements_without_preload_content.call_args_list] == [
            {"limit": 200}, {"limit": 200, "cursor": "C1"}]

    def test_min_ts_is_asked_for_on_every_page(self):
        # A window is sent with the first page and with each page after it
        client = _settlements_client(
            _settlements_page(_LIVE_SETTLEMENTS[:2], cursor="C1"),
            _settlements_page(_LIVE_SETTLEMENTS[2:], cursor=""))
        got = scanner.get_settlements(client, min_ts=1_790_000_000)
        assert [s.ticker for s in got] == [r["ticker"] for r in _LIVE_SETTLEMENTS]
        assert [c.kwargs for c in client.get_settlements_without_preload_content.call_args_list] == [
            {"limit": 200, "min_ts": 1_790_000_000},
            {"limit": 200, "min_ts": 1_790_000_000, "cursor": "C1"}]

    @pytest.mark.parametrize("cursors", [["C1", "C1"], ["C1", "C2", "C1"]])
    def test_a_repeated_cursor_stops_and_gives_none(self, cursors, caplog):
        client = _settlements_client(*[_settlements_page(_LIVE_SETTLEMENTS[:1], cursor=c)
                                       for c in cursors])
        with caplog.at_level(logging.INFO):
            assert scanner.get_settlements(client) is None
        assert client.get_settlements_without_preload_content.call_count == len(cursors)
        assert "cursor did not advance" in caplog.text
        assert "Settlements read" not in caplog.text

    def test_the_page_cap_gives_none(self, monkeypatch, caplog):
        monkeypatch.setattr(scanner, "SCANNER_MAX_PAGES", 2)
        client = _settlements_client(_settlements_page(_LIVE_SETTLEMENTS[:1], cursor="C1"),
                                     _settlements_page(_LIVE_SETTLEMENTS[1:2], cursor="C2"))
        with caplog.at_level(logging.WARNING):
            assert scanner.get_settlements(client) is None
        assert "SCANNER_MAX_PAGES (2)" in caplog.text

    def test_a_list_that_ends_on_the_last_allowed_page_is_complete(self, monkeypatch):
        monkeypatch.setattr(scanner, "SCANNER_MAX_PAGES", 2)
        client = _settlements_client(_settlements_page(_LIVE_SETTLEMENTS[:1], cursor="C1"),
                                     _settlements_page(_LIVE_SETTLEMENTS[1:2], cursor=None))
        assert len(scanner.get_settlements(client)) == 2

    @pytest.mark.parametrize("change", [
        {"ticker": None}, {"ticker": ""}, {"ticker": 5}, {"event_ticker": None},
        {"market_result": None}, {"market_result": ""},
        {"yes_count_fp": "abc"}, {"yes_count_fp": "-1"}, {"yes_count_fp": "nan"},
        {"no_count_fp": "inf"}, {"yes_total_cost_dollars": "-0.01"},
        {"no_total_cost_dollars": [1]}, {"fee_cost": None}, {"fee_cost": "x"},
        {"revenue": "1e400"}, {"revenue": None}, {"revenue": "1e1000002"},
        {"revenue": "23.000000"}, {"revenue": 0.5}, {"settled_time": None},
        {"settled_time": "garbage"}, {"settled_time": "2026-10-08T04:35:27"},
        {"settled_time": "2026-10-08"}, {"settled_time": 1791424800},
        # A JSON true or false is never a number
        {"yes_count_fp": True}, {"no_total_cost_dollars": False}, {"revenue": False},
        {"fee_cost": True},
    ])
    def test_an_unreadable_record_is_left_out_and_counted(self, change, caplog):
        bad = dict(_LIVE_SETTLEMENTS[1], **change)
        client = _settlements_client(_settlements_page([_LIVE_SETTLEMENTS[0], bad]))
        with caplog.at_level(logging.INFO):
            got = scanner.get_settlements(client)
        assert [s.ticker for s in got] == ["KXKENNEDYREOPEN-28-26OCT08"]
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1 and "left out 1 record(s)" in warnings[0]
        assert "Settlements read: 1 (1 unreadable)" in caplog.text

    @pytest.mark.parametrize("field", ["ticker", "yes_count_fp", "fee_cost", "revenue",
                                       "settled_time", "market_result"])
    def test_a_missing_field_is_unreadable(self, field):
        row = {k: v for k, v in _LIVE_SETTLEMENTS[0].items() if k != field}
        assert scanner.get_settlements(_settlements_client(_settlements_page([row]))) == []

    @pytest.mark.parametrize("record", ["x", None, [], 5])
    def test_a_record_that_is_not_an_object_is_counted(self, record, caplog):
        client = _settlements_client(_settlements_page([record]))
        with caplog.at_level(logging.INFO):
            assert scanner.get_settlements(client) == []
        assert "Settlements read: 0 (1 unreadable)" in caplog.text

    def test_unreadable_out_counts_the_records_left_out(self, caplog):
        # Two settlements that could each be the partner of one held market;
        # the second has no UTC offset, so it is left out. The count tells a
        # caller that needs exactly one partner that the list may hide one.
        first = _LIVE_SETTLEMENTS[4]
        second = dict(first, ticker="KXTRUMPAICZARWHEN-26-26OCT09",
                      settled_time="2026-10-09T14:45:33")
        out = {"unreadable": 7}
        got = scanner.get_settlements(_settlements_client(_settlements_page([first, second])),
                                      unreadable_out=out)
        assert [s.ticker for s in got] == [first["ticker"]]
        assert out == {"unreadable": 1}

    def test_unreadable_out_is_zero_on_a_clean_read(self):
        out: dict = {}
        got = scanner.get_settlements(
            _settlements_client(_settlements_page(_LIVE_SETTLEMENTS)), unreadable_out=out)
        assert len(got) == 5 and out == {"unreadable": 0}

    def test_unreadable_out_is_set_before_a_failed_call(self):
        out = {"unreadable": 7}
        client = _settlements_client(
            _settlements_page([_LIVE_SETTLEMENTS[0], "x"], cursor="C1"),
            _error_reply(400, "Bad Request"))
        assert scanner.get_settlements(client, unreadable_out=out) is None
        # The key is there, but a call that gives None leaves nothing to use
        assert "unreadable" in out

    def test_a_legacy_bool_is_refused(self):
        row = dict(_LEGACY_SETTLEMENT, yes_count=True)
        assert scanner.get_settlements(_settlements_client(_settlements_page([row]))) == []

    def test_a_time_with_an_offset_is_read_in_utc(self):
        row = dict(_LIVE_SETTLEMENTS[0], settled_time="2026-10-07T21:35:27-07:00")
        [got] = scanner.get_settlements(_settlements_client(_settlements_page([row])))
        assert got.settled_at == datetime(2026, 10, 8, 4, 35, 27, tzinfo=UTC)
        assert got.settled_at.tzinfo is UTC

    def test_a_failed_page_gives_none_in_one_line(self, caplog):
        client = MagicMock()
        client.get_settlements_without_preload_content.return_value = _error_reply(
            404, "Not Found")
        with caplog.at_level(logging.INFO):
            assert scanner.get_settlements(client) is None
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "Could not read the account's settlements (page 1): HTTP 404 Not Found" in (
            warnings[0])
        assert "\n" not in warnings[0] and "HEADER-DUMP" not in caplog.text
        assert "Settlements read" not in caplog.text

    def test_a_later_page_that_fails_gives_none_not_a_partial_list(self, caplog):
        client = _settlements_client(_settlements_page(_LIVE_SETTLEMENTS[:2], cursor="C1"),
                                     _error_reply(400, "Bad Request"))
        with caplog.at_level(logging.WARNING):
            assert scanner.get_settlements(client) is None
        assert "(page 2): HTTP 400 Bad Request" in caplog.text

    def test_a_connection_failure_is_retried_then_gives_none(self, caplog):
        client = MagicMock()
        client.get_settlements_without_preload_content.side_effect = ProtocolError(
            "Connection broken: IncompleteRead(0 bytes read)")
        with patch.object(_http.time, "sleep"), caplog.at_level(logging.WARNING):
            assert scanner.get_settlements(client) is None
        # Retried like every read-only GET
        assert client.get_settlements_without_preload_content.call_count > 1
        assert ("ProtocolError: Connection broken: IncompleteRead(0 bytes read)"
                in caplog.text)

    def test_a_rate_limited_page_is_retried(self):
        client = _settlements_client(_error_reply(429, "Too Many Requests"),
                                     _settlements_page(_LIVE_SETTLEMENTS[:1]))
        with patch.object(_http.time, "sleep"):
            got = scanner.get_settlements(client)
        assert [s.ticker for s in got] == ["KXKENNEDYREOPEN-28-26OCT08"]

    @pytest.mark.parametrize("body", [
        [], "ok", None, {"settlements": {}}, {"settlements": "x"},
        {"settlements": [], "cursor": 5}, {"settlements": [], "cursor": ["C1"]},
        # No "settlements" key at all: a renamed key, not an account with none
        {}, {"cursor": ""}, {"Settlements": [], "cursor": ""},
    ])
    def test_a_page_that_is_not_a_list_of_settlements_gives_none(self, body, caplog):
        client = MagicMock()
        client.get_settlements_without_preload_content.return_value = _json_reply(body)
        with caplog.at_level(logging.WARNING):
            assert scanner.get_settlements(client) is None
        assert "Could not read the account's settlements" in caplog.text

    @pytest.mark.parametrize("body", [
        {"settlements": [], "cursor": ""}, {"settlements": None, "cursor": None},
        {"settlements": []},
    ])
    def test_no_settlements_is_an_empty_list(self, body, caplog):
        client = MagicMock()
        client.get_settlements_without_preload_content.return_value = _json_reply(body)
        with caplog.at_level(logging.INFO):
            assert scanner.get_settlements(client) == []
        assert "Settlements read: 0 (0 unreadable)" in caplog.text


_RUNG_EARLY = "KXSTAR-14-MAR01"
_RUNG_LATE = "KXSTAR-14-MAR20"


def _star_rung(ticker: str, deadline: str):
    """One listed rung of the Starship ladder (every rung shares one event)."""
    return _ingest_market(ticker, _STAR_EVENT,
                          f"Will SpaceX launch another Starship before {deadline}?",
                          _STAR_TITLE)


def _held(ticker: str, count, exposure=None, fees=None) -> HeldPosition:
    """A held market as get_held_positions reports it."""
    return HeldPosition(ticker, count, exposure, fees)


def _priced_rung(ticker: str, deadline: str, *, yes_ask, no_ask):
    """A listed rung of the Starship ladder with these asks (raw values, as sent)."""
    return _ingest_market(ticker, _STAR_EVENT,
                          f"Will SpaceX launch another Starship before {deadline}?",
                          _STAR_TITLE, yes_ask=yes_ask, no_ask=no_ask)


class TestHeldPairs:
    """held_pairs finds the held pairs a run may add to: exactly two held
    markets on one ladder, one YES and one NO of equal size, both costs
    readable. Every other shape is never added to. An add-on is exempt from
    the one-trade-per-ladder rule only because this isolation holds, so a
    shape it lets through by mistake is a real-money stacking bug. Each pair
    is valued at today's prices, and its stake, the figure sizing reads, is
    that worth plus both markets' fees paid: a wrong stake sizes a real
    add-on too big or too small."""

    @staticmethod
    def _labels(*markets) -> dict:
        """Each market's ladder labels, as resolve_held_ladders' labels_out gives them."""
        return {m.ticker: market_ladder_keys(m) for m in markets}

    @staticmethod
    def _listed(*markets) -> dict:
        """This run's market list as held_pairs reads it: ticker -> market."""
        return {m.ticker: m for m in markets}

    def _ladder(self) -> dict:
        """The two rungs' labels: one event and one question."""
        return self._labels(_star_rung(_RUNG_EARLY, "Mar 1, 2026"),
                            _star_rung(_RUNG_LATE, "Mar 20, 2026"))

    @staticmethod
    def _rung_markets(late_no="0.45") -> dict:
        """The two rungs as this run lists them: early YES asked at 0.30, late NO at late_no."""
        early = _priced_rung(_RUNG_EARLY, "Mar 1, 2026", yes_ask="0.30", no_ask="0.71")
        late = _priced_rung(_RUNG_LATE, "Mar 20, 2026", yes_ask="0.56", no_ask="0.45")
        # Set as sent, so a quote no market dict could be built from still reaches held_pairs
        late.no_ask_dollars = late_no
        return TestHeldPairs._listed(early, late)

    def test_an_exact_time_series_pair(self):
        positions = {_RUNG_EARLY: _held(_RUNG_EARLY, 30.0, 6.00, 0.40),
                     _RUNG_LATE: _held(_RUNG_LATE, -30.0, 12.00, 0.50)}
        [(key, pair)] = held_pairs(positions, self._ladder(), self._rung_markets()).items()
        assert key == frozenset({_RUNG_EARLY, _RUNG_LATE})
        assert pair.sides == ((_RUNG_EARLY, "yes"), (_RUNG_LATE, "no"))
        assert pair.count == 30.0
        assert pair.cost_dollars == pytest.approx(18.90)
        # Each market at the ask of the side held there: YES at 0.30, NO at 0.45
        assert pair.value_dollars == 30.0 * 0.30 + 30.0 * 0.45 == 22.5
        # Both markets' fees paid, $0.40 and $0.50, and the stake sizing
        # subtracts: the worth plus those fees
        assert pair.fees_dollars == 0.40 + 0.50
        assert pair.stake_dollars == 22.5 + (0.40 + 0.50) == pytest.approx(23.40)

    def test_a_same_title_pair_is_joined_by_its_shared_question(self):
        # Two events of two series asking one question
        x = _ingest_market("KXA-1-Y", "KXA-1", "Who wins the game?", "Game", subtitle="Team A",
                           yes_ask="0.45", no_ask="0.56")
        y = _ingest_market("KXB-1-Y", "KXB-1", "Who wins the game?", "Game", subtitle="Team A",
                           yes_ask="0.47", no_ask="0.54")
        labels = self._labels(x, y)
        assert {kind for kind, _ in labels[x.ticker] & labels[y.ticker]} == {"question"}
        positions = {x.ticker: _held(x.ticker, -10.0, 4.00, 0.20),
                     y.ticker: _held(y.ticker, 10.0, 5.00, 0.20)}
        [pair] = held_pairs(positions, labels, self._listed(x, y)).values()
        assert pair.sides == ((x.ticker, "no"), (y.ticker, "yes"))
        # NO held on x at its NO ask, YES held on y at its YES ask (the other
        # way round would read 10 x 0.45 + 10 x 0.54)
        assert pair.value_dollars == 10.0 * 0.56 + 10.0 * 0.47
        # Both markets' fees ($0.20 each) count into the stake
        assert pair.stake_dollars == 10.0 * 0.56 + 10.0 * 0.47 + (0.20 + 0.20)

    @pytest.mark.parametrize("early, late", [
        # Unequal counts, as after a partial unwind: exact, never within a tolerance
        (_held(_RUNG_EARLY, 30.0, 6.0, 0.4), _held(_RUNG_LATE, -29.99, 12.0, 0.5)),
        # The same side held on both
        (_held(_RUNG_EARLY, 30.0, 6.0, 0.4), _held(_RUNG_LATE, 30.0, 12.0, 0.5)),
        (_held(_RUNG_EARLY, -30.0, 6.0, 0.4), _held(_RUNG_LATE, -30.0, 12.0, 0.5)),
        # A count that could not be read
        (_held(_RUNG_EARLY, None, 6.0, 0.4), _held(_RUNG_LATE, -30.0, 12.0, 0.5)),
        # A cost that could not be read
        (_held(_RUNG_EARLY, 30.0, None, 0.4), _held(_RUNG_LATE, -30.0, 12.0, 0.5)),
        (_held(_RUNG_EARLY, 30.0, 6.0, 0.4), _held(_RUNG_LATE, -30.0, 12.0, None)),
        # An exposure of "0.00", or below the finest price times the count
        (_held(_RUNG_EARLY, 30.0, 0.0, 0.4), _held(_RUNG_LATE, -30.0, 12.0, 0.5)),
        (_held(_RUNG_EARLY, 30.0, 6.0, 0.4), _held(_RUNG_LATE, -30.0, 0.0029, 0.5)),
    ], ids=["unequal", "both-yes", "both-no", "count-none", "exposure-none", "fees-none",
            "zero-exposure", "exposure-below-floor"])
    def test_a_shape_that_is_not_an_exact_pair_is_never_added_to(self, early, late):
        positions = {early.ticker: early, late.ticker: late}
        assert held_pairs(positions, self._ladder(), self._rung_markets()) == {}

    def test_a_third_held_market_on_the_ladder_leaves_no_pair(self):
        mid = _star_rung("KXSTAR-14-MAR10", "Mar 10, 2026")
        labels = {**self._ladder(), **self._labels(mid)}
        positions = {_RUNG_EARLY: _held(_RUNG_EARLY, 30.0, 6.0, 0.4),
                     _RUNG_LATE: _held(_RUNG_LATE, -30.0, 12.0, 0.5),
                     mid.ticker: _held(mid.ticker, 5.0, 2.0, 0.1)}
        assert held_pairs(positions, labels, {**self._rung_markets(), **self._listed(mid)}) == {}

    def test_a_lone_leg_whose_partner_paid_out_is_added_to(self):
        # The early rung paid out and left the positions listing; the late
        # rung, alone on its ladder, may be added to, staked at its own worth
        # at today's NO ask plus its own fee (its partner's payout is cash)
        positions = {_RUNG_LATE: _held(_RUNG_LATE, -30.0, 12.0, 0.5)}
        [(key, leg)] = held_pairs(positions, self._ladder(), self._rung_markets()).items()
        assert key == frozenset({_RUNG_LATE})
        assert leg.lone and leg.sides == ((_RUNG_LATE, "no"),)
        assert leg.count == 30.0
        assert leg.cost_dollars == pytest.approx(12.5)
        assert leg.value_dollars == 30.0 * 0.45
        assert leg.fees_dollars == 0.5
        assert leg.stake_dollars == 30.0 * 0.45 + 0.5
        assert leg.labels == self._ladder()[_RUNG_LATE]

    @pytest.mark.parametrize("position", [
        _held(_RUNG_LATE, None, 12.0, 0.5),       # unreadable count
        _held(_RUNG_LATE, -30.0, None, 0.5),      # unreadable exposure
        _held(_RUNG_LATE, -30.0, 12.0, None),     # unreadable fees
        _held(_RUNG_LATE, -30.0, 0.0, 0.5),       # an exposure no contract could cost
    ], ids=["count", "exposure", "fees", "zero-exposure"])
    def test_a_lone_leg_with_an_unreadable_figure_is_never_added_to(self, position):
        assert held_pairs({_RUNG_LATE: position}, self._ladder(), self._rung_markets()) == {}

    def test_a_market_without_labels_joins_nothing(self):
        positions = {_RUNG_EARLY: _held(_RUNG_EARLY, 30.0, 6.0, 0.4),
                     _RUNG_LATE: _held(_RUNG_LATE, -30.0, 12.0, 0.5)}
        labels = self._ladder()
        del labels[_RUNG_LATE]
        assert held_pairs(positions, labels, self._rung_markets()) == {}

    @pytest.mark.parametrize("unknown", ["missing", "empty"])
    def test_a_third_held_market_with_no_known_ladder_leaves_no_pair(self, caplog, unknown):
        # The Mar 10 rung is held and sits on the pair's ladder, but its
        # labels are missing (or empty), so nothing shows the pair is alone
        # on its ladder: adding to it could stack a trade beside the Mar 10 one
        mid = _star_rung("KXSTAR-14-MAR10", "Mar 10, 2026")
        positions = {_RUNG_EARLY: _held(_RUNG_EARLY, 30.0, 6.0, 0.4),
                     _RUNG_LATE: _held(_RUNG_LATE, -30.0, 12.0, 0.5),
                     mid.ticker: _held(mid.ticker, 5.0, 2.0, 0.1)}
        labels = self._ladder()
        if unknown == "empty":
            labels[mid.ticker] = frozenset()
        # Non-vacuous: without the Mar 10 rung the other two are an exact pair
        markets = {**self._rung_markets(), **self._listed(mid)}
        assert held_pairs({t: positions[t] for t in (_RUNG_EARLY, _RUNG_LATE)},
                          labels, markets)
        caplog.clear()
        with caplog.at_level(logging.INFO):
            assert held_pairs(positions, labels, markets) == {}
        # The one line, and nothing else: no pair, and no sizing figure of its own
        assert [r.getMessage() for r in caplog.records] == [
            "Held pairs to add to: none (a held market's ladder is unknown)"]

    def test_the_value_counts_only_the_pair_s_own_markets(self):
        # Other held markets (one unreadable) never count into the pair's
        # worth or its stake: the worth is the pair alone at today's prices,
        # fees left out, and the stake adds only the pair's own two fees
        positions = {_RUNG_EARLY: _held(_RUNG_EARLY, 30.0, 6.0, 0.4),
                     _RUNG_LATE: _held(_RUNG_LATE, -30.0, 12.0, 0.5),
                     "KXRAIN-1": _held("KXRAIN-1", 7.0, 5.0, 0.2),
                     "KXSNOW-1": _held("KXSNOW-1", None, None, None)}
        labels = {**self._ladder(), "KXRAIN-1": frozenset({("event", "KXRAIN")}),
                  "KXSNOW-1": frozenset({("event", "KXSNOW")})}
        found = held_pairs(positions, labels, self._rung_markets())
        pair = found[frozenset({_RUNG_EARLY, _RUNG_LATE})]
        assert pair.value_dollars == 22.5
        assert pair.fees_dollars == 0.4 + 0.5
        assert pair.stake_dollars == 22.5 + (0.4 + 0.5)
        # KXRAIN-1, alone on its ladder, is a lone leg of its own, valued
        # apart (at its exposure: this run does not list it); the unreadable
        # KXSNOW-1 is never added to
        assert found[frozenset({"KXRAIN-1"})].stake_dollars == 5.0 + 0.2
        assert set(found) == {frozenset({_RUNG_EARLY, _RUNG_LATE}), frozenset({"KXRAIN-1"})}

    @pytest.mark.parametrize("raw", [
        None, "", "abc", "0", "0.00", "1", "1.00", "1.5", "-0.20", "nan", "inf", True,
        10 ** 400, [], {"ask": "0.45"},
    ], ids=["none", "empty", "text", "zero", "zero-dollars", "one", "one-dollar",
            "above-one", "negative", "nan", "inf", "bool", "huge-int", "list", "dict"])
    def test_a_side_with_no_usable_ask_counts_at_its_exposure(self, raw):
        # Only a number strictly between 0 and 1 is a price: anything else is
        # a settled, unread or broken quote, and the contracts count at what
        # they cost (the listing's exposure, fees left out); the other market
        # keeps its ask
        positions = {_RUNG_EARLY: _held(_RUNG_EARLY, 30.0, 6.0, 0.4),
                     _RUNG_LATE: _held(_RUNG_LATE, -30.0, 12.0, 0.5)}
        [pair] = held_pairs(positions, self._ladder(), self._rung_markets(late_no=raw)).values()
        assert pair.value_dollars == 30.0 * 0.30 + 12.0 == 21.0
        # The fees still count into the stake
        assert pair.stake_dollars == 21.0 + (0.4 + 0.5)

    def test_a_market_missing_from_the_list_counts_at_its_exposure(self):
        # A held market this run does not list (closed but not yet paid out,
        # or looked up by resolve_held_ladders) has no ask to value it at
        positions = {_RUNG_EARLY: _held(_RUNG_EARLY, 30.0, 6.0, 0.4),
                     _RUNG_LATE: _held(_RUNG_LATE, -30.0, 12.0, 0.5)}
        markets = self._rung_markets()
        del markets[_RUNG_EARLY]
        [pair] = held_pairs(positions, self._ladder(), markets).values()
        assert pair.value_dollars == 6.0 + 30.0 * 0.45 == 19.5
        assert pair.stake_dollars == 19.5 + (0.4 + 0.5)
        # Neither listed: both at their exposure, fees left out of the $18.90 cost
        [pair] = held_pairs(positions, self._ladder(), {}).values()
        assert pair.value_dollars == 6.0 + 12.0
        assert pair.cost_dollars == pytest.approx(18.90)
        # ... and the stake is then exactly what the pair cost, fees included
        assert pair.stake_dollars == 6.0 + 12.0 + (0.4 + 0.5) == pytest.approx(18.90)

    @pytest.mark.parametrize("unusable", ["no-ask", "settled", "unlisted"])
    def test_a_leg_with_no_usable_ask_still_counts_its_fees_in_the_stake(self, unusable):
        # The fees were paid whatever the market quotes today: a leg valued at
        # its exposure (no ask, a settled price, or not listed this run) still
        # adds its fees to the stake, so its add-on is never sized as if they
        # were not spent
        positions = {_RUNG_EARLY: _held(_RUNG_EARLY, 30.0, 6.0, 0.4),
                     _RUNG_LATE: _held(_RUNG_LATE, -30.0, 12.0, 0.5)}
        if unusable == "unlisted":
            markets = self._rung_markets()
            del markets[_RUNG_LATE]
        else:
            markets = self._rung_markets(late_no=None if unusable == "no-ask" else "1.00")
        [pair] = held_pairs(positions, self._ladder(), markets).values()
        # The late leg at its $12.00 exposure, the early one at its 0.30 ask
        assert pair.value_dollars == 30.0 * 0.30 + 12.0 == 21.0
        assert pair.fees_dollars == 0.4 + 0.5
        assert pair.stake_dollars == 21.0 + (0.4 + 0.5) == pytest.approx(21.90)
        # The late leg's own $0.50 fee is in it, not only the early leg's
        assert pair.stake_dollars > pair.value_dollars + 0.4 + 1e-9

    def test_the_run_says_what_it_may_add_to(self, caplog):
        positions = {_RUNG_EARLY: _held(_RUNG_EARLY, 30.0, 6.0, 0.4),
                     _RUNG_LATE: _held(_RUNG_LATE, -30.0, 12.0, 0.5),
                     "KXRAIN-1": _held("KXRAIN-1", 7.0, 5.0, 0.2),
                     "KXSNOW-1": _held("KXSNOW-1", None, None, None)}
        labels = {**self._ladder(), "KXRAIN-1": frozenset({("event", "KXRAIN")}),
                  "KXSNOW-1": frozenset({("event", "KXSNOW")})}
        with caplog.at_level(logging.INFO):
            held_pairs(positions, labels, self._rung_markets())
        # Exactly these lines: the lone leg and the pair, each with its cost
        # (and the fees in it) and its worth today, then the counts, and no
        # sizing figure of held_pairs' own
        assert [r.getMessage() for r in caplog.records] == [
            "Held market to add to (its partner has paid out): YES KXRAIN-1, 7 contracts, "
            "cost $5.20 (fees $0.20), worth $5.00 at today's prices",
            f"Held pair to add to: YES {_RUNG_EARLY} / NO {_RUNG_LATE}, 30 contracts each, "
            "cost $18.90 (fees $0.90), worth $22.50 at today's prices",
            "Held pairs to add to: 1 (other held markets, never added to: 1)",
            "Held markets to add to whose partner has paid out: 1"]

    def test_the_count_line_is_logged_with_no_pair_too(self, caplog):
        with caplog.at_level(logging.INFO):
            assert held_pairs({}, {}, {}) == {}
        assert [r.getMessage() for r in caplog.records] == [
            "Held pairs to add to: 0 (other held markets, never added to: 0)"]


def _named(ticker: str) -> SimpleNamespace:
    """A market with just a ticker, as HeldPair.matches reads one."""
    return SimpleNamespace(ticker=ticker)


class TestHeldPairMatches:
    """HeldPair.matches says whether a candidate buys exactly what the account
    holds: the same two tickers, each leg buying the side held on it. The
    other way round would close the held pair rather than add to it."""

    _PAIR = HeldPair(sides=(("KX-A", "yes"), ("KX-C", "no")), count=30.0,
                     cost_dollars=18.9, value_dollars=18.0, fees_dollars=0.9)

    def test_the_held_sides_match(self):
        # Time-series: YES on market A, NO on market B
        assert self._PAIR.matches(_named("KX-A"), _named("KX-C"), "time_series")
        # Same-title: NO on market A, YES on market B
        assert self._PAIR.matches(_named("KX-C"), _named("KX-A"), "same_title")

    def test_the_legs_the_other_way_round_do_not_match(self):
        assert not self._PAIR.matches(_named("KX-C"), _named("KX-A"), "time_series")
        assert not self._PAIR.matches(_named("KX-A"), _named("KX-C"), "same_title")

    def test_one_ticker_named_twice_does_not_match(self):
        assert not self._PAIR.matches(_named("KX-A"), _named("KX-A"), "time_series")
        assert not self._PAIR.matches(_named("KX-C"), _named("KX-C"), "same_title")

    def test_another_ticker_does_not_match(self):
        assert not self._PAIR.matches(_named("KX-A"), _named("KX-B"), "time_series")

    def test_a_mock_market_does_not_match(self):
        assert not self._PAIR.matches(MagicMock(), MagicMock(), "time_series")

    def test_a_lone_leg_matches_its_held_side_beside_any_other_market(self):
        lone = _lone("KX-C", "no", frozenset())
        assert lone.lone and not self._PAIR.lone
        # Time-series: NO is bought on market B, same-title on market A
        assert lone.matches(_named("KX-NEW"), _named("KX-C"), "time_series")
        assert lone.matches(_named("KX-C"), _named("KX-NEW"), "same_title")
        # The held market bought on its other side, or not bought at all
        assert not lone.matches(_named("KX-C"), _named("KX-NEW"), "time_series")
        assert not lone.matches(_named("KX-A"), _named("KX-B"), "time_series")
        assert not lone.matches(_named("KX-C"), _named("KX-C"), "time_series")


class TestPairHeld:
    """pair_held reads CandidatePair.held by type: only a real HeldPair makes a
    candidate an add-on, never a mock's truthy auto-attribute."""

    def test_a_held_pair_is_returned(self):
        pair = SimpleNamespace(held=TestHeldPairMatches._PAIR)
        assert pair_held(pair) is TestHeldPairMatches._PAIR

    @pytest.mark.parametrize("pair", [
        SimpleNamespace(held=None),
        SimpleNamespace(),
        MagicMock(),
        SimpleNamespace(held={"sides": (("KX-A", "yes"), ("KX-C", "no"))}),
    ], ids=["none", "absent", "mock", "dict"])
    def test_anything_else_is_no_held_pair(self, pair):
        assert pair_held(pair) is None

    def test_a_candidate_pair_holds_nothing_by_default(self):
        pair = CandidatePair(market_a=_named("A"), market_b=_named("B"), pA=0.2, pB=0.6,
                             nA=0.8, tradeable=True, canonical_title="q",
                             pair_type="time_series", nB=0.4)
        assert pair.held is None and pair_held(pair) is None


class TestErrorText:
    """A failed request is described in one short line, never with the
    exchange's full error text."""

    def test_an_http_error_gives_its_status_and_reason(self):
        exc = ApiException(http_resp=_error_reply(404, "Not Found"))
        assert scanner._error_text(exc) == "ApiException (HTTP 404 Not Found)"

    def test_any_other_error_gives_only_the_first_line_of_its_message(self):
        exc = ProtocolError("Connection broken\nX-Header-Dump: HEADER-DUMP")
        assert scanner._error_text(exc) == "ProtocolError: Connection broken"

    def test_an_error_with_no_message_gives_its_type(self):
        assert scanner._error_text(KeyError()) == "KeyError"

    @pytest.mark.parametrize("exc", [
        ApiException(status=500, reason="x" * 500),
        RuntimeError("y" * 500),
    ], ids=["http error", "other error"])
    def test_the_line_is_cut_at_160_characters(self, exc):
        text = scanner._error_text(exc)
        assert len(text) == 160
        assert text.startswith(type(exc).__name__)

    def test_an_error_whose_status_cannot_be_read_is_still_named(self):
        class Broken(Exception):
            @property
            def status(self):
                raise RuntimeError("boom")
        assert scanner._error_text(Broken()) == "Broken"

    def test_an_error_whose_message_cannot_be_read_is_still_named(self):
        class Unprintable(Exception):
            def __str__(self):
                raise RuntimeError("boom")
        assert scanner._error_text(Unprintable()) == "Unprintable"


class TestFetchShardStatuses:
    """GET /exchange/status must be read RAW — the pinned SDK's ExchangeStatus
    model silently drops exchange_index_statuses — and must fail soft."""

    @staticmethod
    def _client(payload, status=200):
        client = MagicMock()
        client.get_exchange_status_without_preload_content = MagicMock(
            return_value=SimpleNamespace(
                status=status, data=json.dumps(payload).encode("utf-8")
            )
        )
        return client

    def test_well_formed_payload_parses_all_fields(self):
        client = self._client({
            "exchange_active": True,
            "exchange_index_statuses": [
                {
                    "exchange_index": 0, "description": "Main",
                    "exchange_active": True, "trading_active": True,
                    "intra_exchange_transfers_active": True,
                },
                {
                    "exchange_index": 1, "description": "Combos",
                    "exchange_active": True, "trading_active": False,
                    "intra_exchange_transfers_active": False,
                },
            ],
        })
        statuses = fetch_shard_statuses(client)
        assert set(statuses) == {0, 1}
        assert statuses[0] == {
            "trading_active": True, "exchange_active": True,
            "intra_exchange_transfers_active": True, "description": "Main",
        }
        assert statuses[1]["trading_active"] is False
        # Read by trader.ensure_shard_collateral()
        assert statuses[1]["intra_exchange_transfers_active"] is False

    def test_string_exchange_index_is_coerced_to_int_key(self):
        client = self._client({"exchange_index_statuses": [
            {"exchange_index": "2", "trading_active": True},
        ]})
        statuses = fetch_shard_statuses(client)
        assert set(statuses) == {2}

    def test_absent_field_returns_none_with_info_log(self, caplog):
        # The sandbox / pre-sharding shape: the breakdown is documented as
        # absent when unavailable. Must degrade to single-shard semantics.
        client = self._client({"exchange_active": True, "trading_active": True})
        with caplog.at_level(logging.INFO):
            assert fetch_shard_statuses(client) is None
        assert "per-shard exchange status unavailable" in caplog.text

    def test_non_list_field_returns_none(self):
        client = self._client({"exchange_index_statuses": {"0": {}}})
        assert fetch_shard_statuses(client) is None

    def test_malformed_entries_are_skipped_not_fatal(self):
        client = self._client({"exchange_index_statuses": [
            "not-a-dict",
            {"description": "no index at all"},
            {"exchange_index": "garbage", "trading_active": True},
            {"exchange_index": 0, "trading_active": True},
        ]})
        statuses = fetch_shard_statuses(client)
        assert set(statuses) == {0}, "one bad record must not discard the good one"

    def test_missing_boolean_fields_default_to_false_but_trading_active_is_none(self):
        # trading_active is deliberately TRI-STATE (TS-04): an absent flag is
        # "unknown", never "halted", because normalising it to False marks
        # every shard inactive and empties the whole ingest. The other two
        # booleans fail CLOSED on anything but a recognised true — a missing
        # transfers flag correctly means "don't move money".
        client = self._client({"exchange_index_statuses": [{"exchange_index": 3}]})
        statuses = fetch_shard_statuses(client)
        assert statuses[3] == {
            "trading_active": None, "exchange_active": False,
            "intra_exchange_transfers_active": False, "description": "",
        }

    def test_api_exception_returns_none_not_raised(self, caplog):
        # An exchange-status hiccup must never abort a scan.
        client = MagicMock()
        client.get_exchange_status_without_preload_content = MagicMock(
            side_effect=RuntimeError("boom")
        )
        with caplog.at_level(logging.INFO):
            assert fetch_shard_statuses(client) is None
        assert "per-shard exchange status unavailable" in caplog.text

    def test_non_2xx_status_returns_none(self):
        # fetch_json_page raises ApiException on non-2xx; that must be caught.
        # 400 (not 5xx) so api_call_with_retry doesn't spend its backoff budget.
        client = self._client({"error": "nope"}, status=400)
        assert fetch_shard_statuses(client) is None

    def test_uses_raw_variant_not_modeled_call(self):
        client = self._client({"exchange_index_statuses": []})
        fetch_shard_statuses(client)
        client.get_exchange_status_without_preload_content.assert_called_once()
        client.get_exchange_status.assert_not_called()


class TestFetchShardStatusesUnknownFlag:
    """TS-04: a renamed or dropped trading_active field must read as UNKNOWN
    (None) and keep the shard. Coercing absence to False marks every shard
    inactive, drops every ingested market, and the run still exits 0 claiming
    full coverage — the exact drift class that already hit markets, positions,
    orders, events and balance."""

    @staticmethod
    def _client(entries):
        client = MagicMock()
        client.get_exchange_status_without_preload_content = MagicMock(
            return_value=SimpleNamespace(
                status=200,
                data=json.dumps({"exchange_index_statuses": entries}).encode("utf-8"),
            )
        )
        return client

    def test_absent_key_is_none_and_warns_once(self, caplog):
        client = self._client([
            {"exchange_index": 0, "description": "Main"},
            {"exchange_index": 1, "description": "Combos"},
        ])
        with caplog.at_level(logging.WARNING):
            statuses = fetch_shard_statuses(client)
        assert statuses[0]["trading_active"] is None
        assert statuses[1]["trading_active"] is None
        drift = [
            r for r in caplog.records if "no trading_active flag" in r.getMessage()
        ]
        assert len(drift) == 1, "one summary WARNING, not one line per shard"
        assert "2 shard(s)" in drift[0].getMessage()

    def test_explicit_null_value_is_also_none(self):
        client = self._client([{"exchange_index": 0, "trading_active": None}])
        assert fetch_shard_statuses(client)[0]["trading_active"] is None

    def test_explicit_false_is_still_false(self, caplog):
        client = self._client([{"exchange_index": 0, "trading_active": False}])
        with caplog.at_level(logging.WARNING):
            statuses = fetch_shard_statuses(client)
        assert statuses[0]["trading_active"] is False
        assert "no trading_active flag" not in caplog.text, "silent at zero unknowns"

    def test_explicit_true_is_still_true(self):
        client = self._client([{"exchange_index": 0, "trading_active": True}])
        assert fetch_shard_statuses(client)[0]["trading_active"] is True

    def test_truthy_non_bool_is_coerced_to_true(self):
        # 1 is a recognised spelling of a wire boolean, exactly as bool(1) read
        # it before; see TestExchangeFlagDrift for the values that do change.
        client = self._client([{"exchange_index": 0, "trading_active": 1}])
        assert fetch_shard_statuses(client)[0]["trading_active"] is True


class TestExchangeFlagDrift:
    """TS-04b: an /exchange/status boolean that arrives RE-TYPED must keep its
    meaning. bool("false") is True, so the bare coercion read a HALTED shard as
    open (scanned and traded) and an un-transferable shard as movable (a real,
    non-idempotent collateral POST). Unrecognised values resolve in the one
    direction that cannot break a correct reading: falsy keeps its False,
    truthy becomes unknown."""

    @staticmethod
    def _client(entries):
        client = MagicMock()
        client.get_exchange_status_without_preload_content = MagicMock(
            return_value=SimpleNamespace(
                status=200,
                data=json.dumps({"exchange_index_statuses": entries}).encode("utf-8"),
            )
        )
        return client

    @pytest.mark.parametrize("raw, expected", [
        # Real booleans and the conventional numeric spellings.
        (True, True), (False, False), (0, False), (1, True),
        (0.0, False), (1.0, True),
        # Recognised re-typings — the cases the bare bool() got WRONG.
        ("false", False), ("FALSE", False), (" false ", False),
        ("no", False), ("0", False), ("off", False), ("f", False), ("n", False),
        ("true", True), ("True", True), ("yes", True), ("1", True), ("on", True),
        # A stringified null is truthy in Python but carries no information:
        # unknown, exactly like an absent key.
        ("null", None), ("NULL", None), ("None", None), ("nil", None),
        ("undefined", None),
        # Unrecognised: falsy keeps its historical False, truthy is unknown.
        ("", False), ([], False), ({}, False),
        (2, None), ("maybe", None), ([1], None), (3.5, None),
        # Absent / null.
        (None, None),
    ])
    def test_flag_table(self, raw, expected):
        assert scanner._status_flag(raw) is expected

    def test_drifted_false_halts_the_shard_end_to_end(self, caplog):
        client = self._client([
            {"exchange_index": 0, "trading_active": "false", "description": "Main"},
            {"exchange_index": 1, "trading_active": True, "description": "Combos"},
        ])
        with caplog.at_level(logging.WARNING):
            statuses = fetch_shard_statuses(client)
        assert statuses[0]["trading_active"] is False
        # Ingest must DROP it, exactly as an explicit JSON false would.
        assert inactive_shard_indexes(statuses) == {0}
        # And the coverage audit must not then CRITICAL about the shard whose
        # markets it deliberately dropped.
        critical, warnings = check_shard_coverage(statuses, {1}, {0, 1})
        assert critical == [] and warnings == []
        # Drift is reported, naming the shard, the flag and the raw value.
        assert "shard 0 trading_active='false' -> False" in caplog.text

    def test_drifted_false_transfers_flag_blocks_money_movement(self):
        client = self._client([{
            "exchange_index": 0, "trading_active": True,
            "intra_exchange_transfers_active": "false",
        }])
        statuses = fetch_shard_statuses(client)
        # The money consequence: trader._transfers_active reads this key.
        assert statuses[0]["intra_exchange_transfers_active"] is False

    def test_unreadable_transfers_flag_fails_closed(self):
        # The enabling flags are stored as `_status_flag(...) is True`, so an
        # UNREADABLE value refuses the transfer instead of coercing truthy and
        # firing a real, non-idempotent, never-retried collateral POST.
        client = self._client([{
            "exchange_index": 0, "trading_active": True,
            "intra_exchange_transfers_active": "maybe", "exchange_active": 2,
        }])
        statuses = fetch_shard_statuses(client)
        assert statuses[0]["intra_exchange_transfers_active"] is False
        assert statuses[0]["exchange_active"] is False

    def test_stringified_null_is_unknown_not_open(self, caplog):
        # "null" is a truthy STRING, so bool() read it as an open shard. It is
        # unknown: the shard stays in the ingest (TS-04) but the money flag
        # still refuses.
        client = self._client([{
            "exchange_index": 0, "trading_active": "null",
            "intra_exchange_transfers_active": "null",
        }])
        with caplog.at_level(logging.WARNING):
            statuses = fetch_shard_statuses(client)
        assert statuses[0]["trading_active"] is None
        assert inactive_shard_indexes(statuses) == set(), "unknown is not halted"
        assert statuses[0]["intra_exchange_transfers_active"] is False
        assert "shard 0 trading_active='null' -> None" in caplog.text
        # The absent-flag counter counts ABSENT keys; a present-but-unreadable
        # one is named individually above instead.
        assert "no trading_active flag" not in caplog.text

    def test_real_booleans_log_no_drift_warning(self, caplog):
        client = self._client([{
            "exchange_index": 0, "trading_active": True,
            "exchange_active": True, "intra_exchange_transfers_active": False,
        }])
        with caplog.at_level(logging.WARNING):
            fetch_shard_statuses(client)
        assert "non-boolean flag value" not in caplog.text, "silent at zero drift"

    def test_absent_flags_are_not_drift_and_keep_their_defaults(self, caplog):
        # Regression pin: absence is unknown (None) for trading_active and
        # fail-CLOSED (False) for the two enabling flags — unchanged.
        client = self._client([{"exchange_index": 3}])
        with caplog.at_level(logging.WARNING):
            statuses = fetch_shard_statuses(client)
        assert statuses[3] == {
            "trading_active": None, "exchange_active": False,
            "intra_exchange_transfers_active": False, "description": "",
        }
        assert "non-boolean flag value" not in caplog.text

    @pytest.mark.parametrize("raw", [0, 0.0, "", [], {}])
    def test_falsy_non_bools_still_halt_the_shard(self, raw):
        # NO-REGRESSION pin against the rejected "non-bool means unknown"
        # design, which turned every one of these from DROP into SCAN+TRADE.
        client = self._client([{"exchange_index": 0, "trading_active": raw}])
        statuses = fetch_shard_statuses(client)
        assert statuses[0]["trading_active"] is False
        assert inactive_shard_indexes(statuses) == {0}

    @pytest.mark.parametrize("raw", [2, "maybe"])
    def test_unrecognised_truthy_values_still_keep_the_shard_scanned(self, raw):
        # These read True before and read None (unknown) now; every
        # trading_active consumer treats the two identically, so the shard is
        # still ingested, still audited and still scannable.
        client = self._client([{"exchange_index": 0, "trading_active": raw}])
        statuses = fetch_shard_statuses(client)
        assert statuses[0]["trading_active"] is None
        assert inactive_shard_indexes(statuses) == set()
        assert check_shard_coverage(statuses, set(), {0}) == (
            ["advertised active shard 0 () produced zero ingested markets but "
             "holds account funds"], [],
        )

    def test_drift_warning_bounds_a_huge_raw_value(self, caplog):
        # TS-02 class: the raw value is whatever the API sent, so an unbounded
        # repr would put a multi-KB line in the log on every single run.
        client = self._client([{"exchange_index": 0, "trading_active": list(range(300))}])
        with caplog.at_level(logging.WARNING):
            fetch_shard_statuses(client)
        line = next(
            r.getMessage() for r in caplog.records if "non-boolean flag value" in r.getMessage()
        )
        assert "(truncated)" in line
        assert len(line) < 300, "one drifted flag must not emit a multi-KB line"


class TestInactiveShardIndexes:
    """Only an EXPLICIT trading_active=False drops a shard at ingest (TS-04).
    One leftover truthiness test here would empty the entire market list on a
    renamed field."""

    @staticmethod
    def _st(trading_active):
        return {
            "trading_active": trading_active,
            "exchange_active": True,
            "intra_exchange_transfers_active": True,
            "description": "",
        }

    @pytest.mark.parametrize("statuses, expected", [
        (None, set()),
        ({}, set()),
        ({0: {"trading_active": True}, 1: {"trading_active": True}}, set()),
        ({0: {"trading_active": True}, 1: {"trading_active": False}}, {1}),
        ({0: {"trading_active": False}, 1: {"trading_active": False}}, {0, 1}),
        ({0: {"trading_active": None}, 1: {"trading_active": None}}, set()),
        ({0: {"trading_active": None}, 1: {"trading_active": False}}, {1}),
        ({0: {}}, set()),
    ])
    def test_table(self, statuses, expected):
        assert inactive_shard_indexes(statuses) == expected

    def test_unknown_flag_keeps_every_shard_scannable(self):
        # The TS-04 production shape: /exchange/status renames the field, so
        # every entry parses to None. Ingest must keep them all.
        statuses = {idx: self._st(None) for idx in range(4)}
        assert inactive_shard_indexes(statuses) == set()


class TestFetchOpenEventsShardTagging:
    """Market data is cross-shard: every shard's markets are INGESTED and
    tagged with exchange_index. Only shards the exchange reports as
    trading-inactive are dropped; each order is routed by its own market's
    exchange_index later."""

    @staticmethod
    def _client(std_events, mve_events=()):
        client = MagicMock()
        client.get_events_without_preload_content = MagicMock(
            return_value=_raw_page(list(std_events))
        )
        client.get_multivariate_events_without_preload_content = MagicMock(
            return_value=_raw_page(list(mve_events))
        )
        return client

    def test_standard_loop_keeps_and_tags_every_shard(self):
        event = {"title": "Weather Event", "markets": [
            _raw_market("STD-SHARD0", "Rain tomorrow", exchange_index=0),
            _raw_market("STD-SHARD1", "Snow tomorrow", exchange_index=1),
            _raw_market("STD-NOFIELD", "Hail tomorrow"),
        ]}
        markets = fetch_open_events_with_markets(self._client([event]))
        by_ticker = {m.ticker: m for m in markets}
        assert set(by_ticker) == {"STD-SHARD0", "STD-SHARD1", "STD-NOFIELD"}, (
            "markets from every shard must survive ingest"
        )
        assert by_ticker["STD-SHARD1"].exchange_index == 1
        assert by_ticker["STD-SHARD0"].exchange_index == 0
        # Missing field is the pre-sharding shape — fail-safe to the default
        assert by_ticker["STD-NOFIELD"].exchange_index == DEFAULT_EXCHANGE_INDEX

    def test_no_non_routable_shard_warning_is_logged(self, caplog):
        # The old drop-at-ingest warning must be gone — a shard-1 market is
        # now a perfectly normal ingest, not an anomaly.
        event = {"title": "Weather Event", "markets": [
            _raw_market("STD-SHARD1", "Snow tomorrow", exchange_index=1),
        ]}
        with caplog.at_level(logging.WARNING):
            markets = fetch_open_events_with_markets(self._client([event]))
        assert {m.ticker for m in markets} == {"STD-SHARD1"}
        assert "non-routable" not in caplog.text
        assert "trading-inactive" not in caplog.text

    @pytest.mark.skipif(not INCLUDE_MVE_MARKETS, reason="MVE scanning disabled in config")
    def test_mve_loop_also_keeps_and_tags_every_shard(self):
        mve_event = {"title": "2024 Election Winner", "markets": [
            _raw_market("MVE-SHARD0", "Trump", status="active", exchange_index=0),
            _raw_market("MVE-SHARD1", "Harris", status="active", exchange_index=1),
        ]}
        markets = fetch_open_events_with_markets(self._client([], [mve_event]))
        by_ticker = {m.ticker: m for m in markets}
        assert set(by_ticker) == {"MVE-SHARD0", "MVE-SHARD1"}
        assert by_ticker["MVE-SHARD1"].exchange_index == 1

    def test_inactive_shard_markets_are_dropped_with_warning(self, caplog):
        event = {"title": "Weather Event", "markets": [
            _raw_market("STD-SHARD0", "Rain tomorrow", exchange_index=0),
            _raw_market("STD-SHARD1-A", "Snow tomorrow", exchange_index=1),
            _raw_market("STD-SHARD1-B", "Hail tomorrow", exchange_index=1),
        ]}
        with caplog.at_level(logging.WARNING):
            markets = fetch_open_events_with_markets(
                self._client([event]), inactive_shards={1}
            )
        assert {m.ticker for m in markets} == {"STD-SHARD0"}
        assert "Skipped 2 markets on trading-inactive exchange shards [1]" in caplog.text

    @pytest.mark.skipif(not INCLUDE_MVE_MARKETS, reason="MVE scanning disabled in config")
    def test_inactive_shard_drop_spans_both_loops_in_one_count(self, caplog):
        event = {"title": "Weather Event", "markets": [
            _raw_market("STD-SHARD1", "Snow tomorrow", exchange_index=1),
        ]}
        mve_event = {"title": "2024 Election Winner", "markets": [
            _raw_market("MVE-SHARD1", "Harris", status="active", exchange_index=1),
        ]}
        with caplog.at_level(logging.WARNING):
            markets = fetch_open_events_with_markets(
                self._client([event], [mve_event]), inactive_shards={1}
            )
        assert markets == []
        # One summary line covering the standard AND MVE loops
        assert "Skipped 2 markets on trading-inactive exchange shards [1]" in caplog.text

    def test_no_warning_when_no_shard_is_inactive(self, caplog):
        event = {"title": "Weather Event", "markets": [
            _raw_market("STD-A", "Rain tomorrow", exchange_index=0),
            _raw_market("STD-B", "Snow tomorrow", exchange_index=1),
        ]}
        with caplog.at_level(logging.WARNING):
            markets = fetch_open_events_with_markets(self._client([event]))
        assert {m.ticker for m in markets} == {"STD-A", "STD-B"}
        assert "trading-inactive" not in caplog.text

    def test_per_shard_ingest_counts_are_logged(self, caplog):
        # Load-bearing diagnostic: the only signal of which shards we actually
        # saw markets on after a category migrates to a new shard.
        event = {"title": "Weather Event", "markets": [
            _raw_market("STD-A", "Rain tomorrow", exchange_index=0),
            _raw_market("STD-B", "Snow tomorrow", exchange_index=1),
            _raw_market("STD-C", "Hail tomorrow", exchange_index=1),
        ]}
        with caplog.at_level(logging.INFO):
            fetch_open_events_with_markets(self._client([event]))
        assert "Ingested markets by shard: {0: 1, 1: 2}" in caplog.text


class TestCheckShardCoverage:
    """Pure comparison of what /exchange/status advertises against what a run
    actually observed. Severity is deliberately empty-vs-funded: an active
    advertised shard with zero markets is only CRITICAL when money is parked
    there (a real blind spot); otherwise it's a WARNING, since an advertised
    shard being legitimately empty is expected during the shard rollout and
    must not cry wolf at critical severity every week."""

    @staticmethod
    def _status(trading_active=True, description="Main"):
        return {
            "trading_active": trading_active,
            "exchange_active": True,
            "intra_exchange_transfers_active": True,
            "description": description,
        }

    def test_full_coverage_is_silent(self):
        advertised = {0: self._status(description="Main"), 1: self._status(description="Combos")}
        critical, warnings = check_shard_coverage(advertised, {0, 1}, {0})
        assert critical == []
        assert warnings == []

    def test_active_shard_zero_markets_zero_funds_is_warning_only(self):
        advertised = {0: self._status(), 1: self._status(description="Combos")}
        critical, warnings = check_shard_coverage(advertised, {0}, set())
        assert critical == []
        assert len(warnings) == 1
        assert "shard 1" in warnings[0]
        assert "Combos" in warnings[0]
        assert "may be legitimately empty" in warnings[0]

    def test_active_shard_zero_markets_with_funds_is_critical(self):
        advertised = {0: self._status(), 1: self._status(description="Combos")}
        critical, warnings = check_shard_coverage(advertised, {0}, {1})
        assert warnings == []
        assert len(critical) == 1
        assert "shard 1" in critical[0]
        assert "Combos" in critical[0]
        assert "holds account funds" in critical[0]

    def test_inactive_advertised_shard_with_zero_markets_is_not_a_problem(self):
        advertised = {
            0: self._status(),
            1: self._status(trading_active=False, description="Crypto"),
        }
        critical, warnings = check_shard_coverage(advertised, {0}, set())
        assert critical == []
        assert warnings == []

    def test_inactive_advertised_shard_with_funds_is_still_not_flagged(self):
        # Its markets were deliberately dropped at ingest (and that drop logs
        # its own warning) — re-reporting it here would be double-alerting.
        advertised = {
            0: self._status(),
            1: self._status(trading_active=False, description="Crypto"),
        }
        critical, warnings = check_shard_coverage(advertised, {0}, {0, 1})
        assert critical == []
        assert warnings == []

    def test_market_shard_not_advertised_is_critical(self):
        advertised = {0: self._status()}
        critical, warnings = check_shard_coverage(advertised, {0, 7}, set())
        assert warnings == []
        assert len(critical) == 1
        assert "shard 7" in critical[0]
        assert "does not advertise" in critical[0]

    def test_balance_shard_not_advertised_is_warning(self):
        advertised = {0: self._status()}
        critical, warnings = check_shard_coverage(advertised, {0}, {0, 9})
        assert critical == []
        assert len(warnings) == 1
        assert "shard 9" in warnings[0]
        assert "does not advertise" in warnings[0]

    def test_advertised_none_is_unknowable_even_with_weird_observed_sets(self):
        critical, warnings = check_shard_coverage(None, {0, 1, 2, 99}, {0, 5})
        assert critical == []
        assert warnings == []

    def test_empty_advertised_dict_is_not_treated_as_none(self):
        # {} is a real (if degenerate) breakdown, not "unavailable": an
        # observed market shard it doesn't list is still a disagreement.
        critical, warnings = check_shard_coverage({}, {0}, set())
        assert len(critical) == 1
        assert "shard 0" in critical[0]

    def test_multiple_simultaneous_problems_all_reported_correctly(self):
        advertised = {
            0: self._status(description="Main"),
            1: self._status(description="Combos"),  # active, zero markets, funds -> critical
            2: self._status(trading_active=False, description="Crypto"),  # never a problem
        }
        market_shards = {0, 7}  # 7 unadvertised -> critical
        balance_shards = {1, 9}  # 1 funds shard 1 -> critical; 9 unadvertised -> warning
        critical, warnings = check_shard_coverage(advertised, market_shards, balance_shards)

        assert len(critical) == 2
        assert any("shard 1" in p and "holds account funds" in p for p in critical)
        assert any("shard 7" in p and "does not advertise" in p for p in critical)

        assert len(warnings) == 1
        assert "shard 9" in warnings[0]
        assert "does not advertise" in warnings[0]

    def test_unknown_trading_active_shard_is_still_audited(self):
        # TS-04: an absent flag (None) leaves the shard IN the ingest, so its
        # coverage must still be checked. Only an explicit False is skipped —
        # a truthiness test here would silently stop auditing drifted shards.
        advertised = {0: self._status(trading_active=None, description="Main")}
        critical, warnings = check_shard_coverage(advertised, set(), {0})
        assert len(critical) == 1
        assert "shard 0" in critical[0]
        assert "holds account funds" in critical[0]
        assert warnings == []

    def test_explicit_false_shard_is_still_never_flagged(self):
        # The contrast case: an explicitly halted shard already warned at
        # ingest, so it is not reported here even while holding funds.
        advertised = {0: self._status(trading_active=False, description="Main")}
        critical, warnings = check_shard_coverage(advertised, set(), {0})
        assert critical == []
        assert warnings == []


class TestSubtitleFallback:
    """The API dropped `subtitle` (2026-08 drift) — ingest must read yes_sub_title."""

    def _parse_one(self, **extra):
        # Route through the real raw-JSON parse path (fetch_open_events_with_markets
        # → _market_from_dict), not the dataclass constructor, so the test covers
        # the actual ingest the live scanner uses.
        event = {"title": "Papal Conclave", "markets": [
            _raw_market("SUB-1", "Who will the next Pope be?", **extra),
        ]}
        client = MagicMock()
        client.get_events_without_preload_content = MagicMock(return_value=_raw_page([event]))
        client.get_multivariate_events_without_preload_content = MagicMock(
            return_value=_raw_page([])
        )
        [m] = fetch_open_events_with_markets(client)
        return m

    def test_explicit_subtitle_wins_over_yes_sub_title(self):
        m = self._parse_one(subtitle="Legacy Label", yes_sub_title="New Label")
        assert m.subtitle == "Legacy Label"

    def test_yes_sub_title_used_when_subtitle_absent(self):
        m = self._parse_one(yes_sub_title="Pierbattista Pizzaballa")
        assert m.subtitle == "Pierbattista Pizzaballa"

    def test_null_subtitle_falls_back_to_yes_sub_title(self):
        # The archive/live payloads sometimes carry an explicit null rather than
        # omitting the key — that must still fall through to yes_sub_title.
        m = self._parse_one(subtitle=None, yes_sub_title="Peter Turkson")
        assert m.subtitle == "Peter Turkson"

    def test_both_absent_yields_empty_string(self):
        m = self._parse_one()
        assert m.subtitle == ""

    def test_no_sub_title_is_not_used(self):
        # no_sub_title is the negated phrasing; using it would make the grouping
        # key asymmetric between the YES and NO framings of the same outcome.
        m = self._parse_one(no_sub_title="Someone else")
        assert m.subtitle == ""


class TestSameTitleSubtitleDiscriminator:
    """Regression: distinct outcomes sharing one title must not be paired.

    Before the yes_sub_title fallback, every market parsed with subtitle="",
    so the (event_title, title, subtitle) grouping key collapsed to
    (event_title, title) — two DIFFERENT outcomes under one shared question
    title on different event tickers would group together and be traded under
    the 95% co-resolution assumption. Real money, wrong contract.
    """

    @staticmethod
    def _pope_event(event_ticker, ticker, sub, yes_ask, no_ask):
        # Same event *title* on both sides so the event_title component of the
        # grouping key matches — the subtitle is the only discriminator left.
        return {"title": "Papal Conclave", "markets": [
            _raw_market(
                ticker, "Who will the next Pope be?",
                event_ticker=event_ticker,
                yes_sub_title=sub,
                yes_ask_dollars=str(yes_ask), no_ask_dollars=str(no_ask),
            ),
        ]}

    @staticmethod
    def _markets(events):
        client = MagicMock()
        client.get_events_without_preload_content = MagicMock(return_value=_raw_page(events))
        client.get_multivariate_events_without_preload_content = MagicMock(
            return_value=_raw_page([])
        )
        return fetch_open_events_with_markets(client)

    def test_distinct_outcomes_under_shared_title_do_not_pair(self):
        # RE-PINNED (DR-02/DR-54): the fixture used EVT-A/EVT-B, one series, so
        # after the one-series rule this assertion would have passed for the
        # wrong reason — the pair would be refused as two instances of one
        # fixture before the subtitle was ever consulted. Two distinct series
        # keep the SUBTITLE the only thing separating these two markets.
        markets = self._markets([
            self._pope_event("EVA-1", "POPE-PIZZABALLA", "Pierbattista Pizzaballa", 0.45, 0.55),
            self._pope_event("EVB-1", "POPE-TURKSON", "Peter Turkson", 0.30, 0.70),
        ])
        # Both parsed subtitles must be populated — otherwise the assertion
        # below would pass for the wrong reason (e.g. a parse failure).
        assert {m.subtitle for m in markets} == {"Pierbattista Pizzaballa", "Peter Turkson"}
        pairs = find_same_title_pairs(markets)
        assert pairs == [], f"Different outcomes must not be same-title paired; got {pairs}"

    def test_identical_outcome_across_events_still_pairs(self):
        # Positive control: same outcome label, different event tickers, price
        # gap well above SAME_TITLE_MIN_PRICE_DIFF — the legitimate arbitrage.
        #
        # RE-PINNED (DR-02/DR-54): EVT-A/EVT-B shared the series prefix "EVT",
        # which the finder now reads as two instances of one recurring fixture
        # and refuses. The tickers name two DIFFERENT series now — one question
        # listed by two independent series, the shape the co-resolution prior
        # was built for — so the assertion is unchanged.
        markets = self._markets([
            self._pope_event("EVA-1", "POPE-A", "Pierbattista Pizzaballa", 0.45, 0.55),
            self._pope_event("EVB-1", "POPE-B", "Pierbattista Pizzaballa", 0.30, 0.70),
        ])
        pairs = find_same_title_pairs(markets)
        assert len(pairs) == 1
        assert {pairs[0].market_a.ticker, pairs[0].market_b.ticker} == {"POPE-A", "POPE-B"}
        # market_a is canonicalized to the more expensive side
        assert pairs[0].market_a.ticker == "POPE-A"


class TestFilterMarketsWithinHorizon:
    def test_none_horizon_returns_markets_unchanged(self):
        markets = [
            _mock_market(ticker="T1", event_ticker="E1", close_time=datetime(2026, 6, 1, tzinfo=UTC)),
            _mock_market(ticker="T2", event_ticker="E2", close_time=None),
        ]
        result = filter_markets_within_horizon(markets, None)
        assert result == markets

    def test_market_within_horizon_is_kept(self):
        now = datetime.now(UTC)
        m = _mock_market(ticker="T1", event_ticker="E1", close_time=now + timedelta(days=3))
        result = filter_markets_within_horizon([m], 7)
        assert result == [m]

    def test_market_beyond_horizon_is_dropped(self):
        now = datetime.now(UTC)
        m = _mock_market(ticker="T1", event_ticker="E1", close_time=now + timedelta(days=30))
        result = filter_markets_within_horizon([m], 7)
        assert result == []

    def test_none_close_time_dropped_when_horizon_active(self):
        # _mock_market's close_time=None falls back to its own default via
        # `close_time or ...`, so build the None case directly
        m = _mock_market(ticker="T1", event_ticker="E1")
        m.close_time = None
        result = filter_markets_within_horizon([m], 7)
        assert result == []

    def test_boundary_close_time_at_cutoff_is_kept(self):
        # cutoff is computed as now + max_horizon_days at call time, so a
        # close_time set slightly earlier than a fresh now() + horizon
        # reliably lands at-or-before the cutoff, exercising the inclusive <=
        now = datetime.now(UTC)
        m = _mock_market(ticker="T1", event_ticker="E1", close_time=now + timedelta(days=7) - timedelta(seconds=1))
        result = filter_markets_within_horizon([m], 7)
        assert result == [m]

    def test_naive_close_time_dropped_not_raised(self):
        m = _mock_market(ticker="T1", event_ticker="E1", close_time=datetime(2026, 12, 1))
        result = filter_markets_within_horizon([m], 7)
        assert result == []


class TestParsePriceRanges:
    """_parse_price_ranges: ingest for the price_ranges tick-band array
    (2026-08). These tests cover the parse itself; tick_size_for_price's
    reading of the parsed bands is covered by TestTickSizeForPrice below."""

    def test_single_band_deci_cent(self):
        bands = _parse_price_ranges(
            [{"start": "0.0000", "end": "1.0000", "step": "0.0010"}]
        )
        assert bands == [PriceRange(start=0.0, end=1.0, step=0.001)]

    def test_multi_band_tapered_market(self):
        raw = [
            {"start": "0.0000", "end": "0.1000", "step": "0.0010"},
            {"start": "0.1000", "end": "0.9000", "step": "0.0100"},
            {"start": "0.9000", "end": "1.0000", "step": "0.0010"},
        ]
        bands = _parse_price_ranges(raw)
        assert bands == [
            PriceRange(start=0.0, end=0.1, step=0.001),
            PriceRange(start=0.1, end=0.9, step=0.01),
            PriceRange(start=0.9, end=1.0, step=0.001),
        ]

    def test_none_returns_none(self):
        assert _parse_price_ranges(None) is None

    def test_not_a_list_returns_none(self):
        assert _parse_price_ranges({"start": "0", "end": "1", "step": "0.01"}) is None

    def test_empty_list_returns_none(self):
        assert _parse_price_ranges([]) is None

    def test_all_bands_missing_keys_returns_none(self):
        # Every band fails to parse -> the result must be None, not [].
        assert _parse_price_ranges([{"foo": "bar"}, {"start": "0"}]) is None

    def test_one_good_one_malformed_band_keeps_only_the_good_one(self):
        raw = [
            {"start": "0.0000", "end": "0.5000", "step": "0.0100"},
            {"start": "0.5000", "end": "1.0000"},  # missing "step"
        ]
        bands = _parse_price_ranges(raw)
        assert bands == [PriceRange(start=0.0, end=0.5, step=0.01)]

    def test_price_range_is_frozen(self):
        band = PriceRange(start=0.0, end=1.0, step=0.01)
        with pytest.raises(dataclasses.FrozenInstanceError):
            band.start = 0.5


class TestTickStructureIngest:
    """Tick-structure fields flow through the real raw-JSON parse path
    (fetch_open_events_with_markets -> _market_from_dict), matching how the
    subtitle-fallback tests above exercise ingest."""

    def _parse_one(self, **extra):
        event = {"title": "Weather Event", "markets": [
            _raw_market("TICK-1", "Rain tomorrow", **extra),
        ]}
        client = MagicMock()
        client.get_events_without_preload_content = MagicMock(return_value=_raw_page([event]))
        client.get_multivariate_events_without_preload_content = MagicMock(
            return_value=_raw_page([])
        )
        [m] = fetch_open_events_with_markets(client)
        return m

    def test_deci_cent_single_band(self):
        m = self._parse_one(
            price_level_structure="deci_cent",
            price_ranges=[{"start": "0.0000", "end": "1.0000", "step": "0.0010"}],
        )
        assert m.price_level_structure == "deci_cent"
        assert m.price_ranges == [PriceRange(start=0.0, end=1.0, step=0.001)]

    def test_tapered_multi_band(self):
        m = self._parse_one(
            price_level_structure="tapered_deci_cent",
            price_ranges=[
                {"start": "0.0000", "end": "0.1000", "step": "0.0010"},
                {"start": "0.1000", "end": "0.9000", "step": "0.0100"},
                {"start": "0.9000", "end": "1.0000", "step": "0.0010"},
            ],
        )
        assert m.price_level_structure == "tapered_deci_cent"
        assert len(m.price_ranges) == 3
        assert [b.step for b in m.price_ranges] == [0.001, 0.01, 0.001]

    def test_absent_fields_default_to_empty_and_none(self):
        m = self._parse_one()
        assert m.price_level_structure == ""
        assert m.price_ranges is None


class TestTickSizeForPrice:
    """tick_size_for_price: maps a price onto the market's own tick grid,
    falling back to the coarse $0.01 grid (valid on every regime, since the
    grids are nested) whenever the structure is uniform-cent or unusable."""

    # center_deci_edge_centi_cent: $0.0001 below $0.01 and above $0.99,
    # $0.001 in between — the regime MVE/combo markets moved to on 2026-08-17.
    _CENTI_BANDS = [
        PriceRange(start=0.0, end=0.01, step=0.0001),
        PriceRange(start=0.01, end=0.99, step=0.001),
        PriceRange(start=0.99, end=1.0, step=0.0001),
    ]

    @staticmethod
    def _market(structure: str, ranges) -> SimpleNamespace:
        """SimpleNamespace market stand-in carrying only the tick fields read."""
        return SimpleNamespace(
            ticker="KXTEST-1", price_level_structure=structure, price_ranges=ranges,
        )

    def test_linear_cent_returns_one_cent(self):
        m = self._market("linear_cent", [PriceRange(start=0.0, end=1.0, step=0.01)])
        assert tick_size_for_price(m, 0.5) == Decimal("0.01")

    def test_empty_structure_returns_one_cent(self):
        m = self._market("", None)
        assert tick_size_for_price(m, 0.5) == Decimal("0.01")

    def test_deci_cent_uniform_band(self):
        m = self._market("deci_cent", [PriceRange(start=0.0, end=1.0, step=0.001)])
        assert tick_size_for_price(m, 0.42) == Decimal("0.001")

    def test_centi_cent_middle_band_is_deci_cent(self):
        m = self._market("center_deci_edge_centi_cent", self._CENTI_BANDS)
        assert tick_size_for_price(m, 0.50) == Decimal("0.001")

    def test_centi_cent_edge_bands(self):
        m = self._market("center_deci_edge_centi_cent", self._CENTI_BANDS)
        assert tick_size_for_price(m, 0.005) == Decimal("0.0001")
        assert tick_size_for_price(m, 0.995) == Decimal("0.0001")

    def test_lower_band_boundary_resolves_to_the_finer_band(self):
        # 0.01 is the end of band 1 ($0.0001) and the start of band 2
        # ($0.001). Both contain it; the FINER one wins. First-match happened
        # to agree here, which is why the old convention looked correct.
        m = self._market("center_deci_edge_centi_cent", self._CENTI_BANDS)
        assert tick_size_for_price(m, 0.01) == Decimal("0.0001")

    def test_upper_band_boundary_resolves_to_the_finer_band(self):
        # TS-10. 0.99 is the end of the $0.001 middle band and the start of
        # the $0.0001 top band. First-match returned the EARLIER band, which
        # at an upper edge is 10x COARSER — so the V2 buy cap
        # (ceil(scanned) + BUY_SLIPPAGE_TICKS x tick) got a 10x tick of
        # slippage exactly there, loosening a cap that is a bid.
        m = self._market("center_deci_edge_centi_cent", self._CENTI_BANDS)
        assert tick_size_for_price(m, 0.99) == Decimal("0.0001")

    def test_finest_wins_regardless_of_band_order(self):
        # The rule is "finest containing", not "last containing" — a coarse
        # band listed after a fine one must not win either.
        m = self._market("tapered_deci_cent", [
            PriceRange(start=0.0, end=1.0, step=0.01),
            PriceRange(start=0.0, end=1.0, step=0.001),
        ])
        assert tick_size_for_price(m, 0.5) == Decimal("0.001")

    def test_malformed_band_does_not_discard_valid_later_bands(self):
        # The old loop `break`s out of the whole list on the first unreadable
        # band, so a single drifted entry silently downgraded the market to
        # the $0.01 default. Scanning on recovers the real grid.
        m = self._market("deci_cent", [
            SimpleNamespace(start=None, end=None, step=None),
            PriceRange(start=0.0, end=1.0, step=0.001),
        ])
        assert tick_size_for_price(m, 0.5) == Decimal("0.001")

    def test_nonpositive_step_does_not_mask_a_valid_containing_band(self):
        # A zero-step band used to `break` the loop too. It must be skipped,
        # not treated as the answer and not treated as the end of the list.
        m = self._market("deci_cent", [
            PriceRange(start=0.0, end=1.0, step=0.0),
            PriceRange(start=0.0, end=1.0, step=0.001),
        ])
        assert tick_size_for_price(m, 0.5) == Decimal("0.001")

    def test_none_ranges_falls_back(self):
        # Structure names a fine grid but the bands failed to parse — the
        # coarse default is still a valid grid point on every regime.
        m = self._market("deci_cent", None)
        assert tick_size_for_price(m, 0.5) == Decimal("0.01")

    def test_no_containing_band_falls_back_with_warning(self, caplog):
        m = self._market("deci_cent", [PriceRange(start=0.0, end=0.4, step=0.001)])
        with caplog.at_level(logging.WARNING):
            assert tick_size_for_price(m, 0.9) == Decimal("0.01")
        assert "KXTEST-1" in caplog.text

    def test_nonpositive_step_falls_back_with_warning(self, caplog):
        m = self._market("deci_cent", [PriceRange(start=0.0, end=1.0, step=0.0)])
        with caplog.at_level(logging.WARNING):
            assert tick_size_for_price(m, 0.5) == Decimal("0.01")
        assert "KXTEST-1" in caplog.text


class TestFetchShardStatusesEmptyBreakdown:
    def test_empty_status_list_means_unavailable_not_zero_shards(self):
        # Regression (adversarial review): an empty exchange_index_statuses
        # list must map to None (single-shard semantics) like every other
        # unavailable shape — {} would falsely CRITICAL every ingested shard
        # in check_shard_coverage and block every collateral transfer.
        client = MagicMock()
        client.get_exchange_status_without_preload_content = MagicMock(
            return_value=SimpleNamespace(
                status=200,
                data=json.dumps({"exchange_index_statuses": []}).encode(),
            )
        )
        assert fetch_shard_statuses(client) is None

    def test_all_unparseable_entries_also_mean_unavailable(self):
        client = MagicMock()
        client.get_exchange_status_without_preload_content = MagicMock(
            return_value=SimpleNamespace(
                status=200,
                data=json.dumps(
                    {"exchange_index_statuses": ["garbage", {"no": "index"}]}
                ).encode(),
            )
        )
        assert fetch_shard_statuses(client) is None


class TestFilterActiveMarketsCloseTimeGuard:
    """A market whose close_time failed to parse must be dropped at the shared
    entry filter, not carried into the finders where a None deadline crashes
    the close_time sort, the orderbook ceiling, and strategy.compute_trade()."""

    @staticmethod
    def _bad_close_time_market(ticker: str, event_ticker: str):
        """Build an otherwise-active market whose close_time is None.

        _mock_market's `close_time or datetime(...)` default turns a None
        argument back into a real datetime, so the field is cleared after
        construction to reproduce _market_from_dict's unparseable-date output.

        Args:
            ticker (str): Ticker for the stand-in market.
            event_ticker (str): Parent event ticker for the stand-in market.

        Returns:
            SimpleNamespace: Active-priced market with close_time set to None.
        """
        m = _mock_market(ticker=ticker, event_ticker=event_ticker, title="Will BTC exceed $80k")
        m.close_time = None
        return m

    def test_filter_active_markets_drops_none_close_time_with_one_warning(self, caplog):
        good_1 = _mock_market(ticker="G1", event_ticker="E1", yes_ask=0.40, no_ask=0.60)
        good_2 = _mock_market(ticker="G2", event_ticker="E2", yes_ask=0.30, no_ask=0.70)
        bad = self._bad_close_time_market("B1", "E3")

        with caplog.at_level(logging.WARNING, logger="root"):
            kept = _filter_active_markets([good_1, bad, good_2])

        assert [m.ticker for m in kept] == ["G1", "G2"]
        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "missing/unparseable close_time" in r.getMessage()
        ]
        assert len(warnings) == 1, f"Expected exactly one summary WARNING; got {warnings}"
        assert "Skipped 1 markets with missing/unparseable close_time" in warnings[0].getMessage()

    def test_filter_active_markets_silent_when_no_bad_close_times(self, caplog):
        markets = [
            _mock_market(ticker="G1", event_ticker="E1"),
            _mock_market(ticker="G2", event_ticker="E2"),
        ]

        with caplog.at_level(logging.WARNING, logger="root"):
            kept = _filter_active_markets(markets)

        assert [m.ticker for m in kept] == ["G1", "G2"]
        assert not [
            r for r in caplog.records
            if "missing/unparseable close_time" in r.getMessage()
        ], "No warning may be emitted when every market has a close_time"

    def test_find_time_series_pairs_ignores_member_with_none_close_time(self):
        # Three markets sharing one date-stripped title on three event tickers:
        # two good ones that qualify as a 15%-tier pair (later contract pricier
        # by 20%), plus one whose close_time is None. The sort in
        # find_time_series_pairs used to raise TypeError comparing None against
        # a datetime.
        early_close = datetime(2026, 3, 1, tzinfo=UTC)
        mA = _mock_market(
            ticker="EARLY", event_ticker="EVT-A",
            title="Will BTC exceed $80k by March 01, 2026",
            yes_ask=0.30, no_ask=0.70,
            close_time=early_close,
        )
        mB = _mock_market(
            ticker="LATE", event_ticker="EVT-B",
            title="Will BTC exceed $80k by March 11, 2026",
            yes_ask=0.50, no_ask=0.50,
            close_time=early_close + timedelta(days=10),
        )
        bad = self._bad_close_time_market("NOCLOSE", "EVT-C")

        pairs = find_time_series_pairs(MagicMock(), held_tickers=set(), markets=[mA, bad, mB])

        assert len(pairs) == 1, f"Expected exactly one pair from the two good markets; got {pairs}"
        assert {pairs[0].market_a.ticker, pairs[0].market_b.ticker} == {"EARLY", "LATE"}


class TestBidsToAskLevelsSubCent:
    """
    _bids_to_ask_levels keeps sub-cent order-book levels (TS-14, levels half).

    The bound here is config.MIN/MAX_ACTIVE_PRICE_DOLLARS (0.0001/0.9999), the
    extreme tradeable levels on Kalshi's FINEST grid — deliberately NOT the
    0.01/0.99 market-eligibility bound, which is unchanged.
    """

    def test_sub_cent_ask_level_is_kept(self):
        # NO bid 0.995 -> YES ask 0.005. Real depth on a centi-cent book; the
        # old 0.01 floor discarded it silently.
        assert _bids_to_ask_levels([["0.995", "40"]]) == [(pytest.approx(0.005), 40.0)]

    def test_level_just_under_one_is_kept(self):
        # NO bid 0.002 -> YES ask 0.998, inside 0.9999 but outside the old 0.99.
        assert _bids_to_ask_levels([["0.002", "12"]]) == [(pytest.approx(0.998), 12.0)]

    def test_levels_outside_the_finest_grid_are_still_dropped(self):
        # 1.0 -> ask 0.0 and 0.0 -> ask 1.0 are settled prices, not depth.
        assert _bids_to_ask_levels([["1.0", "5"], ["0.0", "5"]]) == []

    def test_whole_cent_levels_are_unchanged(self):
        # PIN: the common linear-cent path must be byte-identical.
        assert _bids_to_ask_levels([["0.60", "10"], ["0.55", "20"]]) == [
            (pytest.approx(0.40), 10.0), (pytest.approx(0.45), 20.0),
        ]

    def test_dropped_levels_are_counted_and_logged_once(self, caplog):
        raw = [["1.0", "5"], ["0.60", "10"], ["0.55", "0"], ["bogus", "3"]]
        with caplog.at_level(logging.WARNING):
            levels = _bids_to_ask_levels(raw, "KXTEST-9")
        assert levels == [(pytest.approx(0.40), 10.0)]
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "KXTEST-9" in warnings[0]
        assert "dropped 3 of 4" in warnings[0]

    def test_no_warning_when_nothing_is_dropped(self, caplog):
        with caplog.at_level(logging.WARNING):
            _bids_to_ask_levels([["0.60", "10"]], "KXTEST-9")
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    def test_market_eligibility_bound_is_unchanged(self):
        # TS-14 is levels-only by operator decision: no market excluded today
        # becomes tradeable. Guard, not proof of the fix.
        assert scanner._MIN_ACTIVE_PRICE == 0.01
        assert scanner._MAX_ACTIVE_PRICE == 0.99


@pytest.mark.usefixtures("pre_toggle_defaults")
class TestPriceEpsilonThresholds:
    """
    TS-09: prices are floats parsed from cent-quantized dollar strings, so a
    pair sitting EXACTLY on a documented threshold can evaluate a hair under it
    and be rejected for representation noise rather than for its price.
    Measured over live books: the same-title 5c test rejected 50 of 94
    qualifying pairs, the 15c tier 21 of 84, the 30c tier 15 of 69. The tiers
    bind only with the tier floors on (pre_toggle_defaults).
    """

    def test_the_float_noise_this_exists_for_is_real(self):
        # PIN on the premise, not on the fix: if these ever become exact the
        # epsilon is dead weight and should be revisited.
        assert 0.35 - 0.30 < 0.05
        assert 0.35 - 0.20 < 0.15

    def test_epsilon_is_far_below_the_finest_tick(self):
        # GUARD on magnitude. The finest grid in any regime is $0.0001, and
        # the tolerance is at most 1% of one tick, so it can only absorb
        # representation noise — never a real one-tick price difference.
        assert config.PRICE_EPSILON <= 0.0001 / 100

    @staticmethod
    def _same_title(pA: float, pB: float):
        from datetime import UTC, datetime
        close = datetime(2026, 3, 1, tzinfo=UTC)
        # RE-PINNED (DR-02/DR-54): EV-A/EV-B share the series prefix "EV", which
        # the finder now refuses as two instances of one recurring fixture —
        # both the positive and the negative epsilon assertion below would have
        # gone green for that reason instead of for the price threshold they
        # exist to pin. Two DIFFERENT series keep the 5% test the only gate.
        mA = _mock_market(ticker="A1", event_ticker="EA-1", title="Same question",
                          subtitle="Yes", yes_ask=pA, no_ask=round(1.0 - pA, 4),
                          close_time=close)
        mB = _mock_market(ticker="B1", event_ticker="EB-1", title="Same question",
                          subtitle="Yes", yes_ask=pB, no_ask=round(1.0 - pB, 4),
                          close_time=close)
        return find_same_title_pairs([mA, mB])

    def test_same_title_pair_exactly_at_threshold_qualifies(self):
        # 0.35 - 0.30 == 0.04999999999999999, one ULP under the 5% threshold.
        pairs = self._same_title(0.35, 0.30)
        assert len(pairs) == 1
        assert pairs[0].pA == pytest.approx(0.35)

    def test_same_title_pair_a_cent_under_threshold_is_still_rejected(self):
        # GUARD: the epsilon must not admit a genuinely sub-threshold pair.
        assert self._same_title(0.34, 0.30) == []

    def test_time_series_pair_exactly_at_the_short_tier_qualifies(self):
        # 0.35 - 0.20 == 0.14999999999999997, one ULP under the 15% tier.
        mA, mB = _ts_pair_markets(gap_days=10, pA=0.20, pB=0.35)
        pairs = find_time_series_pairs(MagicMock(), held_tickers=set(), markets=[mA, mB])
        assert len(pairs) == 1
        assert pairs[0].pB == pytest.approx(0.35)

    def test_time_series_pair_a_cent_under_the_tier_is_still_rejected(self):
        mA, mB = _ts_pair_markets(gap_days=10, pA=0.21, pB=0.34)
        assert find_time_series_pairs(MagicMock(), held_tickers=set(), markets=[mA, mB]) == []


class TestValidatePairPriceReachableDepth:
    """
    The pre-execution re-check must ask the question the wire asks: will the
    order we are ABOUT TO SUBMIT fill against the book as it stands? Counting
    every contract that merely clears the gap let a spec whose top levels rest
    above its own FoK limit pass here and then be killed on the exchange,
    reported as "NO leg FoK not filled" (TS-08).
    """

    @staticmethod
    def _client(a_yes_bids, b_no_bids):
        client = MagicMock()

        def _raw(ticker, *a, **k):
            book = {"A1": {"yes": a_yes_bids, "no": []},
                    "B1": {"yes": [], "no": b_no_bids}}[ticker]
            return {"orderbook_fp": {"yes_dollars": book["yes"], "no_dollars": book["no"]}}

        client._get = _raw
        return client

    @staticmethod
    def _pair(nA, pB):
        from datetime import UTC, datetime
        close = datetime(2026, 3, 1, tzinfo=UTC)
        mA = SimpleNamespace(ticker="A1", title="A", subtitle="", event_ticker="EV-A",
                             close_time=close, exchange_index=0,
                             price_level_structure="", price_ranges=None)
        mB = SimpleNamespace(ticker="B1", title="B", subtitle="", event_ticker="EV-B",
                             close_time=close, exchange_index=0,
                             price_level_structure="", price_ranges=None)
        return CandidatePair(
            market_a=mA, market_b=mB, pA=1.0 - nA, pB=pB, nA=nA, nB=1.0 - pB,
            tradeable=True, canonical_title="reachability pair",
            pair_type="same_title",
        )

    def _run(self, monkeypatch, x):
        # NO leg (market_a) ladders 0.32@300 then 0.37@300; YES leg flat 0.30.
        # The spec is priced at the top level, so its NO cap is 0.33 and only
        # the first 300 contracts are reachable.
        pair = self._pair(nA=0.32, pB=0.30)
        spec = SimpleNamespace(pair=pair, x=x)
        client = self._client(
            a_yes_bids=[["0.68", "300"], ["0.63", "300"]],
            b_no_bids=[["0.70", "600"]],
        )
        monkeypatch.setattr(scanner, "_fetch_orderbook",
                            lambda c, t: {"A1": {"yes": [["0.68", "300"], ["0.63", "300"]], "no": []},
                                          "B1": {"yes": [], "no": [["0.70", "600"]]}}[t])
        return validate_pair_price(client, spec)

    def test_spec_within_reachable_depth_passes(self):
        import pytest as _p
        mp = _p.MonkeyPatch()
        try:
            assert self._run(mp, 300) is True
        finally:
            mp.undo()

    def test_spec_beyond_reachable_depth_is_dropped(self):
        # 600 contracts clear the gap, but only 300 rest at or below the cap.
        import pytest as _p
        mp = _p.MonkeyPatch()
        try:
            assert self._run(mp, 600) is False
        finally:
            mp.undo()

    def test_the_rejection_names_reachability_not_thin_depth(self, caplog, monkeypatch):
        with caplog.at_level(logging.WARNING):
            self._run(monkeypatch, 600)
        assert "reachable at the FoK limit" in caplog.text

    def test_the_rejection_line_is_pinned_word_for_word(self, caplog, monkeypatch):
        # The drop logs exactly one WARNING, word for word: 300 of the 600
        # contracts rest at or below the caps
        with caplog.at_level(logging.WARNING):
            assert self._run(monkeypatch, 600) is False
        assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING] == [
            "Pre-execution check failed for 'reachability pair' — only 300.0 contracts"
            " reachable at the FoK limit (need 600); dropping"
        ]


class TestCloseTimeWarningOncePerRun:
    """
    TS-22: _filter_active_markets emits one summary WARNING naming how many
    markets it dropped for a missing close_time. Both finders call it on the
    SAME list in one run, so the line appeared TWICE. CLAUDE.md specifies one
    summary WARNING carrying the count.
    """

    @staticmethod
    def _markets():
        from datetime import UTC, datetime
        close = datetime(2026, 3, 1, tzinfo=UTC)
        # RE-PINNED (DR-02/DR-54): EV-A/EV-B share the series prefix "EV", so
        # the good same-title pair this fixture exists to carry alongside the
        # two bad markets was silently skipped as two instances of one fixture.
        # EA-1/EB-1 are two DIFFERENT series, so the pair forms again and the
        # run still exercises pairing while the close-time assertion below
        # (emitted by _filter_active_markets, upstream of the series rule) is
        # unchanged.
        good_a = _mock_market(ticker="A1", event_ticker="EA-1", title="Q",
                              subtitle="Yes", yes_ask=0.35, no_ask=0.65,
                              close_time=close)
        good_b = _mock_market(ticker="B1", event_ticker="EB-1", title="Q",
                              subtitle="Yes", yes_ask=0.30, no_ask=0.70,
                              close_time=close)
        # _mock_market substitutes a default for a falsy close_time, so the
        # missing-deadline markets are built directly.
        bad_1 = SimpleNamespace(ticker="X1", event_ticker="EV-X", title="Q2",
                                subtitle="Yes", yes_ask_dollars="0.40",
                                no_ask_dollars="0.60", yes_bid_dollars="0.38",
                                close_time=None)
        bad_2 = SimpleNamespace(ticker="X2", event_ticker="EV-Y", title="Q2",
                                subtitle="Yes", yes_ask_dollars="0.40",
                                no_ask_dollars="0.60", yes_bid_dollars="0.38",
                                close_time=None)
        return [good_a, good_b, bad_1, bad_2]

    def test_one_warning_across_both_finders_in_one_run(self, caplog):
        # The duplication happens one level up from _filter_active_markets, so
        # a test that invokes it once is tautologically satisfied and cannot
        # see this. Drive BOTH finders on one list, as both run modes do.
        markets = self._markets()
        with caplog.at_level(logging.WARNING):
            find_time_series_pairs(MagicMock(), held_tickers=set(), markets=markets)
            find_same_title_pairs(markets, held_tickers=set())
        lines = [r.getMessage() for r in caplog.records
                 if "missing/unparseable close_time" in r.getMessage()]
        assert len(lines) == 1
        assert "2" in lines[0]

    def test_the_markets_are_still_dropped_by_the_quiet_caller(self):
        # GUARD: the flag suppresses the REPORT, never the filtering.
        markets = self._markets()
        kept = scanner._filter_active_markets(markets, set(), warn_missing_close=False)
        assert [m.ticker for m in kept] == ["A1", "B1"]

    def test_standalone_caller_still_warns_by_default(self, caplog):
        with caplog.at_level(logging.WARNING):
            scanner._filter_active_markets(self._markets(), set())
        assert "missing/unparseable close_time" in caplog.text


class TestTimeSeriesFallbackForwardsShardFilter:
    """
    TS-27: find_time_series_pairs fetches its own markets when the caller
    supplies none, and that fallback passed NO shard filter — so it would
    ingest and pair markets on shards the exchange reports trading_active=false
    for, the one ingest-time exclusion CLAUDE.md calls mandatory.

    DEAD CODE TODAY: both non-test callers pass markets=, as do all the test
    call sites. Unreachable in production, unreachable in the suite, and wrong
    if ever reached.
    """

    def test_fallback_forwards_the_inactive_shard_set(self, monkeypatch):
        seen = {}

        def _fake_fetch(client, inactive_shards=None):
            seen["inactive_shards"] = inactive_shards
            return []

        monkeypatch.setattr(scanner, "fetch_open_events_with_markets", _fake_fetch)
        find_time_series_pairs(MagicMock(), inactive_shards={2, 3})
        # The VALUE, not merely the parameter's existence: a test that only
        # asserted the kwarg was accepted would pass against an inverted or
        # dropped forward.
        assert seen["inactive_shards"] == {2, 3}

    def test_fallback_defaults_to_excluding_nothing(self, monkeypatch):
        seen = {}

        def _fake_fetch(client, inactive_shards=None):
            seen["inactive_shards"] = inactive_shards
            return []

        monkeypatch.setattr(scanner, "fetch_open_events_with_markets", _fake_fetch)
        find_time_series_pairs(MagicMock())
        assert seen["inactive_shards"] is None

    def test_supplied_markets_skip_the_fetch_entirely(self, monkeypatch):
        def _boom(*a, **k):
            raise AssertionError("fallback must not run when markets= is given")

        monkeypatch.setattr(scanner, "fetch_open_events_with_markets", _boom)
        find_time_series_pairs(MagicMock(), markets=[], inactive_shards={1})


class TestHorizonFilterLogging:
    """
    TS-24: filter_markets_within_horizon logged nothing, so --max-horizon-days
    left no evidence it had taken effect and a live run whose pair counts
    differed could not be attributed to it.
    """

    @staticmethod
    def _markets():
        from datetime import UTC, datetime, timedelta
        now = datetime.now(UTC)
        return [
            _mock_market(ticker="NEAR", event_ticker="E1", close_time=now + timedelta(days=3)),
            _mock_market(ticker="MID", event_ticker="E2", close_time=now + timedelta(days=10)),
            _mock_market(ticker="FAR", event_ticker="E3", close_time=now + timedelta(days=90)),
        ]

    def test_logs_kept_and_total_and_the_cutoff(self, caplog):
        with caplog.at_level(logging.INFO):
            kept = filter_markets_within_horizon(self._markets(), 14)
        assert [m.ticker for m in kept] == ["NEAR", "MID"]
        lines = [r.getMessage() for r in caplog.records if "Horizon filter" in r.getMessage()]
        assert len(lines) == 1
        assert "kept 2 of 3" in lines[0]
        assert "--max-horizon-days 14" in lines[0]

    def test_cutoff_carries_a_time_of_day_not_just_a_date(self, caplog):
        # The cutoff is now + N days, so printing only .date() would imply a
        # midnight boundary the filter does not have.
        with caplog.at_level(logging.INFO):
            filter_markets_within_horizon(self._markets(), 14)
        line = next(r.getMessage() for r in caplog.records if "Horizon filter" in r.getMessage())
        assert "T" in line.split("before ")[1]

    def test_silent_when_the_flag_is_absent(self, caplog):
        with caplog.at_level(logging.INFO):
            out = filter_markets_within_horizon(self._markets(), None)
        assert len(out) == 3
        assert "Horizon filter" not in caplog.text


class TestCoerceIntCents:
    """
    TS-28: _coerce_int_cents had no direct coverage. It decides whether a
    legacy `true`/`false` bid array is really carrying whole cents; getting it
    wrong sends dollars through the cents path (x100) or drops a real level.
    """

    def test_accepts_a_genuine_int(self):
        assert scanner._coerce_int_cents(45) == 45

    def test_accepts_an_integral_float(self):
        assert scanner._coerce_int_cents(45.0) == 45

    def test_accepts_integral_strings_in_both_spellings(self):
        assert scanner._coerce_int_cents("45") == 45
        assert scanner._coerce_int_cents(" 45.0 ") == 45

    def test_rejects_a_fractional_value(self):
        # A fractional "cent" means the array is really dollars. Multiplying it
        # through the cents path would be a 100x price error.
        assert scanner._coerce_int_cents(0.45) is None
        assert scanner._coerce_int_cents("0.45") is None

    def test_rejects_bool_despite_int_subclassing(self):
        # bool is an int subclass, so True would silently become 1 cent.
        assert scanner._coerce_int_cents(True) is None
        assert scanner._coerce_int_cents(False) is None

    def test_rejects_non_numeric_and_none(self):
        assert scanner._coerce_int_cents("abc") is None
        assert scanner._coerce_int_cents(None) is None
        assert scanner._coerce_int_cents([45]) is None


class TestCentsBidsToDollarBids:
    """
    TS-28: _cents_bids_to_dollar_bids converts the legacy integer-cent arrays
    to dollars BEFORE _bids_to_ask_levels, which is dollars-only by contract.
    Feeding cents straight through that parser yields 1 - 45 = -44, which is
    then silently discarded — a full book becomes a silent empty one (BS-12).
    """

    def test_converts_whole_cents_to_dollars(self):
        out = scanner._cents_bids_to_dollar_bids("T", "true", [[45, 100], [40, 50]])
        assert out == [[0.45, 100], [0.40, 50]]

    def test_preserves_order_and_quantity_type(self):
        out = scanner._cents_bids_to_dollar_bids("T", "true", [[60, "10"], [55, "20"]])
        assert out == [[0.60, "10"], [0.55, "20"]]

    def test_drops_out_of_range_levels_with_a_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            out = scanner._cents_bids_to_dollar_bids(
                "KXT-1", "true", [[0, 5], [45, 100], [100, 5]])
        assert out == [[0.45, 100]]
        assert caplog.text.count("KXT-1") == 2

    def test_drops_a_level_that_is_not_a_price_qty_pair(self, caplog):
        with caplog.at_level(logging.WARNING):
            out = scanner._cents_bids_to_dollar_bids("KXT-1", "true", [[45], [40, 10]])
        assert out == [[0.40, 10]]
        assert "not a [price, qty] pair" in caplog.text

    def test_all_malformed_yields_empty_not_an_exception(self):
        assert scanner._cents_bids_to_dollar_bids("T", "true", [["x", 1], [0.5, 1]]) == []


# ─── The sell rule's book arithmetic: walk_bids, bid_ladder, floor_to_tick ──

def _old_ladder_average(ladder: list[list[float]], contracts: float) -> float | None:
    """backtester._ladder_average's walk as it stood before it moved to
    scanner.walk_bids, kept verbatim as the reference the shared walk must
    match bit for bit."""
    left = contracts
    proceeds = 0.0
    used = 0
    for price, size in ladder:
        take = min(size, left)
        proceeds += take * price
        left -= take
        used += 1
        if left <= 0:
            break
    # Ladder sizes carry six decimals, so anything finer is float noise
    if round(left, 6) > 0:
        return None
    return ladder[0][0] if used == 1 else proceeds / contracts


def _random_ladder(rng: random.Random) -> list[list[float]]:
    """A bid ladder, best first: 1 to 6 levels on a cent, deci-cent or
    centi-cent grid, sizes whole, two-decimal, or six-decimal with float
    noise in them."""
    count = rng.randint(1, 6)
    grid = rng.choice([100, 1000, 10_000])
    prices = sorted({rng.randint(1, grid - 1) / grid for _ in range(count)}, reverse=True)
    ladder = []
    for price in prices:
        kind = rng.random()
        if kind < 0.3:
            size = float(rng.randint(1, 500))
        elif kind < 0.6:
            size = round(rng.uniform(0.01, 300.0), 2)
        else:
            size = round(rng.uniform(0.000001, 50.0), 6) + rng.choice([0.0, 1e-9, -1e-9])
        ladder.append([price, size])
    return ladder


class TestWalkBids:
    """walk_bids sells a count down a bid ladder, best bid first: the
    backtest's sale walk (backtester._ladder_average reads its average,
    which must be exactly the old walk's), plus the lowest price the walk
    reached, which bounds a live sale order's price."""

    def test_the_average_is_the_old_walks_bit_for_bit(self):
        rng = random.Random(4242)
        compared = refused = 0
        for _ in range(600):
            ladder = _random_ladder(rng)
            total = sum(size for _price, size in ladder)
            for contracts in (total, total + 1e-7, total + 1e-5, total - 1e-7,
                              total * rng.uniform(0.05, 0.95), ladder[0][1],
                              float(rng.randint(1, 50)), round(total * rng.random(), 6)):
                if contracts <= 0:
                    continue
                old = _old_ladder_average(ladder, contracts)
                walked = scanner.walk_bids(ladder, contracts)
                if old is None:
                    assert walked is None, (ladder, contracts)
                    refused += 1
                    continue
                assert walked is not None, (ladder, contracts)
                # repr tells every float apart, -0.0 from 0.0 included
                assert repr(walked[0]) == repr(old), (ladder, contracts)
                assert walked[1] in [price for price, _size in ladder]
                compared += 1
        # Both outcomes occur, so the comparison is not vacuous
        assert compared > 1000 and refused > 300

    def test_one_level_that_holds_them_all_is_the_best_bid(self):
        assert scanner.walk_bids([[0.62, 100.0], [0.55, 50.0]], 40.0) == (0.62, 0.62)
        assert scanner.walk_bids([[0.62, 100.0]], 100.0) == (0.62, 0.62)

    def test_the_lowest_price_is_the_last_level_reached(self):
        ladder = [[0.62, 10.0], [0.60, 10.0], [0.55, 10.0], [0.40, 10.0]]
        average, lowest = scanner.walk_bids(ladder, 25.0)
        assert lowest == 0.55
        assert average == pytest.approx((10 * 0.62 + 10 * 0.60 + 5 * 0.55) / 25)
        # Exactly the first two levels: the walk stops at the second
        assert scanner.walk_bids(ladder, 20.0)[1] == 0.60

    def test_a_noise_remainder_does_not_reach_a_lower_level(self):
        # 0.1 + 0.2 is 0.30000000000000004, so about 5.6e-17 is left after
        # the second level and taken from the third: the average is the old
        # walk's, but the sale reaches no lower than the second level
        ladder = [[0.70, 0.1], [0.65, 0.2], [0.10, 5.0]]
        assert 0.1 + 0.2 - 0.1 - 0.2 > 0
        average, lowest = scanner.walk_bids(ladder, 0.1 + 0.2)
        assert average == _old_ladder_average(ladder, 0.1 + 0.2)
        assert lowest == 0.65

    def test_too_thin_a_ladder_is_refused(self):
        assert scanner.walk_bids([[0.62, 10.0], [0.55, 5.0]], 15.01) is None
        # Six-decimal noise beyond the ladder's depth is not a shortfall
        assert scanner.walk_bids([[0.62, 10.0], [0.55, 5.0]], 15.0000004) == (
            _old_ladder_average([[0.62, 10.0], [0.55, 5.0]], 15.0000004), 0.55)
        assert scanner.walk_bids([], 1.0) is None

    @pytest.mark.parametrize("contracts", [0, 0.0, -0.0, -1.0, 1e-7, float("nan")])
    def test_a_count_that_is_not_above_zero_has_no_price(self, contracts):
        # A sale of no contracts (or a count that cannot be read) is refused,
        # on a full ladder and an empty one alike. A tiny positive count is a
        # sale, priced at the best bid
        ladder = [[0.62, 10.0], [0.55, 5.0]]
        if contracts > 0:
            assert scanner.walk_bids(ladder, contracts) == (0.62, 0.62)
        else:
            assert scanner.walk_bids(ladder, contracts) is None
            assert scanner.walk_bids([], contracts) is None

    def test_the_lowest_price_on_random_whole_ladders(self):
        rng = random.Random(4243)
        for _ in range(300):
            ladder = [[price, float(rng.randint(1, 20))] for price in
                      sorted({rng.randint(1, 99) / 100 for _ in range(5)}, reverse=True)]
            contracts = float(rng.randint(1, 60))
            walked = scanner.walk_bids(ladder, contracts)
            depth = 0.0
            for price, size in ladder:
                depth += size
                if depth >= contracts:
                    assert walked is not None and walked[1] == price
                    break
            else:
                assert walked is None


class TestBidLadder:
    """bid_ladder reads one side's resting bids from _fetch_orderbook's
    result — dollar-string levels in ascending price order on the wire — as
    floats, best first, the order walk_bids reads."""

    _BOOK = {"yes": [["0.0100", "100.00"], ["0.0200", "257.00"]],
             "no": [["0.4500", "10.00"], ["0.4700", "3.50"], ["0.9600", "1.00"]]}

    def test_a_wire_book_reads_best_first(self):
        assert scanner.bid_ladder(self._BOOK, "yes") == [[0.02, 257.0], [0.01, 100.0]]
        assert scanner.bid_ladder(self._BOOK, "no") == [[0.96, 1.0], [0.47, 3.5], [0.45, 10.0]]
        # Floats from a cents book read the same way
        assert scanner.bid_ladder({"yes": [[0.45, 10], [0.5, 2]], "no": []}, "yes") == [
            [0.5, 2.0], [0.45, 10.0]]

    def test_a_walk_over_a_wire_book(self):
        # Selling 5 NO contracts: 1 at 0.96 and 3.5 at 0.47, then 0.5 at 0.45
        average, lowest = scanner.walk_bids(scanner.bid_ladder(self._BOOK, "no"), 5.0)
        assert lowest == 0.45
        assert average == pytest.approx((0.96 + 3.5 * 0.47 + 0.5 * 0.45) / 5)

    def test_unusable_levels_are_left_out(self):
        book = {"yes": [["x", "1"], ["0.5"], None, 7, ["0.0000", "5"], ["1.0000", "5"],
                        ["0.5000", "0"], ["0.5000", "-1"], ["0.5000", "nan"],
                        ["nan", "5"], ["0.5000", "inf"], ["-0.1", "5"], ["0.3000", "2.00"],
                        {"price": "0.4"}, ["0.6000", "1.50", "extra"]],
                "no": []}
        assert scanner.bid_ladder(book, "yes") == [[0.6, 1.5], [0.3, 2.0]]

    def test_prices_off_the_finest_grid_are_left_out(self):
        # The tradeable levels on Kalshi's finest grid, 0.0001 to 0.9999, the
        # bounds _bids_to_ask_levels keeps too; anything between them and 0
        # or 1 is no tradeable level
        book = {"yes": [["0.00005", "5"], ["0.0001", "1"], ["0.9999", "2"],
                        ["0.99995", "5"]], "no": []}
        assert scanner.bid_ladder(book, "yes") == [[0.9999, 2.0], [0.0001, 1.0]]
        assert [config.MIN_ACTIVE_PRICE_DOLLARS, config.MAX_ACTIVE_PRICE_DOLLARS] == [
            0.0001, 0.9999]

    def test_a_number_too_large_for_a_float_is_left_out(self):
        # float() of an integer past a float's range raises OverflowError
        book = {"yes": [["0.5", 10 ** 400], [10 ** 400, "5"], ["0.40", "2"]], "no": []}
        assert scanner.bid_ladder(book, "yes") == [[0.4, 2.0]]

    def test_left_out_levels_are_counted_in_one_warning(self, caplog):
        book = {"yes": [["0.4000", "10.00"], ["0.4500", "oops"], ["0.5000", "-3"],
                        ["0.99995", "1"]], "no": [["0.30", "1"]]}
        with caplog.at_level(logging.WARNING):
            assert scanner.bid_ladder(book, "yes", ticker="KXT-1") == [[0.4, 10.0]]
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings == [
            "Orderbook for KXT-1: left out 3 of 4 YES bid levels as unusable "
            "(price outside [0.0001, 0.9999], quantity not a positive finite number, "
            "or unreadable)"]

    def test_a_clean_book_logs_nothing(self, caplog):
        with caplog.at_level(logging.DEBUG):
            scanner.bid_ladder(self._BOOK, "no", ticker="KXT-1")
            scanner.bid_ladder(None, "no", ticker="KXT-1")
            scanner.bid_ladder({"yes": [], "no": []}, "no", ticker="KXT-1")
        assert caplog.records == []

    @pytest.mark.parametrize("book", [None, {}, {"no": [["0.5", "1"]]}, {"yes": None},
                                      {"yes": []}, {"yes": "0.5"}, {"yes": 3}, [["0.5", "1"]]])
    def test_no_book_or_no_side_reads_as_no_bids(self, book):
        assert scanner.bid_ladder(book, "yes") == []

    def test_levels_of_one_price_keep_their_order(self):
        book = {"yes": [["0.40", "1"], ["0.50", "2"], ["0.40", "3"], ["0.50", "4"]], "no": []}
        assert scanner.bid_ladder(book, "yes") == [[0.5, 2.0], [0.5, 4.0], [0.4, 1.0],
                                                   [0.4, 3.0]]

    @pytest.mark.parametrize("side", ["YES", "", None, "both"])
    def test_an_unknown_side_is_refused(self, side):
        with pytest.raises(ValueError, match="side"):
            scanner.bid_ladder(self._BOOK, side)


class TestFloorToTick:
    """floor_to_tick rounds a price down onto a tick grid: ceil_to_tick's
    mirror, for the most a bid that buys back a held NO pays. A price that
    started as a float is quantized to six decimals first."""

    @pytest.mark.parametrize("price, tick, expected", [
        ("0.567", "0.01", "0.56"), ("0.56", "0.01", "0.56"), ("0.5699999", "0.01", "0.56"),
        ("0.5678", "0.001", "0.567"), ("0.567", "0.001", "0.567"),
        ("0.00567", "0.0001", "0.0056"), ("0.99995", "0.0001", "0.9999"),
        ("0.0099", "0.01", "0.00"), ("1", "0.01", "1.00"),
    ])
    def test_rounds_down_onto_the_grid(self, price, tick, expected):
        assert scanner.floor_to_tick(Decimal(price), Decimal(tick)) == Decimal(expected)

    def test_a_quantized_float_price_floors_onto_itself(self):
        # Every grid price read from a float, and every complement 1 - p a NO
        # sale's bid starts from, floors onto itself once its float noise is
        # quantized away (Decimal(0.57) alone is a hair below 0.57)
        assert Decimal(0.57) < Decimal("0.57")
        for steps, tick in ((100, "0.01"), (1000, "0.001"), (10_000, "0.0001")):
            tick = Decimal(tick)
            for k in range(1, steps):
                for value in (k / steps, 1.0 - (steps - k) / steps):
                    price = Decimal(str(value)).quantize(scanner._SCANNED_PRICE_QUANTUM)
                    assert scanner.floor_to_tick(price, tick) == Decimal(k) / steps, value

    def test_it_mirrors_ceil_to_tick(self):
        rng = random.Random(4244)
        for _ in range(500):
            tick = Decimal(rng.choice(["0.01", "0.001", "0.0001"]))
            price = Decimal(rng.randint(0, 1_000_000)) / Decimal(1_000_000)
            low = scanner.floor_to_tick(price, tick)
            high = scanner.ceil_to_tick(price, tick)
            assert low <= price <= high
            assert high - low in (Decimal(0), tick)
            assert low / tick == (low / tick).to_integral_value()
            assert (low == price) is (high == price)
