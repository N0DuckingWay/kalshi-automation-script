"""Tests for scanner.py normalize_title() — the core pair-detection function."""
import dataclasses
import json
import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kalshi_betting import config, scanner
from kalshi_betting.config import (
    DEFAULT_EXCHANGE_INDEX,
    INCLUDE_MVE_MARKETS,
    SAME_TITLE_LEG_SIDES,
    TIME_SERIES_LEG_SIDES,
)
from kalshi_betting.scanner import (
    CandidatePair,
    PriceRange,
    _bids_to_ask_levels,
    _fetch_orderbook,
    _filter_active_markets,
    _leg_ask_levels,
    _market_from_dict,
    _parse_price_ranges,
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
    inactive_shard_indexes,
    leg_prices,
    leg_sides,
    normalize_title,
    pair_key,
    prefix_fill_prices,
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
        # days into one time-series group — the premise violation the live
        # scanner cannot detect from prices. These two must stay apart, as
        # they did before the pattern was added.
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
        # pB - pA = 0.25 at a zero-day gap (>= the 15% short-tier threshold,
        # and pA + nB = 0.75 <= the 0.85 ceiling) for the time-series finder.
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
        # pA + nB < 1 is accidental rather than structural. The one-series rule
        # still does not fire — the two legs are worded differently — so this
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
        # earlier/later leg assignment means the opposite of what it says and
        # pA + nB < 1 stops holding.
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
        # cumulative. M05 swaps the two `if any(...)` blocks in
        # _field_phrasing.
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
        # control — kills a dropped re.IGNORECASE flag on either compiled
        # table, or "November" dropped from the shared month list. Both must
        # stay — plain member/flag drops a line-level diff would not
        # otherwise catch.
        assert scanner.deadline_phrasing(
            "", "Before Oct 1, 2026", "",
        ) == scanner.DEADLINE_CUMULATIVE
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


class TestDeadlineGuardFinders:
    """Finder-level pins for the cumulative-deadline guard: the verdict gate,
    the span-presence check, and running the screen before best-pair
    selection.

    The predicate-level tests above prove the RULE; these prove
    find_time_series_pairs actually APPLIES it end to end, on fixtures where
    nothing else — the price tiers, the deadline-gap cap, the one-series rule
    — would independently have refused the pair. Every test that asserts []
    also asserts, inside the test, that its two legs share a group key (so a
    fixture that silently stops grouping together — e.g. a normalize_title
    drift — fails loudly instead of returning [] for the wrong reason) and
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
        # only the spans-differ check does. Mirror of
        # test_backtester.py::TestRunBacktestCrossTypeDedup::
        # test_dated_identical_wording_is_same_title_only.
        title = "Will X happen by Dec 31, 2026?"
        mA = _mock_market(
            ticker="A1", event_ticker="EVA-1", title=title, event_title="EV",
            yes_ask=0.30, no_ask=0.70, close_time=datetime(2026, 12, 1, tzinfo=UTC),
        )
        mB = _mock_market(
            ticker="B1", event_ticker="EVB-1", title=title, event_title="EV",
            yes_ask=0.60, no_ask=0.40, close_time=datetime(2026, 12, 20, tzinfo=UTC),
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
        assert len(find_same_title_pairs([mA, mB])) == 1
        assert find_time_series_pairs(
            MagicMock(), held_tickers=set(), markets=[mA, mB],
        ) == []

        # Positive control: B's wording states a DIFFERENT deadline ("Dec
        # 20" instead of "Dec 31") on the same two series, same prices, same
        # close times. The spans now differ, so the time-series pair forms —
        # proving the [] above comes from the spans-differ conjunct, not from
        # the price tier, the deadline gap, or the two series being distinct.
        mB3 = _mock_market(
            ticker="B1", event_ticker="EVB-1", title="Will X happen by Dec 20, 2026?",
            event_title="EV", yes_ask=0.60, no_ask=0.40,
            close_time=datetime(2026, 12, 20, tzinfo=UTC),
        )
        assert len(find_time_series_pairs(
            MagicMock(), held_tickers=set(), markets=[mA, mB3],
        )) == 1


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


class TestTimeSeriesTieredThreshold:
    """The minimum price gap (later YES ask minus earlier YES ask) is tiered
    by deadline gap: 15% for gaps <= 15 days, 30% for 16-30 days, and gaps
    > 30 days are never candidates. Every fixture has the LATER contract
    pricier (pB > pA) except the direction test."""

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
        # legs cost pA + nB = 1.05. The structural invariant of a
        # cumulative-deadline pair is that buying YES at pA and NO at nB costs
        # LESS than the $1 a win pays, so this is not a candidate at all.
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
        #
        # RE-PINNED (DR-02/DR-54): the two event tickers used to be EVT-A and
        # EVT-B, one series, which the finder now reads as two instances of one
        # recurring fixture and refuses. They name two DIFFERENT series now, so
        # the 20-day assertion below passes UNCHANGED — the close gap is
        # irrelevant to the one-series rule.
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
        pairs = find_same_title_pairs([mA, mB])
        assert len(pairs) == 1


def _raw_book_response(ob: dict) -> SimpleNamespace:
    """Wrap one orderbook_fp side dict as a raw *_without_preload_content response.

    Responses use the raw orderbook_fp JSON wire format (the SDK's modeled
    orderbook response can't deserialize live payloads anymore).
    """
    payload = {"orderbook_fp": ob}
    return SimpleNamespace(status=200, data=json.dumps(payload).encode("utf-8"))


def _ts_orderbook_client(
    *, pA_fill: float, nB_fill: float, qty: int = 100, pB_ref: float | None = None,
):
    """Mock KalshiClient serving TIME-SERIES-shaped depth at exactly one level.

    A time-series pair buys YES on EARLY and NO on LATE. Buying YES on EARLY
    consumes EARLY's NO bids (YES ask = 1 - NO bid) and buying NO on LATE
    consumes LATE's YES bids (NO ask = 1 - YES bid) — so a NO bid of
    (1 - pA_fill) on EARLY and a YES bid of (1 - nB_fill) on LATE yield
    qualifying depth priced at exactly pA_fill + nB_fill. The opposite sides
    are left empty so a wrong-side read shows up as "no depth".

    pB_ref, when given, additionally rests a NO bid of (1 - pB_ref) on LATE so
    LATE's best YES ask is exactly pB_ref — the reference quote
    _reference_yes_ask reads. It is NOT a leg side for this pair type, so it
    changes no fill price; setting it above 1 - (1 - nB_fill), i.e. crossing
    LATE's book, is the only way to drive the post-enrichment direction guard.
    Left None, LATE's NO side stays empty and the pair keeps its scan-time pB.
    """
    def fake_orderbook(ticker):
        if ticker == "EARLY":  # market A — NO bids become YES ask levels
            ob = {"yes_dollars": [],
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


def _ts_multilevel_client(levels: list[tuple[float, float, int]]):
    """Mock KalshiClient serving TIME-SERIES depth at SEVERAL price levels.

    levels is [(pA_fill, nB_fill, qty), ...]. Same side mapping as
    _ts_orderbook_client (EARLY's NO bids -> YES asks, LATE's YES bids -> NO
    asks), just with more than one rung, so the affordability bound has
    somewhere worse to reach when the budget is large.

    The two legs are SEPARATE books that _pair_orderbooks merges with a
    two-pointer sweep, so each column must ascend on its own for the rungs here
    to pair up 1:1 with the slices that sweep emits — a column that dips gets
    re-sorted by _bids_to_ask_levels and the quantities no longer line up.
    """
    def fake_orderbook(ticker):
        if ticker == "EARLY":
            ob = {"yes_dollars": [],
                  "no_dollars": [[str(round(1.0 - pa, 4)), str(q)]
                                 for pa, _, q in levels]}
        else:
            ob = {"yes_dollars": [[str(round(1.0 - nb, 4)), str(q)]
                                  for _, nb, q in levels],
                  "no_dollars": []}
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


class TestEnrichmentBoundsDepthByAffordability:
    """Enrichment must average only the depth this balance could actually buy.

    One pair is capped at BUDGET_FRACTION of the balance, so averaging a liquid
    market's full book priced every pair against levels no single trade can
    reach — inflating the fill price and killing pairs at the profitability gate
    on contracts we would never have bought.
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
        client = _ts_orderbook_client(pA_fill=0.32, nB_fill=0.42, qty=40)
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
            _ts_orderbook_client(pA_fill=0.32, nB_fill=0.42), [ts], _AMPLE_BALANCE_CENTS,
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


class TestOrderbookCeilingTieredByDeadlineGap:
    """enrich_with_orderbook_prices and validate_pair_price must apply the
    deadline-gap-tiered LEG-price-sum ceiling (0.85 for gaps <= 15 days, 0.70
    for 16-30 days), not the old flat 1 - 15% = 0.85.

    The fixtures use a deliberately WIDE later book: with a tight nB = 1 - pB
    the leg sum is exactly 1 - (pB - pA), which is always <= the ceiling once
    the pair has passed the gap filter, so the ceiling could never bind."""

    def test_long_gap_depth_at_075_sum_marked_untradeable(self):
        # 20-day gap → ceiling 0.70. Leg depth priced at pA 0.30 + nB 0.45 =
        # 0.75 would have passed the old flat 0.85 ceiling but must disqualify.
        pair = _ts_candidate(gap_days=20, pA=0.30, pB=0.65, nB=0.45)
        client = _ts_orderbook_client(pA_fill=0.30, nB_fill=0.45)
        [enriched] = enrich_with_orderbook_prices(client, [pair], _AMPLE_BALANCE_CENTS)
        assert enriched.tradeable is False

    def test_short_gap_depth_at_080_sum_qualifies(self):
        # 10-day gap → ceiling 0.85. Leg depth priced at 0.30 + 0.50 = 0.80
        # qualifies and the pair picks up the depth-weighted fill prices in
        # the LEG fields (pA/nB). pB is not a leg price but IS the model's
        # reference quote, so it is refreshed from LATE's NO bids in the same
        # pass (0.62 here, against a scan-time 0.60); nA is reporting-only for
        # this pair type and stays put.
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
        client = _ts_orderbook_client(pA_fill=0.30, nB_fill=0.50)
        assert validate_pair_price(client, spec) is False

    def test_validate_pair_price_accepts_short_gap_at_same_depth(self):
        # Identical depth passes for a 10-day-gap pair (ceiling 0.85)
        pair = _ts_candidate(gap_days=10, pA=0.30, pB=0.60, nB=0.50)
        spec = SimpleNamespace(pair=pair, x=10)
        client = _ts_orderbook_client(pA_fill=0.30, nB_fill=0.50)
        assert validate_pair_price(client, spec) is True

    def test_validate_pair_price_logs_gap_rejection_at_warning(self, caplog):
        # Same rejecting fixture as test_validate_pair_price_rejects_long_gap_at_old_ceiling
        # (0.30 + 0.50 = 0.80 exceeds the 20-day-gap ceiling of 0.70, so this hits
        # the "gap no longer qualifies" branch, not the depth branch). The drop must
        # be logged exactly once, at WARNING, with "; dropping" appended — this is
        # the one log line for the drop; pre_execution_check must not log a second.
        pair = _ts_candidate(gap_days=20, pA=0.30, pB=0.65, nB=0.50)
        spec = SimpleNamespace(pair=pair, x=10)
        client = _ts_orderbook_client(pA_fill=0.30, nB_fill=0.50)
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
        # pB is the reference quote _kelly_p subtracts pA from, so it comes from
        # this same snapshot (LATE's NO bids) rather than the scan
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
        client = _ts_orderbook_client(pA_fill=0.30, nB_fill=0.40, qty=9)
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
    the book it already fetched, and drop any pair whose refreshed reference is
    not above the YES leg's fill.

    Left stale, that quote is compared against a depth-weighted fill by
    strategy._kelly_p, and config.time_series_profit_prob's max(0, pB - pA)
    clamp turns a stale pB at or below a fresh pA into p = 1.0 — a riskless
    model on a directional bet (TS-34)."""

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
        snapshot, the qualifying ceiling avg_yes + avg_no <= 1 - tier makes the
        inversion arithmetically impossible on an uncrossed book, so a crossed
        book is the only shape that can still reach the guard.

        Leg fills: pA 0.54 (EARLY NO bid 0.46) + nB 0.30 (LATE YES bid 0.70) =
        0.84, inside the 10-day-gap ceiling of 0.85 and profitable after fees.
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

        direction_drops = [
            r for r in caplog.records
            if "no longer prices above the YES leg fill" in r.getMessage()
        ]
        assert len(direction_drops) == 1
        assert direction_drops[0].levelno == logging.WARNING

        # The pair IS profitable at those fills — it must not also be reported
        # as unprofitable, which would misattribute the drop
        assert [
            r for r in caplog.records
            if "unprofitable after depth adjustment" in r.getMessage()
        ] == []

    def test_refresh_makes_the_riskless_clamp_unreachable(self):
        # The consequence, not just the flag. Import locally so this scanner
        # test file does not take a module-level dependency on strategy.
        from kalshi_betting.strategy import _kelly_p

        pair, client = self._inverted_pair_and_client()

        # The pre-fix shape: leg fill written to pA while pB stays at its
        # scan-time 0.50. time_series_profit_prob clamps the negative gap to
        # zero and the pair models as RISKLESS.
        stale_shape = dataclasses.replace(pair, pA=0.54)
        assert _kelly_p(stale_shape) == 1.0

        # The fixed enrichment never produces such a pair: the refreshed
        # reference fails the direction guard, so it is not tradeable and
        # compute_trade returns None before _kelly_p is ever consulted.
        [enriched] = enrich_with_orderbook_prices(client, [pair], _AMPLE_BALANCE_CENTS)
        assert enriched.tradeable is False

    def test_reference_ask_falls_back_to_scan_time_when_side_is_empty(self):
        # LATE has no resting NO bids, so no reference ask can be derived —
        # pB keeps its scan-time value and the guard evaluates against that
        pair = _ts_candidate(gap_days=10, pA=0.30, pB=0.60, nB=0.50)
        client = _ts_orderbook_client(pA_fill=0.30, nB_fill=0.50)
        [enriched] = enrich_with_orderbook_prices(client, [pair], _AMPLE_BALANCE_CENTS)
        assert enriched.tradeable is True
        assert enriched.pB == pair.pB
        assert enriched.pA == pytest.approx(0.30)

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


class TestTimeSeriesBestPairPerGroup:
    def test_group_of_three_keeps_largest_later_minus_earlier_gap(self):
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

    def test_scanner_and_ceiling_share_the_gap(self):
        # _pair_max_sum tiers off the same order-independent gap the finder
        # used, so a 16-day pair gets the long-tier ceiling from either side
        pair = _ts_candidate(gap_days=16, pA=0.30, pB=0.65, nB=0.40)
        assert scanner._pair_max_sum(pair) == pytest.approx(0.70)
        swapped = dataclasses.replace(pair, market_a=pair.market_b, market_b=pair.market_a)
        assert scanner._pair_max_sum(swapped) == pytest.approx(0.70)


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
    trading-inactive are dropped. (Routing is enforced at order submission —
    per-leg on the V2 path, trader._legacy_routable on the legacy one — never
    here.)"""

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


class TestPriceEpsilonThresholds:
    """
    TS-09: prices are floats parsed from cent-quantized dollar strings, so a
    pair sitting EXACTLY on a documented threshold can evaluate a hair under it
    and be rejected for representation noise rather than for its price.
    Measured over live books: the same-title 5c test rejected 50 of 94
    qualifying pairs, the 15c tier 21 of 84, the 30c tier 15 of 69.
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

    def test_legacy_path_counts_the_whole_qualifying_book(self, monkeypatch):
        # buy_max_cost is a TOTAL-cost cap and can sweep a ladder.
        monkeypatch.setattr(scanner, "ORDER_API_VERSION", "legacy")
        assert self._run(monkeypatch, 600) is True


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
