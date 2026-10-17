"""
File: glossary.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    The short dictionary of the scores and terms the dashboards show: what
    each one means and why it is useful. Each term is tagged with the pages
    that show it, and a page lists only its own. The backtest dashboard and
    the Data page (not built yet) will both render it from here, so a term
    reads the same on both. Reporting only: nothing that trades, sizes or
    backtests reads it.

Dependencies:
    The standard library only (html). dashboard.py imports it for the backtest
    page's Terms link and section; the Data page's builder will do the same
    once it exists.

Notes:
    Each definition describes how this codebase computes the figure, not the
    textbook version. Change a figure's arithmetic and its entry here in the
    same commit. Every cell is escaped when it is rendered, so a term may
    hold any text; the rows are written as plain text, never as HTML. The
    tier-floor figures in the "Tier floors" entry are typed out here, because
    this module imports only the standard library; a test holds them equal to
    config's MIN_PRICE_DIFF_SHORT_GAP, MIN_PRICE_DIFF_LONG_GAP and
    SHORT_DEADLINE_GAP_DAYS.
"""

import html

# The two pages that show terms
PAGE_BACKTEST = "backtest"
PAGE_DATA = "data"
PAGES = (PAGE_BACKTEST, PAGE_DATA)
_BOTH = (PAGE_BACKTEST, PAGE_DATA)
_BACKTEST = (PAGE_BACKTEST,)
_DATA = (PAGE_DATA,)

# (term, what it means, why it is useful, the pages that show it), in display
# order; a page lists only the terms tagged with it
TERMS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("Total return",
     "How much the portfolio value grew or shrank from the starting balance to the "
     "end of the period, as a percent.",
     "The plainest answer to whether it made money.",
     _BACKTEST),
    ("Win rate",
     "The share of trades whose profit, after fees, was above zero.",
     "Read with the mean and median return: it shows whether gains come from many "
     "small wins or a few large ones.",
     _BACKTEST),
    ("Mean / median return per trade",
     "The average and the middle one of every trade's profit divided by what it "
     "cost with fees; each trade counts once, whatever its size.",
     "A median far below the mean means a few big winners carry the result.",
     _BACKTEST),
    ("Median monthly return",
     "The middle one of the calendar months' returns on the portfolio value; every "
     "month of the period counts, quiet ones too.",
     "A typical month, which one huge month cannot move.",
     _BACKTEST),
    ("Max drawdown",
     "The largest fall of the portfolio value from a high point to a later low, as "
     "a percent of that high; open trades count at market prices.",
     "The worst stretch you would have sat through, so a guide to how much risk "
     "the strategy carries.",
     _BACKTEST),
    ("Portfolio value",
     "Cash plus every open trade valued at market (the ask of the side held).",
     "Returns are measured on it and each new trade is sized on it, not on idle "
     "cash alone.",
     _BACKTEST),
    ("Sharpe ratio",
     "Average daily return above the 8-week T-bill yield (charged only on money in "
     "open trades), divided by how much daily returns vary, scaled to a year.",
     "Return per unit of risk, so strategies with different swings can be compared.",
     _BACKTEST),
    ("Sortino ratio",
     "Like Sharpe, but only the days that fell below the T-bill hurdle count as "
     "risk.",
     "Large gains do not count against the strategy, only the swings that hurt.",
     _BACKTEST),
    ("Time-series pair",
     "Two markets asking the same question with different deadlines; the bot buys "
     "Yes on the earlier one and No on the later one.",
     "A bet that the market overstates the chance the event happens between the "
     "deadlines, so it can lose.",
     _BOTH),
    ("Ladder / cross-event pair",
     "A ladder pair takes both markets from one event that lists the question at "
     "several deadlines; a cross-event pair takes them from two events.",
     "Rungs of one ladder tend to settle alike, so the two kinds are shown "
     "apart.",
     _BOTH),
    ("Same-title pair",
     "Two identically worded markets on different series, filed under the same "
     "Kalshi category, tag and competition and closing within an hour of each other, "
     "whose Yes prices differ by 5 cents or more; the bot buys No on the pricier "
     "and Yes on the cheaper.",
     "If both settle the same way one leg pays, so it is close to a sure gain; it "
     "loses only if the pricier one settles Yes and the cheaper one No.",
     _BACKTEST),
    ("In-between outcome",
     "A time-series pair whose earlier market settled No and later market Yes: "
     "the event happened between the deadlines, and both legs lose.",
     "The one outcome that costs the whole stake; k and k̂ are about how often "
     "it happens.",
     _BOTH),
    ("Mid spread",
     "For a time-series pair, the later market's midpoint price minus the earlier "
     "one's, a midpoint being halfway between a market's Yes ask and Yes bid.",
     "The market's own estimate, after removing the cost of trading, of the chance "
     "the event happens between the deadlines.",
     _BOTH),
    ("Spread band",
     "The range of gaps between the later market's Yes ask and the earlier one's "
     "that a time-series pair must fall in to be entered.",
     "Shows which gaps earned money and which were noise.",
     _BACKTEST),
    ("Tier floors",
     "On, a pair whose deadlines are 15 days apart or fewer needs a Yes-ask gap of "
     "at least 15 cents and one 16 to 30 days apart at least 30 cents; off, only "
     "the spread band's own floor applies.",
     "A wider deadline gap leaves more room for the event to really happen in "
     "between, so a bigger gap is demanded before disputing it.",
     _BACKTEST),
    ("Kelly fraction",
     "The share of the portfolio value the Kelly rule says to stake on a trade, "
     "from its chance of paying and its price after fees.",
     "Sizes each bet for the fastest long-run growth if the chance of paying is "
     "right; the size cap can hold it smaller.",
     _BACKTEST),
    ("Size cap",
     "The most of the portfolio value one new trade may stake, whatever the Kelly "
     "fraction says; same-title pairs also have a cap of their own, and the lower "
     "of the two applies.",
     "Limits how much one wrong forecast can cost.",
     _BACKTEST),
    ("Fills",
     "Whether a trade was priced by walking a modeled order book, whose depth comes "
     "from saved snapshots, or at the top-of-book price in any size.",
     "Walking the book makes a big trade pay more, so it is the more realistic "
     "fill.",
     _BACKTEST),
    ("k",
     "The share of the mid spread the bot believes will really happen in between; "
     "it forecasts a time-series trade's chance of paying as 1 − k × mid spread.",
     "The one setting that decides how much a time-series trade is trusted.",
     _BOTH),
    ("k̂ (empirical k)",
     "What k turned out to be: the share of qualifying time-series pairs (traded or "
     "not) that settled in between, divided by their average mid spread on the "
     "first Monday each qualified.",
     "Checks the setting against what really happened.",
     _BOTH),
    ("k̂ − k",
     "Empirical k minus the k the trades were sized at.",
     "Above 0 means the in-between outcome happened more than assumed, so trades "
     "were sized too large; below 0, too small.",
     _BOTH),
    ("95% CI (under k̂)",
     "A range around k̂ that should hold the true value 95 times out of 100, "
     "treating all of one event's pairs as one piece of evidence because they tend "
     "to settle alike; it never goes below 0.",
     "A wide range means too little data to trust k̂ yet.",
     _BOTH),
    ("Brier score",
     "The average squared gap between a forecast chance and the result (1 if it "
     "happened, 0 if not): the backtest page scores each time-series trade's entry "
     "mid spread against whether it settled in between, the Data page scores prices "
     "against whether the market settled Yes.",
     "One number for how well forecasts match outcomes; 0 is perfect, always "
     "saying 50% scores 0.25, and lower is better.",
     _BOTH),
    ("Log loss",
     "Like the Brier score (each time-series trade's entry mid spread against "
     "whether it settled in between), but a confident forecast that turns out "
     "wrong is punished far more.",
     "Shows whether a forecaster ever claims certainty it has not earned; lower "
     "is better.",
     _BACKTEST),
    ("Calibration curve (reliability chart)",
     "Forecasts grouped into bins, each bin's average forecast plotted against "
     "how often the event really happened.",
     "Points on the diagonal mean the forecasts are fair; points off it show where "
     "they are wrong.",
     _BOTH),
    ("Share settled Yes",
     "Of the markets that settled Yes or No, the fraction that settled Yes.",
     "Shows how often a category's questions come true.",
     _DATA),
    ("Favourite–longshot gap",
     "For each price group, the share settled Yes minus the price: the Yes ask a "
     "set time before close (the midpoint is a second figure).",
     "Below 0 at low prices means long shots are overpriced; on the ask it reads a "
     "little low everywhere, since the ask includes the cost of trading.",
     _DATA),
    ("Slippage",
     "A trade's profit minus what it would have made held to a win, in dollars (not "
     "price slippage).",
     "Zero for a trade held to a win; negative for a loss or a trade sold early; "
     "positive only for a same-title pair where both legs paid.",
     _BACKTEST),
    ("Split-half check (H1 / H2)",
     "The same scenario run again from the starting balance on the entries before, "
     "then after, the middle time-series entry date.",
     "A result that holds in both halves is less likely to be luck.",
     _BACKTEST),
    ("Top event (return without it)",
     "The scenario run again with every entry of the event whose trades earned the "
     "most left out.",
     "Shows whether one lucky event carries the whole result.",
     _BACKTEST),
    ("Combo market",
     "A Kalshi multi-leg ticket (series beginning KXMVE) that pays only if every "
     "leg hits.",
     "Most listed markets are combos; the bot never pairs two of them, so they "
     "are shown apart.",
     _DATA),
)

# The id of the Terms section and the fragment the header link points at
TERMS_ANCHOR = "terms"

_LINK_STYLE = "font-size:13px; font-family:sans-serif;"
# scroll-margin-top keeps a jump to #terms below the page's sticky filter bar
# (dashboard.py's #flt-bar, position:sticky; top:0): the bar's eleven controls
# wrap onto two to four rows of about 35px, with a summary line under them, so
# it can stand about 130px to 190px tall. 220px clears that; a page with no
# sticky bar just scrolls the heading a little below the top
_SCROLL_MARGIN = "scroll-margin-top:220px;"
_HEAD_STYLE = ("margin:40px 0 0 0; padding:0 0 8px 0; border-bottom:2px solid #E0E0E0; "
               "font-size:22px; font-weight:700; color:#212121; font-family:sans-serif; "
               f"{_SCROLL_MARGIN}")
_TABLE_STYLE = ("font-family:sans-serif; font-size:14px; border-collapse:collapse; "
                "margin:16px 0; width:100%;")
_TH_STYLE = "padding:8px 16px; text-align:left;"
_TD_STYLE = "padding:6px 16px; vertical-align:top; border-top:1px solid #E0E0E0;"


def _check_page(page: str) -> str:
    """
    Check that a page name is one of PAGES.

    Args:
        page (str): PAGE_BACKTEST or PAGE_DATA.

    Returns:
        str: The same name.

    Raises:
        ValueError: If it names no page.
    """
    if page not in PAGES:
        raise ValueError(f"unknown page {page!r}; expected one of {PAGES}")
    return page


def terms_for(page: str) -> tuple[tuple[str, str, str], ...]:
    """
    The terms one page shows, in display order.

    Args:
        page (str): PAGE_BACKTEST or PAGE_DATA.

    Returns:
        tuple[tuple[str, str, str], ...]: (term, what it means, why it helps)
            for every TERMS entry tagged with the page. TERMS is read when
            this is called, not at import.

    Raises:
        ValueError: If the page is not one of PAGES.
    """
    _check_page(page)
    return tuple((term, meaning, reason)
                 for term, meaning, reason, pages in TERMS if page in pages)


def terms_link_html(page: str) -> str:
    """
    The small header link that jumps to the page's Terms section.

    Args:
        page (str): PAGE_BACKTEST or PAGE_DATA, the page the link sits on.

    Returns:
        str: An anchor whose href is "#" + TERMS_ANCHOR.

    Raises:
        ValueError: If the page is not one of PAGES.
    """
    _check_page(page)
    return (f'<a href="#{TERMS_ANCHOR}" class="terms-link" style="{_LINK_STYLE}" '
            f'title="What the scores on this page mean and why they help">Terms</a>')


def terms_section_html(page: str) -> str:
    """
    The Terms section: a heading and one short table of the page's terms, what
    each means and why it helps.

    Every cell goes through html.escape, so a term holding markup characters
    shows as text.

    Args:
        page (str): PAGE_BACKTEST or PAGE_DATA; only that page's terms are listed.

    Returns:
        str: HTML for the section; the heading carries id=TERMS_ANCHOR.

    Raises:
        ValueError: If the page is not one of PAGES.
    """
    rows = "".join(
        f"<tr><td style='{_TD_STYLE} font-weight:bold; white-space:nowrap;'>"
        f"{html.escape(term)}</td>"
        f"<td style='{_TD_STYLE}'>{html.escape(meaning)}</td>"
        f"<td style='{_TD_STYLE}'>{html.escape(reason)}</td></tr>"
        for term, meaning, reason in terms_for(page))
    return (
        f'<div id="{TERMS_ANCHOR}" style="{_HEAD_STYLE}">Terms</div>'
        f'<table style="{_TABLE_STYLE}">'
        f'<tr style="background:#E3F2FD;"><th style="{_TH_STYLE}">Term</th>'
        f'<th style="{_TH_STYLE}">What it means</th>'
        f'<th style="{_TH_STYLE}">Why it helps</th></tr>'
        f"{rows}</table>\n")
