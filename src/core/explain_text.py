"""Feature glossary and plain-English reason-code rendering for SHAP outputs."""
from __future__ import annotations

GLOSSARY: dict[str, tuple[str, str]] = {
    # feature -> (short label, human description template)
    "ret_1d": ("1-day return", "yesterday's price move"),
    "ret_5d": ("5-day return", "the last week's price move"),
    "ret_21d": ("1-month return", "the last month's price move"),
    "ret_63d": ("3-month return", "the last quarter's price move"),
    "mom_12_1": ("12-1 momentum", "the 12-month trend excluding the last month"),
    "c_sma10": ("price vs 10d average", "where price sits against its 10-day average"),
    "c_sma50": ("price vs 50d average", "where price sits against its 50-day average"),
    "sma10_sma50": ("trend crossover", "the 10-day vs 50-day average gap"),
    "rsi_14": ("RSI-14", "the 14-day relative strength index"),
    "macd_norm": ("MACD", "the MACD trend signal"),
    "macd_hist_norm": ("MACD histogram", "MACD momentum vs its signal line"),
    "boll_pctb": ("Bollinger %B", "position inside the Bollinger band"),
    "boll_bw": ("Bollinger width", "how squeezed/expanded the band is"),
    "atr_norm": ("ATR", "the average true range (intraday risk)"),
    "rv_21": ("1-month volatility", "realized volatility over 21 days"),
    "rv_63": ("3-month volatility", "realized volatility over 63 days"),
    "rv_ratio": ("vol regime shift", "short-term vs medium-term volatility"),
    "parkinson_21": ("range volatility", "high-low range-based volatility"),
    "range_norm": ("day range", "today's high-low range"),
    "overnight_gap": ("overnight gap", "the open vs yesterday's close"),
    "vol_surprise": ("volume surprise", "volume vs its 21-day median"),
    "vol_trend": ("volume trend", "5-day vs 63-day average volume"),
    "pct_deliv": ("delivery %", "share of volume actually delivered (conviction)"),
    "deliv_dev": ("delivery shift", "delivery % vs its monthly average"),
    "amihud": ("illiquidity", "price impact per rupee traded (Amihud)"),
    "c_vwap": ("close vs VWAP", "close relative to the day's volume-weighted price"),
    "dow": ("day of week", "calendar day-of-week"),
    "month": ("month", "calendar month"),
    "is_month_end": ("month-end", "turn-of-month window"),
    "cs_rank_ret1": ("1-day rank", "today's return ranked across the index"),
    "cs_rank_ret21": ("momentum rank", "1-month return ranked across the index"),
    "cs_rank_rv21": ("volatility rank", "volatility ranked across the index"),
    "cs_rank_volsurp": ("volume rank", "volume surprise ranked across the index"),
    "mkt_ret_1d": ("market return", "the index-wide move yesterday"),
    "mkt_rv21": ("market volatility", "index-wide realized volatility"),
    "beta_252": ("beta", "sensitivity to the market over the past year"),
    "rel_sector_21": ("vs sector", "1-month return relative to the sector"),
    "sector_ret_21": ("sector momentum", "the sector's own 1-month move"),
    "regime": ("volatility regime", "the Calm/Normal/Turbulent market state"),
    "drawdown_252": ("drawdown", "distance below the 52-week high"),
    "symbol_cat": ("stock identity", "stock-specific base behavior"),
    "sector_cat": ("sector identity", "sector-specific base behavior"),
}


def feature_label(feature: str) -> str:
    return GLOSSARY.get(feature, (feature, feature))[0]


def render_reason(feature: str, value: float, shap: float, horizon: int) -> str:
    """One-sentence reason code: direction + driver + magnitude."""
    label, desc = GLOSSARY.get(feature, (feature, feature))
    push = "raised" if shap > 0 else "lowered"
    pct = abs(shap) * 100
    if feature == "rsi_14" and value == value:
        state = "overbought" if value > 70 else "oversold" if value < 30 else f"{value:.0f}"
        return f"{label} at {state} {push} the {horizon}-day outlook by {pct:.2f}%"
    if feature in ("regime",):
        name = {0: "Calm", 1: "Normal", 2: "Turbulent"}.get(int(value), "Unknown")
        return f"the {name} market regime {push} the {horizon}-day outlook by {pct:.2f}%"
    if feature in ("symbol_cat", "sector_cat"):
        return f"{desc} {push} the {horizon}-day outlook by {pct:.2f}%"
    return f"{desc} ({value:+.3g}) {push} the {horizon}-day outlook by {pct:.2f}%"
