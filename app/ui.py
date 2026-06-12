"""Shared figure builders and formatting for the dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

TEMPLATE = "plotly_white"
REGIME_COLORS = {0: "rgba(46,204,113,0.10)", 1: "rgba(0,0,0,0)", 2: "rgba(231,76,60,0.12)"}
REGIME_NAMES = {0: "Calm", 1: "Normal", 2: "Turbulent"}
PROFILE_COLORS = {
    "conservative": "#2e86c1", "balanced": "#117a65", "aggressive": "#ca6f1e",
    "benchmark_ew": "#7f8c8d", "hrp": "#8e44ad", "market_ew": "#7f8c8d",
}


def fmt_pct(x: float, digits: int = 1) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:.{digits}f}%"


def fmt_num(x: float, digits: int = 2) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


def downsample(df: pd.DataFrame, max_points: int = 3000) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    step = int(np.ceil(len(df) / max_points))
    return df.iloc[::step]


def base_layout(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(
        template=TEMPLATE, title=title, height=height,
        margin=dict(l=10, r=10, t=40 if title else 16, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        hovermode="x unified",
    )
    return fig


def regime_shapes(regimes: pd.DataFrame, y0: float = 0, y1: float = 1) -> list[dict]:
    """Background shading rectangles for Calm/Turbulent periods."""
    reg = regimes[["date", "regime"]].dropna().reset_index(drop=True)
    if reg.empty:
        return []
    shapes = []
    run_start = reg.at[0, "date"]
    cur = reg.at[0, "regime"]
    for i in range(1, len(reg)):
        r = reg.at[i, "regime"]
        if r != cur:
            if cur in (0, 2):
                shapes.append(dict(type="rect", xref="x", yref="paper",
                                   x0=run_start, x1=reg.at[i, "date"], y0=y0, y1=y1,
                                   fillcolor=REGIME_COLORS[int(cur)], line_width=0,
                                   layer="below"))
            run_start, cur = reg.at[i, "date"], r
    if cur in (0, 2):
        shapes.append(dict(type="rect", xref="x", yref="paper",
                           x0=run_start, x1=reg.at[len(reg) - 1, "date"], y0=y0, y1=y1,
                           fillcolor=REGIME_COLORS[int(cur)], line_width=0, layer="below"))
    return shapes


def line(df: pd.DataFrame, x: str, y: str, name: str | None = None,
         color: str | None = None, fig: go.Figure | None = None,
         dash: str | None = None) -> go.Figure:
    fig = fig or go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x], y=df[y], mode="lines", name=name or y,
        line=dict(color=color, width=1.6, dash=dash),
    ))
    return fig


def wealth_chart(curves: pd.DataFrame, log_scale: bool = True) -> go.Figure:
    fig = go.Figure()
    for prof, d in curves.groupby("profile"):
        d = downsample(d.sort_values("date"))
        fig.add_trace(go.Scatter(
            x=d["date"], y=d["wealth"], mode="lines", name=prof,
            line=dict(color=PROFILE_COLORS.get(prof), width=1.8),
        ))
    if log_scale:
        fig.update_yaxes(type="log")
    fig.update_yaxes(title="growth of ₹1")
    return base_layout(fig)
