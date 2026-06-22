"""
High-quality visualizations for the NBA Fantasy Value project.

Reads the project's existing CSV outputs (under sim_stats/) and produces a suite
of presentation-ready figures grouped into five themes:

    overview/  - data sanity checks (players per season, retention, PPG dist)
    value/     - fantasy H2H value insights (category breakdown, scarcity, ...)
    thesis/    - pre-draft H2H value vs simulated in-league win shares
    model/     - projection model evaluation (pred vs actual, errors, CIs)
    draft/     - draft simulation results (standings, team strengths)

Dependencies: matplotlib, numpy, scipy, pandas (no seaborn required).

Usage (run from the repo root):
    py visual_helper/generate_visualizations.py
    py visual_helper/generate_visualizations.py --theme value
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

# Resolve paths relative to the repo root (parent of this file's folder),
# so the script works no matter where it is launched from.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SIM_STATS = REPO_ROOT / "sim_stats"
OUT_ROOT = SCRIPT_DIR / "quality_plots"

THEMES = ["overview", "value", "thesis", "model", "draft"]


# ---------------------------------------------------------------------
# Theme / styling
# ---------------------------------------------------------------------

# Categorical palette. Deliberately colorblind-safe: it avoids using red and
# green to carry meaning together (a key perception rule from the course notes).
# Based on Wong's colorblind-safe palette.
PALETTE = [
    "#0072b2",  # blue
    "#e69f00",  # orange
    "#009e73",  # bluish green
    "#cc79a7",  # reddish purple
    "#56b4e9",  # sky blue
    "#d55e00",  # vermillion
    "#f0e442",  # yellow
    "#999999",  # grey
    "#000000",  # black
]

# Two-direction (diverging) encoding that does NOT rely on red/green: blue for
# one direction, orange for the other, with a light middle (perception rule:
# "use a diverging scheme where light colors represent middle values").
BLUE = "#0072b2"
ORANGE = "#d55e00"
HIGHLIGHT = "#e69f00"

# Perceptually uniform colormap for single-quantity bar fills.
SEQUENTIAL = plt.get_cmap("viridis")
# Diverging map for heatmaps: warm red = high, cold blue = low, light middle.
# (Red-blue is a standard colorblind-safe diverging scheme.)
DIVERGING = plt.get_cmap("RdBu_r")


def set_theme() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e2e8f0",
        "grid.linewidth": 0.8,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.titlepad": 12,
        "axes.labelsize": 12,
        "axes.labelcolor": "#1e293b",
        "axes.labelweight": "medium",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "text.color": "#0f172a",
        "legend.frameon": False,
        "legend.fontsize": 10,
        "font.family": "DejaVu Sans",
        "figure.titlesize": 17,
        "figure.titleweight": "bold",
    })
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=PALETTE)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save(fig: plt.Figure, theme: str, name: str) -> None:
    out_dir = ensure_dir(OUT_ROOT / theme)
    out_path = out_dir / name
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] {theme}/{name}")


def load_csv(name: str) -> pd.DataFrame | None:
    path = SIM_STATS / name
    if not path.exists():
        print(f"  [SKIP] missing file: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        print(f"  [SKIP] could not read {name}: {exc}")
        return None


def style_axes(ax: plt.Axes) -> None:
    """Remove top/right spines for a cleaner look."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# The 9 H2H category columns as stored in the value CSVs.
H2H_CATS = ["PTS", "REB", "AST", "STL", "BLK", "FG3M", "TOV", "FG", "FT"]

# Human-readable names for every stat abbreviation, so readers who do not follow
# basketball can understand the figures.
STAT_LABELS = {
    "PTS": "Points",
    "REB": "Rebounds",
    "AST": "Assists",
    "STL": "Steals",
    "BLK": "Blocks",
    "FG3M": "3-Pointers Made",
    "TOV": "Turnovers",
    "FG": "Field Goal %",
    "FT": "Free Throw %",
    "FG%": "Field Goal %",
    "FT%": "Free Throw %",
    "FG_pct": "Field Goal %",
    "FT_pct": "Free Throw %",
    "games_played": "Games Played",
}


def pretty(stat: str) -> str:
    """Map a stat abbreviation to a full readable name (with abbreviation kept)."""
    full = STAT_LABELS.get(stat)
    if full is None:
        return stat
    abbr = stat.replace("_pct", "%")
    return f"{full}\n({abbr})"


def pretty_inline(stat: str) -> str:
    full = STAT_LABELS.get(stat)
    return f"{full} ({stat})" if full else stat


def strategy_label(strategy: str, punts: str, risk: float) -> str:
    """Build a short, readable label for a draft strategy from its parameters."""
    strategy = str(strategy or "baseline").lower()
    # Empty punt sets are stored as NaN (a float) by pandas; treat as no punt.
    punts = punts.strip() if isinstance(punts, str) else ""
    base = {
        "baseline": "Balanced",
        "blocks_heavy": "Blocks-heavy",
        "guard": "Guard-focused",
        "big": "Big-man-focused",
    }.get(strategy, strategy.title())

    if punts:
        nice = ", ".join(STAT_LABELS.get(p, p).replace(" %", "%") for p in punts.split(","))
        base = f"{base}\n(punt {nice})"
    elif strategy == "baseline" and risk and float(risk) > 0.2:
        base = f"{base}\n(low-risk)"
    return base


# ---------------------------------------------------------------------
# Theme: overview
# ---------------------------------------------------------------------

def fig_active_players_per_season(season_df: pd.DataFrame) -> None:
    counts = (
        season_df.dropna(subset=["season"])
        .groupby("season")["player_id"]
        .nunique()
        .reset_index(name="n_players")
        .sort_values("season")
    )
    if counts.empty:
        print("  [SKIP] no season data for active players")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    seasons = counts["season"].astype(int).astype(str)
    bars = ax.bar(seasons, counts["n_players"], color=PALETTE[0], width=0.62)
    for rect, val in zip(bars, counts["n_players"]):
        ax.text(rect.get_x() + rect.get_width() / 2, val + max(counts["n_players"]) * 0.01,
                f"{int(val)}", ha="center", va="bottom", fontsize=11, fontweight="bold",
                color="#1e293b")

    ax.set_title("Number of distinct active players per season (2023-2026)")
    ax.set_xlabel("Season")
    ax.set_ylabel("Number of distinct players")
    ax.set_ylim(0, counts["n_players"].max() * 1.12)
    style_axes(ax)
    fig.text(0.5, -0.02,
             "Each player counted once per season if they have at least one scraped game log "
             "(source: Basketball-Reference).",
             ha="center", fontsize=9, color="#64748b")
    save(fig, "overview", "active_players_per_season.png")


def fig_player_retention(season_df: pd.DataFrame) -> None:
    seasons = sorted(int(s) for s in season_df["season"].dropna().unique())
    if not seasons:
        print("  [SKIP] no seasons for retention")
        return
    latest = seasons[-1]
    latest_players = set(season_df.loc[season_df["season"] == latest, "player_id"])
    if not latest_players:
        print("  [SKIP] no players in latest season")
        return

    rows = []
    for s in seasons:
        players_s = set(season_df.loc[season_df["season"] == s, "player_id"])
        overlap = len(latest_players & players_s)
        rows.append((s, overlap))

    fig, ax = plt.subplots(figsize=(9, 6))
    labels = [str(s) for s, _ in rows]
    vals = [v for _, v in rows]
    colors = [HIGHLIGHT if s == latest else BLUE for s, _ in rows]
    bars = ax.bar(labels, vals, color=colors, width=0.62)
    for rect, val in zip(bars, vals):
        pct = 100.0 * val / len(latest_players)
        ax.text(rect.get_x() + rect.get_width() / 2, val + max(vals) * 0.01,
                f"{val}\n({pct:.0f}%)", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color="#1e293b")

    ax.set_title(f"Players active in {latest} who also appear in each earlier season")
    ax.set_xlabel("Season")
    ax.set_ylabel(f"Players also active in {latest}")
    ax.set_ylim(0, max(vals) * 1.16)
    style_axes(ax)
    fig.text(0.5, -0.02,
             f"Of the {len(latest_players)} players active in {latest}, each bar counts how many "
             f"also have game logs in that earlier season (the {latest} bar is the full pool).",
             ha="center", fontsize=9, color="#64748b")
    save(fig, "overview", "player_retention.png")


def fig_ppg_distribution_by_season(season_df: pd.DataFrame) -> None:
    if "pts_mean" not in season_df.columns:
        print("  [SKIP] no pts_mean column")
        return
    seasons = sorted(int(s) for s in season_df["season"].dropna().unique())
    if not seasons:
        return
    seasons = seasons[:4]  # 2x2 grid

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    for ax in axes:
        ax.set_visible(False)

    for i, s in enumerate(seasons):
        ax = axes[i]
        ax.set_visible(True)
        vals = pd.to_numeric(
            season_df.loc[season_df["season"] == s, "pts_mean"], errors="coerce"
        ).dropna()
        if vals.empty:
            continue
        ax.hist(vals, bins=25, color=PALETTE[0], alpha=0.75, edgecolor="white",
                density=True, label="Players")

        mu, sigma = float(vals.mean()), float(vals.std(ddof=1))
        if sigma > 0:
            xs = np.linspace(vals.min(), vals.max(), 200)
            ax.plot(xs, stats.norm.pdf(xs, mu, sigma), color=ORANGE, lw=2.4,
                    label=f"Normal fit (μ={mu:.1f}, σ={sigma:.1f})")
        ax.axvline(mu, color="#0f172a", ls="--", lw=1.4, label=f"Mean = {mu:.1f}")
        ax.set_title(f"{s} season")
        ax.set_xlabel("Points per game")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8.5)
        style_axes(ax)

    fig.suptitle("Distribution of points per game by season", y=0.99)
    fig.text(0.5, -0.01,
             "One panel per season. Bars are the empirical density of players' points-per-game; "
             "the orange curve is a fitted normal distribution and the dashed line marks the mean.",
             ha="center", fontsize=9, color="#64748b")
    save(fig, "overview", "ppg_distribution_by_season.png")


def run_overview() -> None:
    print("[overview]")
    season_df = load_csv("player_season_stats.csv")
    if season_df is None:
        return
    season_df["season"] = pd.to_numeric(season_df["season"], errors="coerce")
    fig_active_players_per_season(season_df)
    fig_player_retention(season_df)
    fig_ppg_distribution_by_season(season_df)


# ---------------------------------------------------------------------
# Theme: value
# ---------------------------------------------------------------------

def fig_category_breakdown(value_df: pd.DataFrame) -> None:
    cats = [c for c in H2H_CATS if c in value_df.columns]
    if not cats or "rank_H2H" not in value_df.columns:
        print("  [SKIP] missing category/rank columns")
        return

    top = value_df.sort_values("rank_H2H").head(15).copy()
    # A heatmap (position-encoded grid) is used instead of a stacked bar: the
    # course notes warn that stacked bars have a "jiggling baseline" that makes
    # individual segments hard to compare.
    plot = top.set_index("player_id")[cats]

    fig, ax = plt.subplots(figsize=(13, 8.5))
    im = ax.imshow(plot.values, cmap=DIVERGING, aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([pretty(c) for c in cats], fontsize=9)
    ax.set_yticks(range(len(plot.index)))
    ax.set_yticklabels(plot.index)
    for i in range(len(plot.index)):
        for j in range(len(cats)):
            v = plot.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > 0.6 else "#0f172a")
    ax.set_title("Normalized category scores of the top 15 players by H2H value")
    ax.set_xlabel("Scoring category")
    ax.set_ylabel("Player (ranked by H2H value, best at top)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Normalized score: red = league best, blue = low (turnovers stored negative)")
    ax.grid(False)
    fig.text(0.5, -0.02,
             "Each cell is a player's normalized score in one category (range -1 to 1). Warm red = "
             "strong, cold blue = weak. Turnovers are stored as negative because fewer is better.",
             ha="center", fontsize=9, color="#64748b")
    save(fig, "value", "category_breakdown_top15.png")


def fig_category_scarcity(season_df: pd.DataFrame) -> None:
    # Per-game category means -> coefficient of variation across players.
    cat_cols = {
        "PTS": "pts_mean",
        "REB": "trb_mean",
        "AST": "ast_mean",
        "STL": "stl_mean",
        "BLK": "blk_mean",
        "FG3M": "fg3m_mean",
        "TOV": "tov_mean",
    }
    available = {k: v for k, v in cat_cols.items() if v in season_df.columns}
    if not available:
        print("  [SKIP] no per-game mean columns for scarcity")
        return

    rows = []
    for cat, col in available.items():
        vals = pd.to_numeric(season_df[col], errors="coerce").dropna()
        vals = vals[vals >= 0]
        if vals.empty or vals.mean() == 0:
            continue
        cov = vals.std(ddof=1) / vals.mean()
        rows.append((cat, cov, vals.mean()))

    if not rows:
        return
    rows.sort(key=lambda r: r[1])
    cats = [r[0] for r in rows]
    covs = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    norm = (np.array(covs) - min(covs)) / (max(covs) - min(covs) + 1e-9)
    colors = SEQUENTIAL(0.15 + 0.7 * norm)
    bars = ax.bar([pretty(c) for c in cats], covs, color=colors, width=0.66,
                  edgecolor="white")
    for rect, cov in zip(bars, covs):
        ax.text(rect.get_x() + rect.get_width() / 2, cov + max(covs) * 0.01,
                f"{cov:.2f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color="#1e293b")

    ax.set_title("Spread of each statistical category across players (coefficient of variation)")
    ax.set_xlabel("Statistical category")
    ax.set_ylabel("Coefficient of variation (std / mean)")
    ax.set_ylim(0, max(covs) * 1.14)
    style_axes(ax)
    fig.text(0.5, -0.03,
             "Coefficient of variation = standard deviation divided by the mean of each per-game "
             "category across all players. Higher means the category is more spread out / scarcer.",
             ha="center", fontsize=9, color="#64748b")
    save(fig, "value", "category_scarcity.png")


def fig_category_correlation(value_df: pd.DataFrame) -> None:
    cats = [c for c in H2H_CATS if c in value_df.columns]
    if len(cats) < 2:
        print("  [SKIP] not enough category columns for correlation")
        return
    data = value_df[cats].apply(pd.to_numeric, errors="coerce")
    corr = data.corr().values

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1, aspect="equal")
    labels = [pretty(c) for c in cats]
    ax.set_xticks(range(len(cats)))
    ax.set_yticks(range(len(cats)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticklabels(labels, fontsize=8.5)
    for i in range(len(cats)):
        for j in range(len(cats)):
            v = corr[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.6 else "#0f172a", fontsize=9)
    ax.set_title("Correlation between scoring categories across players")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation (red = positive, blue = negative)")
    ax.grid(False)
    fig.text(0.5, -0.02,
             "Pearson correlation between players' normalized category scores. Red = move together, "
             "blue = move oppositely, near-white = unrelated.",
             ha="center", fontsize=9, color="#64748b")
    save(fig, "value", "category_correlation_heatmap.png")


def fig_top_h2h_value(value_df: pd.DataFrame, weekly_df: pd.DataFrame | None = None) -> None:
    if "H2H_value" not in value_df.columns:
        print("  [SKIP] no H2H_value column")
        return
    top = value_df.sort_values("H2H_value", ascending=False).head(30).copy()
    if weekly_df is not None and "games_played_mean" in weekly_df.columns:
        top = top.merge(
            weekly_df[["player_id", "games_played_mean"]], on="player_id", how="left"
        )
    top = top.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 11))
    norm = (top["H2H_value"] - top["H2H_value"].min()) / (
        top["H2H_value"].max() - top["H2H_value"].min() + 1e-9)
    colors = SEQUENTIAL(0.2 + 0.7 * norm.values)
    ax.barh(top["player_id"], top["H2H_value"], color=colors, edgecolor="white",
            height=0.72)
    has_gp = "games_played_mean" in top.columns
    for y, (_, row) in enumerate(top.iterrows()):
        txt = f"{row['H2H_value']:.2f}"
        if has_gp and pd.notna(row.get("games_played_mean")):
            txt += f"   ({row['games_played_mean']:.0f} games)"
        ax.text(row["H2H_value"] + top["H2H_value"].max() * 0.005, y, txt,
                va="center", fontsize=8.5, color="#1e293b")

    ax.set_title("Top 30 players by projected H2H value")
    ax.set_xlabel("Projected H2H value")
    ax.set_ylabel("Player")
    ax.set_xlim(0, top["H2H_value"].max() * 1.15)
    style_axes(ax)
    if has_gp:
        fig.text(0.5, -0.01,
                 "Numbers in parentheses are projected games played out of 82 (from the Monte Carlo "
                 "simulation). Weekly stats already reflect missed games; this shows durability on top.",
                 ha="center", fontsize=9, color="#64748b")
    save(fig, "value", "top30_h2h_value.png")


def fig_h2h_vs_games_played(value_df: pd.DataFrame, weekly_df: pd.DataFrame) -> None:
    if "games_played_mean" not in weekly_df.columns:
        print("  [SKIP] no games_played_mean in weekly projections")
        return
    df = value_df.merge(weekly_df[["player_id", "games_played_mean"]], on="player_id", how="inner")
    df["games_played_mean"] = pd.to_numeric(df["games_played_mean"], errors="coerce")
    df["H2H_value"] = pd.to_numeric(df["H2H_value"], errors="coerce")
    df = df.dropna(subset=["games_played_mean", "H2H_value"])
    if len(df) < 3:
        return

    x = df["games_played_mean"].values
    y = df["H2H_value"].values

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.scatter(x, y, s=36, color=BLUE, alpha=0.55, edgecolor="white", linewidth=0.5, zorder=3)
    ax.axvline(82, color="#94a3b8", ls=":", lw=1.2, label="full season (82 games)")

    slope, intercept, r, _, _ = stats.linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, slope * xs + intercept, color=ORANGE, lw=2.4,
            label=f"Trend (Pearson r = {r:.2f})")

    # Label a few notable players: high value + few games, and high value + many games.
    df["resid"] = y - (slope * x + intercept)
    label_idx = set()
    for subset in [
        df.sort_values("H2H_value", ascending=False).head(3).index,
        df.sort_values(["H2H_value", "games_played_mean"], ascending=[False, True]).head(3).index,
        df.sort_values("games_played_mean", ascending=False).head(2).index,
    ]:
        label_idx.update(subset)
    for idx in label_idx:
        ax.annotate(df.loc[idx, "player_id"],
                    (df.loc[idx, "games_played_mean"], df.loc[idx, "H2H_value"]),
                    fontsize=8, color="#475569", xytext=(4, 3), textcoords="offset points")

    ax.set_title("Projected H2H value versus projected games played")
    ax.set_xlabel("Projected games played (out of 82)")
    ax.set_ylabel("Projected H2H value")
    ax.legend()
    style_axes(ax)
    fig.text(0.5, -0.02,
             "Each point is a player. H2H value is built from projected weekly stats (which already "
             "account for missed games in the simulation). Games played is shown separately here "
             "because durability still affects how much total production a team receives.",
             ha="center", fontsize=9, color="#64748b")
    save(fig, "value", "h2h_value_vs_games_played.png")


def run_value() -> None:
    print("[value]")
    value_df = load_csv("h2h_value_2027.csv")
    weekly_df = load_csv("projected_2027_weekly.csv")
    season_df = load_csv("player_season_stats.csv")
    if value_df is not None:
        fig_category_breakdown(value_df)
        fig_category_correlation(value_df)
        fig_top_h2h_value(value_df, weekly_df)
        if weekly_df is not None:
            fig_h2h_vs_games_played(value_df, weekly_df)
    if season_df is not None:
        fig_category_scarcity(season_df)


# ---------------------------------------------------------------------
# Theme: thesis (pre-draft H2H value vs simulated win shares)
# ---------------------------------------------------------------------

def fig_ws_vs_h2h_scatter(ws_df: pd.DataFrame) -> None:
    needed = {"win_shares", "H2H_value"}
    if not needed.issubset(ws_df.columns):
        print("  [SKIP] missing win_shares/H2H_value")
        return
    df = ws_df.dropna(subset=["win_shares", "H2H_value"]).copy()
    x = pd.to_numeric(df["win_shares"], errors="coerce")
    y = pd.to_numeric(df["H2H_value"], errors="coerce")
    mask = x.notna() & y.notna()
    x, y = x[mask].values, y[mask].values
    if len(x) < 3:
        return

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.scatter(x, y, s=42, color=BLUE, alpha=0.6, edgecolor="white", linewidth=0.6,
               zorder=3)

    slope, intercept, r, _, _ = stats.linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, slope * xs + intercept, color=ORANGE, lw=2.6, zorder=4,
            label=f"Trend (Pearson r = {r:.2f})")

    # Annotate a few notable players (largest divergence from the trend).
    if "player_id" in df.columns:
        resid = y - (slope * x + intercept)
        order = np.argsort(np.abs(resid))[::-1][:6]
        ids = df.loc[mask, "player_id"].values
        for idx in order:
            ax.annotate(ids[idx], (x[idx], y[idx]), fontsize=8.5, color="#475569",
                        xytext=(5, 4), textcoords="offset points")

    ax.set_title("Pre-draft H2H value versus simulated win shares")
    ax.set_xlabel("Simulated win shares (extra weekly matchup wins vs replacement)")
    ax.set_ylabel("Pre-draft H2H value")
    ax.legend()
    style_axes(ax)
    fig.text(0.5, -0.03,
             "Each point is a drafted player (n ≈ 132). Win shares come from a simulated H2H "
             "season (draft/compute_win_shares.py): for each matchup, stats are sampled from "
             "projected weekly distributions and the player is swapped with a replacement-level "
             "undrafted player. H2H value is a static pre-draft category score. Neither axis "
             "reflects actual NBA performance.",
             ha="center", fontsize=9, color="#64748b")
    save(fig, "thesis", "ws_vs_h2h_scatter.png")


def fig_over_under_valued(ws_df: pd.DataFrame) -> None:
    if "rank_diff" not in ws_df.columns or "player_id" not in ws_df.columns:
        print("  [SKIP] missing rank_diff/player_id")
        return
    df = ws_df.dropna(subset=["rank_diff"]).copy()
    df["rank_diff"] = pd.to_numeric(df["rank_diff"], errors="coerce")
    df = df.dropna(subset=["rank_diff"])
    if df.empty:
        return

    top_over = df.sort_values("rank_diff", ascending=False).head(12)
    top_under = df.sort_values("rank_diff").head(12)
    combined = pd.concat([top_under, top_over]).drop_duplicates("player_id")
    combined = combined.sort_values("rank_diff")

    fig, ax = plt.subplots(figsize=(11, 9))
    # Blue / orange instead of green / red so the chart stays colorblind-safe.
    colors = [BLUE if v > 0 else ORANGE for v in combined["rank_diff"]]
    ax.barh(combined["player_id"], combined["rank_diff"], color=colors,
            edgecolor="white", height=0.72)
    ax.axvline(0, color="#0f172a", lw=1.0)
    for y, val in enumerate(combined["rank_diff"]):
        offset = 1 if val >= 0 else -1
        ax.text(val + offset, y, f"{val:+.0f}", va="center",
                ha="left" if val >= 0 else "right", fontsize=8.5, color="#1e293b")

    ax.set_title("Largest rank gaps between pre-draft H2H value and simulated win shares")
    ax.set_xlabel("Rank difference  (simulated win-shares rank − pre-draft H2H rank)")
    ax.set_ylabel("Player")
    style_axes(ax)

    # Manual legend.
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=BLUE, label="Ranked higher pre-draft than in simulation (paper > league)"),
        Patch(color=ORANGE, label="Ranked higher in simulation than pre-draft (league > paper)"),
    ], loc="lower right")
    fig.text(0.5, -0.03,
             "Only drafted players from one simulated 12-team snake draft. Positive (blue) means "
             "the pre-draft H2H score ranks the player much better than their simulated matchup "
             "contribution; negative (orange) means the opposite.",
             ha="center", fontsize=9, color="#64748b")
    save(fig, "thesis", "over_under_valued.png")


def run_thesis() -> None:
    print("[thesis]")
    ws_df = load_csv("ws_vs_h2h_2027.csv")
    if ws_df is None:
        return
    fig_ws_vs_h2h_scatter(ws_df)
    fig_over_under_valued(ws_df)


# ---------------------------------------------------------------------
# Theme: model evaluation
# ---------------------------------------------------------------------

MODEL_STATS = [
    ("PTS", "pred_PTS_mean", "actual_PTS_mean"),
    ("REB", "pred_REB_mean", "actual_REB_mean"),
    ("AST", "pred_AST_mean", "actual_AST_mean"),
    ("STL", "pred_STL_mean", "actual_STL_mean"),
    ("BLK", "pred_BLK_mean", "actual_BLK_mean"),
    ("TOV", "pred_TOV_mean", "actual_TOV_mean"),
    ("FG3M", "pred_FG3M_mean", "actual_FG3M_mean"),
    ("FG%", "pred_FG%_mean", "actual_FG%_mean"),
    ("FT%", "pred_FT%_mean", "actual_FT%_mean"),
]


def fig_pred_vs_actual(eval_df: pd.DataFrame) -> None:
    usable = [(lbl, p, a) for lbl, p, a in MODEL_STATS
              if p in eval_df.columns and a in eval_df.columns]
    if not usable:
        print("  [SKIP] no pred/actual columns")
        return

    n = len(usable)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax in axes:
        ax.set_visible(False)

    for i, (lbl, pcol, acol) in enumerate(usable):
        ax = axes[i]
        ax.set_visible(True)
        pred = pd.to_numeric(eval_df[pcol], errors="coerce")
        act = pd.to_numeric(eval_df[acol], errors="coerce")
        m = pred.notna() & act.notna()
        pred, act = pred[m].values, act[m].values
        if len(pred) < 2:
            continue
        ax.scatter(act, pred, s=24, color=BLUE, alpha=0.5, edgecolor="white",
                   linewidth=0.4, zorder=3)
        lo = float(min(act.min(), pred.min()))
        hi = float(max(act.max(), pred.max()))
        ax.plot([lo, hi], [lo, hi], color="#0f172a", ls="--", lw=1.3, zorder=4,
                label="perfect (y = x)")
        r = np.corrcoef(act, pred)[0, 1]
        ax.set_title(f"{pretty_inline(lbl)}  (r = {r:.2f})", fontsize=11)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.legend(fontsize=8)
        style_axes(ax)

    fig.suptitle("Projected versus actual per-game statistics (2026)", y=0.997)
    fig.text(0.5, -0.01,
             "One panel per projected statistic. Each point is a player; the dashed line is a "
             "perfect prediction (y = x). r is the Pearson correlation for that statistic.",
             ha="center", fontsize=9, color="#64748b")
    save(fig, "model", "predicted_vs_actual.png")


def fig_error_metrics(summary_df: pd.DataFrame) -> None:
    if "stat" not in summary_df.columns:
        print("  [SKIP] no stat column in summary")
        return
    df = summary_df.copy()

    # MAE / RMSE grouped bar (counting raw stats only; pct stats are tiny scale,
    # but we keep all and let the chart speak).
    if {"mae", "rmse"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(12, 6.5))
        x = np.arange(len(df))
        w = 0.4
        ax.bar(x - w / 2, df["mae"], w, label="MAE", color=BLUE, edgecolor="white")
        ax.bar(x + w / 2, df["rmse"], w, label="RMSE", color=ORANGE, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([pretty(s) for s in df["stat"]], fontsize=8.5)
        ax.set_title("Projection error per statistic (MAE and RMSE)")
        ax.set_ylabel("Error (in each statistic's own units)")
        ax.legend()
        style_axes(ax)
        fig.text(0.5, -0.05,
                 "MAE = mean absolute error, RMSE = root mean squared error, computed per statistic "
                 "in that statistic's own units (so scales are not comparable across statistics).",
                 ha="center", fontsize=9, color="#64748b")
        save(fig, "model", "error_mae_rmse.png")

    # Pearson / Spearman grouped bar.
    if {"pearson", "spearman"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(12, 6.5))
        x = np.arange(len(df))
        w = 0.4
        ax.bar(x - w / 2, df["pearson"], w, label="Pearson", color=BLUE,
               edgecolor="white")
        ax.bar(x + w / 2, df["spearman"], w, label="Spearman", color=ORANGE,
               edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([pretty(s) for s in df["stat"]], fontsize=8.5)
        ax.set_ylim(0, 1)
        ax.set_title("Correlation between projected and actual values, per statistic")
        ax.set_ylabel("Correlation with actual (0-1)")
        ax.legend()
        style_axes(ax)
        fig.text(0.5, -0.05,
                 "Pearson (linear) and Spearman (rank) correlation between each player's projected "
                 "and actual value, computed per statistic. Higher means projections rank players "
                 "more like reality.",
                 ha="center", fontsize=9, color="#64748b")
        save(fig, "model", "correlation_pearson_spearman.png")


def fig_ci_calibration(summary_df: pd.DataFrame) -> None:
    if not {"stat", "ci_5_95_coverage"}.issubset(summary_df.columns):
        print("  [SKIP] no ci coverage column")
        return
    df = summary_df.copy()
    cov = pd.to_numeric(df["ci_5_95_coverage"], errors="coerce")

    fig, ax = plt.subplots(figsize=(12, 6.5))
    bars = ax.bar(df["stat"], cov, color=BLUE, edgecolor="white", width=0.66)
    ax.axhline(0.90, color=ORANGE, ls="--", lw=2.0, label="ideal coverage = 0.90")
    for rect, c in zip(bars, cov):
        ax.text(rect.get_x() + rect.get_width() / 2, c + 0.01, f"{c:.2f}",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                color="#1e293b")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([pretty(s) for s in df["stat"]], fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_title("Share of actual values inside the predicted 5-95% interval, per statistic")
    ax.set_ylabel("Fraction of players inside the interval")
    ax.legend()
    style_axes(ax)
    fig.text(0.5, -0.04,
             "Each bar is the fraction of players whose actual value fell inside the model's "
             "predicted 5-95% interval. The dashed line marks the ideal value of 0.90.",
             ha="center", fontsize=9, color="#64748b")
    save(fig, "model", "ci_calibration.png")


def run_model() -> None:
    print("[model]")
    eval_df = load_csv("evaluation_2026.csv")
    summary_df = load_csv("evaluation_2026_summary.csv")
    if eval_df is not None:
        fig_pred_vs_actual(eval_df)
    if summary_df is not None:
        fig_error_metrics(summary_df)
        fig_ci_calibration(summary_df)


# ---------------------------------------------------------------------
# Theme: draft
# ---------------------------------------------------------------------

def _strategy_labels_for(df: pd.DataFrame) -> list[str]:
    """Build readable strategy labels from a frame that has strategy/punts/risk."""
    if "strategy" in df.columns:
        return [strategy_label(r.get("strategy"), r.get("punts", ""), r.get("risk", 0.0))
                for _, r in df.iterrows()]
    # Fallback: plain team labels.
    return [f"Team {int(t)}" if pd.notna(t) and str(t).replace('.', '').isdigit()
            else str(t) for t in df["team"]]


def fig_standings(stand_df: pd.DataFrame) -> None:
    if not {"team", "W"}.issubset(stand_df.columns):
        print("  [SKIP] missing team/W in standings")
        return
    df = stand_df.sort_values("W", ascending=True).copy()
    labels = _strategy_labels_for(df)

    fig, ax = plt.subplots(figsize=(11, 8))
    norm = (df["W"] - df["W"].min()) / (df["W"].max() - df["W"].min() + 1e-9)
    colors = SEQUENTIAL(0.2 + 0.7 * norm.values)
    bars = ax.barh(range(len(df)), df["W"], color=colors, edgecolor="white", height=0.72)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=9)
    has_cat = "cat_pts" in df.columns
    for y, (_, row) in enumerate(df.iterrows()):
        txt = f"{row['W']:.2f} wins"
        if has_cat:
            txt += f"   ({row['cat_pts']:.1f} cat. pts)"
        ax.text(row["W"] + df["W"].max() * 0.01, y, txt, va="center", fontsize=8.5,
                color="#1e293b")

    ax.set_title("Average weekly wins by draft strategy (simulated 2027 season)")
    ax.set_xlabel("Average weekly matchups won (out of 9 categories)")
    ax.set_ylabel("Draft strategy")
    ax.set_xlim(0, df["W"].max() * 1.22)
    style_axes(ax)
    fig.text(0.5, -0.02,
             "Each row is one drafting strategy used to build a team; bars show the average number "
             "of weekly category matchups it won across the simulated season.",
             ha="center", fontsize=9, color="#64748b")
    save(fig, "draft", "standings_wins.png")


def fig_team_strengths(avg_df: pd.DataFrame, stand_df: pd.DataFrame | None) -> None:
    cat_cols = [c for c in ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M",
                            "FG_pct", "FT_pct"] if c in avg_df.columns]
    if "team" not in avg_df.columns or len(cat_cols) < 2:
        print("  [SKIP] missing team/category columns for strengths")
        return

    df = avg_df.copy()
    # Order teams by standings (wins), best at top, and borrow strategy labels.
    label_map = {}
    if stand_df is not None and {"team", "W"}.issubset(stand_df.columns):
        s = stand_df.sort_values("W", ascending=False).reset_index(drop=True)
        order = s["team"].tolist()
        s_labels = _strategy_labels_for(s)
        label_map = {t: lab for t, lab in zip(s["team"], s_labels)}
        df["__ord"] = df["team"].apply(lambda t: order.index(t) if t in order else 1e9)
        df = df.sort_values("__ord")

    data = df[cat_cols].apply(pd.to_numeric, errors="coerce")
    # Column-wise z-score so each category is on the same relative scale.
    # TOV is bad: flip its sign so "strong" is always warm/high.
    z = (data - data.mean()) / (data.std(ddof=0) + 1e-9)
    if "TOV" in z.columns:
        z["TOV"] = -z["TOV"]

    row_labels = [label_map.get(t, f"Team {int(t)}" if pd.notna(t) else str(t))
                  for t in df["team"]]

    fig, ax = plt.subplots(figsize=(13, 9))
    im = ax.imshow(z.values, cmap=DIVERGING, vmin=-2.2, vmax=2.2, aspect="auto")
    ax.set_xticks(range(len(cat_cols)))
    ax.set_xticklabels([pretty(c) for c in cat_cols], fontsize=8.5)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8.5)
    for i in range(len(row_labels)):
        for j in range(len(cat_cols)):
            v = z.values[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                    color="white" if abs(v) > 1.3 else "#0f172a", fontsize=8)
    ax.set_title("Category strengths by draft strategy (z-scores)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Relative strength: red = above league average, blue = below")
    ax.grid(False)
    fig.text(0.5, -0.02,
             "Rows are draft strategies (ordered best-to-worst by wins), columns are the nine "
             "categories. Each cell is a z-score within its category; turnovers are sign-flipped so "
             "red always means stronger.",
             ha="center", fontsize=9, color="#64748b")
    save(fig, "draft", "team_category_strengths.png")


def run_draft() -> None:
    print("[draft]")
    # Prefer the multi-strategy draft (teams = different strategies). Fall back to
    # the single-strategy draft only if the strategic files are absent.
    stand_df = load_csv("strategic_draft_standings_2027.csv")
    avg_df = load_csv("strategic_draft_team_averages_2027.csv")
    if stand_df is None:
        stand_df = load_csv("draft_standings_2027.csv")
    if avg_df is None:
        avg_df = load_csv("draft_team_averages_2027.csv")
    if stand_df is not None:
        fig_standings(stand_df)
    if avg_df is not None:
        fig_team_strengths(avg_df, stand_df)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

RUNNERS = {
    "overview": run_overview,
    "value": run_value,
    "thesis": run_thesis,
    "model": run_model,
    "draft": run_draft,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--theme", choices=THEMES, default=None,
                        help="Only generate one theme (default: all).")
    args = parser.parse_args()

    set_theme()
    ensure_dir(OUT_ROOT)

    themes = [args.theme] if args.theme else THEMES
    print(f"Output directory: {OUT_ROOT}")
    for theme in themes:
        RUNNERS[theme]()

    print("\nDone. Figures saved under:", OUT_ROOT)


if __name__ == "__main__":
    main()
