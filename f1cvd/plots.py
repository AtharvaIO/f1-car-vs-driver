"""Figures: team fingerprint radar, driver heatmap, cumulative delta trace."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .segments import CATEGORIES


def radar(team_fp, teams=None, path="results/team_radar.png"):
    """Outward = faster. Axis values are seconds gained per km of that segment type."""
    teams = list(team_fp.index if teams is None else teams)
    cats = [c for c in CATEGORIES if c in team_fp.columns]
    ang = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist(); ang += ang[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    allv = -team_fp.loc[teams, cats].to_numpy()
    lo, hi = min(-0.1, float(allv.min()) - 0.02), float(allv.max()) + 0.03
    ax.set_ylim(lo, hi)
    ax.plot(ang, [0.0] * len(ang), "k--", lw=1.0)            # dashed ring = field average
    for t in teams:
        vals = (-team_fp.loc[t, cats]).tolist(); vals += vals[:1]
        ax.plot(ang, vals, label=t, lw=1.8); ax.fill(ang, vals, alpha=0.06)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=9)
    ax.set_rlabel_position(200); ax.tick_params(axis="y", labelsize=7)
    ax.set_title("Car fingerprint: s/km gained vs field (outward = faster, dashed = field average)", pad=22, fontsize=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def driver_heatmap(driver_fp, path="results/driver_heatmap.png"):
    cats = [c for c in CATEGORIES if c in driver_fp.columns]
    data = -driver_fp[cats].sort_values(cats[0]).to_numpy()
    lim = np.abs(data).max() or 1
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(driver_fp) + 2))
    im = ax.imshow(data, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, rotation=35, ha="right")
    ax.set_yticks(range(len(driver_fp))); ax.set_yticklabels(driver_fp[cats].sort_values(cats[0]).index)
    ax.set_title("Driver skill after removing the car (green = faster than teammates' baseline)")
    fig.colorbar(im, label="s/km gained"); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def delta_trace(ev, a, b, segs, path="results/delta_trace.png"):
    """Cumulative time gap of driver a vs b along the lap, shaded by segment type."""
    ia, ib = list(ev.drivers).index(a), list(ev.drivers).index(b)
    gap = ev.time[ia] - ev.time[ib]
    fig, ax = plt.subplots(figsize=(11, 4))
    colors = {"straight": "#dddddd", "slow": "#f4b6b6", "medium": "#f7e1a0", "fast": "#b6d7f4"}
    for _, s in segs.iterrows():
        key = "straight" if s.category == "straight" else s.category.split("_")[0]
        ax.axvspan(ev.distance[s.start], ev.distance[s.end], color=colors[key], alpha=0.6, lw=0)
    ax.plot(ev.distance, gap, color="k")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("distance (m)"); ax.set_ylabel(f"{a} minus {b} (s)")
    ax.set_title(f"{ev.key}: where {a} gains/loses on {b}  (below zero = {a} ahead)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
