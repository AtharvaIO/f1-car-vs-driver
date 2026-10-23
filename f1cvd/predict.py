"""Can a team's fingerprint predict its pace at a track it was not trained on?

For each held-out event:
  predicted team delta = sum over segment types of (team's mean rate in OTHER events,
                         s/km) x (km of that segment type at the held-out track).
Baseline = the team's mean total delta in the other events, ignoring the layout.
If the fingerprint carries real information, it should beat the baseline.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _rho(x, y):
    """Spearman rank correlation, NaN (not a warning) when either input is constant."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float("nan") if np.ptp(x) == 0 or np.ptp(y) == 0 else float(spearmanr(x, y)[0])


def loeo_predict(cat: pd.DataFrame, min_train_events: int = 4) -> pd.DataFrame:
    team_rate = (cat.groupby(["event", "team", "category"])["rate"].mean().reset_index())
    km = cat.groupby(["event", "category"])["length_m"].mean().unstack(fill_value=0) / 1000.0
    total = (cat.groupby(["event", "driver", "team"])["delta"].sum()
                .groupby(["event", "team"]).mean())
    events = list(km.index)
    rows = []
    for e in events:
        train = team_rate[team_rate["event"] != e]
        if train["event"].nunique() < min_train_events:
            continue
        rate = train.groupby(["team", "category"])["rate"].mean().unstack()
        base = total[total.index.get_level_values("event") != e].groupby("team").mean()
        actual = total.xs(e, level="event")
        teams = [t for t in actual.index if t in rate.index and t in base.index]
        if len(teams) < 4:
            continue
        cols = [c for c in rate.columns if c in km.columns]
        pred = pd.Series({t: float((rate.loc[t, cols].fillna(0) * km.loc[e, cols]).sum()) for t in teams})
        pred -= pred.mean()
        b = base[teams] - base[teams].mean()
        a = actual[teams] - actual[teams].mean()
        rows.append({
            "event": e, "n_teams": len(teams),
            "spearman_fingerprint": _rho(pred, a),
            "spearman_baseline": _rho(b, a),
            "mae_fingerprint_s": float((pred - a).abs().mean()),
            "mae_baseline_s": float((b - a).abs().mean())})
    return pd.DataFrame(rows)
