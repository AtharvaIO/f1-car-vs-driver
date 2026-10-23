"""Separate car from driver.

Two complementary views:

1. Teammate split (model-free). Within one event and segment type, the team mean is
   the "car level" (it also contains the average driver quality of the pair), and each
   driver's gap to the team mean isolates the driver, because the car is shared.

2. Crossed random-effects model:  y = mu + team_effect + driver_effect + noise,
   fitted as BLUPs via ridge regression. Penalties (the inverse variance ratios) are
   chosen by cross-validation grouped by event, so no event leaks between train and
   validation. This matches a mixed model with crossed random effects, but needs only
   numpy. It uses the information from drivers who change teams, and shrinks noisy
   estimates towards zero instead of over-trusting a few races.

Driver effects are therefore "relative to teammates (and to the drivers they swapped with)".
Sign convention: y is time LOST vs the field, so negative = faster.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


def teammate_split(cat: pd.DataFrame, col: str = "rate") -> pd.DataFrame:
    out = cat.copy()
    g = out.groupby(["event", "team", "category"])[col]
    out["car_level"] = g.transform("mean")
    out["n_mates"] = g.transform("count")
    out["driver_vs_mate"] = np.where(out["n_mates"] >= 2, out[col] - out["car_level"], np.nan)
    return out


@dataclass
class Fit:
    team: pd.Series
    driver: pd.Series
    lam_team: float
    lam_driver: float


def _onehot(values, levels):
    return (np.asarray(values)[:, None] == np.asarray(levels)[None, :]).astype(float)


def _solve(X, y, pen):
    mu = y.mean()
    beta = np.linalg.solve(X.T @ X + np.diag(pen), X.T @ (y - mu))
    return mu, beta


def fit_crossed(df: pd.DataFrame, col: str, grid=(0.1, 1, 5, 20, 100, 500),
                n_folds: int = 5, seed: int = 0, team_mode: str = "fixed") -> Fit:
    """team_mode="fixed": team levels are unshrunk, so driver effects are identified from
    teammate gaps (and drivers who change teams). The team effect is then the team's
    overall level INCLUDING the average quality of its drivers.
    team_mode="ridge": team and driver both shrunk; when most teammate pairs never change
    teams this arbitrarily splits their shared level between car and driver."""
    d = df.dropna(subset=[col]).reset_index(drop=True)
    teams, drivers = sorted(d["team"].unique()), sorted(d["driver"].unique())
    X = np.hstack([_onehot(d["team"], teams), _onehot(d["driver"], drivers)])
    nt = len(teams)
    y = d[col].to_numpy(float)

    events = d["event"].unique()
    k = max(2, min(n_folds, len(events)))
    order = np.random.default_rng(seed).permutation(events)
    fold_of = {e: i % k for i, e in enumerate(order)}
    fold = d["event"].map(fold_of).to_numpy()

    best = None
    for lt in ((1e-6,) if team_mode == "fixed" else grid):
        for ld in grid:
            pen = np.r_[np.full(nt, lt), np.full(len(drivers), ld)]
            err = 0.0
            for f in range(k):
                tr, te = fold != f, fold == f
                if tr.sum() == 0 or te.sum() == 0:
                    continue
                mu, b = _solve(X[tr], y[tr], pen)
                err += float(((y[te] - (mu + X[te] @ b)) ** 2).sum())
            if best is None or err < best[0]:
                best = (err, lt, ld)
    _, lt, ld = best
    pen = np.r_[np.full(nt, lt), np.full(len(drivers), ld)]
    _, b = _solve(X, y, pen)
    return Fit(pd.Series(b[:nt], index=teams), pd.Series(b[nt:], index=drivers), lt, ld)


def fit_all(cat: pd.DataFrame, lap: pd.DataFrame, team_mode: str = "fixed") -> dict:
    """Fit one model per segment category (rate, s/km) and one for total lap delta (s)."""
    team_fp, driver_fp, lams = {}, {}, {}
    for c in sorted(cat["category"].unique()):
        f = fit_crossed(cat[cat["category"] == c], "rate", team_mode=team_mode)
        team_fp[c], driver_fp[c] = f.team, f.driver
        lams[c] = (f.lam_team, f.lam_driver)
    total = fit_crossed(lap, "delta", team_mode=team_mode)
    return {
        "team_fp": pd.DataFrame(team_fp), "driver_fp": pd.DataFrame(driver_fp),
        "team_total": total.team, "driver_total": total.driver, "lambdas": lams,
        "spread": {"car_sd_s": float(total.team.std()), "driver_sd_s": float(total.driver.std())},
    }
