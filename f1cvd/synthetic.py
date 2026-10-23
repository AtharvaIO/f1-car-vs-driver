"""Synthetic qualifying telemetry with PLANTED car and driver effects.

Used to validate the method offline: if the pipeline cannot recover effects we put in
ourselves, it has no business interpreting real data. Effects are fractional speed
changes applied per segment category (positive = faster).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from .data import EventData
from .segments import CATEGORIES

STEP = 5.0


def _track(rng):
    length = rng.uniform(4500, 6500)
    grid = np.arange(0, length, STEP)
    n_c = int(rng.integers(11, 17))
    pos = np.linspace(300, length - 300, n_c) + rng.uniform(-100, 100, n_c)
    cls = rng.choice(3, n_c, p=rng.dirichlet([2, 2, 2]))            # 0 slow, 1 medium, 2 fast
    apex = np.select([cls == 0, cls == 1], [rng.uniform(70, 110, n_c), rng.uniform(130, 185, n_c)],
                     rng.uniform(210, 260, n_c))
    vmax = rng.uniform(310, 335)
    drop = vmax - apex
    w = np.where(grid[None, :] < pos[:, None], 1.2 * drop[:, None], 1.6 * drop[:, None])
    f = np.exp(-(((grid[None, :] - pos[:, None]) / w) ** 2))          # (n_c, g)
    base = np.clip(vmax - (drop[:, None] * f).max(axis=0), 50, None)
    names = ["slow", "medium", "fast"]
    masks = np.zeros((len(CATEGORIES), len(grid)))
    for i, cat in enumerate(CATEGORIES):
        if cat == "straight":
            masks[i] = 1 - f.max(axis=0)
            continue
        c_name, phase = cat.split("_")
        for k in np.where(cls == names.index(c_name))[0]:
            side = grid < pos[k] if phase == "entry" else grid >= pos[k]
            masks[i] = np.maximum(masks[i], f[k] * side)
    return grid, base, masks


def _lap(base, masks, effect, noise):
    v = base * (1 + effect @ masks) * (1 + noise)
    t = np.r_[0.0, np.cumsum(STEP / (v[1:] / 3.6))]
    return v, t


def make_season(n_events=18, n_teams=8, n_seasons=3, seed=0, rotate=True):
    """Return (events, truth). truth has per-team / per-driver planted total effects (s).

    Second drivers rotate to the next team each season. Without such moves a driver's
    shared quality is mathematically indistinguishable from the car's.
    """
    rng = np.random.default_rng(seed)
    teams = [f"Team{chr(65 + i)}" for i in range(n_teams)]
    drivers = [f"D{i:02d}" for i in range(2 * n_teams)]
    def team_of(i, season):
        return teams[(i // 2 + season) % n_teams] if (i % 2 and rotate) else teams[i // 2]
    nc = len(CATEGORIES)
    t_eff = rng.normal(0, 0.010, (n_teams, nc))
    d_eff = rng.normal(0, 0.004, (len(drivers), nc))

    events, tot_team, tot_drv = [], np.zeros(n_teams), np.zeros(len(drivers))
    for e in range(n_events):
        season = e * n_seasons // n_events
        grid, base, masks = _track(rng)
        _, t0 = _lap(base, masks, np.zeros(nc), 0.0)
        speeds, times = [], []
        for i, d in enumerate(drivers):
            ti = teams.index(team_of(i, season))
            noise = uniform_filter1d(rng.normal(0, 0.004, len(grid)), 20)
            v, t = _lap(base, masks, t_eff[ti] + d_eff[i], noise)
            speeds.append(v); times.append(t)
            tot_team[ti] += (_lap(base, masks, t_eff[ti], 0.0)[1][-1] - t0[-1]) / 2   # /2: two drivers
            tot_drv[i] += _lap(base, masks, d_eff[i], 0.0)[1][-1] - t0[-1]
        times = np.array(times)
        events.append(EventData(2000 + season, f"Synthetic GP {e + 1}", grid, np.array(drivers),
                                np.array([team_of(i, season) for i in range(len(drivers))]), times[:, -1],
                                np.array(speeds), times))
    truth = {
        "team_total": pd.Series(tot_team / n_events, index=teams),
        "driver_total": pd.Series(tot_drv / n_events, index=drivers),
        "team_fp": pd.DataFrame(t_eff, index=teams, columns=CATEGORIES),
        "driver_fp": pd.DataFrame(d_eff, index=drivers, columns=CATEGORIES),
    }
    return events, truth
