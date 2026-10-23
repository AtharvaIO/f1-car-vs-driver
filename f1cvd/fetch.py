"""Download qualifying telemetry with FastF1 and resample onto a distance grid.

Why qualifying: one flat-out lap per driver, low fuel, little traffic, so car and
driver differences are far less contaminated than in race data.
Why Q3 only: Q1/Q2 laps are set on a different track state, with different engine
modes, and by drivers who then get eliminated. Comparing them to Q3 laps is unfair.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .data import EventData, canon_team


def _fastest_laps(session, quali_part: int):
    """Return {driver: lap} using each driver's fastest lap in the chosen Q segment."""
    parts = session.laps.split_qualifying_sessions()   # [Q1, Q2, Q3]
    laps = parts[quali_part - 1]
    out = {}
    if laps is None or len(laps) == 0:
        return out
    for drv in laps["Driver"].unique():
        lap = laps.pick_drivers(drv).pick_fastest()
        if lap is None or lap["LapTime"] != lap["LapTime"]:   # None or NaT
            continue
        out[drv] = lap
    return out


def fetch_event(year: int, event: str, cache_dir="data/cache", step_m: float = 5.0,
                quali_part: int = 3, min_drivers: int = 8) -> EventData | None:
    import fastf1  # imported lazily so the rest of the package works without it
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    ses = fastf1.get_session(year, event, "Q")
    ses.load(telemetry=True, weather=False, messages=False)

    laps = _fastest_laps(ses, quali_part)
    if len(laps) < min_drivers and quali_part > 1:  # e.g. red-flagged Q3
        laps = _fastest_laps(ses, quali_part - 1)
    if len(laps) < min_drivers:
        print(f"  skip {year} {event}: only {len(laps)} usable laps")
        return None

    raw = {}
    for drv, lap in laps.items():
        tel = lap.get_car_data().add_distance()
        d = tel["Distance"].to_numpy(float)
        t = tel["Time"].dt.total_seconds().to_numpy(float)
        v = tel["Speed"].to_numpy(float)
        ok = np.r_[True, np.diff(d) > 0]            # strictly increasing distance
        if ok.sum() < 100:
            continue
        d, t, v = d[ok] - d[ok][0], t[ok] - t[ok][0], v[ok]
        raw[drv] = (d, t, v, lap["LapTime"].total_seconds(), canon_team(lap["Team"]))
    if len(raw) < min_drivers:
        print(f"  skip {year} {event}: telemetry too sparse")
        return None

    length = float(np.median([r[0][-1] for r in raw.values()]))
    grid = np.arange(0.0, length, step_m)
    drivers, teams, lts, spd, tim = [], [], [], [], []
    for drv, (d, t, v, lt, team) in raw.items():
        d = d * (length / d[-1])                    # remove integration drift in distance
        t = t * (lt / t[-1])                        # pin elapsed time to the official lap time
        spd.append(np.interp(grid, d, v)); tim.append(np.interp(grid, d, t))
        drivers.append(drv); teams.append(team); lts.append(lt)
    return EventData(year, event, grid, np.array(drivers), np.array(teams),
                     np.array(lts), np.array(spd), np.array(tim))


def event_names(year: int) -> list[str]:
    import fastf1
    sched = fastf1.get_event_schedule(year, include_testing=False)
    return list(sched["EventName"])


def fetch_season(year: int, out_dir="data/processed", cache_dir="data/cache", events=None):
    for name in events or event_names(year):
        try:
            ev = fetch_event(year, name, cache_dir)
        except Exception as exc:                    # one bad weekend should not kill the run
            print(f"  failed {year} {name}: {exc}")
            continue
        if ev is not None:
            print(f"  saved {ev.save(out_dir)}  ({len(ev.drivers)} drivers)")
