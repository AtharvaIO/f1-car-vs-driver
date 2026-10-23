"""Fetch qualifying telemetry from the OpenF1 API (https://openf1.org) instead of FastF1.

Why: since ~15 Oct 2026 F1's timing archive, which FastF1 reads, rejects anonymous requests.
OpenF1 is free for historical data (2023 onward), needs no login, and is rate limited to
3 requests/s and 30 requests/min. This client sleeps between calls and caches every
response on disk, so re-runs are free. Please keep it to personal / research use.

How the data is turned into a lap:
  * Q3 laps are identified by matching each driver's official Q3 time (session_result)
    to the lap with that exact lap_duration (falls back to Q2, then to fastest laps).
  * Car speed arrives at ~3.7 Hz (a sample every ~20 m at racing speed). Distance is the
    integral of speed, rescaled so every lap has the same length.
  * `date_start` of a lap is documented as approximate, so each lap's window is nudged
    by up to +-1 s to best match the field's median speed-vs-distance trace. Elapsed
    time is always pinned to the official lap_duration.
"""
from __future__ import annotations
import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np

from .data import EventData, canon_team

BASE = "https://api.openf1.org/v1"
DT = 0.02   # seconds, resolution used to integrate speed into distance


class Client:
    """Tiny OpenF1 client: throttled, retrying, disk-cached."""

    def __init__(self, cache_dir="data/cache/openf1", min_interval=2.2, timeout=60, verbose=True):
        self.cache = Path(cache_dir); self.cache.mkdir(parents=True, exist_ok=True)
        self.min_interval, self.timeout, self.verbose = min_interval, timeout, verbose
        self._last = 0.0

    @staticmethod
    def url(endpoint, params=None, where=()):
        q = [f"{k}={quote(str(v), safe=':')}" for k, v in (params or {}).items()]
        q += [f"{k}{op}{quote(str(v), safe=':')}" for k, op, v in where]   # e.g. date>=...
        return f"{BASE}/{endpoint}?" + "&".join(q)

    def get(self, endpoint, params=None, where=()):
        url = self.url(endpoint, params, where)
        f = self.cache / (hashlib.sha1(url.encode()).hexdigest() + ".json")
        if f.exists():
            return json.loads(f.read_text())
        data = self._request(url)
        if data:                                  # never cache empties: data may appear later
            f.write_text(json.dumps(data))
        return data

    def _request(self, url):
        for attempt in range(5):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "f1cvd (non-commercial research)"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                if e.code == 404:                 # OpenF1 answers 404 when a query matches nothing
                    return []
                if e.code in (429, 500, 502, 503, 504):
                    if self.verbose:
                        print(f"    server said {e.code}, waiting...")
                    time.sleep(10 * (attempt + 1)); continue
                raise
            except (urllib.error.URLError, TimeoutError):
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"OpenF1 request kept failing: {url}")


def _ts(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="milliseconds")


# ---------------------------------------------------------------- lap selection
def _target_laps(laps, results, part, tol=0.002):
    """{driver_number: lap row} for drivers who set a time in Q`part`."""
    out = {}
    for r in results:
        dur = r.get("duration")
        if not isinstance(dur, list) or len(dur) < part or dur[part - 1] in (None, ""):
            continue
        target = float(dur[part - 1])
        for lap in laps:
            if (lap["driver_number"] == r["driver_number"] and lap.get("lap_duration") is not None
                    and lap.get("date_start") and abs(lap["lap_duration"] - target) <= tol):
                out[r["driver_number"]] = lap; break
    return out


def _fastest_per_driver(laps, top=10):
    best = {}
    for lap in laps:
        d = lap.get("lap_duration")
        if d is None or lap.get("is_pit_out_lap") or not lap.get("date_start"):
            continue
        if lap["driver_number"] not in best or d < best[lap["driver_number"]]["lap_duration"]:
            best[lap["driver_number"]] = lap
    ranked = sorted(best.items(), key=lambda kv: kv[1]["lap_duration"])[:top]
    return dict(ranked)


# ---------------------------------------------------------------- telemetry -> lap profile
def _car_trace(client, session_key, drv, start, T, margin=3.0):
    rows = client.get("car_data", {"session_key": session_key, "driver_number": drv},
                      where=[("date", ">=", _iso(start - margin)), ("date", "<=", _iso(start + T + margin))])
    if not rows:
        return None
    ts = np.array([_ts(r["date"]) for r in rows]); v = np.array([r["speed"] for r in rows], float)
    o = np.argsort(ts)
    return ts[o], v[o]


def _usable(ts, start, T) -> bool:
    if len(ts) < 20 or ts[0] > start - 1.0 or ts[-1] < start + T + 1.0:
        return False
    inside = ts[(ts >= start - 1.0) & (ts <= start + T + 1.0)]
    return len(inside) > 20 and np.diff(inside).max() < 1.5


def _window(ts, v, start, T):
    """Speed and integrated distance over [start, start+T] on a fine time grid."""
    tg = np.arange(0.0, T + DT / 2, DT)
    vg = np.interp(start + tg, ts, v)
    d = np.r_[0.0, np.cumsum((vg[1:] + vg[:-1]) / 2 / 3.6 * DT)]
    return tg, vg, d


def _to_grid(tg, vg, d, grid, length):
    dn = np.maximum.accumulate(d * (length / d[-1]))
    dn = dn + np.arange(len(dn)) * 1e-9            # guarantee strictly increasing for interp
    return np.interp(grid, dn, vg), np.interp(grid, dn, tg)


def build_profiles(traces, step_m=5.0, align=True, max_shift=1.0, shift_step=0.05, iters=2):
    """traces: {drv: (ts, v, start, T)} -> (grid, {drv: (speed, time)}, {drv: shift_s})."""
    ends = [_window(ts, v, s, T)[2][-1] for ts, v, s, T in traces.values()]
    length = float(np.median(ends))
    grid = np.arange(0.0, length, step_m)
    shifts = {d: 0.0 for d in traces}
    offsets = np.arange(-max_shift, max_shift + 1e-9, shift_step) if align else np.array([0.0])

    def profile(drv, shift):
        ts, v, s, T = traces[drv]
        return _to_grid(*_window(ts, v, s + shift, T), grid, length)

    for _ in range(iters if align else 0):
        ref = np.median([profile(d, shifts[d])[0] for d in traces], axis=0)
        for d in traces:
            errs = [np.mean((profile(d, o)[0] - ref) ** 2) for o in offsets]
            shifts[d] = float(offsets[int(np.argmin(errs))])
    return grid, {d: profile(d, shifts[d]) for d in traces}, shifts


# ---------------------------------------------------------------- events
def fetch_event(client, session, step_m=5.0, quali_part=3, min_drivers=8, verbose=True):
    sk, year = session["session_key"], int(session["year"])
    name = session.get("circuit_short_name") or session.get("location") or str(sk)
    info = {d["driver_number"]: (d["name_acronym"], canon_team(d["team_name"]))
            for d in client.get("drivers", {"session_key": sk})}
    laps = client.get("laps", {"session_key": sk})
    results = client.get("session_result", {"session_key": sk})
    if not info or not laps:
        print(f"  skip {year} {name}: no driver/lap data yet"); return None

    if results:
        chosen = _target_laps(laps, results, quali_part)
        if len(chosen) < min_drivers and quali_part > 1:        # e.g. red-flagged Q3
            chosen = _target_laps(laps, results, quali_part - 1)
    else:                                                       # session_result is a beta endpoint
        chosen = _fastest_per_driver(laps)
    chosen = {d: l for d, l in chosen.items() if d in info}

    traces = {}
    for drv, lap in chosen.items():
        start, T = _ts(lap["date_start"]), float(lap["lap_duration"])
        tr = _car_trace(client, sk, drv, start, T)
        if tr is not None and _usable(tr[0], start, T):
            traces[drv] = (tr[0], tr[1], start, T)
        elif verbose:
            print(f"    drop {info[drv][0]}: telemetry gap")
    if len(traces) < min_drivers:
        print(f"  skip {year} {name}: only {len(traces)} usable laps"); return None

    grid, prof, shifts = build_profiles(traces, step_m)
    order = list(traces)
    if verbose:
        print(f"    window nudges (s): median |shift| {np.median(np.abs(list(shifts.values()))):.2f}")
    return EventData(
        year, name, grid,
        np.array([info[d][0] for d in order]), np.array([info[d][1] for d in order]),
        np.array([traces[d][3] for d in order]),
        np.array([prof[d][0] for d in order]), np.array([prof[d][1] for d in order]))


def _matches(session, tokens):
    hay = " ".join(str(session.get(k, "")) for k in ("circuit_short_name", "location", "country_name")).lower()
    return any(t.lower() in hay for t in tokens)


def fetch_season(year, out_dir="data/processed", events=None, client=None, step_m=5.0):
    client = client or Client()
    sessions = client.get("sessions", {"year": year, "session_name": "Qualifying"})
    now = datetime.now(timezone.utc).timestamp()
    sessions = sorted([s for s in sessions if not s.get("is_cancelled") and _ts(s["date_end"]) < now],
                      key=lambda s: s["date_start"])
    if events:
        sessions = [s for s in sessions if _matches(s, events)]
    if not sessions:
        print(f"  no finished qualifying sessions matched for {year}"); return []
    print(f"  {len(sessions)} qualifying sessions (about {len(sessions) * 0.5:.0f} min at the free rate limit)")
    saved = []
    for s in sessions:
        label = s.get("circuit_short_name") or s.get("location")
        try:
            ev = fetch_event(client, s, step_m)
        except Exception as exc:                                  # one bad weekend should not kill the run
            print(f"  failed {year} {label}: {exc}"); continue
        if ev is not None:
            print(f"  saved {ev.save(out_dir)}  ({len(ev.drivers)} drivers)")
            saved.append(ev)
    return saved
