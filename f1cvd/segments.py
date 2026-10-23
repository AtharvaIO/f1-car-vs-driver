"""Split a lap into straights and corner entry/exit phases, classified by apex speed.

Method (run on the fastest lap of each event, then applied to every driver):
  1. Smooth the speed trace and find apexes = prominent local speed minima.
  2. Corner ENTRY  = braking point (highest speed since the previous apex) -> apex.
     Corner EXIT   = apex -> point where half the speed gain to the next peak is back.
  3. Everything else is STRAIGHT.
  4. Corners are classed slow / medium / fast by apex speed.

Entry time is mostly braking + turn-in; exit time is mostly traction. Segment times
partition the lap, so per-segment deltas sum exactly to the lap-time delta.
Limitation: flat-out sweeps with no real speed minimum are absorbed into straights.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

CATEGORIES = ["straight", "slow_entry", "slow_exit", "medium_entry", "medium_exit",
              "fast_entry", "fast_exit"]


def corner_class(apex_kmh: float, slow=120.0, fast=200.0) -> str:
    return "slow" if apex_kmh < slow else ("medium" if apex_kmh < fast else "fast")


def build_segments(distance, speed, slow=120.0, fast=200.0,
                   min_prominence=20.0, smooth_m=30.0) -> pd.DataFrame:
    n = len(speed)
    step = float(distance[1] - distance[0])
    v = uniform_filter1d(speed, size=max(1, int(round(smooth_m / step))), mode="nearest")
    apexes, _ = find_peaks(-v, prominence=min_prominence)

    rows, cursor, corner_no = [], 0, 0
    for k, a in enumerate(apexes):
        prev_a = apexes[k - 1] if k > 0 else 0
        next_a = apexes[k + 1] if k + 1 < len(apexes) else n - 1
        entry = max(prev_a + int(np.argmax(v[prev_a:a + 1])), cursor)
        window = v[a:next_a + 1]
        target = v[a] + 0.5 * (window.max() - v[a])
        exit_ = max(a + int(np.argmax(window >= target)), a + 1)
        if entry >= a:
            continue
        cls = corner_class(v[a], slow, fast)
        corner_no += 1
        if entry > cursor:
            rows.append((cursor, entry, "straight", 0, np.nan))
        rows.append((entry, a, f"{cls}_entry", corner_no, v[a]))
        rows.append((a, exit_, f"{cls}_exit", corner_no, v[a]))
        cursor = exit_
    if cursor < n - 1:
        rows.append((cursor, n - 1, "straight", 0, np.nan))

    segs = pd.DataFrame(rows, columns=["start", "end", "category", "corner", "apex_kmh"])
    segs = segs[segs.end > segs.start].reset_index(drop=True)
    segs["length_m"] = distance[segs.end.values] - distance[segs.start.values]
    return segs
