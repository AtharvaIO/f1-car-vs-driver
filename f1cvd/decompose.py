"""Turn distance-aligned laps into per-segment time deltas vs the field."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .data import EventData
from .segments import build_segments


def decompose_event(ev: EventData, **seg_kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (long_df, segments). The reference lap for segmentation is the fastest lap."""
    ref = int(np.argmin(ev.lap_times))
    segs = build_segments(ev.distance, ev.speed[ref], **seg_kwargs)
    seg_time = ev.time[:, segs.end.values] - ev.time[:, segs.start.values]   # (n, m)
    delta = seg_time - seg_time.mean(axis=0, keepdims=True)                  # vs field mean
    n, m = seg_time.shape
    long = pd.DataFrame({
        "event": ev.key,
        "driver": np.repeat(ev.drivers, m), "team": np.repeat(ev.teams, m),
        "seg": np.tile(segs.index.values, n), "category": np.tile(segs.category.values, n),
        "seg_time": seg_time.ravel(), "delta": delta.ravel(),
        "length_m": np.tile(segs.length_m.values, n)})
    return long, segs


def category_table(long: pd.DataFrame) -> pd.DataFrame:
    """Sum segment deltas per (event, driver, category); add a rate in seconds per km.

    Rate = time lost (positive) or gained (negative) per km of that segment type, which
    makes tracks with different layouts comparable.
    """
    g = long.groupby(["event", "driver", "team", "category"], as_index=False).agg(
        delta=("delta", "sum"), length_m=("length_m", "sum"))
    g["rate"] = g["delta"] / (g["length_m"] / 1000.0)
    return g


def lap_delta(cat: pd.DataFrame) -> pd.DataFrame:
    """Total lap delta per (event, driver); equals lap time minus field mean."""
    return cat.groupby(["event", "driver", "team"], as_index=False)["delta"].sum()
