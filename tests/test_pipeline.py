"""Run with `pytest` (or `python tests/test_pipeline.py`). Uses synthetic data only."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from f1cvd.synthetic import make_season
from f1cvd.decompose import decompose_event, category_table, lap_delta
from f1cvd.pipeline import run_analysis

EVENTS, TRUTH = make_season(seed=0)


def test_segments_partition_lap_and_deltas_sum_to_lap_delta():
    ev = EVENTS[0]
    long, segs = decompose_event(ev)
    assert segs.start.iloc[0] == 0 and segs.end.iloc[-1] == len(ev.distance) - 1
    assert (segs.start.values[1:] == segs.end.values[:-1]).all()      # contiguous, no gaps
    lap = lap_delta(category_table(long)).set_index("driver")["delta"]
    expected = dict(zip(ev.drivers, ev.time[:, -1] - ev.time[:, -1].mean()))
    assert max(abs(lap[d] - expected[d]) for d in lap.index) < 1e-6   # exact identity


def test_recovers_planted_car_and_driver_effects(tmp_path):
    res = run_analysis(EVENTS, tmp_path, verbose=False, team_scope="all")
    fit = res["fit"]
    assert fit["team_total"].corr(TRUTH["team_total"]) > 0.9
    assert fit["driver_total"].corr(TRUTH["driver_total"]) > 0.8
    # per-category fingerprints: planted effect is a speed gain, recovered rate is time lost
    cors = [fit["team_fp"][c].corr(-TRUTH["team_fp"][c]) for c in fit["team_fp"].columns]
    assert np.nanmean(cors) > 0.6, cors


def test_fixed_teammate_pairs_do_not_collapse_car_into_driver(tmp_path):
    """Regression: with pairs that never change teams, shrinking both effects pushed the car
    effect towards zero and inflated drivers (seen on real 2023-25 data)."""
    ev, tr = make_season(rotate=False, seed=4)
    dt = tr["driver_total"]; tm = dict(zip(ev[0].drivers, ev[0].teams))
    rel = dt - dt.groupby(dt.index.map(tm)).transform("mean")      # driver relative to teammate
    fit = run_analysis(ev, tmp_path, verbose=False, team_scope="all")["fit"]
    assert fit["driver_total"].corr(rel) > 0.9
    assert fit["spread"]["car_sd_s"] > 0.7 * tr["team_total"].std()


def test_fingerprint_prediction_beats_layout_blind_baseline(tmp_path):
    pred = run_analysis(EVENTS, tmp_path, verbose=False, team_scope="all")["pred"]
    assert pred["mae_fingerprint_s"].mean() < pred["mae_baseline_s"].mean()


if __name__ == "__main__":
    import tempfile
    for fn in (test_segments_partition_lap_and_deltas_sum_to_lap_delta,
               test_recovers_planted_car_and_driver_effects,
               test_fixed_teammate_pairs_do_not_collapse_car_into_driver,
               test_fingerprint_prediction_beats_layout_blind_baseline):
        fn(pathlib.Path(tempfile.mkdtemp())) if fn.__code__.co_argcount else fn()
        print("PASS", fn.__name__)
