"""Offline tests for the OpenF1 path: a fake API serves synthetic laps in OpenF1's format."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from datetime import datetime, timezone
from f1cvd.synthetic import make_season
from f1cvd.openf1 import Client, fetch_event, _target_laps
from f1cvd.pipeline import run_analysis

EVENTS, TRUTH = make_season(seed=0)


def _iso(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="microseconds")


class FakeClient:
    """Mimics OpenF1: 3.7 Hz integer speeds, approximate lap date_start, Q3 only for the top n."""

    def __init__(self, ev, n_q3=16, start_error=0.15, seed=1):
        rng = np.random.default_rng(seed)
        self.ev, self.n_q3 = ev, n_q3
        self.t0 = 1.7e9 + rng.uniform(0, 1e6)
        self.err = rng.normal(0, start_error, len(ev.drivers))
        self.rng = rng
        self.order = np.argsort(ev.lap_times)

    def get(self, endpoint, params=None, where=()):
        ev = self.ev
        if endpoint == "drivers":
            return [{"driver_number": i + 1, "name_acronym": d, "team_name": t}
                    for i, (d, t) in enumerate(zip(ev.drivers, ev.teams))]
        if endpoint == "laps":    # date_start is only approximate in the real API
            return [{"driver_number": i + 1, "lap_number": 12, "is_pit_out_lap": False,
                     "lap_duration": float(ev.lap_times[i]), "date_start": _iso(self.t0 + self.err[i])}
                    for i in range(len(ev.drivers))]
        if endpoint == "session_result":
            rows = []
            for rank, i in enumerate(self.order):
                q3 = float(ev.lap_times[i]) if rank < self.n_q3 else None
                rows.append({"driver_number": int(i) + 1, "position": rank + 1,
                             "duration": [float(ev.lap_times[i]) + 1.0, float(ev.lap_times[i]) + 0.5, q3]})
            return rows
        if endpoint == "car_data":
            i = params["driver_number"] - 1
            lo = [datetime.fromisoformat(v).timestamp() for k, op, v in where if op == ">="][0]
            hi = [datetime.fromisoformat(v).timestamp() for k, op, v in where if op == "<="][0]
            ts = np.arange(lo, hi, 0.27) + self.rng.uniform(-0.03, 0.03, len(np.arange(lo, hi, 0.27)))
            sp = np.round(np.interp(ts - self.t0, ev.time[i], ev.speed[i]))
            return [{"date": _iso(t), "speed": int(s), "driver_number": i + 1} for t, s in zip(ts, sp)]
        raise KeyError(endpoint)


SESSION = {"session_key": 1, "year": 2024, "circuit_short_name": "Testville"}


def test_url_building_encodes_plus_and_keeps_operators():
    u = Client.url("car_data", {"session_key": 9158, "driver_number": 1},
                   [("date", ">=", "2024-09-01T13:00:00.000+00:00")])
    assert u.endswith("date>=2024-09-01T13:00:00.000%2B00:00") and "session_key=9158" in u


def test_q3_selection_uses_exact_official_time_and_top_ten_only():
    fc = FakeClient(EVENTS[0], n_q3=10)
    chosen = _target_laps(fc.get("laps"), fc.get("session_result"), part=3)
    assert len(chosen) == 10
    assert set(chosen) == {int(i) + 1 for i in fc.order[:10]}


def test_lap_profiles_match_ground_truth_despite_bad_date_start():
    ev = EVENTS[0]
    out = fetch_event(FakeClient(ev), SESSION, verbose=False)
    assert out is not None and len(out.drivers) == 16
    idx = {d: i for i, d in enumerate(ev.drivers)}
    # official lap time is preserved exactly
    assert max(abs(out.lap_times[k] - ev.lap_times[idx[d]]) for k, d in enumerate(out.drivers)) < 1e-9
    # time elapsed at each point of the lap is close to the truth (3.7 Hz data, 0.15 s start error)
    errs = [np.abs(out.time[k][: len(ev.distance)] - ev.time[idx[d]][: len(out.distance)]).max()
            for k, d in enumerate(out.drivers)]
    assert np.median(errs) < 0.25, errs


def test_full_pipeline_on_openf1_style_data_still_separates_car_and_driver(tmp_path):
    events = [fetch_event(FakeClient(ev, seed=10 + k), dict(SESSION, year=ev.year), verbose=False)
              for k, ev in enumerate(EVENTS)]
    for o, e in zip(events, EVENTS):
        o.event = e.event                                   # keep unique event names
    fit = run_analysis(events, tmp_path, verbose=False, team_scope="all")["fit"]
    assert fit["team_total"].corr(TRUTH["team_total"]) > 0.85
    assert fit["driver_total"].corr(TRUTH["driver_total"]) > 0.6


if __name__ == "__main__":
    import tempfile
    for fn in (test_url_building_encodes_plus_and_keeps_operators,
               test_q3_selection_uses_exact_official_time_and_top_ten_only,
               test_lap_profiles_match_ground_truth_despite_bad_date_start,
               test_full_pipeline_on_openf1_style_data_still_separates_car_and_driver):
        fn(pathlib.Path(tempfile.mkdtemp())) if fn.__code__.co_argcount else fn()
        print("PASS", fn.__name__)
