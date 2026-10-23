from __future__ import annotations
from pathlib import Path
import pandas as pd
from .decompose import decompose_event, category_table, lap_delta
from .model import fit_all, teammate_split
from .predict import loeo_predict
from . import plots


def run_analysis(events, out_dir="results", verbose=True, team_scope="season", team_mode="fixed") -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    longs, segs_by_event = [], {}
    for ev in events:
        long, segs = decompose_event(ev)
        longs.append(long); segs_by_event[ev.key] = segs
    long = pd.concat(longs, ignore_index=True)
    cat = category_table(long)
    if team_scope == "season":
        # A team's car changes every year, so each team-season is its own "car".
        # Drivers keep one identity, and drivers moving between team-seasons are what
        # allows car and driver to be told apart.
        cat["team"] = cat["team"] + " " + cat["event"].str[:4]
    lap = lap_delta(cat)

    cat.to_csv(out / "category_deltas.csv", index=False)
    teammate_split(cat).to_csv(out / "teammate_split.csv", index=False)
    fit = fit_all(cat, lap, team_mode)
    fit["team_fp"].to_csv(out / "team_fingerprint.csv")
    fit["driver_fp"].to_csv(out / "driver_fingerprint.csv")
    n_team = lap.groupby("team")["event"].nunique(); n_drv = lap.groupby("driver")["event"].nunique()
    pd.DataFrame({"effect_s": fit["team_total"], "n_events": n_team}).sort_values("effect_s").to_csv(
        out / "team_total_effect.csv", index_label="team")
    pd.DataFrame({"effect_s": fit["driver_total"], "n_events": n_drv}).sort_values("effect_s").to_csv(
        out / "driver_total_effect.csv", index_label="driver")
    pred = loeo_predict(cat)
    pred.to_csv(out / "layout_prediction.csv", index=False)

    fastest = fit["team_total"].sort_values().index[:4]          # four fastest cars overall
    plots.radar(fit["team_fp"], teams=fastest, path=out / "team_radar.png")
    plots.driver_heatmap(fit["driver_fp"], path=out / "driver_heatmap.png")
    ev = events[-1]
    order = ev.lap_times.argsort()
    plots.delta_trace(ev, ev.drivers[order[0]], ev.drivers[order[1]], segs_by_event[ev.key],
                      path=out / "delta_trace.png")

    if verbose:
        s = fit["spread"]
        print(f"Events: {len(events)}  Drivers: {lap.driver.nunique()}  Teams: {lap.team.nunique()}")
        print(f"Spread of lap-time effect (std, seconds): car {s['car_sd_s']:.3f} | driver {s['driver_sd_s']:.3f}")
        if len(pred):
            print("Layout prediction (mean over held-out events):")
            print(pred.drop(columns=["event"]).mean().round(3).to_string())
            wins = int((pred["mae_fingerprint_s"] < pred["mae_baseline_s"]).sum())
            print(f"Fingerprint beats layout-blind baseline (MAE) in {wins} of {len(pred)} held-out events")
    return {"cat": cat, "lap": lap, "fit": fit, "pred": pred, "segments": segs_by_event}
