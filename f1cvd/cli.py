from __future__ import annotations
import argparse
from .data import load_all


def main(argv=None):
    p = argparse.ArgumentParser(prog="f1cvd", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download qualifying telemetry via FastF1")
    f.add_argument("--year", type=int, nargs="+", required=True)
    f.add_argument("--source", choices=["openf1", "fastf1"], default="openf1",
                   help="openf1 (default, 2023+) or fastf1 (broken while F1 blocks its archive)")
    f.add_argument("--events", nargs="*", help="e.g. Monza Spa (matches circuit, city or country; default: all)")
    f.add_argument("--out", default="data/processed")
    f.add_argument("--cache", default="data/cache")

    a = sub.add_parser("analyse", help="decompose, fit models, predict, plot")
    a.add_argument("--data", default="data/processed")
    a.add_argument("--out", default="results")
    a.add_argument("--team-scope", choices=["season", "all"], default="season",
                   help="season: a team's car is re-estimated every year (recommended); all: one car for all years")

    a.add_argument("--team-mode", choices=["fixed", "ridge"], default="fixed",
                   help="fixed: unshrunk team levels, driver = gap to teammates (recommended)")

    d = sub.add_parser("demo", help="run on synthetic data with planted effects")
    d.add_argument("--out", default="results/demo")

    args = p.parse_args(argv)
    if args.cmd == "fetch":
        for y in args.year:
            print(f"Season {y}")
            if args.source == "openf1":
                from .openf1 import fetch_season
                fetch_season(y, args.out, args.events)
            else:
                from .fetch import fetch_season
                fetch_season(y, args.out, args.cache, args.events)
    elif args.cmd == "analyse":
        from .pipeline import run_analysis
        events = load_all(args.data)
        if len(events) < 6:
            raise SystemExit(f"Only {len(events)} events in {args.data}; fetch more first (6+ recommended).")
        run_analysis(events, args.out, team_scope=args.team_scope, team_mode=args.team_mode)
    else:
        from .pipeline import run_analysis
        from .synthetic import make_season
        events, truth = make_season()
        res = run_analysis(events, args.out, team_scope="all")   # planted team effects are constant
        corr = res["fit"]["team_total"].corr(truth["team_total"])
        corr_d = res["fit"]["driver_total"].corr(truth["driver_total"])
        print(f"Recovery of planted effects: car r={corr:.2f}, driver r={corr_d:.2f}")


if __name__ == "__main__":
    main()
