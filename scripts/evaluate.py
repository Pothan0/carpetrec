"""Phases 2/4: run the evaluation harness (recall@k, mAP; config comparison).

    python -m scripts.evaluate                      # synthetic + real query sets
    python -m scripts.evaluate --set mobile          # mobile (robustness) set
    python -m scripts.evaluate --compare             # baseline vs gated preprocessing
    python -m scripts.evaluate --baseline-json data/eval/baseline.json   # dump metrics
    python -m scripts.evaluate --against data/eval/baseline.json         # print deltas vs baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from carpet_search.config import load_settings
from carpet_search.eval.run_eval import run_all, compare_configs


def _print_deltas(baseline: dict, current: dict) -> None:
    print("\n=== deltas vs baseline ===")
    for set_name, cur in current.items():
        base = baseline.get(set_name, {})
        print(f"[{set_name}]")
        for metric, val in cur.items():
            if isinstance(val, float):
                b = base.get(metric)
                d = (val - b) if isinstance(b, (int, float)) else float("nan")
                print(f"  {metric:12s} {val:.3f}  (Δ {d:+.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Carpet search evaluation")
    parser.add_argument("--compare", action="store_true",
                        help="compare baseline vs gated preprocessing configs")
    parser.add_argument("--set", choices=["synthetic", "real", "mobile", "both", "all"],
                        default="both", help="which query set(s) to evaluate")
    parser.add_argument("--baseline-json", type=str, default=None,
                        help="dump the metrics dict to this JSON path (regression baseline)")
    parser.add_argument("--against", type=str, default=None,
                        help="load a baseline JSON and print per-metric deltas")
    args = parser.parse_args()

    settings = load_settings()
    if args.compare:
        which = args.set if args.set in ("synthetic", "real", "mobile") else "mobile"
        compare_configs(settings, which=which)
        return

    results = run_all(settings, which=args.set)

    if args.baseline_json:
        Path(args.baseline_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.baseline_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nbaseline written -> {args.baseline_json}")

    if args.against:
        with open(args.against, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        _print_deltas(baseline, results)


if __name__ == "__main__":
    main()
