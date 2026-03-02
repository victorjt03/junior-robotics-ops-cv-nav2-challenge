#!/usr/bin/env python3
import argparse
import csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_csv", required=True)
    ap.add_argument("--metric", default="metrics/mAP_0.5:0.95",
                    help="Column to maximize (default: metrics/mAP_0.5:0.95)")
    args = ap.parse_args()

    with open(args.results_csv, "r", newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        rows = list(reader)

    if not rows:
        raise SystemExit("No rows found in results.csv")

    metric = args.metric
    if metric not in rows[0]:
        raise SystemExit(f"Metric '{metric}' not found. Available: {list(rows[0].keys())}")

    def to_float(x):
        try:
            return float(x)
        except Exception:
            return float("-inf")

    best = max(rows, key=lambda r: to_float(r.get(metric, "")))

    cols = ["epoch", "metrics/precision", "metrics/recall", "metrics/mAP_0.5", "metrics/mAP_0.5:0.95"]
    print(f"Best epoch by {metric}:")
    for c in cols:
        print(f"  {c}: {best.get(c)}")

if __name__ == "__main__":
    main()