"""
Run the anomaly detector over every case in a dataset folder and report
how well it locates the true fault-injection time.

Usage:
    python run_anomaly_detection.py --dataset-root "data/RE1/RE1-OB" --limit 20
    python run_anomaly_detection.py --dataset-root "data/RE1/RE1-OB"   # all cases

This is a validation/tuning script, not the final pipeline. Use it to
pick window / theta / min_columns / gap_seconds before wiring the
detector into the retrieval layer.
"""
from __future__ import annotations

import argparse

import pandas as pd

from data_loader import discover_cases
from anamoly_detection import detect_anomalous_points, form_incidents, evaluate_against_ground_truth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True,
                         help="Path to a dataset folder, e.g. data/RE1-OB")
    parser.add_argument("--window", type=int, default=60, help="Rolling baseline window (samples)")
    parser.add_argument("--theta", type=float, default=3.0, help="Z-score threshold")
    parser.add_argument("--min-columns", type=int, default=5,
                         help="Metrics that must co-fire before a row counts as anomalous")
    parser.add_argument("--gap-seconds", type=int, default=60, help="Merge gap for incident formation")
    parser.add_argument("--tolerance-seconds", type=int, default=120,
                         help="How close to inject_time counts as 'detected'")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N cases")
    args = parser.parse_args()

    cases = discover_cases(args.dataset_root)
    if not cases:
        print(f"No cases found under {args.dataset_root}. "
              f"Check the path -- it should point at a folder like 'RE1-OB', "
              f"and contain subfolders like 'frontend_cpu/1/data.csv'.")
        return

    if args.limit:
        cases = cases[: args.limit]

    print(f"Found {len(cases)} cases. Running detector "
          f"(window={args.window}, theta={args.theta}, min_columns={args.min_columns}, "
          f"gap={args.gap_seconds}s)...\n")

    rows = []
    for case in cases:
        if case.inject_time is None:
            print(f"[skip] {case.service_fault_dir}/{case.case_id}: no inject_time.txt found")
            continue
        try:
            df = case.load_metrics()
        except Exception as e:
            print(f"[error] {case.service_fault_dir}/{case.case_id}: {e}")
            continue

        z = detect_anomalous_points(
            df, window=args.window, theta=args.theta, min_columns=args.min_columns
        )
        incidents = form_incidents(z, gap_seconds=args.gap_seconds)
        result = evaluate_against_ground_truth(
            incidents, case.inject_time, tolerance_seconds=args.tolerance_seconds
        )

        normal_fp_rate = z[z["time"] < case.inject_time]["is_anomaly"].mean()

        rows.append({
            "service_fault": case.service_fault_dir,
            "case_id": case.case_id,
            "true_service": case.root_cause_service,
            "fault_type": case.fault_type,
            "n_incidents": result["n_incidents"],
            "detected": result["detected"],
            "delay_seconds": result["delay_seconds"],
            "normal_fp_rate": round(normal_fp_rate, 3),
        })

    report = pd.DataFrame(rows)
    if report.empty:
        print("No cases were successfully processed.")
        return

    print(report.to_string(index=False))
    print()
    print(f"Detection rate: {report['detected'].mean():.1%} "
          f"({report['detected'].sum()}/{len(report)} cases)")
    print(f"Mean normal-period false-positive rate: {report['normal_fp_rate'].mean():.1%}")
    print(f"Median delay when detected: "
          f"{report.loc[report['detected'], 'delay_seconds'].median()}s")

    out_path = "anamoly_detection_report.csv"
    report.to_csv(out_path, index=False)
    print(f"\nFull per-case report written to {out_path}")


if __name__ == "__main__":
    main()