"""
Incident Summary Builder -- the "Telemetry Retrieval" block of Section 3.2.

Turns an Incident (from anamoly_detection.form_incidents) into a compact,
structured TEXT summary of what happened, using the case's own telemetry.

That summary has two jobs downstream:
  1. It is the QUERY the retrieval layer searches with.
  2. It is the FIRST block of evidence handed to the LLM.

The SAME builder is used two ways: on the current/test incident (-> retrieval
query) and on each history incident (-> knowledge-base entry). So it is
label-free -- it only describes telemetry. The root-cause label is attached
later, during KB prep.

RANKING: metrics are ranked by significance = |peak - baseline| / wiggle,
where 'wiggle' is the baseline standard deviation BUT floored at a few percent
of the baseline value. The floor matters: without it, smooth slow-drifting
metrics (e.g. memory in bytes) get a near-zero std, so even a trivial 1% move
looks like a huge z-score and drowns out the real signal. The floor makes a
metric prove it moved by a meaningful fraction of itself, so genuine spikes
(a CPU pinned at 100%, a latency 10x) rank above byte-scale noise.

Currently metrics-only (RE1). Logs/traces (RE2, RE3) are marked TODO.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from anamoly_detection import Incident


@dataclass
class MetricChange:
    column: str
    baseline: float
    peak_value: float
    peak_time: int
    direction: str          # "rose" or "fell"
    ratio: float | None     # peak / baseline, or None if baseline ~ 0
    significance: float     # |peak - baseline| / floored-wiggle

    def as_line(self) -> str:
        base = f"{self.baseline:.4g}"
        peak = f"{self.peak_value:.4g}"
        if self.ratio is not None and np.isfinite(self.ratio):
            mag = f"{self.ratio:.1f}x"
        else:
            mag = f"0 -> {self.peak_value:.4g}"  # baseline was ~0
        return (f"  {self.column:<26} baseline~={base:<11} -> "
                f"{self.direction} to {peak} at t={self.peak_time} ({mag})")


def _summarise_metric(
    column: str,
    times: np.ndarray,
    values: np.ndarray,
    incident_start: int,
    incident_end: int,
    noise_floor_frac: float,
) -> MetricChange | None:
    during_mask = (times >= incident_start) & (times <= incident_end)
    if not during_mask.any():
        return None

    baseline_mask = times < incident_start
    if baseline_mask.sum() >= 3:
        baseline = float(np.median(values[baseline_mask]))
        baseline_std = float(np.std(values[baseline_mask]))
    else:
        baseline = float(np.median(values))
        baseline_std = float(np.std(values))

    during_vals = values[during_mask]
    during_times = times[during_mask]
    deviations = during_vals - baseline
    peak_idx = int(np.argmax(np.abs(deviations)))
    peak_value = float(during_vals[peak_idx])
    peak_time = int(during_times[peak_idx])
    deviation = abs(peak_value - baseline)
    if deviation == 0:
        return None

    direction = "rose" if peak_value > baseline else "fell"
    ratio = (peak_value / baseline) if abs(baseline) > 1e-9 else None

    # significance with a NOISE FLOOR: the wiggle can't be smaller than
    # noise_floor_frac of the baseline. Stops smooth big-scale metrics from
    # faking a huge z on a trivial percentage move.
    wiggle = max(baseline_std, noise_floor_frac * abs(baseline), 1e-9)
    significance = min(deviation / wiggle, 50.0)  # cap: avoids blow-up when baseline~=0

    return MetricChange(column, baseline, peak_value, peak_time,
                        direction, ratio, significance)


def build_incident_summary(
    incident: Incident,
    df: pd.DataFrame,
    time_col: str = "time",
    buffer_seconds: int = 60,
    max_metrics: int = 12,
    min_significance: float = 3.0,
    min_shown: int = 5,
    noise_floor_frac: float = 0.05,
) -> str:
    """
    Build the structured text summary for one incident.

    Args:
        incident: the Incident to describe.
        df: the SAME case's telemetry the incident was detected in.
        buffer_seconds: buffer b for the retrieval window (Eq. 3-5).
        max_metrics: cap on metrics listed, most significant first.
        min_significance: hide metrics below this significance (drops noise).
        min_shown: always show at least this many, even if the filter is strict.
        noise_floor_frac: a metric's wiggle is floored at this fraction of its
            baseline (0.05 = 5%). Higher -> stricter about relative movement.
    """
    start, end = incident.retrieval_window(buffer_seconds)
    window_df = df[(df[time_col] >= start) & (df[time_col] <= end)].sort_values(time_col)
    times = window_df[time_col].to_numpy()

    candidate_cols = list(incident.affected_columns) or \
                     [c for c in df.columns if c != time_col]

    changes: list[MetricChange] = []
    for col in candidate_cols:
        if col not in window_df.columns:
            continue
        values = window_df[col].to_numpy(dtype=float)
        change = _summarise_metric(col, times, values,
                                   incident.t_start, incident.t_end, noise_floor_frac)
        if change is not None:
            changes.append(change)

    changes.sort(key=lambda c: (c.significance,
                                abs(c.ratio - 1) if (c.ratio and np.isfinite(c.ratio)) else 0.0),
                 reverse=True)
    significant = [c for c in changes if c.significance >= min_significance]
    if len(significant) < min_shown:
        significant = changes[:min_shown]
    shown = significant[:max_metrics]
    hidden = len(changes) - len(shown)

    # ---- format for RETRIEVAL: lead with the discriminative signal, and
    # drop the boilerplate (headers, timestamps, all-13-services list) that
    # made every summary look ~98% identical to the embedder. ----
    if not shown:
        return "No significant anomaly detected in the incident window."

    def _mag(c: MetricChange) -> str:
        if c.ratio is not None and np.isfinite(c.ratio):
            return f"{c.ratio:.1f}x"
        return f"0 to {c.peak_value:.3g}"

    def _svc(col: str) -> str:
        for suf in ("_cpu", "_mem", "_load", "_latency", "_error",
                    "_workload", "_disk"):
            if col.endswith(suf):
                return col[:-len(suf)]
        return col.rsplit("_", 1)[0]

    top = shown[0]
    lines = [
        f"Primary anomaly: {top.column} {top.direction} {_mag(top)} "
        f"(baseline {top.baseline:.3g} to peak {top.peak_value:.3g})."
    ]
    if len(shown) > 1:
        others = "; ".join(f"{c.column} {c.direction} {_mag(c)}" for c in shown[1:6])
        lines.append(f"Other anomalies: {others}.")

    # top services BY SIGNIFICANCE (not all affected services) -- this line
    # now actually differs between incidents, so it helps discrimination.
    top_services: list[str] = []
    for c in shown:
        s = _svc(c.column)
        if s not in top_services:
            top_services.append(s)
    lines.append(f"Services involved: {', '.join(top_services[:5])}.")

    # TODO(RE2/RE3): logs section (error keywords, chronological)
    # TODO(RE2/RE3): traces section (service-op-duration-status tuples)
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        from data_loader import discover_cases
        from anamoly_detection import detect_anomalous_points, form_incidents
        cases = discover_cases(sys.argv[1])
        if not cases:
            print(f"No cases under {sys.argv[1]}"); sys.exit(1)
        case = cases[0]
        df = case.load_metrics()
        incidents = form_incidents(detect_anomalous_points(df))
        print(f"Case {case.service_fault_dir}/{case.case_id}: {len(incidents)} incident(s)\n")
        for inc in incidents:
            print(build_incident_summary(inc, df)); print()
    else:
        print("Run:  python incident_summary.py <dataset_root>")