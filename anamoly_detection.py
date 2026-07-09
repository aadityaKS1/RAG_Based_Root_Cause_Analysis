"""
Anomaly Detection module -- Section 3.2.1 of the proposal.

Implements:
  Eq 3-1: rolling mean / std over a window W
  Eq 3-2: rolling z-score
  Eq 3-3: anomaly flag when |z(t)| > theta
  Eq 3-4: merge nearby flagged points into one incident, gap G
  Eq 3-5: retrieval window = incident interval extended by buffer b

This module only detects and localizes. It does not diagnose.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Incident:
    t_start: int
    t_end: int
    affected_services: list[str]
    affected_columns: list[str]

    def retrieval_window(self, buffer_seconds: int) -> tuple[int, int]:
        """Eq 3-5: window = [t_start - b, t_end + b]."""
        return (self.t_start - buffer_seconds, self.t_end + buffer_seconds)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Eq 3-1 and 3-2. Uses the trailing `window` samples, excluding
    the current point, so the baseline isn't contaminated by the point
    being scored."""
    mu = series.shift(1).rolling(window=window, min_periods=max(3, window // 2)).mean()
    sigma = series.shift(1).rolling(window=window, min_periods=max(3, window // 2)).std()
    sigma_safe = sigma.replace(0, np.nan)
    z = (series - mu) / sigma_safe
    return z.fillna(0)


def detect_anomalous_points(
    df: pd.DataFrame,
    time_col: str = "time",
    window: int = 60,
    theta: float = 3.0,
    exclude_cols: tuple[str, ...] = (),
    min_columns: int = 3,
) -> pd.DataFrame:
    """
    Eq 3-1 to 3-3. Runs rolling z-score over every metric column.

    IMPORTANT: with ~50-70 metric columns per case, requiring only ONE
    column to exceed theta triggers on noise almost every row (each
    column has its own ~2-6% false positive rate in real telemetry,
    which compounds across many columns: P(at least one fires) climbs
    fast). `min_columns` requires several metrics to exceed theta AT
    THE SAME TIMESTAMP before the row counts as anomalous, which is a
    much better real-world corroboration rule than "any one column".

    Returns a DataFrame indexed the same as df, with:
      - one z-score column per metric (suffix "_z")
      - "n_triggering": how many columns exceeded theta at that row
      - "is_anomaly": bool, True if n_triggering >= min_columns
      - "anomaly_columns": list of metric names that triggered
    """
    metric_cols = [c for c in df.columns if c != time_col and c not in exclude_cols]
    z_scores = pd.DataFrame(index=df.index)
    z_scores[time_col] = df[time_col]

    for col in metric_cols:
        z_scores[f"{col}_z"] = rolling_zscore(df[col], window=window)

    z_cols = [f"{c}_z" for c in metric_cols]
    exceed = z_scores[z_cols].abs() > theta

    z_scores["n_triggering"] = exceed.sum(axis=1)
    z_scores["is_anomaly"] = z_scores["n_triggering"] >= min_columns
    z_scores["anomaly_columns"] = exceed.apply(
        lambda row: [metric_cols[i] for i, v in enumerate(row) if v], axis=1
    )
    return z_scores


def form_incidents(
    z_scores: pd.DataFrame,
    time_col: str = "time",
    gap_seconds: int = 60,
) -> list[Incident]:
    """
    Eq 3-4. Merge flagged points closer than `gap_seconds` into a single
    incident. affected_services is derived from the service prefix of
    each triggering column (e.g. "frontend_cpu_z" -> "frontend").
    """
    flagged = z_scores[z_scores["is_anomaly"]].copy()
    if flagged.empty:
        return []

    flagged = flagged.sort_values(time_col).reset_index(drop=True)
    incidents: list[Incident] = []

    cur_start = flagged.loc[0, time_col]
    cur_end = flagged.loc[0, time_col]
    cur_cols: set[str] = set(flagged.loc[0, "anomaly_columns"])

    for i in range(1, len(flagged)):
        t = flagged.loc[i, time_col]
        if t - cur_end < gap_seconds:
            cur_end = t
            cur_cols.update(flagged.loc[i, "anomaly_columns"])
        else:
            incidents.append(_build_incident(cur_start, cur_end, cur_cols))
            cur_start = t
            cur_end = t
            cur_cols = set(flagged.loc[i, "anomaly_columns"])

    incidents.append(_build_incident(cur_start, cur_end, cur_cols))
    return incidents


def _build_incident(t_start: int, t_end: int, cols: set[str]) -> Incident:
    services = sorted({_service_from_column(c) for c in cols})
    return Incident(
        t_start=int(t_start),
        t_end=int(t_end),
        affected_services=services,
        affected_columns=sorted(cols),
    )


def _service_from_column(col: str) -> str:
    """'frontend_cpu' -> 'frontend'. Strips the trailing metric suffix."""
    known_suffixes = ("_cpu", "_mem", "_workload", "_error", "_latency")
    for suf in known_suffixes:
        if col.endswith(suf):
            return col[: -len(suf)]
    return col.rsplit("_", 1)[0]


def evaluate_against_ground_truth(
    incidents: list[Incident],
    inject_time: int,
    tolerance_seconds: int = 120,
) -> dict:
    """
    Sanity-check: did we detect an incident near the real fault
    injection time? Not a proposal metric, just a debugging aid while
    the detector is being tuned.
    """
    if not incidents:
        return {"detected": False, "delay_seconds": None, "n_incidents": 0}

    best = min(incidents, key=lambda inc: abs(inc.t_start - inject_time))
    delay = best.t_start - inject_time
    detected = abs(delay) <= tolerance_seconds or (best.t_start <= inject_time <= best.t_end)
    return {
        "detected": detected,
        "delay_seconds": delay,
        "n_incidents": len(incidents),
        "matched_incident": best,
    }