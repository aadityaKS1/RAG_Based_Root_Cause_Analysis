"""
Loader for RCAEval-style dataset folders (RE1 / RE2 / RE3).

Real on-disk layout, confirmed against the RCAEval GitHub repo (main.py):

    <dataset_root>/<service>_<fault>/<case_id>/data.csv
    <dataset_root>/<service>_<fault>/<case_id>/inject_time.txt
    <dataset_root>/<service>_<fault>/<case_id>/logs.csv      (RE2, RE3 only)
    <dataset_root>/<service>_<fault>/<case_id>/traces.csv    (RE2, RE3 only)

The ground-truth root-cause service and fault type are NOT in a separate
label file -- they are encoded in the folder name itself:
    "frontend_cpu"   -> service="frontend", fault="cpu"
    "cartservice_mem" -> service="cartservice", fault="mem"

RE1 = metrics only. RE2/RE3 = metrics + logs + traces.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Case:
    """One labeled failure case."""
    dataset_root: str
    service_fault_dir: str      # e.g. "frontend_cpu"
    case_id: str                # e.g. "1"
    root_cause_service: str     # parsed from folder name
    fault_type: str             # parsed from folder name
    case_dir: str                # full path to the case folder
    inject_time: Optional[int] = None  # unix timestamp, ground truth

    metrics_path: Optional[str] = None
    logs_path: Optional[str] = None
    traces_path: Optional[str] = None

    def load_metrics(self) -> pd.DataFrame:
        if not self.metrics_path:
            raise FileNotFoundError(f"No data.csv found in {self.case_dir}")
        df = pd.read_csv(self.metrics_path)
        df = df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        # RCAEval convention: drop latency-50, rename latency-90 -> latency
        df = df.loc[:, ~df.columns.str.endswith("_latency-50")]
        df = df.rename(columns={
            c: c.replace("_latency-90", "_latency")
            for c in df.columns if c.endswith("_latency-90")
        })
        return df

    def load_logs(self) -> Optional[pd.DataFrame]:
        if not self.logs_path:
            return None
        return pd.read_csv(self.logs_path)

    def load_traces(self) -> Optional[pd.DataFrame]:
        if not self.traces_path:
            return None
        return pd.read_csv(self.traces_path)


def _parse_service_fault(dirname: str) -> tuple[str, str]:
    """
    Folder names are "<service>_<fault>". Service names can themselves
    contain underscores (rare), so we split on the LAST underscore-separated
    token against the known fault vocabulary; fall back to a plain rsplit.
    """
    known_faults = {"cpu", "mem", "disk", "delay", "loss", "socket",
                     "f1", "f2", "f3", "f4", "f5"}
    parts = dirname.split("_")
    if len(parts) >= 2 and parts[-1] in known_faults:
        fault = parts[-1]
        service = "_".join(parts[:-1])
        return service, fault
    # fallback: last token as fault type regardless
    service, _, fault = dirname.rpartition("_")
    return service or dirname, fault or "unknown"


def discover_cases(dataset_root: str) -> list[Case]:
    """
    Walk a dataset root (e.g. path to RE1-OB) and return one Case per
    case folder that contains a data.csv.
    """
    cases: list[Case] = []
    pattern = os.path.join(dataset_root, "**", "data.csv")
    for data_path in sorted(glob.glob(pattern, recursive=True)):
        case_dir = os.path.dirname(data_path)
        service_fault_dir = os.path.basename(os.path.dirname(case_dir))
        case_id = os.path.basename(case_dir)
        service, fault = _parse_service_fault(service_fault_dir)

        inject_time = None
        inject_path = os.path.join(case_dir, "inject_time.txt")
        if os.path.exists(inject_path):
            with open(inject_path) as f:
                inject_time = int(f.readlines()[0].strip())

        logs_path = os.path.join(case_dir, "logs.csv")
        traces_path = os.path.join(case_dir, "traces.csv")

        cases.append(Case(
            dataset_root=dataset_root,
            service_fault_dir=service_fault_dir,
            case_id=case_id,
            root_cause_service=service,
            fault_type=fault,
            case_dir=case_dir,
            inject_time=inject_time,
            metrics_path=data_path,
            logs_path=logs_path if os.path.exists(logs_path) else None,
            traces_path=traces_path if os.path.exists(traces_path) else None,
        ))
    return cases


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/RE1-OB"
    cases = discover_cases(root)
    print(f"Found {len(cases)} cases under {root}")
    for c in cases[:5]:
        print(f"  service={c.root_cause_service:20s} fault={c.fault_type:8s} "
              f"case={c.case_id:4s} inject_time={c.inject_time} "
              f"logs={'yes' if c.logs_path else 'no'} traces={'yes' if c.traces_path else 'no'}")