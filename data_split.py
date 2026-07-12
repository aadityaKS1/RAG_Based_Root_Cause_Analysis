"""
History / Test split -- the gate that protects every evaluation number.

Splits the labeled cases into two groups:
  - HISTORY: goes into the knowledge base (searchable during retrieval)
  - TEST:    held out, used ONLY to score diagnoses at the very end

Why this matters: the system diagnoses a test incident by retrieving similar
past incidents from the history set. If a test case leaked into history, the
system could 'retrieve' the answer to the exact question it is being graded
on, and every accuracy number would be meaningless. So the split is created
ONCE and reused everywhere.

Two guarantees:
  1. STRATIFIED -- both sides contain every fault scenario in (roughly) the
     same proportion, so we never test on a scenario the history has never
     seen.
  2. DETERMINISTIC + SAVED -- a fixed seed plus an on-disk JSON file mean the
     split is identical on every run and every team member's machine.

The split is at the CASE level, not the incident level: all incidents from
one case go to the same side, otherwise incidents from the same case would
leak across the boundary.
"""
from __future__ import annotations

import json
import os

import numpy as np

from data_loader import Case


def case_key(case: Case) -> str:
    """Stable unique id for a case, e.g. 'adservice_cpu/1'. Used on disk."""
    return f"{case.service_fault_dir}/{case.case_id}"


def _stratify_value(case: Case, stratify_by: str) -> str:
    if stratify_by == "fault_type":
        return case.fault_type
    if stratify_by == "service_fault":
        return case.service_fault_dir          # e.g. 'adservice_cpu'
    raise ValueError(f"unknown stratify_by: {stratify_by}")


def make_split(
    cases: list[Case],
    test_frac: float = 0.2,
    seed: int = 42,
    stratify_by: str = "service_fault",
) -> tuple[list[Case], list[Case]]:
    """
    Split cases into (history, test), stratified and deterministic.

    Within each stratification group the cases are shuffled with a seeded RNG
    and split by test_frac. Groups with a single case go entirely to history
    (so a test scenario always has at least one history example to retrieve).
    """
    rng = np.random.default_rng(seed)

    # group cases by the stratification key, in a deterministic order
    groups: dict[str, list[Case]] = {}
    for case in cases:
        groups.setdefault(_stratify_value(case, stratify_by), []).append(case)

    history: list[Case] = []
    test: list[Case] = []

    for key in sorted(groups):                 # sorted -> deterministic
        group = sorted(groups[key], key=case_key)
        idx = rng.permutation(len(group))      # seeded shuffle
        shuffled = [group[i] for i in idx]

        n = len(shuffled)
        if n == 1:
            history.extend(shuffled)           # keep lone scenarios searchable
            continue
        n_test = round(n * test_frac)
        n_test = max(1, min(n_test, n - 1))    # both sides non-empty
        test.extend(shuffled[:n_test])
        history.extend(shuffled[n_test:])

    # safety: the two sides must never overlap
    hk, tk = {case_key(c) for c in history}, {case_key(c) for c in test}
    assert hk.isdisjoint(tk), "LEAKAGE: a case is in both history and test"
    return history, test


def save_split(history: list[Case], test: list[Case], path: str,
               seed: int, test_frac: float, stratify_by: str) -> None:
    payload = {
        "seed": seed,
        "test_frac": test_frac,
        "stratify_by": stratify_by,
        "history": sorted(case_key(c) for c in history),
        "test": sorted(case_key(c) for c in test),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_split(cases: list[Case], path: str) -> tuple[list[Case], list[Case]]:
    """Reload a saved split and map the ids back onto the given Case objects."""
    with open(path) as f:
        payload = json.load(f)
    by_key = {case_key(c): c for c in cases}
    history = [by_key[k] for k in payload["history"] if k in by_key]
    test = [by_key[k] for k in payload["test"] if k in by_key]
    return history, test


def get_or_create_split(
    cases: list[Case],
    path: str = "split.json",
    test_frac: float = 0.2,
    seed: int = 42,
    stratify_by: str = "service_fault",
) -> tuple[list[Case], list[Case]]:
    """
    Main entry point. Loads the split from `path` if it exists (so it never
    changes), otherwise creates it, saves it, and returns it.
    """
    if os.path.exists(path):
        return load_split(cases, path)
    history, test = make_split(cases, test_frac, seed, stratify_by)
    save_split(history, test, path, seed, test_frac, stratify_by)
    return history, test


def summarize_split(history: list[Case], test: list[Case]) -> None:
    """Print counts per fault type so you can eyeball the balance."""
    def counts(cases: list[Case]) -> dict[str, int]:
        d: dict[str, int] = {}
        for c in cases:
            d[c.fault_type] = d.get(c.fault_type, 0) + 1
        return d

    hc, tc = counts(history), counts(test)
    faults = sorted(set(hc) | set(tc))
    print(f"{'fault_type':<12} {'history':>8} {'test':>6}")
    for f in faults:
        print(f"{f:<12} {hc.get(f,0):>8} {tc.get(f,0):>6}")
    print(f"{'TOTAL':<12} {len(history):>8} {len(test):>6}")


if __name__ == "__main__":
    import sys
    from data_loader import discover_cases

    root = sys.argv[1] if len(sys.argv) > 1 else "data/RE1-OB"
    cases = discover_cases(root)
    if not cases:
        print(f"No cases under {root}")
        sys.exit(1)

    history, test = get_or_create_split(cases, path="split.json")
    print(f"Loaded/created split for {len(cases)} cases -> "
          f"{len(history)} history, {len(test)} test\n")
    summarize_split(history, test)
    print("\nSplit saved to split.json (reused on every future run).")