"""
Knowledge Base builder -- Section 3.1 "Knowledge Base".

Turns the HISTORY cases into a collection of labeled, searchable entries.
Each entry = one incident summary (the searchable content) + its verified
root-cause label (service + fault type, known from the folder name).

This collection is the SHARED foundation for both retrieval modes:
  - the vector index (FAISS) is built from these entries
  - the hierarchical tree is built from these SAME entries
Both indexes MUST come from this one list, or the vector-vs-vectorless
comparison is unfair (any accuracy gap could come from different data
instead of the retrieval method).

CRITICAL: only ever pass the HISTORY split here. If a test case became a KB
entry, retrieval could return the answer to the question being graded.

One entry per case: a case may yield several detected incidents, so we keep
the one that actually corresponds to the injected fault (the incident nearest
the known inject_time). History is allowed to use inject_time because its
labels are known; the test side is not.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Optional

from data_loader import Case
from anamoly_detection import Incident, detect_anomalous_points, form_incidents
from incident_summary import build_incident_summary
from data_split import case_key


@dataclass
class KBEntry:
    entry_id: str              # 'adservice_cpu/1'
    summary: str               # incident summary text -- what retrieval searches
    root_cause_service: str    # verified label
    fault_type: str            # verified label
    t_start: int
    t_end: int


def _pick_representative(incidents: list[Incident],
                         inject_time: Optional[int]) -> Incident:
    """The incident that actually matches the injected fault: the one whose
    window contains inject_time, else the one whose start is nearest to it.
    Falls back to the longest incident if inject_time is unknown."""
    if inject_time is None:
        return max(incidents, key=lambda inc: inc.t_end - inc.t_start)
    for inc in incidents:
        if inc.t_start <= inject_time <= inc.t_end:
            return inc
    return min(incidents, key=lambda inc: abs(inc.t_start - inject_time))


def build_knowledge_base(
    history_cases: list[Case],
    buffer_seconds: int = 60,
    verbose: bool = True,
) -> list[KBEntry]:
    """
    Run each history case through detect -> form incidents -> summarise, and
    tag the result with the case's verified label.
    """
    entries: list[KBEntry] = []
    skipped = 0
    for i, case in enumerate(history_cases):
        try:
            df = case.load_metrics()
        except Exception as e:
            if verbose:
                print(f"[skip] {case_key(case)}: {e}")
            skipped += 1
            continue

        incidents = form_incidents(detect_anomalous_points(df))
        if not incidents:
            skipped += 1
            continue

        inc = _pick_representative(incidents, case.inject_time)
        summary = build_incident_summary(inc, df, buffer_seconds=buffer_seconds)
        entries.append(KBEntry(
            entry_id=case_key(case),
            summary=summary,
            root_cause_service=case.root_cause_service,
            fault_type=case.fault_type,
            t_start=inc.t_start,
            t_end=inc.t_end,
        ))
        if verbose and (i + 1) % 25 == 0:
            print(f"  ...built {len(entries)} entries ({i+1}/{len(history_cases)} cases)")

    if verbose:
        print(f"Knowledge base: {len(entries)} entries, {skipped} case(s) skipped.")
    return entries


def save_kb(entries: list[KBEntry], path: str = "knowledge_base.json") -> None:
    with open(path, "w") as f:
        json.dump([asdict(e) for e in entries], f, indent=2)


def load_kb(path: str = "knowledge_base.json") -> list[KBEntry]:
    with open(path) as f:
        return [KBEntry(**d) for d in json.load(f)]


if __name__ == "__main__":
    import sys
    from data_loader import discover_cases
    from data_split import get_or_create_split

    root = sys.argv[1] if len(sys.argv) > 1 else "data/RE1-OB"
    cases = discover_cases(root)
    if not cases:
        print(f"No cases under {root}")
        sys.exit(1)

    history, test = get_or_create_split(cases, path="split.json")
    print(f"{len(cases)} cases -> {len(history)} history, {len(test)} test")
    print("Building knowledge base from HISTORY only...\n")

    entries = build_knowledge_base(history)
    save_kb(entries, "knowledge_base.json")
    print(f"\nSaved {len(entries)} entries to knowledge_base.json")
    if entries:
        print("\n--- example entry ---")
        e = entries[0]
        print(f"id={e.entry_id}  label=({e.root_cause_service}, {e.fault_type})")
        print(e.summary)