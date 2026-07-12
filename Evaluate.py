"""
Evaluation harness -- Section 3.2.3 + Chapter 4. The capstone.

Runs every TEST case through the full pipeline under three configurations:
    - no retrieval   (ablation baseline: LLM sees only the current incident)
    - vector         (retrieval mode 1)
    - vectorless     (retrieval mode 2)

and scores the diagnoses against ground truth. Produces the head-to-head
vector-vs-vectorless table that is the whole point of the project.

Metrics:
    AC@k     -- true root-cause service is in the top-k predicted services
    Avg@5    -- mean of AC@1..AC@5
    Faithful -- (proxy) fraction of cited evidence items that exist AND
                support the predicted root cause. Automatable stand-in for
                "claims backed by cited evidence".
    dRAG     -- ablation delta: AC@1(mode) - AC@1(no retrieval)

Only the query's telemetry is used to build its summary; inject_time is NOT
used to pick the test incident (that would peek at ground truth), so the most
sustained detected incident is used.

The comparison is fair because all three configs share the same LLM, prompt,
scoring, and test set -- only retrieval changes.
"""
from __future__ import annotations

import csv

from data_loader import discover_cases, Case
from anamoly_detection import detect_anomalous_points, form_incidents
from incident_summary import build_incident_summary
from data_split import get_or_create_split
from knowledge_base import build_knowledge_base, _pick_representative
from vector_retrieval import VectorRetriever
from vectorless_retrieval import VectorlessRetriever
from Diagnosis import DiagnosisEngine

MODEL = "llama3.2:3b"      # <-- single place to change the LLM for the whole run


# ---------- metrics ----------

def ac_at_k(ranked: list[str], true: str, k: int) -> float:
    return 1.0 if true in ranked[:k] else 0.0


def avg_at_k(ranked: list[str], true: str, k: int) -> float:
    return sum(ac_at_k(ranked, true, j) for j in range(1, k + 1)) / k


def faithfulness(dx, retrieved) -> float | None:
    """Proxy: of the evidence items the diagnosis cited, what fraction exist
    and share the predicted root-cause service. None if nothing was cited."""
    if not retrieved or not dx.citations:
        return None
    ev = {f"E{i+1}": e for i, (e, _) in enumerate(retrieved)}
    valid = sum(1 for c in dx.citations
                if c in ev and ev[c].root_cause_service == dx.root_cause_service)
    return valid / len(dx.citations)


# ---------- build the test query ----------

def build_query(case: Case, buffer_seconds: int = 60) -> str | None:
    df = case.load_metrics()
    incidents = form_incidents(detect_anomalous_points(df))
    if not incidents:
        return None
    inc = _pick_representative(incidents, None)   # inject_time=None -> no peeking
    return build_incident_summary(inc, df, buffer_seconds=buffer_seconds)


# ---------- the harness ----------

def run_evaluation(
    dataset_root: str,
    model: str = MODEL,
    k: int = 5,
    buffer_seconds: int = 60,
    limit: int | None = None,
    # injectable components (None -> real MiniLM / Ollama); used for testing
    vector_embed_fn=None,
    vectorless_choose_fn=None,
    llm_fn=None,
    out_csv: str = "evaluation_results.csv",
):
    cases = discover_cases(dataset_root)
    history_cases, test_cases = get_or_create_split(cases, path="split.json")
    if limit:
        test_cases = test_cases[:limit]

    print(f"{len(cases)} cases -> {len(history_cases)} history, "
          f"{len(test_cases)} test (evaluating {len(test_cases)})")

    print("Building knowledge base + indexes...")
    kb = build_knowledge_base(history_cases, buffer_seconds=buffer_seconds, verbose=False)
    vector = VectorRetriever(embed_fn=vector_embed_fn).build(kb)
    vectorless = VectorlessRetriever(choose_fn=vectorless_choose_fn, model=model).build(kb)
    engine = DiagnosisEngine(llm_fn=llm_fn, model=model)

    configs = ["no_retrieval", "vector", "vectorless"]
    rows = []

    for n, case in enumerate(test_cases, 1):
        query = build_query(case, buffer_seconds)
        if query is None:
            continue
        true_service = case.root_cause_service
        row = {"case": f"{case.service_fault_dir}/{case.case_id}",
               "true_service": true_service, "true_fault": case.fault_type}

        for cfg in configs:
            if cfg == "no_retrieval":
                retrieved = []
            elif cfg == "vector":
                retrieved = vector.retrieve(query, k=k)
            else:
                retrieved = vectorless.retrieve(query, k=k)

            dx = engine.diagnose(query, retrieved)
            ranked = dx.ranked_services
            row[f"{cfg}_pred"] = ranked[0] if ranked else "unknown"
            row[f"{cfg}_ac1"] = ac_at_k(ranked, true_service, 1)
            row[f"{cfg}_ac3"] = ac_at_k(ranked, true_service, 3)
            row[f"{cfg}_avg5"] = avg_at_k(ranked, true_service, 5)
            f = faithfulness(dx, retrieved)
            row[f"{cfg}_faith"] = f if f is not None else ""
        rows.append(row)
        print(f"  [{n}/{len(test_cases)}] {row['case']:<22} "
              f"vec={'OK' if row['vector_ac1'] else 'x'} "
              f"vl={'OK' if row['vectorless_ac1'] else 'x'}")

    _print_report(rows, configs)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nPer-case results saved to {out_csv}")
    return rows


def _mean(vals):
    vals = [v for v in vals if v != "" and v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _print_report(rows, configs):
    print("\n" + "=" * 58)
    print("RESULTS  (mean over test set)")
    print("=" * 58)
    print(f"{'Configuration':<16}{'AC@1':>7}{'AC@3':>7}{'Avg@5':>8}{'Faithful':>10}")
    ac1 = {}
    for cfg in configs:
        a1 = _mean([r[f'{cfg}_ac1'] for r in rows])
        a3 = _mean([r[f'{cfg}_ac3'] for r in rows])
        a5 = _mean([r[f'{cfg}_avg5'] for r in rows])
        fa = _mean([r[f'{cfg}_faith'] for r in rows])
        ac1[cfg] = a1
        fstr = f"{fa:>10.2f}" if cfg != "no_retrieval" else f"{'-':>10}"
        print(f"{cfg:<16}{a1:>7.2f}{a3:>7.2f}{a5:>8.2f}{fstr}")

    print("-" * 58)
    print(f"Ablation delta (AC@1 vs no-retrieval):")
    print(f"  vector      dRAG = {ac1['vector'] - ac1['no_retrieval']:+.2f}")
    print(f"  vectorless  dRAG = {ac1['vectorless'] - ac1['no_retrieval']:+.2f}")

    # per-fault-type AC@1 -- "which strategy for which failure category"
    print("-" * 58)
    print("AC@1 by fault type:")
    faults = sorted({r['true_fault'] for r in rows})
    print(f"{'fault':<12}{'vector':>8}{'vectorless':>12}")
    for flt in faults:
        sub = [r for r in rows if r['true_fault'] == flt]
        v = _mean([r['vector_ac1'] for r in sub])
        vl = _mean([r['vectorless_ac1'] for r in sub])
        print(f"{flt:<12}{v:>8.2f}{vl:>12.2f}")
    print("=" * 58)


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/RE1-OB"
    run_evaluation(root, model=MODEL)