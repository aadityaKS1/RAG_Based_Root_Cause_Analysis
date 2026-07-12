"""
RAG Diagnosis Engine -- Section 3.2.3.

Takes the current incident summary plus the retrieved past incidents (from
EITHER retrieval mode) and asks the LLM for a structured diagnosis:
  - a RANKED list of the most likely root-cause services (so AC@1/AC@3/Avg@5
    have a ranking to score)
  - the fault type
  - an explanation, with citations to the evidence items it used

Retrieval-mode-agnostic: `retrieved` is the list[(KBEntry, score)] returned by
either VectorRetriever or VectorlessRetriever, so this one engine serves both
arms of the comparison unchanged.

The LLM call is swappable: default is a local model via Ollama; inject an
llm_fn for testing without a model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from knowledge_base import KBEntry


@dataclass
class Diagnosis:
    ranked_services: list[str]        # most likely first -> used for AC@k
    fault_type: str
    explanation: str
    citations: list[str] = field(default_factory=list)
    raw: str = ""                     # raw LLM text, for debugging

    @property
    def root_cause_service(self) -> str:
        return self.ranked_services[0] if self.ranked_services else "unknown"


def _build_prompt(query_summary: str, retrieved: list[tuple[KBEntry, float]]) -> str:
    ev_lines = []
    for i, (e, score) in enumerate(retrieved, start=1):
        ev_lines.append(
            f"[E{i}] root_cause_service={e.root_cause_service} "
            f"fault_type={e.fault_type}\n     {e.summary.splitlines()[0]}"
        )
    evidence = "\n".join(ev_lines) if ev_lines else "(no evidence retrieved)"

    return (
        "You are a site reliability engineer diagnosing a microservice failure.\n\n"
        "CURRENT INCIDENT:\n"
        f"{query_summary}\n\n"
        "SIMILAR PAST INCIDENTS (evidence, each with its verified root cause):\n"
        f"{evidence}\n\n"
        "Using the current incident and the evidence, identify the root cause. "
        "List the most likely root-cause services in order (most likely first). "
        "Cite the evidence items you relied on by their tag (e.g. E1).\n\n"
        "Respond with ONLY a JSON object, no other text:\n"
        '{"ranked_services": ["...", "..."], "fault_type": "...", '
        '"explanation": "...", "citations": ["E1"]}'
    )


def _parse_diagnosis(text: str) -> Diagnosis:
    """Robustly pull the JSON object out of the LLM response."""
    raw = text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return Diagnosis([], "unknown", "could not parse LLM output", [], raw)
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return Diagnosis([], "unknown", "invalid JSON from LLM", [], raw)

    services = obj.get("ranked_services") or []
    if isinstance(services, str):
        services = [services]
    return Diagnosis(
        ranked_services=[str(s) for s in services],
        fault_type=str(obj.get("fault_type", "unknown")),
        explanation=str(obj.get("explanation", "")),
        citations=[str(c) for c in (obj.get("citations") or [])],
        raw=raw,
    )


def make_ollama_llm(model: str = "llama3.2:3b"):
    """Default LLM: local model via Ollama. Set `model` to your pulled tag."""
    import ollama

    def llm(prompt: str) -> str:
        return ollama.generate(model=model, prompt=prompt)["response"]

    return llm


class DiagnosisEngine:
    def __init__(self, llm_fn=None, model: str = "llama3.2:3b", runs: int = 1):
        self._llm_fn = llm_fn
        self._model = model
        self.runs = runs              # >1 -> repeat and majority-vote (LLM is noisy)

    def _llm(self):
        if self._llm_fn is None:
            self._llm_fn = make_ollama_llm(self._model)
        return self._llm_fn

    def diagnose(self, query_summary: str,
                 retrieved: list[tuple[KBEntry, float]]) -> Diagnosis:
        prompt = _build_prompt(query_summary, retrieved)
        llm = self._llm()
        if self.runs == 1:
            return _parse_diagnosis(llm(prompt))

        # multiple runs -> majority vote on the top service (non-determinism)
        results = [_parse_diagnosis(llm(prompt)) for _ in range(self.runs)]
        from collections import Counter
        top = Counter(d.root_cause_service for d in results
                      if d.ranked_services).most_common(1)
        winner_service = top[0][0] if top else "unknown"
        # return the first run whose top service is the winner (keeps its text)
        for d in results:
            if d.root_cause_service == winner_service:
                return d
        return results[0]


if __name__ == "__main__":
    from knowledge_base import load_kb
    from vector_retrieval import VectorRetriever

    entries = load_kb("knowledge_base.json")
    retriever = VectorRetriever().build(entries)      # real MiniLM
    engine = DiagnosisEngine(model="llama3.2:3b")     # real Ollama

    q = entries[0]
    retrieved = retriever.retrieve(q.summary, k=5)
    dx = engine.diagnose(q.summary, retrieved)
    print(f"Query {q.entry_id}  (true: {q.root_cause_service}, {q.fault_type})")
    print(f"Predicted services (ranked): {dx.ranked_services}")
    print(f"Predicted fault: {dx.fault_type}")
    print(f"Explanation: {dx.explanation}")
    print(f"Citations: {dx.citations}")