"""
Vectorless retrieval -- retrieval MODE 2 (structure-aware). Your contribution.

Instead of embedding incidents into vectors and comparing similarity, this
mode organizes the knowledge base into a TREE and has the LLM navigate it,
the way a reader uses a table of contents (this is the RAPTOR idea, applied
to fault diagnosis).

Tree shape (built from the HISTORY entries, using their known labels):

    root
    |- adservice
    |   |- cpu   -> [adservice_cpu/1, adservice_cpu/2, ...]
    |   \\- mem   -> [adservice_mem/1, ...]
    |- cartservice
    |   |- cpu   -> [...]
    ...

Navigation (Algorithm 1 in the proposal):
    node = root
    while node is not a leaf:
        show the child headings + the query summary to the LLM
        node = the child the LLM judges most relevant
    return the incidents in that leaf

Only the query's SUMMARY TEXT is used to navigate -- never its label (which
is unknown at test time). The headings come from the history labels, which is
legitimate because history is the known knowledge base.

Same interface as VectorRetriever: retrieve(query_summary, k) -> [(entry, score)].
So the diagnosis engine treats both modes identically, which is what makes the
comparison fair.

The 'chooser' (which child to descend into) is swappable. The default uses a
local LLM via Ollama; a keyword chooser is provided for testing without an LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from knowledge_base import KBEntry, load_kb


@dataclass
class TreeNode:
    heading: str
    children: list["TreeNode"] = field(default_factory=list)
    entries: list[KBEntry] = field(default_factory=list)  # only at leaves

    def is_leaf(self) -> bool:
        return not self.children


def build_tree(entries: list[KBEntry]) -> TreeNode:
    """root -> service -> fault -> [entries]."""
    by_service: dict[str, dict[str, list[KBEntry]]] = {}
    for e in entries:
        by_service.setdefault(e.root_cause_service, {}) \
                  .setdefault(e.fault_type, []).append(e)

    root = TreeNode("root")
    for svc in sorted(by_service):
        svc_node = TreeNode(svc)
        for flt in sorted(by_service[svc]):
            svc_node.children.append(TreeNode(flt, entries=by_service[svc][flt]))
        root.children.append(svc_node)
    return root


# ---- choosers (decide which child heading to descend into) ----

def keyword_chooser(query: str, headings: list[str]) -> str:
    """No-LLM stand-in: pick the heading mentioned most in the query text.
    Good enough to test navigation mechanics."""
    ql = query.lower()
    scored = sorted(headings, key=lambda h: ql.count(h.lower()), reverse=True)
    return scored[0] if ql.count(scored[0].lower()) > 0 else headings[0]


def make_ollama_chooser(model: str = "llama3.1:8b"):
    """Default chooser: ask a local LLM which heading is most relevant.
    Set `model` to whatever tag you pulled in Ollama (e.g. the proposal's
    Llama model). Requires `pip install ollama` and Ollama running."""
    import ollama

    def choose(query: str, headings: list[str]) -> str:
        prompt = (
            "An incident was observed with these symptoms:\n"
            f"{query}\n\n"
            "Which ONE of the following categories is the most likely match?\n"
            f"Options: {', '.join(headings)}\n"
            "Answer with exactly one option name and nothing else."
        )
        resp = ollama.generate(model=model, prompt=prompt)
        text = resp["response"].strip().lower()
        for h in headings:                    # tolerant match
            if h.lower() in text:
                return h
        return headings[0]                    # fallback

    return choose


class VectorlessRetriever:
    def __init__(self, choose_fn=None, model: str = "llama3.1:8b"):
        # choose_fn=None -> real LLM via Ollama (lazy). Inject one for tests.
        self._choose_fn = choose_fn
        self._model = model
        self.tree: TreeNode | None = None
        self.last_path: list[str] = []        # headings taken on the last query

    def _chooser(self):
        if self._choose_fn is None:
            self._choose_fn = make_ollama_chooser(self._model)
        return self._choose_fn

    def build(self, entries: list[KBEntry]) -> "VectorlessRetriever":
        self.tree = build_tree(entries)
        return self

    def retrieve(self, query_summary: str, k: int = 5) -> list[tuple[KBEntry, float]]:
        if self.tree is None:
            raise RuntimeError("call build() first")
        choose = self._chooser()
        node = self.tree
        self.last_path = []
        while not node.is_leaf():
            headings = [c.heading for c in node.children]
            chosen = choose(query_summary, headings)
            self.last_path.append(chosen)
            node = next((c for c in node.children if c.heading == chosen),
                        node.children[0])
        # vectorless has no similarity score; report 1.0 for interface parity
        return [(e, 1.0) for e in node.entries[:k]]


if __name__ == "__main__":
    entries = load_kb("knowledge_base.json")
    print(f"Loaded {len(entries)} KB entries. Building tree...")
    r = VectorlessRetriever(model="llama3.2:3b").build(entries)

    q = entries[0]
    print(f"\nQuery = {q.entry_id} (true: {q.root_cause_service}, {q.fault_type})")
    print("Navigating tree with the LLM...")
    results = r.retrieve(q.summary, k=5)
    print(f"Path taken: root -> {' -> '.join(r.last_path)}")
    print("Retrieved incidents:")
    for e, _ in results:
        print(f"  {e.entry_id:<24} ({e.root_cause_service}, {e.fault_type})")