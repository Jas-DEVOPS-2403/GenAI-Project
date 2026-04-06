# Phase 3 — RAG retrieval benchmark
#
# For every exercise/fault pair in FAULT_QUERIES, scores the top-k retrieved
# cues using Precision, Recall, and F1 against expected keywords.
#
#   Precision@k  = fraction of the top-k cues that are relevant
#                  (contain at least one expected keyword)
#   Recall@k     = 1 if any top-k cue is relevant, else 0
#                  (did we surface anything useful at all?)
#   F1@k         = 2 × (Precision × Recall) / (Precision + Recall)
#
# Run from: Mediapipe + Auditor + RAG/
#   python benchmark.py        # k=3 (default)
#   python benchmark.py 5      # k=5

import sys
from rag_coach import RAGCoach, FAULT_QUERIES

# ---------------------------------------------------------------------------
# Expected keywords per fault type.
# A cue is RELEVANT if it contains at least one keyword from this list.
# ---------------------------------------------------------------------------
EXPECTED_KEYWORDS = {
    "knee_valgus":    ["knee", "valgus", "caving", "inward", "push.*out", "spread"],
    "shallow_depth":  ["depth", "parallel", "floor", "deep", "range", "low"],
    "rounded_back":   ["spine", "neutral", "rounded", "back", "chest up"],
    "stuck":          ["drive", "press", "push", "explode", "squeeze"],
    "sagging_hips":   ["hips", "core", "plank", "straight", "brace"],
    "low_drive":      ["drive", "height", "reach", "high", "hip height"],
    "good_form":      ["good", "strong", "keep", "great", "solid", "consistent"],
    "elbow_flare":    ["elbow", "45", "tuck", "wide", "flare"],
    "forward_lean":   ["torso", "upright", "lean", "chest", "tall"],
    "landing":        ["land", "soft", "absorb", "bend"],
    "asymmetry":      ["arms", "legs", "sync", "together", "both"],
    "timing":         ["rhythm", "sync", "coordinate", "together", "cadence"],
    "rotation":       ["hips", "level", "rotation", "rotate"],
    "balance_loss":   ["balance", "gaze", "fix", "standing", "glute"],
    "rep_milestone":  ["rep", "keep", "going", "pace", "strong"],
}


def cue_is_relevant(cue: str, fault_type: str) -> bool:
    """Returns True if the cue contains at least one expected keyword."""
    keywords = EXPECTED_KEYWORDS.get(fault_type, [])
    if not keywords:
        return True  # no ground truth defined — treat as relevant
    cue_lower = cue.lower()
    return any(kw.lower() in cue_lower for kw in keywords)


def score_pair(cues: list, fault_type: str, k: int) -> dict:
    """Compute Precision@k, Recall@k, F1@k for one exercise/fault pair."""
    relevance = [cue_is_relevant(c, fault_type) for c in cues[:k]]
    relevant_count = sum(relevance)

    precision = relevant_count / k if k > 0 else 0.0
    recall    = 1.0 if relevant_count > 0 else 0.0
    denom     = precision + recall
    f1        = 2 * precision * recall / denom if denom > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1,
            "relevant_count": relevant_count, "relevance": relevance}


def run_benchmark(k: int = 3):
    print(f"\n[RAGCoach] Loading encoder...", flush=True)
    coach = RAGCoach(model_name="llama3.2:3b")

    all_precision, all_recall, all_f1 = [], [], []
    misses = []

    col_ex  = 24
    col_flt = 20
    col_p   = 6
    col_r   = 6
    col_f1  = 6

    print(f"\n{'='*85}")
    print(f"  RAG RETRIEVAL BENCHMARK — top-{k} cues")
    print(f"{'='*85}")
    print(f"  {'Exercise':<{col_ex}} {'Fault':<{col_flt}} {'P@k':>{col_p}} {'R@k':>{col_r}} {'F1':>{col_f1}}  Top cue")
    print(f"  {'-'*83}")

    for exercise, faults in FAULT_QUERIES.items():
        if exercise == "_generic":
            continue
        for fault in faults:
            cues   = coach._retrieve(fault, exercise, k=k)
            scores = score_pair(cues, fault, k)

            all_precision.append(scores["precision"])
            all_recall.append(scores["recall"])
            all_f1.append(scores["f1"])

            top = cues[0][:48] + "..." if cues and len(cues[0]) > 48 else (cues[0] if cues else "—")
            flag = "" if scores["recall"] == 1.0 else " ←"
            print(
                f"  {exercise:<{col_ex}} {fault:<{col_flt}} "
                f"{scores['precision']:>{col_p}.2f} "
                f"{scores['recall']:>{col_r}.2f} "
                f"{scores['f1']:>{col_f1}.2f}  {top}{flag}"
            )
            if scores["recall"] == 0.0:
                misses.append((exercise, fault, cues))

    # ── Aggregate ────────────────────────────────────────────────────────────
    n = len(all_f1)
    macro_p  = sum(all_precision) / n if n else 0.0
    macro_r  = sum(all_recall)    / n if n else 0.0
    macro_f1 = sum(all_f1)        / n if n else 0.0

    print(f"\n  {'─'*83}")
    print(f"  {'MACRO AVERAGE':<{col_ex+col_flt+1}} "
          f"{macro_p:>{col_p}.2f} {macro_r:>{col_r}.2f} {macro_f1:>{col_f1}.2f}")
    print(f"\n  {n} exercise/fault pairs evaluated at k={k}")
    print(f"  Precision@{k}: {macro_p:.3f}  |  Recall@{k}: {macro_r:.3f}  |  F1@{k}: {macro_f1:.3f}")

    if misses:
        print(f"\n{'='*85}")
        print(f"  RECALL=0 MISSES — no relevant cue found in top-{k}:")
        print(f"  Tune FAULT_QUERIES or KNOWLEDGE_BASE for these pairs.")
        print(f"{'='*85}")
        for ex, fault, cues in misses:
            print(f"\n  [{ex} / {fault}]")
            for i, c in enumerate(cues, 1):
                print(f"    {i}. {c}")
    else:
        print(f"\n  All {n} pairs returned at least one relevant cue — knowledge base well-aligned.")

    print()
    return {"precision": macro_p, "recall": macro_r, "f1": macro_f1}


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    run_benchmark(k)
