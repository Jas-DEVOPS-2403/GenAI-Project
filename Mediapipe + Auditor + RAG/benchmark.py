# Phase 3 — RAG retrieval benchmark (hit@k)
#
# For every exercise/fault pair in FAULT_QUERIES, checks whether the top-k
# retrieved cues contain at least one expected keyword. Produces a score
# you can track across tuning iterations.
#
# Run from: Mediapipe + Auditor/
#   python benchmark.py        # k=3 (default)
#   python benchmark.py 5      # k=5

import sys
from rag_coach import RAGCoach, FAULT_QUERIES

# Ground-truth keywords per fault type.
# A retrieval is a HIT if any top-k cue contains at least one keyword.
EXPECTED_KEYWORDS = {
    "knee_valgus":   ["knee", "valgus", "caving", "inward", "push.*out", "spread"],
    "shallow_depth": ["depth", "parallel", "floor", "deep", "range", "low"],
    "rounded_back":  ["spine", "neutral", "rounded", "back", "chest up"],
    "stuck":         ["drive", "press", "push", "explode", "squeeze"],
    "sagging_hips":  ["hips", "core", "plank", "straight", "brace"],
    "low_drive":     ["drive", "height", "reach", "high", "hip height"],
    "good_form":     ["good", "strong", "keep", "great", "solid", "consistent"],
    "elbow_flare":   ["elbow", "45", "tuck", "wide", "flare"],
    "forward_lean":  ["torso", "upright", "lean", "chest", "tall"],
    "landing":       ["land", "soft", "absorb", "bend"],
    "asymmetry":     ["arms", "legs", "sync", "together", "both"],
    "timing":        ["rhythm", "sync", "coordinate", "together", "cadence"],
    "hinge":         ["hips", "hinge", "hamstring", "neutral"],
    "hip_drop":      ["hips", "level", "rotation", "rotate"],
}

def is_hit(cues: list, fault_type: str) -> bool:
    keywords = EXPECTED_KEYWORDS.get(fault_type, [])
    if not keywords:
        return True  # no ground truth defined — skip penalising
    combined = " ".join(cues).lower()
    return any(kw.lower() in combined for kw in keywords)


def run_benchmark(k: int = 3):
    print(f"\n[RAGCoach] Loading encoder...", flush=True)
    coach = RAGCoach(model_name="llama3.2:3b")

    hits = 0
    total = 0
    misses = []

    col_ex  = 22
    col_flt = 18
    col_res = 6

    print(f"\n{'='*70}")
    print(f"  BENCHMARK RESULTS — hit@{k}")
    print(f"{'='*70}")
    print(f"  {'Exercise':<{col_ex}} {'Fault':<{col_flt}} {'Result':<{col_res}}  Top cue")
    print(f"  {'-'*(col_ex+col_flt+col_res+30)}")

    for exercise, faults in FAULT_QUERIES.items():
        if exercise == "_generic":
            continue
        for fault in faults:
            cues = coach._retrieve(fault, exercise, k=k)
            hit = is_hit(cues, fault)
            hits += hit
            total += 1
            tag = "HIT " if hit else "MISS"
            top = cues[0][:55] + "..." if cues and len(cues[0]) > 55 else (cues[0] if cues else "—")
            print(f"  {exercise:<{col_ex}} {fault:<{col_flt}} {tag:<{col_res}}  {top}")
            if not hit:
                misses.append((exercise, fault, cues))

    pct = 100 * hits / total if total else 0
    print(f"\n  {'-'*60}")
    print(f"  Score: {hits}/{total} = {pct:.1f}%  (hit@{k})")

    if misses:
        print(f"\n{'='*70}")
        print(f"  MISSES — tune FAULT_QUERIES or KNOWLEDGE_BASE for these:")
        print(f"{'='*70}")
        for ex, fault, cues in misses:
            print(f"\n  [{ex} / {fault}]")
            for i, c in enumerate(cues, 1):
                print(f"    {i}. {c}")
    else:
        print("\n  All pairs hit — knowledge base is well-aligned.")

    print()
    return pct


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    run_benchmark(k)
