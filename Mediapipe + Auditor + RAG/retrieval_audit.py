# Phase 2 — Retrieval audit
# Prints the top-3 retrieved cues for every exercise/fault pair in FAULT_QUERIES.
# Use this to spot weak or off-topic retrievals before tuning FAULT_QUERIES / KNOWLEDGE_BASE.
#
# Run from: Mediapipe + Auditor/
#   python retrieval_audit.py
#   python retrieval_audit.py squats          # filter to one exercise
#   python retrieval_audit.py squats stuck    # filter to one exercise + fault

import sys
from rag_coach import RAGCoach, FAULT_QUERIES

coach = RAGCoach(model_name="llama3.2:3b")

filter_exercise = sys.argv[1] if len(sys.argv) > 1 else None
filter_fault    = sys.argv[2] if len(sys.argv) > 2 else None

total = 0
for exercise, faults in FAULT_QUERIES.items():
    if filter_exercise and exercise != filter_exercise:
        continue
    print(f"\n{'='*60}")
    print(f"  EXERCISE: {exercise}")
    print(f"{'='*60}")
    for fault, query in faults.items():
        if filter_fault and fault != filter_fault:
            continue
        cues = coach._retrieve(fault, exercise)
        print(f"\n  [{fault}]  query: \"{query}\"")
        for i, cue in enumerate(cues, 1):
            print(f"    {i}. {cue}")
        total += 1

print(f"\n{'='*60}")
print(f"Checked {total} exercise/fault pairs.")
print("Look for cues that don't match the fault — those query strings need tuning.")
