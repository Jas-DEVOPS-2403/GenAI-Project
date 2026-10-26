#python -m pip install --upgrade pip
#python -m pip install ollama faiss-cpu sentence-transformers


import sys

# A Windows console defaults to cp1252 and raises UnicodeEncodeError on the
# en/em dashes carried in the knowledge-base cues printed below.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rag_coach import RAGCoach

coach = RAGCoach(model_name="llama3.2:3b")

print("\n=== RETRIEVAL CHECK ===")
for fault in ["stuck", "shallow_depth", "knee_valgus", "good_form"]:
    cues = coach._retrieve(fault)
    print(f"\n[{fault}]")
    for c in cues:
        print(f"  → {c}")

print("\n=== LLM CHECK ===")
feedback = coach.get_feedback(fault_type="stuck", phase="down", angle=75)
print(f"LLM output: {feedback}")

print("\n=== SESSION MEMORY CHECK ===")
for i in range(3):
    result = coach.get_feedback(fault_type="stuck", phase="down", angle=75)
    print(f"  Call {i+1}: {repr(result)}")
print("(3rd call should be None — suppressed by session memory)")

print("\n=== ALL CHECKS DONE ===")
sys.exit(0)