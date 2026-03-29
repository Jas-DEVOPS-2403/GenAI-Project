from rag_coach import RAGCoach

# Single shared instance — loaded once, reused every audit call
_coach = None

def _get_coach():
    global _coach
    if _coach is None:
        _coach = RAGCoach(model_name="llama3.2:3b")
    return _coach


def vlm_auditor(image_path: str, exercise_type: str, fault_type: str = "stuck",
                phase: str = "down", angle: float = 100.0):
    """
    Agentic coach auditor.
    - image_path: saved snapshot (kept for logging / future VLM upgrade)
    - exercise_type: e.g. "squat"
    - fault_type: detected issue passed in from orchestrator
    - phase: current movement phase
    - angle: current joint angle
    Returns dict with 'feedback' key, matching the original mock interface.
    """
    print(f"[Auditor] Triggered — fault={fault_type} phase={phase} angle={int(angle)}")

    coach = _get_coach()
    feedback = coach.get_feedback(fault_type=fault_type, phase=phase, angle=angle)

    if feedback is None:
        feedback = ""  # suppressed by session memory

    return {"feedback": feedback}
