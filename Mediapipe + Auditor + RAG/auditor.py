from rag_coach import RAGCoach

# Single shared instance — loaded once, reused every audit call
_coach = None

def _get_coach():
    global _coach
    if _coach is None:
        _coach = RAGCoach(model_name="llama3.2:3b")
    return _coach


def vlm_auditor(image_path: str, exercise_name: str, fault_type: str = "stuck",
                phase: str = "down", angle: float = 100.0):
    """
    Agentic coach auditor.
    - image_path: saved snapshot (kept for logging / future VLM upgrade)
    - exercise_name: active exercise, e.g. "squats", "push-ups", "high_knees"
    - fault_type: detected issue passed in from orchestrator
    - phase: current movement phase
    - angle: current joint angle (0.0 for non-angle exercises)
    Returns dict with 'feedback' key.
    """
    print(f"[Auditor] Triggered — exercise={exercise_name} fault={fault_type} "
          f"phase={phase} angle={int(angle)}")

    coach = _get_coach()
    feedback = coach.get_feedback(
        fault_type=fault_type,
        exercise_name=exercise_name,
        phase=phase,
        angle=angle
    )

    return {"feedback": feedback or ""}
