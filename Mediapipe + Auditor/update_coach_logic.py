from typing import TypedDict

# You must define the structure here so the function recognizes the type
class CoachState(TypedDict):
    reps: int
    phase: str
    is_anomaly: bool
    consecutive_stuck_frames: int

# Thresholds for a Squat
THRESHOLD_DOWN = 90   
THRESHOLD_UP = 160     
STUCK_LIMIT = 45       

def update_coach_logic(state: CoachState, current_angle: float):
    # 1. REP COUNTING LOGIC (The Fast Loop)
    if current_angle < THRESHOLD_DOWN and state["phase"] == "up":
        state["phase"] = "down"
        state["consecutive_stuck_frames"] = 0 
        
    elif current_angle > THRESHOLD_UP and state["phase"] == "down":
        state["phase"] = "up"
        state["reps"] += 1
        state["consecutive_stuck_frames"] = 0
        print(f"REPS: {state['reps']}")

    # 2. ANOMALY TRIGGER (The Agentic Gate)
    if state["phase"] == "down" and current_angle < 110:
        state["consecutive_stuck_frames"] += 1
    else:
        state["consecutive_stuck_frames"] = 0

    if state["consecutive_stuck_frames"] > STUCK_LIMIT:
        state["is_anomaly"] = True 
    
    return state