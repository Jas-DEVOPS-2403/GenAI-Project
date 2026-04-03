from typing import TypedDict

# You must define the structure here so the function recognizes the type
class CoachState(TypedDict):
    reps: int
    phase: str
    is_anomaly: bool
    consecutive_stuck_frames: int
    angle_buffer: list        # rolling window of last N raw angles
    min_angle_this_rep: float # lowest smoothed angle seen in current rep

# Thresholds for a Squat
THRESHOLD_DOWN = 90
THRESHOLD_UP = 160
STUCK_LIMIT = 45
ANGLE_BUFFER_SIZE = 5

def update_coach_logic(state: CoachState, current_angle: float):
    # Rolling average (noise smoothing)
    state["angle_buffer"].append(current_angle)
    if len(state["angle_buffer"]) > ANGLE_BUFFER_SIZE:
        state["angle_buffer"].pop(0)
    smoothed_angle = sum(state["angle_buffer"]) / len(state["angle_buffer"])

    # 1. REP COUNTING LOGIC with hysteresis (±5° dead zone)
    if smoothed_angle < (THRESHOLD_DOWN - 5) and state["phase"] == "up":
        state["phase"] = "down"
        state["consecutive_stuck_frames"] = 0
        state["min_angle_this_rep"] = smoothed_angle

    elif smoothed_angle > (THRESHOLD_UP + 5) and state["phase"] == "down":
        if state["min_angle_this_rep"] > 95:
            state["is_anomaly"] = True   # shallow rep — didn't hit depth
        state["phase"] = "up"
        state["reps"] += 1
        state["consecutive_stuck_frames"] = 0
        state["min_angle_this_rep"] = 180.0
        print(f"REPS: {state['reps']}")

    # 2. ANOMALY TRIGGER (stuck gate)
    if state["phase"] == "down":
        if smoothed_angle < state["min_angle_this_rep"]:
            state["min_angle_this_rep"] = smoothed_angle
        if smoothed_angle < 110:
            state["consecutive_stuck_frames"] += 1
        else:
            state["consecutive_stuck_frames"] = 0
    else:
        state["consecutive_stuck_frames"] = 0

    if state["consecutive_stuck_frames"] > STUCK_LIMIT:
        state["is_anomaly"] = True

    return state
