from typing import TypedDict, Dict, Any

class CoachState(TypedDict):
    reps: int
    phase: str
    is_anomaly: bool
    consecutive_stuck_frames: int
    angle_buffer: list          # rolling 5-frame noise smoothing (angle branch)
    min_angle_this_rep: float   # lowest smoothed angle this rep — depth check
    cooldown_frames: int        # frames until next rep can be counted

# --- Constants ---
STUCK_LIMIT       = 45    # frames stuck in "down" before anomaly fires (~1.5s at 30fps)
ANGLE_BUFFER_SIZE = 5

DIST_THRESHOLD_IN  = 0.30  # spatial "down" (limbs close together)
DIST_THRESHOLD_OUT = 0.50  # spatial "up"  (limbs apart)

HEIGHT_THRESHOLD_UP = 0.02   # height_val threshold — landmark "up" relative to hip
HEIGHT_THRESHOLD_DN = -0.02  # height_val threshold — landmark "down"

COOLDOWN_DEFAULT = 10  # frames between rep counts (prevents double-counting)


def make_initial_state() -> dict:
    """Returns a correctly initialised state dict for any exercise."""
    return {
        "reps": 0,
        "phase": "up",
        "is_anomaly": False,
        "consecutive_stuck_frames": 0,
        "angle_buffer": [],
        "min_angle_this_rep": 180.0,
        "cooldown_frames": 0,
        "vlm_feedback": "",
        "audit_in_progress": False,
        "timer": 0.0,
    }


def update_coach_logic(state: CoachState, data: Dict[str, Any], exercise_name: str,
                       threshold_down: int = 90, threshold_up: int = 160) -> CoachState:
    """
    Unified state machine for all 23 Table 5 exercises.
    Dispatches on the data shape returned by tracker.process_frame():
      'angle'      — squats, lunges, push-ups, good mornings, etc.
      'hand_y_diff'— jumping jacks (open/closed)
      'dist_val'   — plank taps, mountain climbers, floor touches, etc.
      'height_val' — high knees, butt kickers, quick feet, standing kicks
      'status'     — vlm_only (isometric holds — no rep counting here)
    """
    if not data:
        return state

    # ── ANGLE BRANCH ────────────────────────────────────────────────────────
    if "angle" in data:
        current_angle = data["angle"]

        # Rolling average — smooths out single-frame noise spikes
        state["angle_buffer"].append(current_angle)
        if len(state["angle_buffer"]) > ANGLE_BUFFER_SIZE:
            state["angle_buffer"].pop(0)
        smoothed = sum(state["angle_buffer"]) / len(state["angle_buffer"])

        # Phase transitions with ±5° hysteresis dead zone
        if smoothed < (threshold_down - 5) and state["phase"] == "up":
            state["phase"] = "down"
            state["consecutive_stuck_frames"] = 0
            state["min_angle_this_rep"] = smoothed

        elif smoothed > (threshold_up + 5) and state["phase"] == "down":
            if state["min_angle_this_rep"] > 95:
                state["is_anomaly"] = True   # completed rep without hitting depth
            state["phase"] = "up"
            state["reps"] += 1
            state["consecutive_stuck_frames"] = 0
            state["min_angle_this_rep"] = 180.0
            print(f"REPS: {state['reps']}")

        # Track lowest angle and stuck counter
        if state["phase"] == "down":
            if smoothed < state["min_angle_this_rep"]:
                state["min_angle_this_rep"] = smoothed
            if smoothed < 110:
                state["consecutive_stuck_frames"] += 1
            else:
                state["consecutive_stuck_frames"] = 0
        else:
            state["consecutive_stuck_frames"] = 0

        if state["consecutive_stuck_frames"] > STUCK_LIMIT:
            state["is_anomaly"] = True

    # ── JUMPING JACKS (open / closed) ───────────────────────────────────────
    elif "hand_y_diff" in data:
        is_open = data["hand_y_diff"] > 0.05 and data["foot_distance"] > 0.25
        if is_open and state["phase"] == "closed":
            state["phase"] = "open"
        elif not is_open and state["phase"] == "open":
            state["phase"] = "closed"
            if state["cooldown_frames"] == 0:
                state["reps"] += 1
                state["cooldown_frames"] = COOLDOWN_DEFAULT
                print(f"REPS: {state['reps']}")
        if state["cooldown_frames"] > 0:
            state["cooldown_frames"] -= 1

    # ── DIST_VAL (plank taps, mountain climbers, floor touches, toe touchers) ─
    elif "dist_val" in data:
        val = data["dist_val"]
        if val < DIST_THRESHOLD_IN and state["phase"] == "up":
            state["phase"] = "down"
        elif val > DIST_THRESHOLD_OUT and state["phase"] == "down":
            if state["cooldown_frames"] == 0:
                state["reps"] += 1
                state["cooldown_frames"] = COOLDOWN_DEFAULT
                print(f"REPS: {state['reps']}")
            state["phase"] = "up"
        if state["cooldown_frames"] > 0:
            state["cooldown_frames"] -= 1

    # ── HEIGHT_VAL (high knees, butt kickers, quick feet, standing kicks) ───
    elif "height_val" in data:
        val = data["height_val"]
        if val > HEIGHT_THRESHOLD_UP and state["phase"] == "low":
            state["phase"] = "high"
        elif val < HEIGHT_THRESHOLD_DN and state["phase"] == "high":
            state["phase"] = "low"
            if state["cooldown_frames"] == 0:
                state["reps"] += 1
                state["cooldown_frames"] = COOLDOWN_DEFAULT
                print(f"REPS: {state['reps']}")
        if state["cooldown_frames"] > 0:
            state["cooldown_frames"] -= 1

    # ── VLM_ONLY / ISOMETRIC (plank, stretches — time-driven, no rep counting) ─
    # No state changes here; audit timing is managed by test_tracker.py

    return state
