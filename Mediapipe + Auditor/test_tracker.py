# pip install opencv-python mediapipe==0.10.14
# pip install ollama faiss-cpu sentence-transformers

import cv2
import time
import os
import threading
from tracker import PoseTracker
from update_coach_logic import update_coach_logic, make_initial_state
from auditor import vlm_auditor, _get_coach

# ---------------------------------------------------------------------------
# All 23 Table 5 exercises mapped to keyboard keys
# ---------------------------------------------------------------------------
EXERCISE_MAP = {
    # --- WARM-UP (Q-T) ---
    ord('q'): "jumping_jacks",
    ord('w'): "high_knees",
    ord('e'): "butt_kickers",
    ord('r'): "air_jump_rope",
    ord('t'): "good_mornings",

    # --- MAIN WORKOUT (1-0, U-P) ---
    ord('1'): "push-ups",
    ord('2'): "plank_taps",
    ord('3'): "moving_plank",
    ord('4'): "squats",
    ord('5'): "walking_lunges",
    ord('6'): "lunge_jumps",
    ord('7'): "puddle_jumps",
    ord('8'): "mountain_climbers",
    ord('9'): "floor_touches",
    ord('0'): "quick_feet",
    ord('u'): "squat_jumps",
    ord('i'): "squat_kicks",
    ord('o'): "standing_kicks",
    ord('p'): "boxing_squat_punches",

    # --- COOL-DOWN (A-F) ---
    ord('a'): "deltoid_stretch",
    ord('s'): "quad_stretch",
    ord('d'): "shoulder_gators",
    ord('f'): "toe_touchers",
}

# Per-exercise angle thresholds for the angle branch
ANGLE_THRESHOLDS = {
    "squats":               {"down": 90,  "up": 160},
    "walking_lunges":       {"down": 100, "up": 160},
    "lunge_jumps":          {"down": 100, "up": 160},
    "squat_jumps":          {"down": 90,  "up": 160},
    "squat_kicks":          {"down": 90,  "up": 160},
    "good_mornings":        {"down": 100, "up": 150},
    "push-ups":             {"down": 90,  "up": 155},
    "moving_plank":         {"down": 150, "up": 170},
    "boxing_squat_punches": {"down": 90,  "up": 160},
    "_default":             {"down": 90,  "up": 160},
}

# Exercises where timer is shown instead of rep count
STRETCH_EXERCISES = {"deltoid_stretch", "quad_stretch", "shoulder_gators", "toe_touchers"}


def classify_fault(state: dict, data: dict) -> str:
    """Determines the fault type for RAG retrieval based on current state and data."""
    if "angle" in data:
        angle = data["angle"]
        if state["min_angle_this_rep"] > 95:
            return "shallow_depth"   # rep completed without hitting depth
        elif angle < 80:
            return "stuck"           # deep but can't drive up
        elif angle < 110:
            return "shallow_depth"
        else:
            return "knee_valgus"
    elif "dist_val" in data:
        return "shallow_depth"       # didn't reach full extension/contraction
    elif "height_val" in data:
        return "low_drive"           # landmark not reaching target height
    elif "hand_y_diff" in data:
        return "asymmetry"
    else:
        return "sagging_hips"        # vlm_only / isometric default


def run_audit(snapshot_path, exercise_name, fault_type, phase, angle, state_ref):
    """Runs RAGCoach audit in a background thread."""
    result = vlm_auditor(
        image_path=snapshot_path,
        exercise_name=exercise_name,
        fault_type=fault_type,
        phase=phase,
        angle=angle,
    )
    state_ref["vlm_feedback"] = result["feedback"]
    state_ref["audit_in_progress"] = False


def main():
    tracker = PoseTracker()
    exercise_name = "squats"
    state = make_initial_state()

    _get_coach()  # pre-warm RAGCoach so first audit fires immediately

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    audit_cooldown = 0
    AUDIT_COOLDOWN_FRAMES = 90  # ~3s between audit triggers

    print("Agentic Tracker Running...")
    print("Q-T: Warm-up | 1-0 / U-P: Main | A-F: Cool-down | ESC: Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── KEYBOARD: exercise switching ────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key in EXERCISE_MAP:
            new_exercise = EXERCISE_MAP[key]
            if new_exercise != exercise_name:
                exercise_name = new_exercise
                state = make_initial_state()
                # Context-aware initial phase
                if "jacks" in exercise_name:
                    state["phase"] = "closed"
                elif any(x in exercise_name for x in ["high_knees", "butt", "quick", "kicks", "standing"]):
                    state["phase"] = "low"
                audit_cooldown = 0
                print(f"[Switched] {exercise_name}")

        cfg = tracker.get_exercise_config(exercise_name)
        logic_type = cfg["type"]
        thresholds = ANGLE_THRESHOLDS.get(exercise_name, ANGLE_THRESHOLDS["_default"])
        is_stretch = exercise_name in STRETCH_EXERCISES

        # ── PERCEPTION ──────────────────────────────────────────────────────
        data = tracker.process_frame(frame, exercise_name)

        if data:
            # ── REASONING ───────────────────────────────────────────────────
            state = update_coach_logic(
                state, data, exercise_name,
                threshold_down=thresholds["down"],
                threshold_up=thresholds["up"],
            )

            if audit_cooldown > 0:
                audit_cooldown -= 1

            # ── STRETCH TIMER ────────────────────────────────────────────────
            if is_stretch:
                state["timer"] += 1 / 30
                if state["timer"] >= 5.0 and not state["audit_in_progress"] and audit_cooldown == 0:
                    state["timer"] = 0.0
                    if not os.path.exists("audits"):
                        os.makedirs("audits")
                    snapshot_path = f"audits/audit_{int(time.time())}.jpg"
                    cv2.imwrite(snapshot_path, frame)
                    state["audit_in_progress"] = True
                    audit_cooldown = AUDIT_COOLDOWN_FRAMES
                    threading.Thread(
                        target=run_audit,
                        args=(snapshot_path, exercise_name, "sagging_hips",
                              state["phase"], 0.0, state),
                        daemon=True,
                    ).start()

            # ── AGENTIC GATE ─────────────────────────────────────────────────
            elif state["is_anomaly"] and not state["audit_in_progress"] and audit_cooldown == 0:
                if not os.path.exists("audits"):
                    os.makedirs("audits")
                snapshot_path = f"audits/audit_{int(time.time())}.jpg"
                cv2.imwrite(snapshot_path, frame)
                fault_type = classify_fault(state, data)
                angle_val  = data.get("angle", 0.0)

                state["audit_in_progress"] = True
                audit_cooldown = AUDIT_COOLDOWN_FRAMES
                state["is_anomaly"] = False
                state["consecutive_stuck_frames"] = 0

                threading.Thread(
                    target=run_audit,
                    args=(snapshot_path, exercise_name, fault_type,
                          state["phase"], angle_val, state),
                    daemon=True,
                ).start()

            # ── VISUAL OVERLAYS ──────────────────────────────────────────────
            # Header bar
            cv2.rectangle(frame, (0, 0), (640, 45), (0, 0, 0), -1)
            cv2.putText(frame, "Q-T: Warmup | 1-P: Main | A-F: Cooldown | ESC: Quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Exercise name
            cv2.putText(frame, f"MODE: {exercise_name.upper()}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Reps or hold timer
            if is_stretch:
                cv2.putText(frame, f"HOLD: {int(state['timer'])}s", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            else:
                cv2.putText(frame, f"REPS: {state['reps']}", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

            # Coach feedback (orange while processing, green when ready)
            coach_color = (0, 165, 255) if state["audit_in_progress"] else (0, 255, 0)
            cv2.putText(frame, f"COACH: {state['vlm_feedback'][:50]}", (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, coach_color, 2)

            # Angle + joint dot for angle-based exercises
            if logic_type == "angle" and "knee_coords" in data:
                h, w, _ = frame.shape
                cx = int(data["knee_coords"][0] * w)
                cy = int(data["knee_coords"][1] * h)
                cv2.putText(frame, str(int(data["angle"])), (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

        cv2.imshow("Agentic Fitness Coach", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
