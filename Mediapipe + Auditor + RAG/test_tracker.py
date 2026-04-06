# pip install opencv-python mediapipe==0.10.14
# pip install ollama faiss-cpu sentence-transformers

import cv2
import time
import os
import threading
from tracker import PoseTracker
from update_coach_logic import update_coach_logic, make_initial_state
from auditor import vlm_auditor, _get_coach

# Auto-detection (optional — only active after model is trained)
try:
    from exercise_detector import ExerciseDetector
    _DETECTOR_AVAILABLE = True
except ImportError:
    _DETECTOR_AVAILABLE = False

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

# Stretch exercises rotate through these fault types (timer-based, no fault detection)
STRETCH_FAULT_ROTATION = ["shallow_depth", "good_form", "good_form"]

# Rep counts that trigger a milestone acknowledgment
REP_MILESTONE_COUNTS = {5, 10, 15, 20, 25, 30}

# How many reps a feedback cue lingers before it can be overwritten by good-form
FEEDBACK_LINGER_REPS = 2


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
    try:
        result = vlm_auditor(
            image_path=snapshot_path,
            exercise_name=exercise_name,
            fault_type=fault_type,
            phase=phase,
            angle=angle,
        )
        # Only overwrite feedback if the LLM returned something (not suppressed by session memory)
        if result["feedback"]:
            state_ref["vlm_feedback"] = result["feedback"]
            state_ref["feedback_set_reps"] = state_ref.get("reps", 0)
    except Exception as e:
        print(f"[Audit] Error: {e}")
    finally:
        state_ref["audit_in_progress"] = False


def main():
    tracker = PoseTracker()
    exercise_name = "squats"
    state = make_initial_state()
    state["feedback_set_reps"] = -FEEDBACK_LINGER_REPS  # allows first cue at rep 3
    good_rep_counter = 0
    stretch_fault_idx = 0

    # Auto-detection setup
    auto_detect = False
    detector = None
    if _DETECTOR_AVAILABLE:
        try:
            detector = ExerciseDetector()
        except FileNotFoundError:
            print("[AutoDetect] No model found — run extract_pose_features.py then train_classifier.py")

    _get_coach()  # pre-warm RAGCoach so first audit fires immediately

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("Agentic Fitness Coach", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Agentic Fitness Coach", 1280, 720)

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
        if key == ord('z') and detector is not None:
            auto_detect = not auto_detect
            if auto_detect:
                detector.reset(exercise_name)
            print(f"[AutoDetect] {'ON  — pose classifier active' if auto_detect else 'OFF — manual keyboard mode'}")
        if key in EXERCISE_MAP:
            new_exercise = EXERCISE_MAP[key]
            if new_exercise != exercise_name:
                exercise_name = new_exercise
                state = make_initial_state()
                state["feedback_set_reps"] = -FEEDBACK_LINGER_REPS
                # Context-aware initial phase
                if "jacks" in exercise_name:
                    state["phase"] = "closed"
                elif any(x in exercise_name for x in ["high_knees", "butt", "quick", "kicks", "standing"]):
                    state["phase"] = "low"
                audit_cooldown = 0
                good_rep_counter = 0
                stretch_fault_idx = 0
                if detector is not None:
                    detector.reset(exercise_name)
                print(f"[Switched] {exercise_name}")

        # ── AUTO-DETECTION ───────────────────────────────────────────────────
        if auto_detect and detector is not None:
            detected = detector.update(frame)
            if detected != exercise_name:
                exercise_name = detected
                state = make_initial_state()
                state["feedback_set_reps"] = -FEEDBACK_LINGER_REPS
                if "jacks" in exercise_name:
                    state["phase"] = "closed"
                elif any(x in exercise_name for x in ["high_knees", "butt", "quick", "kicks", "standing"]):
                    state["phase"] = "low"
                audit_cooldown = 0
                good_rep_counter = 0
                stretch_fault_idx = 0
                print(f"[AutoDetect] → {exercise_name}")

        cfg = tracker.get_exercise_config(exercise_name)
        logic_type = cfg["type"]
        thresholds = ANGLE_THRESHOLDS.get(exercise_name, ANGLE_THRESHOLDS["_default"])
        is_stretch = exercise_name in STRETCH_EXERCISES

        # ── PERCEPTION ──────────────────────────────────────────────────────
        data = tracker.process_frame(frame, exercise_name)

        if data:
            # ── REASONING ───────────────────────────────────────────────────
            prev_reps = state["reps"]
            state = update_coach_logic(
                state, data, exercise_name,
                threshold_down=thresholds["down"],
                threshold_up=thresholds["up"],
            )

            # Track clean reps for positive reinforcement
            if state["reps"] > prev_reps:
                good_rep_counter += 1

            if audit_cooldown > 0:
                audit_cooldown -= 1

            # ── STRETCH TIMER ────────────────────────────────────────────────
            if is_stretch:
                state["timer"] += 1 / 30
                if state["timer"] >= 5.0 and not state["audit_in_progress"] and audit_cooldown == 0:
                    state["timer"] = 0.0
                    fault_type = STRETCH_FAULT_ROTATION[stretch_fault_idx % len(STRETCH_FAULT_ROTATION)]
                    stretch_fault_idx += 1
                    if not os.path.exists("audits"):
                        os.makedirs("audits")
                    snapshot_path = f"audits/audit_{int(time.time())}.jpg"
                    cv2.imwrite(snapshot_path, frame)
                    state["audit_in_progress"] = True
                    audit_cooldown = AUDIT_COOLDOWN_FRAMES
                    threading.Thread(
                        target=run_audit,
                        args=(snapshot_path, exercise_name, fault_type,
                              state["phase"], 0.0, state),
                        daemon=True,
                    ).start()

            # ── REP MILESTONE (5, 10, 15, 20 … reps) ────────────────────────
            elif (state["reps"] > prev_reps and state["reps"] in REP_MILESTONE_COUNTS
                  and not state["audit_in_progress"] and audit_cooldown == 0):
                if not os.path.exists("audits"):
                    os.makedirs("audits")
                snapshot_path = f"audits/audit_{int(time.time())}.jpg"
                cv2.imwrite(snapshot_path, frame)
                state["audit_in_progress"] = True
                audit_cooldown = AUDIT_COOLDOWN_FRAMES
                threading.Thread(
                    target=run_audit,
                    args=(snapshot_path, exercise_name, "rep_milestone",
                          state["phase"], data.get("angle", 0.0), state),
                    daemon=True,
                ).start()

            # ── GOOD-FORM GATE (every 3 clean reps, respects linger window) ──
            elif (good_rep_counter > 0 and good_rep_counter % 3 == 0
                  and not state["is_anomaly"]
                  and not state["audit_in_progress"]
                  and audit_cooldown == 0
                  and state["reps"] - state.get("feedback_set_reps", 0) >= FEEDBACK_LINGER_REPS):
                if not os.path.exists("audits"):
                    os.makedirs("audits")
                snapshot_path = f"audits/audit_{int(time.time())}.jpg"
                cv2.imwrite(snapshot_path, frame)
                state["audit_in_progress"] = True
                audit_cooldown = AUDIT_COOLDOWN_FRAMES
                good_rep_counter = 0  # reset so next trigger is 5 more reps later
                threading.Thread(
                    target=run_audit,
                    args=(snapshot_path, exercise_name, "good_form",
                          state["phase"], data.get("angle", 0.0), state),
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

                # Keep old feedback visible while LLM processes — new cue replaces it on arrival
                state["feedback_set_reps"] = state["reps"]

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
            fh, fw, _ = frame.shape

            # ── Header bar ───────────────────────────────────────────────────
            cv2.rectangle(frame, (0, 0), (fw, 50), (20, 20, 20), -1)
            mode_tag = "AUTO" if auto_detect else "MANUAL"
            header   = f"[{mode_tag}] Q-T: Warmup | 1-P: Main | A-F: Cooldown | Z: AutoDetect | ESC: Quit"
            cv2.putText(frame, header,
                        (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

            # ── Exercise name ────────────────────────────────────────────────
            cv2.putText(frame, f"MODE: {exercise_name.upper()}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            # ── Reps or hold timer ───────────────────────────────────────────
            if is_stretch:
                cv2.putText(frame, f"HOLD: {int(state['timer'])}s", (20, 145),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 220, 255), 3)
            else:
                cv2.putText(frame, f"REPS: {state['reps']}", (20, 145),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)

            # ── Coach feedback — dark bg for readability ─────────────────────
            feedback = state.get("vlm_feedback") or ""
            if feedback:
                full_text  = "COACH: " + feedback
                line1      = full_text[:52]
                line2      = full_text[52:100] if len(full_text) > 52 else ""
                box_bottom = 245 if line2 else 215
                overlay    = frame.copy()
                cv2.rectangle(overlay, (10, 158), (fw - 10, box_bottom), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
                coach_color = (0, 200, 255) if state["audit_in_progress"] else (100, 255, 100)
                cv2.putText(frame, line1, (18, 192),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, coach_color, 2)
                if line2:
                    cv2.putText(frame, line2, (18, 230),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, coach_color, 2)

            # ── Angle + joint dots ───────────────────────────────────────────
            if logic_type == "angle" and "all_knee_coords" in data:
                for coords in data["all_knee_coords"]:
                    cx = int(coords[0] * fw)
                    cy = int(coords[1] * fh)
                    cv2.circle(frame, (cx, cy), 12, (0, 255, 100), -1)
                cx0 = int(data["all_knee_coords"][0][0] * fw)
                cy0 = int(data["all_knee_coords"][0][1] * fh)
                cv2.putText(frame, str(int(data["angle"])), (cx0 + 12, cy0 - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # ── Landmark dots for spatial/height exercises ───────────────────
            elif "landmark_coords" in data:
                for coords in data["landmark_coords"]:
                    if coords is not None:
                        cx = int(coords[0] * fw)
                        cy = int(coords[1] * fh)
                        cv2.circle(frame, (cx, cy), 10, (0, 255, 255), -1)

        cv2.imshow("Agentic Fitness Coach", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
