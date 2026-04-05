# Offline benchmark bridge: MediaPipe + RAGCoach → predictions.json
#
# Processes each exercise segment from the eval manifest through our pipeline
# and outputs the predictions.json format expected by stage3_eval.py.
#
# Usage:
#   python generate_predictions.py --manifest ../eval/benchmark_manifest.json --output predictions.json
#   python generate_predictions.py --manifest ../eval/benchmark_manifest.json --output predictions.json --limit 5

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# --- eval framework imports (direct path, no src. prefix needed) ---
sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
from stage3.manifest import load_segment_manifest, load_video_timestamps
from stage3.predictions import save_predictions

# --- pipeline imports ---
from tracker import PoseTracker
from update_coach_logic import update_coach_logic, make_initial_state
from auditor import vlm_auditor, _get_coach

# ---------------------------------------------------------------------------
# Exercise name map: benchmark names → tracker config names
# ---------------------------------------------------------------------------
EXERCISE_NAME_MAP = {
    "good morning beginner": "good_mornings",
    "pushups":               "push-ups",
    "armcrosschest left":    "deltoid_stretch",
    "armcrosschest right":   "deltoid_stretch",
    "quad stretch left":     "quad_stretch",
    "quad stretch right":    "quad_stretch",
}

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

STRETCH_EXERCISES = {"deltoid_stretch", "quad_stretch", "shoulder_gators", "toe_touchers"}

AUDIT_COOLDOWN_FRAMES = 90  # ~3s at 30fps


def normalize_exercise_name(name: str) -> str:
    name = name.lower().strip()
    if name in EXERCISE_NAME_MAP:
        return EXERCISE_NAME_MAP[name]
    return name.replace(" ", "_")


def classify_fault(state: dict, data: dict) -> str:
    if "angle" in data:
        angle = data["angle"]
        if state["min_angle_this_rep"] > 95:
            return "shallow_depth"
        elif angle < 80:
            return "stuck"
        elif angle < 110:
            return "shallow_depth"
        else:
            return "knee_valgus"
    elif "dist_val" in data:
        return "shallow_depth"
    elif "height_val" in data:
        return "low_drive"
    elif "hand_y_diff" in data:
        return "asymmetry"
    else:
        return "sagging_hips"


def process_segment(segment, tracker) -> tuple[list[str], list[float]]:
    tracker_name = normalize_exercise_name(segment.exercise_name)
    is_stretch = tracker_name in STRETCH_EXERCISES

    timestamps = load_video_timestamps(segment.video_timestamps_path)
    start_ts = segment.exercise_start_timestamp
    end_ts = segment.exercise_end_timestamp

    start_frame = int(np.searchsorted(timestamps, start_ts))
    end_frame = int(np.searchsorted(timestamps, end_ts, side="right"))

    cap = cv2.VideoCapture(segment.video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    state = make_initial_state()
    # Set correct initial phase (mirrors test_tracker.py exercise-switch logic)
    if "jacks" in tracker_name:
        state["phase"] = "closed"
    elif any(x in tracker_name for x in ["high_knees", "butt", "quick", "kicks", "standing"]):
        state["phase"] = "low"
    thresholds = ANGLE_THRESHOLDS.get(tracker_name, ANGLE_THRESHOLDS["_default"])
    pred_feedbacks: list[str] = []
    pred_timestamps: list[float] = []
    audit_cooldown = 0
    good_rep_counter = 0
    stretch_timer = 0.0

    for frame_idx in range(start_frame, min(end_frame, total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        if segment.rotate_90_cw:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        frame_ts = float(timestamps[frame_idx]) if frame_idx < len(timestamps) else start_ts
        rel_ts = frame_ts - start_ts

        data = tracker.process_frame(frame, tracker_name)
        if not data:
            continue

        prev_reps = state["reps"]
        state = update_coach_logic(
            state, data, tracker_name,
            threshold_down=thresholds["down"],
            threshold_up=thresholds["up"],
        )

        if state["reps"] > prev_reps:
            good_rep_counter += 1

        if audit_cooldown > 0:
            audit_cooldown -= 1

        fault_type = None

        if is_stretch:
            stretch_timer += 1 / 30
            if stretch_timer >= 5.0 and audit_cooldown == 0:
                stretch_timer = 0.0
                fault_type = "sagging_hips"
        elif (good_rep_counter > 0 and good_rep_counter % 5 == 0
              and not state["is_anomaly"] and audit_cooldown == 0):
            fault_type = "good_form"
            good_rep_counter = 0
        elif state["is_anomaly"] and audit_cooldown == 0:
            fault_type = classify_fault(state, data)
            state["is_anomaly"] = False
            state["consecutive_stuck_frames"] = 0

        if fault_type:
            os.makedirs("audits", exist_ok=True)
            snap = f"audits/snap_{int(time.time()*1000)}.jpg"
            cv2.imwrite(snap, frame)
            result = vlm_auditor(
                image_path=snap,
                exercise_name=tracker_name,
                fault_type=fault_type,
                phase=state["phase"],
                angle=data.get("angle", 0.0),
            )
            fb = result.get("feedback", "")
            if fb:
                pred_feedbacks.append(fb)
                pred_timestamps.append(round(rel_ts, 3))
            audit_cooldown = AUDIT_COOLDOWN_FRAMES

    cap.release()
    return pred_feedbacks, pred_timestamps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate predictions.json from benchmark manifest.")
    parser.add_argument("--manifest", required=True, help="Path to benchmark_manifest.json")
    parser.add_argument("--output",   required=True, help="Where to write predictions.json")
    parser.add_argument("--limit",    type=int, default=None, help="Cap number of segments (for testing)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    segments = load_segment_manifest(args.manifest)
    if args.limit:
        segments = segments[:args.limit]

    print(f"[generate_predictions] {len(segments)} segments to process")
    _get_coach()  # pre-warm RAGCoach encoder
    tracker = PoseTracker()

    predictions = []
    for i, seg in enumerate(segments):
        print(f"  [{i+1}/{len(segments)}] {seg.segment_id}  ({seg.exercise_name})", flush=True)
        feedbacks, ts_list = process_segment(seg, tracker)
        predictions.append({
            "segment_id":              seg.segment_id,
            "video_id":                seg.video_id,
            "exercise_name":           seg.exercise_name,
            "pred_feedbacks":          feedbacks,
            "pred_feedback_timestamps": ts_list,
        })

    save_predictions(predictions, args.output)
    print(f"\n[generate_predictions] Saved {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
