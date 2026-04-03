import cv2
import mediapipe as mp
import numpy as np

class PoseTracker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def get_exercise_config(self, exercise_name):
        """
        Maps all 23 Table 5 exercises to joint IDs and logic types.
        Angle-based exercises use 6 joints: [L_A, L_B, L_C, R_A, R_B, R_C]
        for bilateral averaging. Single-side exercises keep 3 joints.
        """
        configs = {
            # --- WARM-UP ---
            "jumping_jacks": {"joints": [15, 16, 27, 28, 0], "type": "spatial", "label": "Width/Height"},
            "high_knees":    {"joints": [23, 25, 24, 26],    "type": "height",  "label": "Knee Drive"},
            "butt_kickers":  {"joints": [25, 27, 26, 28],    "type": "height",  "label": "Heel Drive"},
            "air_jump_rope": {"joints": [15, 16],             "type": "spatial", "label": "Wrist Circle"},
            "good_mornings": {"joints": [11, 23, 25, 12, 24, 26], "type": "angle", "label": "Hip Hinge"},

            # --- MAIN WORKOUT ---
            "push-ups":             {"joints": [11, 13, 15, 12, 14, 16],     "type": "angle",   "label": "Elbow Angle"},
            "plank_taps":           {"joints": [15, 16],                      "type": "spatial", "label": "Hand Tap"},
            "moving_plank":         {"joints": [11, 23, 25, 12, 24, 26],     "type": "angle",   "label": "Core Flatness"},
            "squats":               {"joints": [23, 25, 27, 24, 26, 28],     "type": "angle",   "label": "Knee Angle"},
            "walking_lunges":       {"joints": [23, 25, 27, 24, 26, 28],     "type": "angle",   "label": "Lead Knee"},
            "lunge_jumps":          {"joints": [23, 25, 27, 24, 26, 28],     "type": "angle",   "label": "Explosive Depth"},
            "puddle_jumps":         {"joints": [23, 24],                      "type": "spatial", "label": "Lateral Shift"},
            "mountain_climbers":    {"joints": [13, 25, 14, 26],              "type": "spatial", "label": "Knee-to-Elbow"},
            "floor_touches":        {"joints": [15, 27],                      "type": "spatial", "label": "Reach"},
            "quick_feet":           {"joints": [27, 28],                      "type": "height",  "label": "Foot Cadence"},
            "squat_jumps":          {"joints": [23, 25, 27, 24, 26, 28],     "type": "angle",   "label": "Jump Depth"},
            "squat_kicks":          {"joints": [23, 25, 27, 24, 26, 28],     "type": "angle",   "label": "Knee Flexion"},
            "standing_kicks":       {"joints": [23, 25],                      "type": "height",  "label": "Kick Height"},
            "boxing_squat_punches": {"joints": [23, 25, 27, 24, 26, 28],     "type": "angle",   "label": "Squat-Punch"},

            # --- COOL-DOWN ---
            "deltoid_stretch":  {"joints": [13, 14, 11, 12], "type": "vlm_only", "label": "Shoulder Hold"},
            "quad_stretch":     {"joints": [25, 27, 26, 28], "type": "vlm_only", "label": "Balance Hold"},
            "shoulder_gators":  {"joints": [13, 14],          "type": "spatial",  "label": "Arm Opening"},
            "toe_touchers":     {"joints": [15, 27, 16, 28],  "type": "spatial",  "label": "Reach"},
        }
        return configs.get(exercise_name, configs["squats"])

    def calculate_angle(self, a, b, c):
        """Calculates the angle at point B given points A, B, and C."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180.0 else angle

    def process_frame(self, frame, exercise_name="squats"):
        """Processes frame and extracts the relevant metric for all 23 exercises."""
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            return None

        lm = results.pose_landmarks.landmark
        config = self.get_exercise_config(exercise_name)

        # 1. ANGLE-BASED (bilateral where 6 joints provided)
        if config["type"] == "angle":
            joints = config["joints"]
            left_ids = joints[:3]
            right_ids = joints[3:6] if len(joints) >= 6 else None

            left_vis = all(lm[i].visibility > 0.5 for i in left_ids)
            right_vis = right_ids is not None and all(lm[i].visibility > 0.5 for i in right_ids)

            angles = []
            coords = []
            if left_vis:
                p1, p2, p3 = [[lm[i].x, lm[i].y] for i in left_ids]
                angles.append(self.calculate_angle(p1, p2, p3))
                coords.append([lm[left_ids[1]].x, lm[left_ids[1]].y])
            if right_vis:
                p1, p2, p3 = [[lm[i].x, lm[i].y] for i in right_ids]
                angles.append(self.calculate_angle(p1, p2, p3))
                coords.append([lm[right_ids[1]].x, lm[right_ids[1]].y])

            if not angles:
                return None

            avg_angle = sum(angles) / len(angles)
            return {
                "angle": avg_angle,
                "knee_coords": coords[0],       # primary dot (backwards-compat)
                "all_knee_coords": coords,       # all visible side coords for multi-dot display
                "label": config["label"],
            }

        # 2. SPATIAL-BASED
        elif config["type"] == "spatial":
            if exercise_name == "jumping_jacks":
                hand_y = (lm[15].y + lm[16].y) / 2
                return {
                    "hand_y_diff": lm[0].y - hand_y,
                    "foot_distance": np.abs(lm[27].x - lm[28].x),
                    "landmark_coords": [         # wrists + ankles for display dots
                        [lm[15].x, lm[15].y],
                        [lm[16].x, lm[16].y],
                        [lm[27].x, lm[27].y],
                        [lm[28].x, lm[28].y],
                    ],
                    "label": config["label"],
                }
            else:
                p1, p2 = lm[config["joints"][0]], lm[config["joints"][1]]
                dist = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
                return {"dist_val": dist, "label": config["label"]}

        # 3. HEIGHT-BASED
        elif config["type"] == "height":
            j = config["joints"]
            l_val = lm[j[0]].y - lm[j[1]].y
            r_val = lm[j[2]].y - lm[j[3]].y if len(j) > 2 else l_val
            landmark_coords = [
                [lm[j[1]].x, lm[j[1]].y],
                [lm[j[3]].x, lm[j[3]].y] if len(j) > 2 else None,
            ]
            return {
                "height_val": max(l_val, r_val),
                "landmark_coords": landmark_coords,
                "label": config["label"],
            }

        # 4. VLM ONLY (isometric holds — no metric, time-driven from test_tracker)
        elif config["type"] == "vlm_only":
            return {"label": config["label"], "status": "holding"}

        return None
