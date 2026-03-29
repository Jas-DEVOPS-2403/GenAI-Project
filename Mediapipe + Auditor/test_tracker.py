# Run pip install opencv-python mediapipe
# Uninstall the current version
# pip uninstall mediapipe -y
# Install a compatible stable version (0.10.14 or 0.10.9)
# pip install mediapipe==0.10.14


import cv2
import time
import os
from tracker import PoseTracker
from update_coach_logic import update_coach_logic
from auditor import vlm_auditor

def trigger_vlm_audit(frame, state):
    """Saves visual evidence for the VLM specialist."""
    if not os.path.exists("audits"):
        os.makedirs("audits")
    
    timestamp = int(time.time())
    filename = f"audits/anomaly_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"--- VLM AUDIT TRIGGERED: Snapshot saved to {filename} ---")
    return filename

def main():
    tracker = PoseTracker()
    # Updated state to hold the VLM's feedback
    state = {
        "reps": 0, 
        "phase": "up", 
        "is_anomaly": False, 
        "consecutive_stuck_frames": 0,
        "vlm_feedback": "" 
    }
    
    cap = cv2.VideoCapture(0)

    print("Agentic Tracker Running... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        data = tracker.process_frame(frame)

        if data:
            angle = data["angle"]
            knee_coords = data["knee_coords"]
            
            # 1. PERCEPTION & REASONING: Update movement state
            state = update_coach_logic(state, angle)
            
            # 2. AGENTIC BRIDGE: Trigger RAG Coach if Anomaly Found
            if state["is_anomaly"]:
                # Capture the image (kept for logging / future VLM upgrade)
                snapshot_path = trigger_vlm_audit(frame, state)

                # Classify fault type from angle so the RAG retrieval is targeted
                if angle < 80:
                    fault_type = "stuck"          # deep but can't drive up
                elif angle < 110:
                    fault_type = "shallow_depth"  # not hitting depth
                else:
                    fault_type = "knee_valgus"    # up-phase issue

                # Call the RAG Coach — retrieves cues + calls LLM
                audit_result = vlm_auditor(
                    snapshot_path, "squat",
                    fault_type=fault_type,
                    phase=state["phase"],
                    angle=angle
                )

                # Update state with grounded feedback
                state["vlm_feedback"] = audit_result["feedback"]

                # Reset triggers to return to the Fast Loop
                state["is_anomaly"] = False
                state["consecutive_stuck_frames"] = 0

            # 3. ACTION: Visual Overlays
            h, w, _ = frame.shape
            cx, cy = int(knee_coords[0] * w), int(knee_coords[1] * h)
            
            # Display Reps and Phase
            cv2.putText(frame, f"Reps: {state['reps']} | Phase: {state['phase']}", 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Display VLM Feedback if it exists
            if state["vlm_feedback"]:
                cv2.putText(frame, f"COACH: {state['vlm_feedback']}", 
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Display Angle and Knee Marker
            cv2.putText(frame, str(int(angle)), (cx + 10, cy - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

        cv2.imshow("Agentic Fitness Prototype", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()