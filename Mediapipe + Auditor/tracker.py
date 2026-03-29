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

    def calculate_angle(self, a, b, c):
        """Calculates the angle at point B given points A, B, and C."""
        a = np.array(a) # Hip
        b = np.array(b) # Knee
        c = np.array(c) # Ankle

        # Calculate the angle using arctan2 for better stability
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle

    def process_frame(self, frame):
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # Extract coordinates for Hip (23), Knee (25), and Ankle (27)
            hip = [lm[23].x, lm[23].y]
            knee = [lm[25].x, lm[25].y]
            ankle = [lm[27].x, lm[27].y]

            # Only return data if all three joints are visible
            if lm[23].visibility > 0.5 and lm[25].visibility > 0.5 and lm[27].visibility > 0.5:
                angle = self.calculate_angle(hip, knee, ankle)
                return {
                    "knee_coords": knee, 
                    "angle": angle
                }
                
        return None