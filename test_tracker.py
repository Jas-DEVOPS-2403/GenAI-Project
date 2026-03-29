# Run pip install opencv-python mediapipe
# Uninstall the current version
# pip uninstall mediapipe -y
# Install a compatible stable version (0.10.14 or 0.10.9)
# pip install mediapipe==0.10.14


import cv2
from tracker import PoseTracker

def main():
    # Initialize the tracker you built
    tracker = PoseTracker()
    
    # Open the default webcam (index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Testing Tracker... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame to get the dictionary of data
        data = tracker.process_frame(frame)

        if data:
            # Extract values from the dictionary
            angle = data["angle"]
            knee_coords = data["knee_coords"]
            
            # Print angle to console for debugging
            print(f"Knee Angle: {int(angle)} degrees")
            
            h, w, _ = frame.shape
            cx, cy = int(knee_coords[0] * w), int(knee_coords[1] * h)

            # Visual feedback: Draw the angle text near the knee
            cv2.putText(frame, str(int(angle)), 
                        (cx + 10, cy - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

        # Show the video feed
        cv2.imshow("Tracker Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()