import cv2
import numpy as np


# Define Eye Aspect Ratio (EAR) calculation function
def eye_aspect_ratio(eye):
    # Calculate the Euclidean distances between the eye landmarks
    A = np.linalg.norm(eye[1] - eye[5])  # Vertical distance
    B = np.linalg.norm(eye[2] - eye[4])  # Vertical distance
    C = np.linalg.norm(eye[0] - eye[3])  # Horizontal distance

    # EAR formula
    ear = (A + B) / (2.0 * C)
    return ear


# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Threshold for EAR to indicate drowsiness
EAR_THRESHOLD = 0.2
CONSECUTIVE_FRAMES = 30
frame_count = 0

# Start video capture (camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for better performance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through the faces detected
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Get the region of interest (ROI) for eye detection
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes within the ROI using Hough Circle Transform
        eyes = cv2.HoughCircles(roi_gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=30, param1=50, param2=30, minRadius=10,
                                maxRadius=30)

        if eyes is not None:
            # Convert the coordinates of the circle center to integer
            eyes = np.uint16(np.around(eyes))

            # Loop through the detected circles (eyes)
            for i in range(eyes.shape[1]):
                # Get the center (x, y) and radius (r) of the eye circle
                ex, ey, er = eyes[0][i]
                cv2.circle(roi_color, (ex, ey), er, (0, 255, 0), 2)  # Draw the circle around the eye

                # Select points on the circumference for EAR calculation (simplified)
                eye_points = [
                    np.array([ex, ey + er]),  # Bottom center point
                    np.array([ex, ey - er]),  # Top center point
                    np.array([ex + er, ey]),  # Right center point
                    np.array([ex - er, ey])  # Left center point
                ]

                # Calculate EAR for this eye
                ear = eye_aspect_ratio(np.array(eye_points))

                # If EAR is below the threshold, increment frame count
                if ear < EAR_THRESHOLD:
                    frame_count += 1

                    # If eyes are closed for more than a certain number of frames, alert for drowsiness
                    if frame_count >= CONSECUTIVE_FRAMES:
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    frame_count = 0  # Reset the frame count if the eyes are open

    # Display the resulting frame with annotations
    cv2.imshow("Drowsiness Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
