import cv2
import numpy as np
import winsound  # For Windows

# Define constants
EAR_THRESHOLD = 1.1  # Adjust threshold as needed
CONSEC_FRAMES_DROWSY = 10 # Adjust frames threshold as needed

# Load the pre-trained Haar cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize frame counter, drowsy counter, and drowsy status
frame_count = 0
drowsy_counter = 0
drowsy_status = False

# Beep sound function
def beep():
    duration = 1000  # milliseconds
    frequency = 1000  # Hz
    winsound.Beep(frequency, duration)  # For Windows
    # For other platforms:
    # import os
    # os.system("printf '\a'")

# Function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)

    return ear

# Open video stream (or capture from camera)
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop over the face detections
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Loop over the detected eyes
        for (ex, ey, ew, eh) in eyes:
            eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]

            # Calculate eye aspect ratio (EAR)
            ear = eye_aspect_ratio(eye_roi)
            print(ear)
            # Check if eye aspect ratio is below the threshold
            if ear < EAR_THRESHOLD:
                drowsy_counter += 1
                if drowsy_counter >= CONSEC_FRAMES_DROWSY:
                    if not drowsy_status:
                        drowsy_status = True
                        beep()  # Emit beep sound when eyes are closed
            else:
                drowsy_counter = 0
                drowsy_status = False

            # Draw bounding box around the eye region
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture
video_capture.release()
cv2.destroyAllWindows()