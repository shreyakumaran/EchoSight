import cv2
from simpleFaceRecognition import SimpleFacerec
import pyttsx3
from collections import defaultdict

# Initialize SimpleFacerec
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")  # Load images from the folder

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speed of speech
engine.setProperty('volume', 0.9)  # Set volume (0.0 to 1.0)

# Track previously announced names to avoid repetition
announced_names = defaultdict(int)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)

    # Draw rectangles and display names
    for face_loc, name in zip(face_locations, face_names):
        top, right, bottom, left = face_loc

        # Draw rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

        # Add background for text
        cv2.rectangle(frame, (left, bottom + 10), (right, bottom + 40), (0, 255, 0), cv2.FILLED)

        # Put the name on the rectangle
        cv2.putText(
            frame,
            name,
            (left + 10, bottom + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Convert name to audio and announce it once
        if name not in announced_names or announced_names[name] > 20:  # Prevent frequent repetition
            engine.say(f"Hello, {name}")
            engine.runAndWait()
            announced_names[name] = 0  # Reset counter

        announced_names[name] += 1

    # Display the video frame with boxes and names
    cv2.imshow("Face Recognition", frame)

    # Exit on pressing 'Esc' key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
engine.stop()
