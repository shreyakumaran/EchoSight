import cv2
import numpy as np
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust the speaking rate
engine.setProperty('volume', 1.0)  # Set volume level (0.0 to 1.0)

# Load YOLOv4-Tiny weights and config
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize webcam
cap = cv2.VideoCapture(0)

# Keep track of spoken labels to avoid repetition
spoken_labels = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare input for YOLO
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    # Process detections
    current_frame_labels = set()
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Adjust this threshold as needed
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                label = f"{classes[class_id]}: {int(confidence * 100)}%"
                current_frame_labels.add(classes[class_id])  # Add to current frame labels
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Provide audio feedback for new labels
    new_labels = current_frame_labels - spoken_labels
    for label in new_labels:
        engine.say(f"Detected {label}")
        engine.runAndWait()

    # Update spoken labels
    spoken_labels.update(new_labels)

    # Display output
    cv2.imshow("Echo Sight - YOLOv4-Tiny", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
