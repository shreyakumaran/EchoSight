import pyttsx3

engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

def provide_audio_feedback(data):
    feedback_text = ""

    # Parsing object detection data
    if "objects" in data and data["objects"]:
        for obj in data["objects"]:
            name = obj.get("name", "unknown object")
            distance = obj.get("distance", "an unknown distance")
            direction = obj.get("direction", "unknown direction")
            feedback_text += f"There is a {name} {distance} ahead on your {direction}. "

    # Parsing face recognition data
    if "faces" in data and data["faces"]:
        for face in data["faces"]:
            feedback_text += f"{face} is nearby. "

    # Default message if no data
    if not feedback_text:
        feedback_text = "No objects or faces detected."

    # Convert to speech
    engine.say(feedback_text)
    engine.runAndWait()

# Sample
sample_data = {
    "objects": [
        {"name": "chair", "distance": "2 meters", "direction": "left"},
        {"name": "bottle", "distance": "1 meter", "direction": "right"}
    ],
    "faces": ["John"]
}
provide_audio_feedback(sample_data)
