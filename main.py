
import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path

# Project folder
BASE_DIR = Path(__file__).resolve().parent

# Load face detection model
cascade_path = BASE_DIR / "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(str(cascade_path))

if face_cascade.empty():
    raise FileNotFoundError(
        "Haarcascade XML file nahi mili! "
        "XML file ko main.py ke same folder mein rakho."
    )

# Load trained emotion model
model_path = BASE_DIR / "improved_emotion_model.keras"
emotion_model = load_model(str(model_path))

# Emotion labels (training class order)
emotion_labels = [
    "Angry", "Disgust", "Fear", "Happy",
    "Neutral", "Sad", "Surprise"
]

# Emotion counts
emotion_counts = {label: 0 for label in emotion_labels}


def show_graph():
    """Display emotion statistics after the camera closes."""
    total = sum(emotion_counts.values())

    if total == 0:
        messagebox.showinfo(
            "Result",
            "Koi emotion data nahi mila. Camera dobara check karo."
        )
        return

    labels = list(emotion_counts.keys())
    counts = list(emotion_counts.values())
    percentages = [count / total * 100 for count in counts]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, percentages)
    plt.title("Facial Emotion Detection Statistics")
    plt.xlabel("Emotions")
    plt.ylabel("Percentage (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=20)

    for bar, value in zip(bars, percentages):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.1f}%",
            ha="center"
        )

    plt.tight_layout()
    plt.show()


def start_camera():
    """Open webcam and detect faces and facial expressions."""
    # Reset previous session counts
    for label in emotion_labels:
        emotion_counts[label] = 0

    # Use DirectShow backend on Windows
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Try the second camera if the default camera fails
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap.release()
        messagebox.showerror(
            "Camera Error",
            "Camera open nahi ho raha. Camera permission aur connection check karo."
        )
        return

    # Request a common webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    messagebox.showinfo(
        "Instructions",
        "Camera start ho raha hai.\n\n"
        "Q = Quit\n"
        "S = Screenshot"
    )

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("Camera se frame nahi mil raha.")
            break

        # Mirror view
        frame = cv2.flip(frame, 1)

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        # Process every detected face
        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]

            # Prepare image for model
            face = cv2.resize(face, (48, 48))
            face = face.astype("float32")
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=-1)

            # Predict emotion
            predictions = emotion_model.predict(
                face, verbose=0
            )[0]

            emotion_index = int(np.argmax(predictions))
            emotion = emotion_labels[emotion_index]
            confidence = float(predictions[emotion_index]) * 100

            # Update emotion count
            emotion_counts[emotion] += 1

            # Draw face rectangle
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Draw emotion and confidence
            label = f"{emotion}: {confidence:.1f}%"
            cv2.putText(
                frame,
                label,
                (x, max(y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # Display face count and instructions
        cv2.putText(
            frame,
            f"Faces: {len(faces)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Q: Quit | S: Screenshot",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # Show camera
        cv2.imshow("Emotion Detection System", frame)

        key = cv2.waitKey(1) & 0xFF

        # Save screenshot
        if key == ord("s"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = BASE_DIR / f"screenshot_{timestamp}.jpg"

            if cv2.imwrite(str(screenshot_path), frame):
                print(f"Screenshot saved: {screenshot_path}")
            else:
                print("Screenshot save nahi ho paya.")

        # Quit camera
        if key == ord("q") or key == 27:
            break

    # Release camera resources
    cap.release()
    cv2.destroyAllWindows()

    # Display statistics
    show_graph()


# Main GUI
root = tk.Tk()
root.title("Emotion Detection and Facial Expression Recognition")
root.geometry("550x400")
root.resizable(False, False)
root.configure(bg="#202A44")

title_label = tk.Label(
    root,
    text="EMOTION DETECTION",
    font=("Arial", 22, "bold"),
    fg="white",
    bg="#202A44"
)
title_label.pack(pady=30)

subtitle_label = tk.Label(
    root,
    text="Facial Expression Recognition System",
    font=("Arial", 12),
    fg="#D6E4FF",
    bg="#202A44"
)
subtitle_label.pack(pady=5)

start_button = tk.Button(
    root,
    text="START CAMERA",
    font=("Arial", 14, "bold"),
    bg="#27AE60",
    fg="white",
    width=20,
    height=2,
    command=start_camera
)
start_button.pack(pady=35)

exit_button = tk.Button(
    root,
    text="EXIT",
    font=("Arial", 12, "bold"),
    bg="#C0392B",
    fg="white",
    width=15,
    command=root.destroy
)
exit_button.pack(pady=10)

footer_label = tk.Label(
    root,
    text="Developed using Python, OpenCV and CNN",
    font=("Arial", 9),
    fg="#D6E4FF",
    bg="#1A4DCE"
)
footer_label.pack(side="bottom", pady=15)

root.mainloop()