import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import math
import pyttsx3
import winsound
import time
import threading

# ================= VOICE SETUP =================

engine = pyttsx3.init()

engine.setProperty('rate', 170)
engine.setProperty('volume', 1)

last_alert_time = 0
alert_delay = 4   # seconds


# ================= SOUND FUNCTION =================

def speak(text):
    def run():
        engine.say(text)
        engine.runAndWait()

    t = threading.Thread(target=run)
    t.start()


def beep():
    winsound.Beep(1200, 600)


# ================= LOAD MODEL =================

model = load_model("model/mask_detector.h5")


# ================= FACE DETECTOR =================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ================= CAMERA =================

cap = cv2.VideoCapture(0)

IMG_SIZE = 224
DISTANCE_THRESHOLD = 150


# ================= MAIN LOOP =================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5
    )

    centers = []

    current_time = time.time()

    for (x, y, w, h) in faces:

        # -------- FACE PREPROCESS --------

        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (224, 224))
        face_img = face_img / 255.0
        face_img = np.reshape(face_img, (1, 224, 224, 3))

        # -------- PREDICTION --------

        pred = model.predict(face_img, verbose=0)[0][0]


        # -------- MASK / NO MASK --------

        if pred < 0.5:

            label = "Mask"
            color = (0, 255, 0)

            if current_time - last_alert_time > alert_delay:

                speak("Green zone. Thank you for wearing mask")
                last_alert_time = current_time

        else:

            label = "No Mask"
            color = (0, 0, 255)

            if current_time - last_alert_time > alert_delay:

                beep()
                speak("Red zone. Please wear your mask")
                last_alert_time = current_time


        # -------- CENTER POINT --------

        cx = x + w // 2
        cy = y + h // 2

        centers.append((cx, cy))


        # -------- DRAW BOX --------

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        cv2.putText(frame, label,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2)


    # ================= DISTANCE CHECK =================

    for i in range(len(centers)):
        for j in range(i+1, len(centers)):

            dist = math.dist(centers[i], centers[j])

            if dist < DISTANCE_THRESHOLD:

                cv2.putText(frame,
                            "Too Close!",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            3)

                beep()


    # ================= SHOW WINDOW =================

    cv2.imshow("Face Mask & Distance Alert System", frame)


    # ================= QUIT WITH Q =================

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ================= CLEANUP =================

cap.release()
cv2.destroyAllWindows()


