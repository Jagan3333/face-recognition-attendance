
import cv2
import time
from attendance import mark_attendance

# Load the trained face recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")

# Load registered names
names = {}

with open("labels.txt", "r") as file:
    for line in file:
        label, name = line.strip().split(",", 1)
        names[int(label)] = name

# Face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not working")
    exit()

print("Face recognition started!")
print("Attendance will be marked automatically.")
print("Press Q to quit.")

# Avoid repeated attendance entries for the same person
# during this camera session
marked_names = set()
last_marked_time = {}
attendance_cooldown = 10

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5
    )

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]

        label, confidence = recognizer.predict(face)

        if confidence < 80 and label in names:
            name = names[label]
            color = (0, 255, 0)

            current_time = time.time()

            if (
                name not in last_marked_time
                or current_time - last_marked_time[name]
                >= attendance_cooldown
            ):
                mark_attendance(name)
                last_marked_time[name] = current_time
                marked_names.add(name)

            status = "Recognized"
        else:
            name = "Unknown Person"
            color = (0, 0, 255)
            status = "Not Registered"

        cv2.rectangle(
            frame, (x, y), (x+w, y+h), color, 2
        )

        cv2.putText(
            frame, name, (x, y-30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

        cv2.putText(
            frame, status, (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )

    cv2.putText(
        frame,
        f"People marked present: {len(marked_names)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.imshow("Face Recognition Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Face recognition stopped.")
print("People marked present:", len(marked_names))
