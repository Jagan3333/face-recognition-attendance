
import cv2
import os
import numpy as np

dataset_folder = "dataset"

faces = []
labels = []

label_id = 0
label_names = {}

for filename in os.listdir(dataset_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(dataset_folder, filename)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        faces.append(image)

        name = filename.rsplit("_", 1)[0]
        if name not in label_names:
            label_names[name] = label_id
            label_id += 1

        labels.append(label_names[name])

if len(faces) == 0:
    print("No face photos found in the dataset folder.")
else:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    recognizer.save("face_model.yml")

    with open("labels.txt", "w") as file:
        for name, label in label_names.items():
            file.write(f"{label},{name}\n")

    print("Training completed successfully!")
    print("Photos used:", len(faces))
    print("Model saved as face_model.yml")
