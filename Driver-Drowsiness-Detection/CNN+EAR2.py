from scipy.spatial import distance
from imutils import face_utils
import dlib
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import numpy as np
from playsound import playsound
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

# CORRECT - no compile parameter for dlib
#predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# CORRECT - compile=False only for TensorFlow
# This works with TensorFlow 2.13!
model = Sequential()
# First convolution
model.add(Conv2D(16, 3, activation='relu', input_shape=(145, 145, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D())

# Second convolution
model.add(Conv2D(32, 5, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

# Third convolution
model.add(Conv2D(64, 10, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

# Fourth convolution
model.add(Conv2D(128, 12, activation='relu'))
model.add(BatchNormalization())

# Flatten and Dense layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

# Output layer
model.add(Dense(1, activation='sigmoid'))

# Compile
model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")

print("Model architecture built successfully!")
model.summary()

# Load the weights from your .h5 file
try:
    model.load_weights("fatigue_detection_CNN.h5")
    print("\n✓ Model weights loaded successfully!")

    # Save in modern format for future use
    model.save("fatigue_detection_modern.h5")
    print("✓ Saved as fatigue_detection_modern.h5")

except Exception as e:
    print(f"\n✗ Error loading weights: {e}")


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


init_ear_list = []
dynamic_threshold = None
EYE_AR_CONSEC_FRAMES = 20
IMG_SIZE = 145
INIT_FRAMES = 40
EAR_SCALE = 0.5  # EAR threshold is 40% of baseline EAR

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # dlib requires gray pic
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # faces detector
    faces = detector(gray)

    status = "No Face Detected"
    color = (100, 100, 100)

    for face in faces:
        shape = predictor(gray, face)
        shape_np = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

        left_eye = shape_np[LEFT_EYE]
        right_eye = shape_np[RIGHT_EYE]

        avg_EAR = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

        cv2.polylines(frame, [left_eye.reshape(-1, 1, 2)], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye.reshape(-1, 1, 2)], True, (0, 255, 0), 1)

        if dynamic_threshold is None:
            if len(init_ear_list) < INIT_FRAMES:
                init_ear_list.append(avg_EAR)
                status = f"Collecting initial EAR... ({len(init_ear_list)}/{INIT_FRAMES})"
                color = (255, 255, 0)
            else:
                baseline_ear = np.mean(init_ear_list)
                dynamic_threshold = baseline_ear * EAR_SCALE
                frame_counter = 0
                status = "Active"
                print(f"[INFO] Dynamic EAR threshold set to {dynamic_threshold:.3f}")
                continue  # skip one frame before starting detection
        else:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            face_img = frame[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE)) / 255.0
            face_img = np.expand_dims(face_img, axis=0)
            cnn_prediction = (model.predict(face_img, verbose=0))[0][0]

            if avg_EAR < dynamic_threshold:
                frame_counter += 1

            else:
                frame_counter = 0

            if frame_counter >= EYE_AR_CONSEC_FRAMES or cnn_prediction < 0.5:
                status = "Fatigue Detected!"
                color = (0, 0, 255)
                # playsound("alarm.wav")
            else:
                status = "Active"
                color = (0, 255, 0)

        break  # only process one face for now

    cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Fatigue Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()