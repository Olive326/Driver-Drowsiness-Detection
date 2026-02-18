from scipy.spatial import distance
from imutils import face_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import imutils
import dlib
import cv2
import tensorflow as tf
import numpy as np
import time
import os
from playsound import playsound

# load face detector and 68_face_landmarks feature extraction model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# eye coordinate
LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

#model = tf.keras.models.load_model('fatigue_detection_modern.h5')
# Load TFLite model using TensorFlow
interpreter = tf.lite.Interpreter(model_path="fatigue_detection_lite.tflite")
interpreter.allocate_tensors()
# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("✓ TFLite model loaded!")
print(f"Input shape: {input_details[0]['shape']}")
print(f"Input type: {input_details[0]['dtype']}")
print(f"Output shape: {output_details[0]['shape']}")

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
def predict_with_tflite(interpreter, face_img):
    """Run inference with TFLite model"""
    input_data = np.array(face_img, dtype=np.float32)

    #set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    #run inference
    interpreter.invoke()

    #get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0][0]

# parameters  ADJUSTED FOR 3-4 SECOND DETECTION
EYE_AR_THRESH = 0.2  # EAR threshold
EYE_AR_CONSEC_FRAMES = 60  # activate warning after closing eyes for more than 90 frames（3s）
IMG_SIZE = 145  # CNN input size

frame_counter = 0
start_time = time.time()

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Camera FPS: {fps}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # dlib requires gray pic
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # faces detector
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)  # feature points
        shape_np = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

        left_eye = shape_np[LEFT_EYE]
        right_eye = shape_np[RIGHT_EYE]

        # EAR
        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        avg_EAR = (left_EAR + right_EAR) / 2.0

        # plot eyes zone
        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)

        # CNN preprocessing
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())  # get face region
        face_img = frame[y:y + h, x:x + w]  # cut
        face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE)) / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        cnn_prediction = predict_with_tflite(interpreter, face_img)

        # if cnn predicts as fatigue or ear less than threshold
        if avg_EAR < EYE_AR_THRESH:
            frame_counter += 1

           # if frame_counter >= EYE_AR_CONSEC_FRAMES or cnn_prediction < 0.5:
            if frame_counter >= EYE_AR_CONSEC_FRAMES:
                status = "Fatigue Detected!"
                color = (0, 0, 255)
                # playsound("alarm.wav")
            else:
                seconds_remaining = (EYE_AR_CONSEC_FRAMES - frame_counter) / 30
                status = f"Eyes Closed ({seconds_remaining:.1f}s)"
                color = (255, 165, 0)
        else:
            frame_counter = 0
            status = "Active"
            color = (0, 255, 0)

        # show informations on frame
        cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # results
    cv2.imshow("Fatigue Detection -TFLite", frame)

    # press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()