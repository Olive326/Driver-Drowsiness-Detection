# Driver-Drowsiness-Detection
A real-time driver drowsiness detection system that combines **Convolutional Neural Network (CNN)** and **Eye Aspect Ratio (EAR)** method to monitor driver alertness and prevent accidents caused by fatigue.

## Detection Method
### **1. CNN (Convolutional Neural Network)**
CNN is widely used in image feature extraction and pattern recognition; It could detect subtle changes in a driver's facial expression(closed eyes, yawning, head titlting); It could handle variations in lighting conditions, filming angles.
<img width="466" height="275" alt="image" src="https://github.com/user-attachments/assets/308dc3f7-8b05-4e1d-95ca-09eee73523f0" />


### **2. EAR (Eye Aspect Ratio)** 

### **3. Hybrid Approach (CNN + EAR)** 
- Combines strengths of both methods
- CNN for complex pattern recognition
- EAR for precise eye state measurement
- Improved accuracy and reduced false positives

## Dataset
### **Drowsiness Prediction Dataset**
- This dataset includes 4560 Active Subjects and 4560 Fatigue Subjects.
- link: (https://www.kaggle.com/datasets/rakibuleceruet/drowsiness-prediction-dataset/data)

### **Data Preprocessing**

#### Image Rescaling and Augmentation
- Reascaling: Normalize pixel values to [0,1]
- Augmentation: Random zoom, flip, rotation, translation, shear and brightness
- Purpose: Boost model generalization and robustness
  
#### Face Landmark Detection with Dlib
- Used shape_predictor_68_landmarks.dat to detect 68 facial landmarks
- Helps in:
-  Face alignment
-  Feature extraction(eyes, mouth, jawline, etc)
-  Preprocessing input for expression recognition, pose estimation. etc


## Results
- **Results Comparison**
CNN+EAR1:
CNN+EAR2:

## Deployment On Raspberry Pi
convert model file to Tflite version







