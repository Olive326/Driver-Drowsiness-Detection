# Driver-Drowsiness-Detection
A real-time driver drowsiness detection system that combines **Convolutional Neural Network (CNN)** and **Eye Aspect Ratio (EAR)** method to monitor driver alertness and prevent accidents caused by fatigue.

## Detection Method
### **1. CNN (Convolutional Neural Network)**
CNN is widely used in image feature extraction and pattern recognition; It could detect subtle changes in a driver's facial expression(closed eyes, yawning, head titlting); It could handle variations in lighting conditions, filming angles.
The general pattern 

Here I used a custom-designed CNN for binary classification(the Signmoid at the end outputs 0 or 1). 
<img width="466" height="275" alt="image" src="https://github.com/user-attachments/assets/308dc3f7-8b05-4e1d-95ca-09eee73523f0" />
- Four conv blocks: Enought depth for progressive feature extraction: face/eye
- Filters increase 16-32-64-128: To capture increasingly complex visual patterns; Each blocks include batch normalization for training stability and max pooling for spatial reduction
- Two dropout layers: Extra protection against overfitting
- Sigmoid : Binary classification (Drowsy or Awake)

** What Each Layer Does? **
Conv(3x3 kernel): Slides a small window across the image to detect patterns
MaxPool: Shrinks the image by keeping only the strongest features
BatchNorm:Normalizea values so training is faster and more stable
Dropout: Randomly turns off neurons during training to prevent overfitting
Dense: Fully connected layers that combine all features to make a decision
Sigmoid: Outputs probability between 0 and 1

### **2. EAR (Eye Aspect Ratio)** 
<img width="436" height="175" alt="image" src="https://github.com/user-attachments/assets/d12e1948-edef-4c6a-8591-109accb9cd39" />
- Hard fatigue-state EAR threshold: EAR < 0.2 for more than 20 frames --- Fatigue!
- Adaptive fatigue-state EAR threshold: EAR < 0.4 normal-state EAR for more than 20 frames --- Fatigue!

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
CNN-tflite+EAR2:
- closed eyes detect  
- fatigue detect(2s)
![IMB_j816ie](https://github.com/user-attachments/assets/c327d7a0-d2e0-433c-b351-f10f48624063)
Video Demo:(https://drive.google.com/file/d/11aNNPDIpcoKXOtF-aOYugxB5BTcFCzQn/view?usp=drive_link)
## Deployment On Raspberry Pi
convert model file to Tflite version







