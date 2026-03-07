from scipy.spatial import distance
import dlib
import cv2
import sys
import tensorflow as tf
import numpy as np
import time
import subprocess
import gpiod # for LED Blinky
import time
import threading
from gpiod.line import Direction, Value

# led config
LED_PIN = 17
BLINK_HZ = 2
GPIO_CHIP   = "/dev/gpiochip4"  

# LED controller
class LEDController:
    def __init__(self, chip_path, pin, hz=2):
        self.chip_path = chip_path
        self.pin = pin
        self.interval = 1.0 / (hz * 2)   # half-period
        self._blink   = False
        self._thread  = None

          # v2.x: request the line via context-managed request object
        self._req = gpiod.request_lines(
            chip_path,
            consumer="LED",
            config={pin: gpiod.LineSettings(direction=Direction.OUTPUT,
                                            output_value=Value.INACTIVE)}
        ) 

    def _run(self):
        state = False
        while self._blink:
            state = not state
            self._req.set_value(self.pin, Value.ACTIVE if state else Value.INACTIVE)
            time.sleep(self.interval)
        self._req.set_value(self.pin, Value.INACTIVE) # LED OFF when done
        
    
    def start_blink(self):
        if self._blink:               # already blinking, do nothing
            return
        self._blink  = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop_blink(self):
        if not self._blink:           # already stopped, do nothing
            return
        self._blink = False
        if self._thread:
            self._thread.join(timeout=1)

    def cleanup(self):
        self.stop_blink()
        self._req.set_value(self.pin, Value.INACTIVE)
        self._req.release()           # <-- note: release() not release


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def predict_with_tflite(interpreter, input_details, output_details, face_img):
    input_data = np.array(face_img, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0][0]

def main():
    print("="*60)
    print("OPTIMIZED Raspberry Pi Camera Drowsiness Detection")
    print("="*60)

    # LED setup
    led = LEDController(chip_path=GPIO_CHIP, pin=LED_PIN, hz=BLINK_HZ)

    # Load face detector and landmarks
    print("Loading models...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    print("✓ Face detector loaded")

    # Eye coordinates
    LEFT_EYE = list(range(42, 48))
    RIGHT_EYE = list(range(36, 42))

    # Parameters
    EYE_AR_THRESH = 0.2
    EYE_AR_CONSEC_FRAMES = 90
    IMG_SIZE = 145

    frame_counter = 0
    frame_count = 0
    fatigue_active = False # track current state

    # Load TFLite model
    try:
        import tflite_runtime.interpreter as tflite
        print("✓ Using TFLite Runtime")
    except ImportError:
        tflite = tf.lite
        print("✓ Using TensorFlow Lite")

    interpreter = tflite.Interpreter(model_path="fatigue_detection_lite.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✓ Model loaded")

    # ===== OPTIMIZATION SETTINGS =====
    WIDTH = 480   # Reduced from 640
    HEIGHT = 360  # Reduced from 480
    FPS = 30
    
    FACE_DETECT_INTERVAL = 5  # Detect face every 5 frames (big speedup!)
    CNN_SKIP = True  # Only run CNN when needed
    
    print("\n⚡ Optimizations enabled:")
    print(f"  - Resolution: {WIDTH}x{HEIGHT}")
    print(f"  - Face detection every {FACE_DETECT_INTERVAL} frames")
    print(f"  - Smart CNN skipping: {CNN_SKIP}")

    # ===== CAMERA SETUP =====
    RPICAM_CMD = [
        'rpicam-vid',
        '--width', str(WIDTH),
        '--height', str(HEIGHT),
        '--framerate', str(FPS),
        '--timeout', '0',
        '--codec', 'mjpeg',
        '-o', '-',
 
    ]

    print("\nStarting camera...")
    try:
        cam = subprocess.Popen(
            RPICAM_CMD,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**8
        )
    except FileNotFoundError:
        print("Trying libcamera-vid...")
        RPICAM_CMD[0] = 'libcamera-vid'
        cam = subprocess.Popen(
            RPICAM_CMD,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**8
        )

    print("✓ Camera started")
    time.sleep(2)
    print("Press Ctrl+C to stop\n")

    start_time = time.time()
    jpeg_buffer = b''
    
    # Optimization variables
    last_face = None
    last_shape = None
    
    try:
        while True:
            # Read data
            chunk = cam.stdout.read(8192)  # Larger chunks
            if not chunk:
                break
            
            jpeg_buffer += chunk
            
            # Find JPEG frame
            start_marker = b'\xff\xd8'
            end_marker = b'\xff\xd9'
            
            start = jpeg_buffer.find(start_marker)
            end = jpeg_buffer.find(end_marker)
            
            if start != -1 and end != -1 and end > start:
                jpeg_data = jpeg_buffer[start:end + 2]
                jpeg_buffer = jpeg_buffer[end + 2:]
                
                frame = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                
                frame_count += 1
                
                # ===== OPTIMIZATION 1: Skip face detection on most frames =====
                if frame_count % FACE_DETECT_INTERVAL == 0 or last_face is None:
                    # Run full face detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = detector(gray)
                    
                    if len(faces) > 0:
                        last_face = faces[0]
                        # Get landmarks
                        shape = predictor(gray, last_face)
                        last_shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
                    else:
                        last_face = None
                        last_shape = None
                else:
                    # Reuse last detection (much faster!)
                    faces = [last_face] if last_face is not None else []

                status = "No Face"
                color = (100, 100, 100)

                if last_face is not None and last_shape is not None:
                    # Use cached shape or detect new one if needed
                    if frame_count % FACE_DETECT_INTERVAL != 0:
                        shape_np = last_shape
                    else:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        shape = predictor(gray, last_face)
                        shape_np = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
                        last_shape = shape_np

                    left_eye = shape_np[LEFT_EYE]
                    right_eye = shape_np[RIGHT_EYE]

                    left_EAR = eye_aspect_ratio(left_eye)
                    right_EAR = eye_aspect_ratio(right_eye)
                    avg_EAR = (left_EAR + right_EAR) / 2.0

                    # ===== OPTIMIZATION 2: Smart CNN usage =====
                    # Only run CNN when eyes are suspicious
                    if CNN_SKIP and avg_EAR > EYE_AR_THRESH * 1.2:
                        # Eyes clearly open, skip CNN
                        cnn_prediction = 1.0
                    else:
                        # Eyes closing or suspicious, run CNN
                        x, y, w, h = (last_face.left(), last_face.top(), 
                                     last_face.width(), last_face.height())
                        
                        x = max(0, x)
                        y = max(0, y)
                        w = min(w, frame.shape[1] - x)
                        h = min(h, frame.shape[0] - y)
                        
                        if w > 0 and h > 0:
                            face_img = frame[y:y + h, x:x + w]
                            face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE)) / 255.0
                            face_img = np.expand_dims(face_img, axis=0)
                            cnn_prediction = predict_with_tflite(interpreter, input_details, output_details, face_img)
                        else:
                            cnn_prediction = 1.0

                    # Detection logic
                    if avg_EAR < EYE_AR_THRESH:
                        frame_counter += 1
                        
                        if frame_counter >= EYE_AR_CONSEC_FRAMES:
                            status = "⚠️  FATIGUE!"
                            color = (0, 0, 255)
                            if not fatigue_active:
                                fatigue_active = True
                                led.start_blink()
                                print(f"\n{status} EAR: {avg_EAR:.2f}, CNN:{cnn_prediction:.2f}\n")
               
                        else:
                            seconds_remaining = (EYE_AR_CONSEC_FRAMES - frame_counter) / FPS
                            status = f"Eyes Closed ({seconds_remaining:.1f}s)"
                            color = (255, 165, 0)
                    else:
                        frame_counter = 0
                        if fatigue_active:
                            fatigue_active = False
                            led.stop_blink()
                        status = "Active"
                        color = (0, 255, 0)

                # Show FPS every 30 frames
                if frame_count % 30 == 0:
                    fps_calc = frame_count / (time.time() - start_time)
                    print(f"[{frame_count:5d}] Status: {status:30s} | FPS: {fps_calc:5.1f}")
                
                # Clean buffer periodically
                if len(jpeg_buffer) > 1000000:
                    jpeg_buffer = jpeg_buffer[-100000:]

    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped by user")
    
    finally:
        try:
            cam.terminate()
            cam.wait(timeout=2)
        except:
            try:
                cam.kill()
            except:
                pass
        
        elapsed_time = time.time() - start_time
        avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"Session Statistics:")
        print(f"{'='*60}")
        print(f"  Total frames: {frame_count}")
        print(f"  Total time: {elapsed_time:.1f}s")
        print(f"  Average FPS: {avg_fps:.1f}")
        print(f"  Target: 25+ FPS")
        print(f"  Status: {'✅ GOOD' if avg_fps >= 25 else '⚠️ NEEDS OPTIMIZATION'}")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    main()