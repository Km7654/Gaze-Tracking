import cv2
import numpy as np
import pyautogui
import time
import mediapipe as mp
import joblib
import os
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)

# Define eye landmarks
LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]

# Blink detection constants
EYE_AR_THRESHOLD = 0.25
EYE_AR_CONSEC_FRAMES = 2
BLINK_COUNTER = 0
BLINK_TOTAL = 0
LAST_BLINK_TIME = 0
CLICK_COOLDOWN = 1.5
MODEL_FILE = 'eye_tracker_model.joblib'

def eye_aspect_ratio(eye_points):
    """Calculate eye aspect ratio for blink detection"""
    vertical1 = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
    vertical2 = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
    horizontal = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
    return (vertical1 + vertical2) / (2.0 * horizontal)

def get_eye_centers(landmarks, frame_shape):
    """Get precise eye centers using MediaPipe landmarks"""
    centers = []
    for eye_indices in [LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS]:
        eye_points = []
        for idx in eye_indices:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * frame_shape[1])
            y = int(landmark.y * frame_shape[0])
            eye_points.append((x, y))
        
        centroid = np.mean(eye_points, axis=0).astype(int)
        centers.append(tuple(centroid))
    return centers

def detect_blinks(landmarks, frame_shape):
    """Detect blinks and trigger click actions"""
    global BLINK_COUNTER, BLINK_TOTAL, LAST_BLINK_TIME
    
    left_eye_points = [(int(landmarks.landmark[idx].x * frame_shape[1]),
                       int(landmarks.landmark[idx].y * frame_shape[0])) 
                      for idx in LEFT_EYE_LANDMARKS]
    
    right_eye_points = [(int(landmarks.landmark[idx].x * frame_shape[1]),
                        int(landmarks.landmark[idx].y * frame_shape[0])) 
                       for idx in RIGHT_EYE_LANDMARKS]
    
    ear_left = eye_aspect_ratio(left_eye_points)
    ear_right = eye_aspect_ratio(right_eye_points)
    ear_avg = (ear_left + ear_right) / 2.0

    current_time = time.time()
    
    if ear_avg < EYE_AR_THRESHOLD:
        BLINK_COUNTER += 1
    else:
        if BLINK_COUNTER >= EYE_AR_CONSEC_FRAMES:
            BLINK_TOTAL += 1
            LAST_BLINK_TIME = current_time
        BLINK_COUNTER = 0

    # Handle click actions
    if current_time - LAST_BLINK_TIME > 0.4:  # 400ms blink sequence timeout
        if BLINK_TOTAL == 3:
            pyautogui.click()
            print("Left click triggered!")
            BLINK_TOTAL = 0
            LAST_BLINK_TIME = current_time + CLICK_COOLDOWN
        elif BLINK_TOTAL == 4:
            pyautogui.rightClick()
            print("Right click triggered!")
            BLINK_TOTAL = 0
            LAST_BLINK_TIME = current_time + CLICK_COOLDOWN
        elif BLINK_TOTAL > 0 and current_time - LAST_BLINK_TIME > 1.0:
            BLINK_TOTAL = 0

    return ear_avg

def calibrate_and_save_model():
    """Perform one-time calibration and save the model"""
    cap = cv2.VideoCapture(0)
    screen_w, screen_h = pyautogui.size()
    X_calib, y_calib = [], []
    
    calib_points = [(x/6, y/4) for x in range(1, 6) for y in range(1, 4)]
    calib_points += [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
    calib_points += [(x/12, y/8) for x in range(1, 12) for y in range(1, 8)]

    try:
        for rel_x, rel_y in calib_points:
            target_x = int(rel_x * screen_w)
            target_y = int(rel_y * screen_h)
            
            target_x = np.clip(target_x, int(0.03*screen_w), int(0.97*screen_w))
            target_y = np.clip(target_y, int(0.03*screen_h), int(0.97*screen_h))
            
            pyautogui.moveTo(target_x, target_y)
            print(f"Look at ({target_x}, {target_y}) for 7 seconds...")
            
            start_time = time.time()
            samples_collected = 0
            
            while (time.time() - start_time) < 7:
                ret, frame = cap.read()
                if not ret:
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0]
                    eye_centers = get_eye_centers(landmarks, frame.shape)
                    
                    if len(eye_centers) == 2:
                        midpoint = (
                            int((eye_centers[0][0] + eye_centers[1][0]) // 2),
                            int((eye_centers[0][1] + eye_centers[1][1]) // 2)
                        )
                        
                        for _ in range(5):
                            jitter_x = np.random.normal(0, 2)
                            jitter_y = np.random.normal(0, 2)
                            X_calib.append([midpoint[0] + jitter_x, midpoint[1] + jitter_y])
                            y_calib.append([target_x + np.random.uniform(-3, 3), 
                                          target_y + np.random.uniform(-3, 3)])
                        
                        samples_collected += 5
                        cv2.circle(frame, midpoint, 5, (0, 255, 0), -1)
                        cv2.line(frame, midpoint, (target_x, target_y), (0, 255, 255), 2)

                cv2.putText(frame, f"Samples: {samples_collected}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Calibration', frame)
                cv2.waitKey(1)

            print(f"Collected {samples_collected} samples for this point")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=3),
        Ridge(alpha=0.5)
    )
    model.fit(np.array(X_calib), np.array(y_calib))
    joblib.dump(model, MODEL_FILE)
    print("Calibration complete! Model saved.")

def track_eyes():
    """Run eye tracking with blink detection and camera preview"""
    try:
        model = joblib.load(MODEL_FILE)
    except FileNotFoundError:
        print("No trained model found! Please run calibration first.")
        return

    screen_w, screen_h = pyautogui.size()
    cap = cv2.VideoCapture(0)
    smooth_x, smooth_y = screen_w//2, screen_h//2
    BORDER_BUFFER = 30
    smoothing_factor = 0.3
    error_history = []
    last_update = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                eye_centers = get_eye_centers(landmarks, frame.shape)
                
                if len(eye_centers) == 2:
                    # Blink detection
                    ear = detect_blinks(landmarks, frame.shape)
                    
                    # Eye tracking
                    midpoint = (
                        int((eye_centers[0][0] + eye_centers[1][0]) // 2),
                        int((eye_centers[0][1] + eye_centers[1][1]) // 2)
                    )
                    
                    # Visual feedback
                    cv2.circle(frame, eye_centers[0], 5, (0, 255, 0), -1)
                    cv2.circle(frame, eye_centers[1], 5, (0, 255, 0), -1)
                    cv2.line(frame, eye_centers[0], eye_centers[1], (0, 255, 0), 1)
                    cv2.circle(frame, midpoint, 3, (0, 0, 255), -1)
                    
                    # Prediction and smoothing
                    pred = model.predict([midpoint])[0]
                    current_error = np.linalg.norm(pred - np.array([smooth_x, smooth_y]))
                    error_history.append(current_error)
                    
                    if time.time() - last_update > 2:
                        avg_error = np.mean(error_history[-50:])
                        smoothing_factor = np.clip(0.4 - (avg_error/1000), 0.1, 0.7)
                        last_update = time.time()
                    
                    pred = np.clip(pred, BORDER_BUFFER, screen_w - BORDER_BUFFER)
                    smooth_x = smoothing_factor * pred[0] + (1 - smoothing_factor) * smooth_x
                    smooth_y = smoothing_factor * pred[1] + (1 - smoothing_factor) * smooth_y
                    
                    final_pos = np.clip([smooth_x, smooth_y], 
                                      BORDER_BUFFER, 
                                      [screen_w - BORDER_BUFFER, 
                                       screen_h - BORDER_BUFFER])
                    pyautogui.moveTo(int(final_pos[0]), int(final_pos[1]))

                    # Display info
                    cv2.putText(frame, f"Pred: ({int(pred[0])}, {int(pred[1])})", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Smooth: ({int(smooth_x)}, {int(smooth_y)})", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Blinks: {BLINK_TOTAL}", (10, 90),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"EAR: {ear:.2f}", (10, 120),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Eye Tracking - Blink 3x=Left Click, 4x=Right Click | Q to Quit', frame)
            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    pyautogui.FAILSAFE = False
    if not os.path.exists(MODEL_FILE):
        print("First-time calibration required...")
        calibrate_and_save_model()
    print("Starting eye tracking...")
    track_eyes()