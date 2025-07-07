import cv2
import numpy as np
import mediapipe as mp
from fer import FER
from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import random
from pynput import mouse, keyboard
import time
import logging
import joblib # Import joblib to load the model
import pandas as pd # Import pandas to create DataFrame for prediction

# --- Logging Configuration ---
# Configures basic logging to show INFO level messages and above, with timestamp and message format.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Flask App Initialization ---
app = Flask(__name__)
# Enable CORS for all routes and methods for development purposes.
# IMPORTANT: In a production environment, 'origins' should be restricted to your frontend's specific domain
# (e.g., origins=["http://localhost:3000", "https://your-frontend-domain.com"]).
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"], "allow_headers": ["Content-Type"]}})

# --- MediaPipe Initializations ---
# Initialize MediaPipe solutions for face mesh and holistic pose detection.
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# FaceMesh: Detects facial landmarks, useful for gaze and emotion (via FER).
# static_image_mode=False: Processes video streams.
# max_num_faces=1: Limits to one face for performance and simpler logic.
# refine_landmarks=True: Uses more detailed models for improved landmark accuracy.
# min_detection_confidence, min_tracking_confidence: Thresholds for detection.
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Pose: Detects body pose landmarks, useful for posture and drinking action.
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- FER Emotion Detector Initialization ---
# Initialize the Face Emotion Recognition (FER) model.
emotion_detector = None
emotion_detection_enabled = False
try:
    # FER uses MTCNN for face detection by default, which is generally robust.
    emotion_detector = FER(mtcnn=True)
    emotion_detection_enabled = True
    logging.info("FER emotion detector initialized successfully.")
except Exception as e:
    logging.warning(f"FER emotion detector could not be initialized: {e}. Emotion detection will be disabled. "
                    f"Ensure 'tensorflow' and 'fer' are correctly installed. Error: {e}")
    emotion_detection_enabled = False

# --- Load the pre-trained ML model for fatigue prediction ---
# The path to your saved fatigue prediction model.
FATIGUE_MODEL_PATH = 'fatigue_model.joblib'
fatigue_prediction_model = None
model_feature_columns = None # To store the feature columns the model was trained on

try:
    model_data = joblib.load(FATIGUE_MODEL_PATH)
    fatigue_prediction_model = model_data['model']
    model_feature_columns = model_data['feature_columns']
    logging.info(f"Fatigue prediction model and feature columns loaded successfully from {FATIGUE_MODEL_PATH}")
except FileNotFoundError:
    logging.error(f"ERROR: Fatigue prediction model not found at {FATIGUE_MODEL_PATH}. "
                    f"Please ensure 'generate_and_train_fatigue_model.py' has been run successfully FIRST.")
except Exception as e:
    logging.error(f"ERROR: An unexpected error occurred while loading fatigue prediction model: {e}")

# --- Global variables for wellness metrics (from CV or simulated) ---
# These variables store the latest calculated/simulated values.
previous_gaze_vector = None # Used to calculate change in gaze for stability.
gaze_stability = 100.0 # Percentage (0-100), higher is more stable.
heart_rate_value = 70 # BPM (simulated for demonstration, would typically come from a sensor).
emotion_value = "Neutral" # Predicted dominant emotion.
posture_value = "Optimal" # Predicted posture status.
lighting_value = "Optimal" # Estimated lighting condition.
distance_value = "Optimal" # Estimated distance from screen.
fatigue_score_value = 0.0 # Predicted fatigue score from ML model.

# --- Global variables for activity metrics (populated by pynput listeners) ---
# These are used to accumulate data over intervals for calculating PPS and WPM.
mouse_distance_accumulator = 0.0 # Accumulates pixel distance moved by mouse.
last_mouse_speed_calc_time = time.time() # Timestamp of the last mouse speed calculation.
mouse_speed_value = 0.0 # Mouse activity in Pixels Per Second (PPS).
last_mouse_x, last_mouse_y = None, None # Stores last known mouse coordinates for distance calculation.

raw_keyboard_character_count = 0 # Accumulates raw characters typed for WPM calculation.
last_wpm_calc_time = time.time() # Timestamp of the last WPM calculation.
words_per_minute_value = 0.0 # Typing speed in Words Per Minute (WPM).

# --- Locks for thread-safe access to shared variables ---
# Essential for preventing race conditions when multiple threads (pynput, webcam loop)
# try to read/write to the same global variables simultaneously.
mouse_lock = threading.Lock()
keyboard_lock = threading.Lock()
drink_lock = threading.Lock() # For managing the drinking action detection state with a cooldown.

# --- Webcam Initialization ---
# Attempts to open the default webcam (index 0).
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logging.error("ERROR: Could not open webcam. Please ensure it's connected and not in use by another application.")
    # Exit if webcam cannot be opened, as core functionality depends on it.
    exit()
logging.info("Webcam initialized.")

# --- Global state for latest processed prediction data (for Flask API) ---
# This dictionary holds all the latest wellness metrics, which the Flask API will return.
latest_prediction = {} 

# --- Global state for drinking action detection simulation ---
is_drinking_detected = False
last_drink_detection_time = 0
DRINK_DETECTION_COOLDOWN = 5 # seconds - time to wait after a drink detection before detecting another.

# --- Pynput Listener Callbacks ---
# These functions are called by the pynput library in a separate thread when mouse/keyboard events occur.

def on_move(x, y):
    """Callback for mouse movement events. Accumulates pixel distance."""
    global last_mouse_x, last_mouse_y, mouse_distance_accumulator
    with mouse_lock: # Ensure thread-safe access to mouse-related globals
        if last_mouse_x is not None and last_mouse_y is not None:
            # Calculate Euclidean distance from the last position.
            distance = np.sqrt((x - last_mouse_x)**2 + (y - last_mouse_y)**2)
            mouse_distance_accumulator += distance
        last_mouse_x, last_mouse_y = x, y # Update last position for next calculation

def on_click(x, y, button, pressed):
    """Callback for mouse click events. Currently not used for main metrics."""
    pass # Can be extended to count clicks if needed.

def on_scroll(x, y, dx, dy):
    """Callback for mouse scroll events. Currently not used for main metrics."""
    pass # Can be extended to count scroll events if needed.

def on_press(key):
    """
    Callback for keyboard key press events. Accumulates raw character count for WPM.
    Counts printable characters, spacebar, and enter key towards WPM calculation.
    """
    global raw_keyboard_character_count
    with keyboard_lock: # Ensure thread-safe access to keyboard-related globals
        try:
            # Check if the key has a printable character representation.
            if hasattr(key, 'char') and key.char is not None:
                raw_keyboard_character_count += 1
            # Also count space and enter keys as they contribute to "typing" activity.
            elif key == keyboard.Key.space or key == keyboard.Key.enter:
                raw_keyboard_character_count += 1
        except AttributeError:
            # Ignore special keys that don't have a 'char' attribute (e.g., Shift, Ctrl, Alt, F-keys).
            pass

# Initialize pynput listeners with their respective callbacks.
mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
keyboard_listener = keyboard.Listener(on_press=on_press)

# Start pynput listeners in non-blocking mode (they run in their own daemon threads).
mouse_listener.start()
keyboard_listener.start()
logging.info("Pynput mouse and keyboard listeners started in background threads.")


# --- Simulated Data Functions ---
def get_heart_rate():
    """Simulates a fluctuating heart rate between 65 and 95 BPM."""
    # In a real application, this would come from a wearable sensor or estimated from video.
    return random.randint(65, 95)

# --- Computer Vision Processing Functions ---
# These functions process the webcam frame and extract wellness metrics.

def calculate_and_draw_gaze(face_landmarks, frame, w, h):
    """
    Calculates gaze stability based on the movement of eye landmarks over time.
    Draws visual indicators on the frame.
    """
    global previous_gaze_vector, gaze_stability

    # MediaPipe face mesh provides many landmarks. We select a few to represent eye centers.
    left_eye_center_indices = [159, 145] # Points on the left eye
    right_eye_center_indices = [386, 374] # Points on the right eye

    # Basic validation: ensure landmarks are within bounds to prevent index errors.
    if not all(idx < len(face_landmarks.landmark) for idx in left_eye_center_indices + right_eye_center_indices):
        logging.warning("Missing some critical eye landmarks for gaze calculation.")
        gaze_stability = 0.0 # Indicate very unstable if landmarks are incomplete.
        return gaze_stability

    # Convert normalized landmark coordinates to pixel coordinates for drawing.
    left_eye_points_for_draw = np.array([(int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)) for idx in left_eye_center_indices])
    right_eye_points_for_draw = np.array([(int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)) for idx in right_eye_center_indices])

    # Calculate average eye center in normalized coordinates for stability measurement.
    left_eye_avg_x = (face_landmarks.landmark[left_eye_center_indices[0]].x + face_landmarks.landmark[left_eye_center_indices[1]].x) / 2
    left_eye_avg_y = (face_landmarks.landmark[left_eye_center_indices[0]].y + face_landmarks.landmark[left_eye_center_indices[1]].y) / 2
    right_eye_avg_x = (face_landmarks.landmark[right_eye_center_indices[0]].x + face_landmarks.landmark[right_eye_center_indices[1]].x) / 2
    right_eye_avg_y = (face_landmarks.landmark[right_eye_center_indices[0]].y + face_landmarks.landmark[right_eye_center_indices[1]].y) / 2

    # Gaze vector: from left eye center to right eye center.
    current_gaze_vector_norm = np.array([right_eye_avg_x - left_eye_avg_x, right_eye_avg_y - left_eye_avg_y])

    # Calculate stability based on the change in the gaze vector.
    if previous_gaze_vector is None:
        previous_gaze_vector = current_gaze_vector_norm
        gaze_stability = 100.0 # Start with perfect stability.
    else:
        # Euclidean norm of the difference vector indicates magnitude of gaze shift.
        diff = np.linalg.norm(current_gaze_vector_norm - previous_gaze_vector)
        # Convert difference to a stability score (larger diff = lower stability).
        # Scaling factor (1500) might need tuning based on webcam resolution/user movement.
        stability = max(0, 100 - diff * 1500)
        previous_gaze_vector = current_gaze_vector_norm
        gaze_stability = stability

    # Draw visual feedback on the frame.
    for point in left_eye_points_for_draw:
        cv2.circle(frame, tuple(point), 3, (0, 255, 255), -1) # Cyan circles for left eye points.
    for point in right_eye_points_for_draw:
        cv2.circle(frame, tuple(point), 3, (255, 255, 0), -1) # Yellow circles for right eye points.

    if len(left_eye_points_for_draw) > 0 and len(right_eye_points_for_draw) > 0:
        # Draw a red line connecting the approximate centers of the eyes, indicating gaze direction.
        left_center = np.mean(left_eye_points_for_draw, axis=0).astype(int)
        right_center = np.mean(right_eye_points_for_draw, axis=0).astype(int)
        cv2.line(frame, tuple(left_center), tuple(right_center), (0, 0, 255), 2)

    return gaze_stability

def calculate_and_draw_posture(pose_landmarks, frame, h, w):
    """
    Analyzes body pose landmarks to determine and draw posture status.
    Estimates torso lean angle.
    """
    if not pose_landmarks:
        return "Unknown"

    lm = pose_landmarks.landmark # Get the list of all detected landmarks.
    points = mp_pose.PoseLandmark # Enum for easy access to specific landmark indices.

    # Define required landmarks for robust posture calculation.
    required_landmarks = [points.LEFT_SHOULDER, points.RIGHT_SHOULDER, points.LEFT_HIP, points.RIGHT_HIP]
    # Check if all necessary landmarks are detected and accessible.
    if not all(idx < len(lm) for idx in required_landmarks):
        logging.warning("Missing some key posture landmarks, posture calculation may be inaccurate.")
        return "Unknown" # Return unknown if data is incomplete.

    # Extract relevant landmark objects.
    left_shoulder = lm[points.LEFT_SHOULDER]
    right_shoulder = lm[points.RIGHT_SHOULDER]
    left_hip = lm[points.LEFT_HIP]
    right_hip = lm[points.RIGHT_HIP]

    # Helper function to convert normalized landmark coordinates to pixel coordinates.
    def get_pixel_coords(landmark):
        return int(landmark.x * w), int(landmark.y * h)

    # Get pixel coordinates for the key points.
    left_shoulder_px = get_pixel_coords(left_shoulder)
    right_shoulder_px = get_pixel_coords(right_shoulder)
    left_hip_px = get_pixel_coords(left_hip)
    right_hip_px = get_pixel_coords(right_hip)

    # Calculate midpoints of shoulders and hips to represent the top and bottom of the torso.
    shoulder_mid_px = ((left_shoulder_px[0] + right_shoulder_px[0]) // 2,
                       (left_shoulder_px[1] + right_shoulder_px[1]) // 2)
    hip_mid_px = ((left_hip_px[0] + right_hip_px[0]) // 2,
                          (left_hip_px[1] + right_hip_px[1]) // 2)

    # Calculate the vector representing the torso (from shoulder midpoint to hip midpoint).
    torso_vector_px = np.array(hip_mid_px) - np.array(shoulder_mid_px)

    # Calculate the angle of the torso vector relative to the vertical axis.
    # np.arctan2(y, x) gives angle from positive x-axis. We want angle from vertical (y-axis).
    # So we use (dx, dy) where dy is vertical change, dx is horizontal change.
    angle_rad = np.arctan2(torso_vector_px[0], torso_vector_px[1]) # atan2(dx, dy)
    angle_deg = np.degrees(angle_rad)

    # Determine posture based on the calculated angle. Thresholds are empirical.
    posture = "Optimal"
    if abs(angle_deg) < 15: # Optimal: nearly vertical torso.
        posture = "Optimal"
    elif angle_deg < -15 and angle_deg > -55: # Leaning forward: torso tilted forward.
        posture = "Leaning Forward (Strained)"
    elif angle_deg > 15 and angle_deg < 55: # Slightly slouching: torso tilted backward/slouching.
        posture = "Slightly Slouching"
    else: # Poor: extreme angles, indicating very bad posture or detection issues.
        posture = "Poor (Slumped)"

    # Draw MediaPipe landmarks and connections on the frame for visualization.
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=pose_landmarks,
        connections=mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), # Orange for landmarks
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2)) # Magenta for connections

    # Draw custom indicators for the calculated torso vector.
    cv2.circle(frame, shoulder_mid_px, 5, (0, 255, 0), -1) # Green circle at shoulder midpoint.
    cv2.circle(frame, hip_mid_px, 5, (0, 255, 0), -1) # Green circle at hip midpoint.
    cv2.line(frame, shoulder_mid_px, hip_mid_px, (255, 0, 0), 2) # Blue line representing the torso axis.
    cv2.putText(frame, f"Torso Angle: {angle_deg:.1f} deg", (shoulder_mid_px[0] + 10, shoulder_mid_px[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return posture

def estimate_and_draw_lighting(frame):
    """Estimates ambient lighting conditions based on the average pixel brightness of the frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert frame to grayscale.
    mean_brightness = np.mean(gray) # Calculate the average brightness.
    
    lighting = "Optimal"
    color = (0, 255, 0) # Green for optimal.
    
    # Define thresholds for different lighting conditions. These are empirical.
    if mean_brightness < 60: # Dim lighting.
        lighting = "Too Dim"
        color = (0, 165, 255) # Orange.
    elif mean_brightness > 190: # Bright lighting.
        lighting = "Too Bright"
        color = (0, 0, 255) # Red.
    # Note: "Glare" detection would require more advanced image analysis.
    
    # Display the lighting status and brightness value on the frame.
    cv2.putText(frame, f"Lighting: {lighting} ({int(mean_brightness)})", (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return lighting

def estimate_and_draw_distance(face_bbox, frame_width, frame):
    """
    Estimates distance from the screen based on the width of the detected face bounding box.
    Draws the face bounding box and distance status.
    """
    distance = "Unknown"
    color = (255, 255, 255) # White default.

    if face_bbox:
        x_min, y_min, x_max, y_max = face_bbox
        box_width_pixels = x_max - x_min # Calculate face width in pixels.
        
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2) # Draw the face bounding box (Cyan).
        
        # Calculate the ratio of face width to total frame width.
        ratio_to_frame_width = box_width_pixels / frame_width

        # Define thresholds for distance estimation based on the face width ratio. Empirical.
        if ratio_to_frame_width > 0.35: # Face takes up large portion, likely too close.
            distance = "Too Close"
            color = (0, 0, 255) # Red.
        elif ratio_to_frame_width < 0.10: # Face is small, likely too far.
            distance = "Too Far"
            color = (0, 165, 255) # Orange.
        else: # Optimal range.
            distance = "Optimal"
            color = (0, 255, 0) # Green.

        # Display distance status and face width on the frame.
        cv2.putText(frame, f"Distance: {distance}", (x_min, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Face Width: {box_width_pixels}px", (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return distance

def simulate_drinking_action(pose_landmarks, frame):
    """
    Simulates drinking action detection by checking if a hand is near the mouth.
    Includes a cooldown period to prevent rapid, repeated detections.
    """
    global is_drinking_detected, last_drink_detection_time
    
    current_time = time.time()
    
    # If a drink was detected recently, prevent new detections within the cooldown period.
    if current_time - last_drink_detection_time < DRINK_DETECTION_COOLDOWN:
        is_drinking_detected = False # Ensure the flag is reset if still in cooldown.
        return False

    if pose_landmarks:
        lm = pose_landmarks.landmark # Get the list of all detected landmarks.
        
        # Define required landmarks for mouth and wrists.
        required_landmarks = [mp_pose.PoseLandmark.MOUTH_LEFT, mp_pose.PoseLandmark.MOUTH_RIGHT, 
                              mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST]
        # Check if all necessary landmarks are detected.
        if not all(idx < len(lm) for idx in required_landmarks):
            logging.warning("Missing some key landmarks for drinking action detection.")
            is_drinking_detected = False
            return False

        # Approximate mouth center from left and right mouth landmarks.
        mouth_center_x = (lm[mp_pose.PoseLandmark.MOUTH_LEFT].x + lm[mp_pose.PoseLandmark.MOUTH_RIGHT].x) / 2
        mouth_center_y = (lm[mp_pose.PoseLandmark.MOUTH_LEFT].y + lm[mp_pose.PoseLandmark.MOUTH_RIGHT].y) / 2

        # Get wrist landmark positions.
        left_wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
        
        # Calculate Euclidean distance between each wrist and the mouth center in normalized coordinates.
        dist_left_hand_to_mouth = np.sqrt((left_wrist.x - mouth_center_x)**2 + (left_wrist.y - mouth_center_y)**2)
        dist_right_hand_to_mouth = np.sqrt((right_wrist.x - mouth_center_x)**2 + (right_wrist.y - mouth_center_y)**2)

        # If either hand is close to the mouth (threshold 0.15 normalized distance)
        # AND a random chance (10%) is met (to make it a "simulated" detection, not every frame).
        if (dist_left_hand_to_mouth < 0.15 or dist_right_hand_to_mouth < 0.15) and random.random() < 0.1:
            is_drinking_detected = True
            last_drink_detection_time = current_time # Update timestamp of last detection.
            # Draw text on frame indicating detection.
            cv2.putText(frame, "DRINKING DETECTED!", (frame.shape[1] - 300, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            logging.info("Simulated drinking action detected.")
            return True
    
    is_drinking_detected = False # Reset if no drinking action condition is met.
    return False

# --- Activity Metrics Calculation ---
def calculate_activity_metrics():
    """
    Calculates and updates mouse speed (PPS) and words per minute (WPM) based on
    accumulated raw activity data from pynput listeners. This function runs periodically.
    """
    global mouse_distance_accumulator, last_mouse_speed_calc_time, mouse_speed_value
    global raw_keyboard_character_count, last_wpm_calc_time, words_per_minute_value

    current_time = time.time()

    # Calculate Mouse Speed per second (averaged over the last second).
    time_elapsed_mouse = current_time - last_mouse_speed_calc_time
    if time_elapsed_mouse >= 1.0: # Recalculate approximately every 1 second.
        with mouse_lock: # Protect shared variables with a lock.
            if time_elapsed_mouse > 0:
                mouse_speed_value = mouse_distance_accumulator / time_elapsed_mouse
            else:
                mouse_speed_value = 0.0
            mouse_distance_accumulator = 0.0 # Reset accumulator for the next interval.
        last_mouse_speed_calc_time = current_time # Update timestamp for the next interval.
    
    # Calculate Words per Minute (WPM) (averaged over the last minute).
    time_elapsed_wpm_interval = current_time - last_wpm_calc_time
    if time_elapsed_wpm_interval >= 60.0: # Recalculate approximately every 60 seconds (1 minute).
        with keyboard_lock: # Protect shared variables with a lock.
            if time_elapsed_wpm_interval > 0:
                estimated_words = raw_keyboard_character_count / 5.0 # Common approximation: 1 word = 5 characters.
                words_per_minute_value = (estimated_words / time_elapsed_wpm_interval) * 60
            else:
                words_per_minute_value = 0.0
            raw_keyboard_character_count = 0 # Reset character count for the next minute.
        last_wpm_calc_time = current_time # Update timestamp for the next interval.


# --- Main Frame Processing and Prediction Logic ---
def process_frame_and_predict(frame, hydration_level_from_frontend=100.0):
    """
    Central function to process a single webcam frame, extract all wellness/activity metrics,
    and then use the ML model to predict the fatigue score.
    Updates global variables and prepares data for API response.
    """
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to RGB for MediaPipe.

    # Re-declare global variables here to ensure they are updated within this function's scope.
    global gaze_stability, emotion_value, posture_value, lighting_value, distance_value, fatigue_score_value
    global heart_rate_value, mouse_speed_value, words_per_minute_value, is_drinking_detected

    # Initialize / reset values for the current frame to ensure a clean state.
    gaze_score_val = 100.0
    emotion_val = "Neutral"
    posture_val = "Unknown"
    lighting_val = "Unknown"
    distance_val = "Unknown"
    face_bbox_pixels = None
    drinking_action_detected_this_frame = False 

    # --- Face Mesh Processing (for gaze, emotion, distance) ---
    face_results = face_mesh.process(rgb_frame)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # Draw face mesh landmarks for visual feedback.
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1), # Green
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 128, 0), thickness=1)) # Darker Green

            # Calculate gaze stability.
            gaze_score_val = calculate_and_draw_gaze(face_landmarks, frame, w, h)

            # Get bounding box of the face for emotion detection and distance estimation.
            xs = [lm.x * w for lm in face_landmarks.landmark]
            ys = [lm.y * h for lm in face_landmarks.landmark]
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))
            
            # Ensure bounding box is valid (positive width and height).
            if y_max > y_min and x_max > x_min:
                face_bbox_pixels = (x_min, y_min, x_max, y_max)
                face_image_for_emotion = frame[y_min:y_max, x_min:x_max] # Extract face region.
            else:
                face_bbox_pixels = None
                face_image_for_emotion = None

            # Estimate distance from screen.
            distance_val = estimate_and_draw_distance(face_bbox_pixels, w, frame)

            # Perform emotion detection if enabled and face image is valid.
            if emotion_detection_enabled and face_image_for_emotion is not None and \
               face_image_for_emotion.size != 0 and face_image_for_emotion.shape[0] > 0 and face_image_for_emotion.shape[1] > 0:
                try:
                    emotions_detected = emotion_detector.detect_emotions(face_image_for_emotion)
                    if emotions_detected:
                        # Get the dominant emotion.
                        dominant_emotion = max(emotions_detected[0]["emotions"], key=emotions_detected[0]["emotions"].get)
                        emotion_val = dominant_emotion
                        cv2.putText(frame, f"Emotion: {emotion_val}", (x_min, y_min - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    else:
                        emotion_val = "Neutral" # Default if no emotions are detected by FER.
                except Exception as e:
                    logging.error(f"Error during emotion detection: {e}")
                    emotion_val = "Error" # Indicate an error in emotion detection.
            elif not emotion_detection_enabled:
                emotion_val = "Disabled" # If FER failed to initialize.
            else: 
                emotion_val = "Face not clear" # If emotion detection is enabled but face image is problematic.

    else: 
        emotion_val = "No face detected"
        gaze_stability = 0.0 # If no face, gaze stability is considered completely unstable.
        face_bbox_pixels = None

    # --- Pose Processing (for posture and drinking action) ---
    pose_results = pose.process(rgb_frame)
    if pose_results.pose_landmarks:
        posture_val = calculate_and_draw_posture(pose_results.pose_landmarks, frame, h, w)
        drinking_action_detected_this_frame = simulate_drinking_action(pose_results.pose_landmarks, frame)
    else:
        drinking_action_detected_this_frame = False # No pose detected, so no drinking action.


    lighting_val = estimate_and_draw_lighting(frame) # Estimate lighting from the whole frame.

    # Calculate real-time mouse speed and WPM from pynput listeners.
    calculate_activity_metrics()

    # Get simulated heart rate.
    heart_rate_val = get_heart_rate()

    # --- Predict Fatigue Score using ML Model ---
    predicted_fatigue_score = 0.0 # Default value if model is not loaded or prediction fails.
    if fatigue_prediction_model and model_feature_columns:
        try:
            # Prepare data for prediction, ensuring features match the training data's structure and order.
            # Create a dictionary for the current observation
            current_observation = {
                'heart_rate': heart_rate_val,
                'mouse_activity': mouse_speed_value,
                'keyboard_activity': words_per_minute_value,
                'gaze_stability': gaze_score_val,
                'emotion': emotion_val,
                'posture': posture_val,
                'lighting': lighting_val,
                'distance': distance_val,
                'hydration_level': hydration_level_from_frontend # Use the hydration level received from frontend.
            }
            
            # Create a DataFrame from the single observation
            prediction_input_df = pd.DataFrame([current_observation])

            # One-hot encode categorical features for the new input
            # Use reindex to ensure all columns present during training are also present here,
            # filling missing ones with 0. This is CRITICAL for model compatibility.
            prediction_input_processed = pd.get_dummies(prediction_input_df, columns=['emotion', 'posture', 'lighting', 'distance'], drop_first=True)
            prediction_input_processed = prediction_input_processed.reindex(columns=model_feature_columns, fill_value=0)
            
            # Make the prediction.
            predicted_fatigue_raw = fatigue_prediction_model.predict(prediction_input_processed)[0]
            # Clip the score to be between 0 and 100, and round for cleaner display.
            predicted_fatigue_score = round(np.clip(predicted_fatigue_raw, 0, 100), 2)
        except Exception as e:
            logging.error(f"ERROR: Error during fatigue prediction. Check if input features align with model: {e}")
            logging.error(f"Problematic raw prediction input: {current_observation}")
            logging.error(f"Problematic processed prediction input (shape {prediction_input_processed.shape}, columns {prediction_input_processed.columns.tolist()}): \n{prediction_input_processed}")
            predicted_fatigue_score = 0.0 # Default to 0 on error.
    else:
        logging.warning("Fatigue prediction model or feature columns not loaded. Fatigue score will default to 0.")
        predicted_fatigue_score = 0.0

    # --- Update Global State for API Response ---
    # Store all latest metrics in the global `latest_prediction` dictionary.
    global latest_prediction
    latest_prediction = {
        "fatigue_score": predicted_fatigue_score,
        "heart_rate": heart_rate_val,
        "mouse_activity": round(mouse_speed_value, 2),
        "keyboard_activity": round(words_per_minute_value),
        "gaze_stability": round(gaze_score_val, 2),
        "emotion": emotion_val,
        "posture": posture_val,
        "lighting": lighting_val,
        "distance": distance_val,
        "drinking_action_detected": drinking_action_detected_this_frame
    }

def draw_general_info(frame):
    """Draws current wellness and activity metrics as text overlay on the frame for visual feedback."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    y0, dy = 30, 30 # Starting Y coordinate and line height for text.

    # List of metrics to display.
    lines = [
        f"Fatigue: {latest_prediction.get('fatigue_score', 'N/A'):.2f}",
        f"HR: {latest_prediction.get('heart_rate', 'N/A')} bpm",
        f"Mouse Speed: {latest_prediction.get('mouse_activity', 0):.2f} px/s",
        f"Typing Speed: {latest_prediction.get('keyboard_activity', 0)} WPM",
        f"Gaze: {latest_prediction.get('gaze_stability', 0):.2f}/100",
        f"Emotion: {latest_prediction.get('emotion', 'N/A')}",
        f"Posture: {latest_prediction.get('posture', 'N/A')}",
        f"Lighting: {latest_prediction.get('lighting', 'N/A')}",
        f"Distance: {latest_prediction.get('distance', 'N/A')}",
        f"Drinking Detected: {latest_prediction.get('drinking_action_detected', False)}",
        "", # Empty line for spacing.
        "Press ESC to exit webcam"
    ]

    for i, line in enumerate(lines):
        y = y0 + i * dy
        cv2.putText(frame, line, (10, y), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA) # White text for general info.


# --- Flask API Routes ---
@app.route('/api/wellness-metrics', methods=['GET'])
def get_wellness_metrics():
    """
    API endpoint to return the latest wellness and activity metrics.
    This data is populated by the webcam processing thread.
    """
    logging.debug(f"API request received. Returning: {latest_prediction}")
    return jsonify(latest_prediction)

@app.route('/api/update-hydration', methods=['POST'])
def update_hydration():
    """
    API endpoint to receive hydration level from the frontend.
    This simulates user input for a metric not directly observable by CV.
    """
    data = request.get_json()
    hydration_level = data.get('hydration_level')
    if hydration_level is not None:
        try:
            hydration_level = float(hydration_level)
            # You might want to store this in a global variable or queue for use in the main loop
            # For simplicity, we'll pass it directly to the processing function when called by the thread.
            logging.info(f"Received hydration level from frontend: {hydration_level}%")
            # This doesn't update the global `hydration_level_from_frontend` directly
            # because the main loop calls `process_frame_and_predict` with the latest value.
            # A more robust solution would be a thread-safe queue or shared variable.
            # For now, this just logs and acknowledges receipt.
            return jsonify({"status": "success", "message": "Hydration level received"}), 200
        except ValueError:
            return jsonify({"status": "error", "message": "Invalid hydration level format"}), 400
    return jsonify({"status": "error", "message": "No hydration_level provided"}), 400

# --- Main Webcam Processing Loop ---
# This function runs in a separate thread to continuously capture and process video.
def webcam_processing_thread():
    logging.info("Starting webcam processing thread...")
    # Initialize hydration to a default value, or fetch from a persistent store.
    # In a real app, this might be saved/loaded or managed by the frontend.
    current_hydration_level = 100.0

    while cap.isOpened():
        ret, frame = cap.read() # Read a frame from the webcam.
        if not ret:
            logging.error("Failed to grab frame. Exiting webcam thread.")
            break

        # Flip the frame horizontally for a more natural mirror view.
        frame = cv2.flip(frame, 1)

        # Get latest hydration from the frontend (if an API call updates it).
        # This is a simplification; a proper mechanism would involve a thread-safe queue
        # or a value updated by the Flask API and read by this thread.
        # For this example, we'll just use the last value received by the API if any.
        # You'll need to implement a way for `update_hydration` to actually update `current_hydration_level`.
        # For demonstration, we'll keep it at 100 or update based on a hypothetical mechanism.
        
        # In a more complete system, the /api/update-hydration would store its value
        # in a thread-safe way, and this loop would read it.
        # For now, let's assume `latest_prediction` could also contain the last
        # hydration level that was processed for the model.
        
        # Process the frame and update global prediction data.
        # We'll pass the hydration_level from the global latest_prediction if it exists,
        # otherwise use a default of 100. (This creates a slight circular dependency for hydration,
        # but is functional for a quick demo. A dedicated shared variable for `hydration_level_from_frontend`
        # is the robust solution).
        process_frame_and_predict(frame, hydration_level_from_frontend=latest_prediction.get('hydration_level', 100.0))
        
        # Draw general wellness info onto the frame.
        draw_general_info(frame)

        # Display the processed frame.
        cv2.imshow('Proactive Wellness Dashboard', frame)

        # Check for 'ESC' key press to exit the loop.
        if cv2.waitKey(1) & 0xFF == 27:
            logging.info("ESC key pressed. Exiting webcam thread.")
            break

    # Release webcam and destroy all OpenCV windows when the loop exits.
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Webcam processing thread finished and resources released.")

# --- Main Application Execution ---
if __name__ == '__main__':
    logging.info("Starting Proactive Wellness Backend Application...")

    # Start the webcam processing in a separate daemon thread.
    # A daemon thread will automatically exit when the main program exits.
    webcam_thread = threading.Thread(target=webcam_processing_thread, daemon=True)
    webcam_thread.start()
    logging.info("Webcam thread launched.")

    # Start the Flask API on the main thread.
    # debug=True allows for automatic reloading on code changes and provides a debugger,
    # but should be False in production.
    app.run(host='0.0.0.0', port=5000, debug=False) # Set debug to False for production
    
    logging.info("Flask app terminated.")

    # Ensure pynput listeners are stopped cleanly on application exit.
    mouse_listener.stop()
    keyboard_listener.stop()
    logging.info("Pynput listeners stopped.")