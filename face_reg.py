import cv2
import face_recognition
import os
import numpy as np
import pyautogui
import mediapipe as mp

# Load and encode known faces
known_face_encodings = []
known_face_names = []

# Path to the directory containing known person images
known_faces_dir = r'C:\Users\Subhashree\Downloads\known faces'

for filename in os.listdir(known_faces_dir):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
        image_path = os.path.join(known_faces_dir, filename)
        
        # Load image
        img = face_recognition.load_image_file(image_path)
        
        # Ensure image is in RGB format
        if img.ndim == 2:  # grayscale image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] != 3:  # not RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Check if any face is detected
        face_locations = face_recognition.face_locations(img)
        if not face_locations:
            print(f"No face detected in image {filename}")
            continue
        
        # Encode face
        encoding = face_recognition.face_encodings(img)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(os.path.splitext(filename)[0])

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break

    rgb_frame = frame[:, :, ::-1]  # Convert to RGB

    # Detect faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    recognized_names = []  # List to store recognized names

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            recognized_names.append(name)  # Add the name to the recognized names list

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Hand Gesture Detection
            hand_gesture_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(hand_gesture_frame)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Get landmarks for gesture detection
                    landmarks = hand_landmarks.landmark
                    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
                    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

                    # Detect Fist Gesture
                    if (index_tip.y > middle_tip.y > ring_tip.y > pinky_tip.y) and (thumb_tip.x < index_tip.x):
                        pyautogui.hotkey('ctrl', 'n')  # Open Notepad
                        print("Fist detected - Opening Notepad")

                    # Detect Index and Little Finger Gesture (Rock Gesture)
                    elif (index_tip.y < middle_tip.y) and (pinky_tip.y < ring_tip.y) and (thumb_tip.x > index_tip.x):
                        pyautogui.hotkey('ctrl', 't')  # Open Google Chrome
                        print("Rock gesture detected - Opening Google Chrome")

    # Display recognized names at the top of the frame
    if recognized_names:
        names_text = "Recognized: " + ", ".join(recognized_names)
        cv2.putText(frame, names_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
video_capture.release()
cv2.destroyAllWindows()

