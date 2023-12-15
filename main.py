import cv2
import numpy as np
import os
from pygame import mixer
from keras.models import load_model
import time

# Load Haar cascade classifiers
left_eye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')
face_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')

# Load pre-trained Keras model for eye state detection
model = load_model('cnnimg.h5')

# Initialize Pygame mixer for sound
mixer.init()
warning_alarm = mixer.Sound('warning_sound.wav')

# Initialize video capture from default camera (0)
capture_video = cv2.VideoCapture(0)

# Set initial variables
text_font = cv2.FONT_HERSHEY_SIMPLEX
eyes_closed_timer = None
drowsiness_detected = False
left_eye_label = 'Open'
right_eye_label = 'Open'
show_drowsy_text = False
drowsy_text_timer = None

while True:
    ret, frame = capture_video.read()

    # Convert frame to grayscale for processing
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Crop region of interest for both eyes
        roi_gray = gray_frame[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        left_eye = left_eye_cascade.detectMultiScale(roi_gray)
        right_eye = right_eye_cascade.detectMultiScale(roi_gray)

        # Check for right eye state
        if len(right_eye) > 0:
            for (ex, ey, ew, eh) in right_eye:
                right_eye_img = roi_color[ey: ey + eh, ex: ex + ew]
                right_eye_img = cv2.resize(right_eye_img, (24, 24))
                right_eye_img = cv2.cvtColor(right_eye_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                right_eye_img = right_eye_img / 255.0
                right_eye_img = np.expand_dims(right_eye_img, axis=-1)  # Expand dimensions to (24, 24, 1)
                right_eye_img = np.expand_dims(right_eye_img, axis=0)  # Add batch dimension

                # Perform prediction for right eye
                right_eye_prediction = model.predict(right_eye_img)
                right_eye_label = 'Open' if np.argmax(right_eye_prediction) else 'Closed'
                cv2.putText(frame, f"Right: {right_eye_label}", (x + w // 2, y - 20), text_font, 0.5, (255, 255, 0), 2)

        # Check for left eye state
        if len(left_eye) > 0:
            for (ex, ey, ew, eh) in left_eye:
                left_eye_img = roi_color[ey: ey + eh, ex: ex + ew]
                left_eye_img = cv2.resize(left_eye_img, (24, 24))
                left_eye_img = cv2.cvtColor(left_eye_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                left_eye_img = left_eye_img / 255.0
                left_eye_img = np.expand_dims(left_eye_img, axis=-1)  # Expand dimensions to (24, 24, 1)
                left_eye_img = np.expand_dims(left_eye_img, axis=0)  # Add batch dimension

                # Perform prediction for left eye
                left_eye_prediction = model.predict(left_eye_img)
                left_eye_label = 'Open' if np.argmax(left_eye_prediction) else 'Closed'
                cv2.putText(frame, f"Left: {left_eye_label}", (x + w // 2, y + h + 20), text_font, 0.5, (255, 255, 0), 2)

                # Drowsiness detection logic
                if left_eye_label == 'Closed' and right_eye_label == 'Closed':
                    if eyes_closed_timer is None:
                        eyes_closed_timer = time.time()
                    else:
                        elapsed_time = time.time() - eyes_closed_timer
                        if elapsed_time > 1 and not drowsiness_detected:
                            show_drowsy_text = True
                            drowsy_text_timer = time.time()
                            warning_alarm.play()
                            drowsiness_detected = True
                else:
                    eyes_closed_timer = None
                    drowsiness_detected = False

    # Display "Drowsy" text for 2 seconds
    if show_drowsy_text:
        cv2.putText(frame, 'Drowsy', (50, 50), text_font, 1, (0, 0, 255), 2)
        if time.time() - drowsy_text_timer > 2:
            show_drowsy_text = False
    
    cv2.imshow('Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture_video.release()
cv2.destroyAllWindows()
