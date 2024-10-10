import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Load the pre-trained emotion classification model
# Ensure that your model is saved as 'emotion_model.h5'
model = load_model('emotion_model.h5')

# Define the emotion labels based on the dataset
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to detect and classify emotions
def classify_emotion(frame, face_detector, model):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for facial detection
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y+h, x:x+w]  # Region of interest (face)
        roi_gray_resized = cv2.resize(roi_gray, (48, 48))  # Resize to model input size (48x48)

        # Normalize the image to match the training process
        roi = roi_gray_resized.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        # Predict the emotion
        prediction = model.predict(roi)
        emotion_index = np.argmax(prediction)
        emotion = emotion_labels[emotion_index]

        # Draw a rectangle around the face and put the emotion label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    return frame

# Initialize the Streamlit app
st.title("Live Facial Emotion Detection")

# Start video capture
video_capture = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Stream the live video
stframe = st.empty()  # Placeholder for video frames

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Detect faces and classify emotions
    processed_frame = classify_emotion(frame, face_detector, model)

    # Display the processed frame in Streamlit app
    stframe.image(processed_frame, channels='BGR')

# Release the video capture when the loop ends
video_capture.release()
cv2.destroyAllWindows()
