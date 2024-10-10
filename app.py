import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

model = load_model('emotion_model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def classify_emotion(frame, face_detector, model):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y+h, x:x+w]  
        roi_gray_resized = cv2.resize(roi_gray, (48, 48))  

        roi = roi_gray_resized.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        prediction = model.predict(roi)
        emotion_index = np.argmax(prediction)
        emotion = emotion_labels[emotion_index]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    return frame

st.title("Live Facial Emotion Detection")

video_capture = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

stframe = st.empty()  

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    processed_frame = classify_emotion(frame, face_detector, model)

    stframe.image(processed_frame, channels='BGR')

video_capture.release()
cv2.destroyAllWindows()
