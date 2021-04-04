# Import libraries
import cv2

# Load haar cascade files
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

class FaceDetector:
    def __init__(self):
        pass
   
    # Function to detect face in an image
    def detect(gray, frame):
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            roi_gray = gray[x:x+w, y:y+h]
            roi_color = frame[x:x+w, y:y+h]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 8)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
        return frame