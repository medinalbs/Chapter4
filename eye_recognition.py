import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

if face_cascade.empty():
    raise IOError('No fue posible obtener el archivo XML clasificador de rostros')
if eye_cascade.empty():
    raise  IOError('No fue posible obtener el archivo XML clasificador de ojos')

cap = cv2.VideoCapture(0)
ds_factor = 1

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=1)
    for(x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+h]
        roi_color = frame[y:y+h, x:x+h]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (x_eye, y_eye, w_eye, h_eye) in eyes:
            center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye))
            radius = int(0.2*(w_eye + h_eye))
            color = (255, 0, 255)
            thickness = 2
            cv2.circle(roi_color, center, radius, color, thickness)

    cv2.imshow('Eye detector', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()

