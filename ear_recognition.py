import cv2
import numpy as np

left_ear_cascade = cv2.CascadeClassifier('haarcascade_mcs_leftear.xml')
right_ear_cascade = cv2.CascadeClassifier('haarcascade_mcs_rightear.xml')

if left_ear_cascade.empty():
    raise IOError('Unable to load the left ear cascade classifier xml file')

if right_ear_cascade.empty():
    raise IOError('Unable to load the right ear cascade classifier xml file')

cap = cv2.VideoCapture(0)
scaling_factor = 1
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    left_ear = left_ear_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)
    right_ear = right_ear_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)
    for (x, y, w, h) in left_ear:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    for (x, y, w, h) in right_ear:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)

    cv2.imshow('Ear Detector', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()