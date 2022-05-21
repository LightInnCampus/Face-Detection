import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist
from pathlib import Path

DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor(str(Path('./Weights/landmarks/shape_predictor_68_face_landmarks.dat')))

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])
    # return the eye aspect ratio
    return (A+B)/(2.0*C)

def are_eyes_closed(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    rects = DETECTOR(gray, 0)
    for rect in rects:
        # get the facial landmarks
        landmarks = np.matrix([[p.x, p.y] for p in PREDICTOR(frame, rect).parts()])
        # get the left eye landmarks
        left_eye = landmarks[list(range(42, 48))]
        # get the right eye landmarks
        right_eye = landmarks[list(range(36, 42))]
        # draw contours on the eyes
        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        # cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1) # (image, [contour], all_contours, color, thickness)
        # cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
        # compute the EAR for the left eye
        ear_left = eye_aspect_ratio(left_eye)
        # compute the EAR for the right eye
        ear_right = eye_aspect_ratio(right_eye)
        # compute the average EAR
        ear_avg = (ear_left + ear_right) / 2.0
        return ear_avg < 0.22
        
