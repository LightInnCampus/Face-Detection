import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])
    # return the eye aspect ratio
    return (A+B)/(2.0*C)


def main(pre, dec):
    COUNTER = 0
    TOTAL = 0
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
            rects = dec(gray, 0)
            for rect in rects:
                x = rect.left()
                y = rect.top()
                xright = rect.right()
                ybottom = rect.bottom()
                # get the facial landmarks
                landmarks = np.matrix([[p.x, p.y] for p in pre(frame, rect).parts()])
                # get the left eye landmarks
                left_eye = landmarks[list(range(42, 48))]
                # get the right eye landmarks
                right_eye = landmarks[list(range(36, 42))]
                # draw contours on the eyes
                left_eye_hull = cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)
                cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1) # (image, [contour], all_contours, color, thickness)
                cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
                # compute the EAR for the left eye
                ear_left = eye_aspect_ratio(left_eye)
                # compute the EAR for the right eye
                ear_right = eye_aspect_ratio(right_eye)
                # compute the average EAR
                ear_avg = (ear_left + ear_right) / 2.0
                # detect the eye blink
                if ear_avg < 0.22:
                    COUNTER += 1
                else:
                    if COUNTER >= 3:
                        TOTAL += 1
                        print("Eye blinked")
                    COUNTER = 0
                cv2.putText(frame, "Blinks{}".format(TOTAL), (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 1)
                cv2.putText(frame, "EAR {}".format(ear_avg), (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 1)
            cv2.imshow("Winks Found", frame)
            key = cv2.waitKey(1) & 0xFF
            # When key 'Q' is pressed, exit
            if key is ord('q'):
                break
    # release all resources
    cap.release()
    # destroy all windows
    cv2.destroyAllWindows()

    
if __name__=="__main__":
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('/home/harryphan/dev/LIC/project/Face-Recognition/sample/Eye-Blink-Detection/shape_predictor_68_face_landmarks.dat')

    main(predictor, detector)

