import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist
from pathlib import Path
from datetime import datetime
import pytz
import gspread
from google.oauth2.service_account import Credentials

DETECTOR = dlib.get_frontal_face_detector()
# PREDICTOR = dlib.shape_predictor(str(Path('./Weights/landmarks/shape_predictor_68_face_landmarks.dat')))
PREDICTOR= None

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
        
def get_time_delta_secs(prev_day,now_day,date_format):
    '''
    Find differences between 2 timestamp following date_format
    Return results in seconds
    '''
    prev_day = datetime.strptime(prev_day,date_format)
    now_day = datetime.strptime(now_day,date_format)
    time_delt = (now_day - prev_day).total_seconds()
    return time_delt

def is_new_day(prev_day):
    '''
    Return True if time now is >1 day away from prev_day
    '''
    dt_HCM = datetime.now(pytz.timezone('Asia/Ho_Chi_Minh'))
    now_day = dt_HCM.strftime(r"%Y-%m-%d")
    time_delt = get_time_delta_secs(prev_day,now_day,r'%Y-%m-%d')
    return time_delt >= (24*60*60)

def is_time_delta_meq_than(prev_time,curr_time,thres):
    '''
    Return True if curr_time is more than or equal to prev_time by thres seconds
    '''
    time_delt = get_time_delta_secs(prev_time,curr_time,r'%H:%M:%S')
    return time_delt>=thres


def insert_to_spreadsheet(current_name,current_day,current_timestamp):
    credentials = Credentials.from_service_account_file(str(Path("./credentials/lic_face_rec.json")),
            scopes=['https://www.googleapis.com/auth/spreadsheets','https://www.googleapis.com/auth/drive'])
    ss_obj = gspread.authorize(credentials)
    # most time consuming
    ss = ss_obj.open('attendance')

    all_wsheets = [ws.title for ws in ss.worksheets()]
    if current_day not in all_wsheets:
        # create new sheet for the day
        ss.add_worksheet(title=current_day,rows="1",cols="3")
        tmp = ss.worksheet(current_day)
        tmp.insert_row(['ID','Check-in Time','Day'],1)
    
    sheet = ss.worksheet(current_day)
    l = len(sheet.col_values(2))

    sheet.insert_row([current_name,current_timestamp,current_day],l+1)