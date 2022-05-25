
import gspread
from google.oauth2.service_account import Credentials
from pathlib import Path
from datetime import datetime


def some_func():
    date_format_str = '%H:%M:%S'
    grace_minutes = 5
    current_name='thu'
    current_day,current_time = '2022-25-6','11:20:16'
    credentials = Credentials.from_service_account_file(str(Path("../credentials/lic_face_rec.json")),
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
    checkin_times = sheet.col_values(2)
    names = sheet.col_values(1)
    l = len(checkin_times)

    for t,n in zip(checkin_times[-1:],names[-1:]):
        if n==current_name:
            time_delt = (datetime.strptime(current_time,date_format_str) - datetime.strptime(t,date_format_str)).total_seconds()/60
            if time_delt < grace_minutes:
                return
            break
    sheet.insert_row([current_name,current_time,current_day],l+1)

if __name__=="__main__":
    some_func()








    