# from unittest import result
import cv2
from pathlib import Path

cam  = cv2.VideoCapture(0)

count = 0

folder = Path('./Raw_photo/')
if not folder.exists():
    folder.mkdir(parents=True)
    
while True:
    ret,frame= cam.read()
    if not ret:
        print('failed to grab frame')
        break
    cv2.imshow('frame',frame)
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = folder + "person{}.png".format(count)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        count += 1
cam.release()
cam.destroyAllWindows()