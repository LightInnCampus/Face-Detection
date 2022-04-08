
import face_recognition
import cv2
import numpy as np
import time
from utils.CamStream import CamStream
from utils.FPS import FPS
import argparse
import cv2

def main(display_frame=1,src=0,resz = 0.6):
    vs = CamStream(src).start()
    fps = FPS().start()
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()

        small_frame = cv2.resize(frame, (0, 0), fx=resz, fy=resz)
        # check to see if the frame should be displayed to our screen
        if display_frame > 0:
            cv2.imshow("Frame", small_frame)

        fps.update()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

if __name__=="__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--display", type=int, default=1,
        help="Whether or not frames should be displayed (1 or 0)")
    ap.add_argument("-s", "--source", type=int, default=0,
        help="Webcam source (0 for first cam, 1 for second ...)")  
    args = vars(ap.parse_args())

    display_frame,src = args.values()
    # print(num_frames,src)
    main(display_frame,src)