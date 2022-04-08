import cv2
import numpy as np
from utils.CamStream import CamStream
from utils.FPS import FPS
import argparse
import cv2
from models.FaceRecModel import FaceRecModel


def show_frame_and_bb(frame,face_locations,face_names,resz=None):
    # show bounding box and names
    font = cv2.FONT_HERSHEY_DUPLEX

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations 
        if resz:
            top = int(top/resz)
            right = int(right/resz)
            bottom = int(bottom/resz)
            left = int(left/resz)

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with a name below the face
        # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), 1)
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.4, (255, 255, 255), 1)
    cv2.imshow("Frame", frame)


def main(display_frame=1,src=0,resz = 0.6,thres=0.4):
    # Initiate and preprocess model
    frm = FaceRecModel(enc_model_size='small',frame_resz=resz,upsample=1,model='hog',num_jitters=1)
    frm.preprocess()

    # start stream
    vs = CamStream(src).start()
    fps = FPS().start()

    while True:
        # grab the frame from the threaded video stream and resize it
        frame = vs.read()
        frame_to_show = frame.copy()
        frame = cv2.resize(frame, (0, 0), fx=resz, fy=resz)
        
        # model predict
        face_locations,face_names = frm.predict(frame,thres)
        print(face_names)

        # check to see if the frame should be displayed to our screen
        if display_frame > 0:
            show_frame_and_bb(frame_to_show,face_locations,face_names,resz=resz)

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
    main(display_frame,src,0.4)