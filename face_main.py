import cv2
from utils.CamStream import CamStream
from pathlib import Path
from utils.FPS import FPS
import argparse
import cv2
from models.FaceRecModel import FaceRecModel
import yaml

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


def main(args,model_config):
    # Initiate and preprocess model
    resz = args.framesize # resolution to downsize
    frm = FaceRecModel(frame_resz = resz,**model_config)
    frm.preprocess(args.enc_list)

    # start stream
    vs = CamStream(args.source).start()
    fps = FPS().start()

    while True:
        # grab the frame from the threaded video stream and resize it
        frame = vs.read()
        frame_to_show = frame.copy()
        frame = cv2.resize(frame, (0, 0), fx=resz, fy=resz)
        
        # model predict
        face_locations,face_names = frm.predict(frame)
        print(f'Face predicted: {face_names}')
        # check to see if the frame should be displayed to our screen
        if args.display > 0:
            show_frame_and_bb(frame_to_show,face_locations,face_names,resz=resz)
            # show_frame_and_bb(frame_to_show,[(126, 205, 216, 116)],['whatever'],resz=resz)
        
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
    ap.add_argument('-fs',"--framesize",type=float,default=0.6,
        help="Frame resize ratio to original frame. To speed up process")
    ap.add_argument("-c","--config", default="config/facerec.yaml",
        help="Configuration file for model")
    args = ap.parse_args()

    # read yaml file for config
    with open(str(Path(args.config))) as f:
        config = yaml.safe_load(f)

    for key in config:
        for k,v in config[key].items(): setattr(args,k,v)

    main(args,config['Model'])  