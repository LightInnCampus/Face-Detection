from functools import partial
import cv2
# from utils.CamStream import CamStream
from pathlib import Path
from utils.FPS import FPS
import argparse
import cv2
from models.FaceRecModel import FaceRecModel
import yaml
from utils.util import are_eyes_closed
from multiprocessing import Pool, Queue
import time
from utils.facerec_utils import *

# https://stackoverflow.com/questions/67567464/multi-threading-in-image-processing-video-python-opencv
def init_pool(d_a,d_b):
    global frame_buffer
    global pred_buffer
    frame_buffer,pred_buffer = d_a,d_b

def show_frame_and_bb(resz):
    while True:
        frame = frame_buffer.get()
        if frame is not None:
            # show bounding box and names
            font = cv2.FONT_HERSHEY_DUPLEX
            if not pred_buffer.empty():
                face_locations,face_names = pred_buffer.get()
                for (top, right, bottom, left) in face_locations:
                    # Scale back up face locations 
                    top = int(top/resz)
                    right = int(right/resz)
                    bottom = int(bottom/resz)
                    left = int(left/resz)

                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    # Draw a label with a name below the face
                    # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), 1)
                cv2.putText(frame, f"Welcome {', '.join(face_names)}", (4, 20), font, 0.6, (0, 0, 0), 1)
            cv2.imshow("Frame", frame)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return

def get_names_from_encodings(enc,frm):
    name="Unknown"
    face_distances = face_distance(frm.face_encs, enc)
    best_match_index = np.argmin(face_distances)
    if face_distances[best_match_index]<=frm.thres:
        name = frm.face_names[best_match_index]
    return name

def predict(frame,frm):
    current_names=[]
    # current_locations = face_locations(frame,number_of_times_to_upsample=frm.upsample,model=frm.model)

    # if len(current_locations):
    #     current_encodings = face_encodings(frame,current_locations,model=frm.enc_model_size,num_jitters = frm.num_jitters)
    #     current_names = [get_names_from_encodings(enc,frm) for enc in current_encodings]
    #     # with ThreadPoolExecutor(max_workers=4) as executor:
    #     #     current_names = executor.map(get_names_from_encodings,current_encodings)


    # return current_locations,list(current_names)
    # return current_locations,['hihi']
    return ['hihi']



def predict_async(frame,frm=None):
    try:
        if frame is not None:
            current_locations,current_names = predict(frame,frm)
            if len(current_locations):
                pred_buffer.put((current_locations,current_names))
    except Exception as e:
        print('Something wrong')
        print(f'{e}')

        # time.sleep(1)


def main(args,model_config):

    frame_count=0
    # Initiate and preprocess model
    resz = args.framesize # resolution to downsize

    frm = FaceRecModel(frame_resz = resz,**model_config)
    frm.preprocess(args.enc_list)


    frame_buffer,pred_buffer = Queue(),Queue()
    pools = Pool(None, initializer=init_pool, initargs=(frame_buffer,pred_buffer))
    
    # showing frame on 1 pool
    show_frame_aresult = pools.apply_async(show_frame_and_bb,args=(resz,))

    stream = cv2.VideoCapture(args.source)
    async_result_list=[]

    while True:
        ret,frame = stream.read()
        if ret:
            # frame_buffer.put(frame)
            frame_rsz = cv2.resize(frame, (0, 0), fx=frm.frame_resz, fy=frm.frame_resz)
            locations = face_locations(frame_rsz,number_of_times_to_upsample=frm.upsample,model=frm.model)
            frame_buffer.put(frame,locations)

            aresult = pools.apply_async(predict_async,args=(frame_rsz,frm,))
            async_result_list.append(aresult)
            frame_count+=1
        else: 
            break
    
    print('Program ending...')
    # wait for all the frame-putting tasks to complete:
    for f in async_result_list:
        f.get() # Return the result when it arrives

    # signal the "show" task to end by placing None in the queue
    frame_buffer.put(None)
    # pred_buffer.put(None)
    show_frame_aresult.get()



if __name__=="__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--display", type=int, default=1,
        help="Whether or not frames should be displayed (1 or 0)")
    ap.add_argument("-s", "--source", type=int, default=0,
        help="Webcam source (0 for first cam, 1 for second ...)")
    ap.add_argument('-fs',"--framesize",type=float,default=0.8,
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