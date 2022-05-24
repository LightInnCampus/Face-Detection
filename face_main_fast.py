from functools import partial
import cv2
from pathlib import Path
import argparse
import cv2
from models.FaceRecModel import FaceRecModel
import yaml
# from utils.util import are_eyes_closed
from multiprocessing import Pool, Queue
from datetime import datetime
import pytz
from utils.facerec_utils import *
from collections import defaultdict

# https://stackoverflow.com/questions/67567464/multi-threading-in-image-processing-video-python-opencv
# def init_pool(d_a,d_b,d_c,d_e,d_f):
#     global FRAME_BUFFER
#     global PRED_BUFFER
#     global NAME_BUFFER
#     global CURRENT_PRED
#     global PREV_PRED
#     FRAME_BUFFER,PRED_BUFFER,NAME_BUFFER,CURRENT_PRED,PREV_PRED = d_a,d_b,d_c,d_e,d_f
def init_pool(d_a,d_b,d_c):
    global FRAME_BUFFER
    global PRED_BUFFER
    global NAME_BUFFER
    FRAME_BUFFER,PRED_BUFFER,NAME_BUFFER = d_a,d_b,d_c

def show_frame_and_bb(resz):
    while True:
        frame = FRAME_BUFFER.get()
        if frame is not None:
            # show bounding box and names
            font = cv2.FONT_HERSHEY_DUPLEX
            if not PRED_BUFFER.empty():
                face_locations,face_names = PRED_BUFFER.get()

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
                # for i,(n,t) in enumerate(face_names):
                #     cv2.putText(frame, f"Welcome {n} at {t}", (4, 20+ i*15), font, 0.6, (0, 0, 0), 1)
            
            # current_name,current_time = read_singlevalue_queue(CURRENT_PRED)
            current_name='hihi'
            current_time='haha'
            cv2.putText(frame, f"Welcome {current_name}, checking in at {current_time}", (4, 20), font, 0.6, (0, 0, 0), 1)
            cv2.imshow("Frame", frame)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return

def get_names_from_encodings(enc,frm):
    name="Unknown"
    dt_HCM = datetime.now(pytz.timezone('Asia/Ho_Chi_Minh'))
    date_now = dt_HCM.strftime("%Y-%m-%d %H:%M:%S")
    face_distances = face_distance(frm.face_encs, enc)
    best_match_index = np.argmin(face_distances)
    if face_distances[best_match_index]<=frm.thres:
        name = frm.face_names[best_match_index]
        
    return name,date_now

def get_single_prediction(maxsize,thres):
    name_list=[]

    name_freq = defaultdict(int)
    date_dict = {}

    max_freq=0
    max_name=''

    while not NAME_BUFFER.empty():
        name_list.append(NAME_BUFFER.get())
    
    if len(name_list) >= maxsize: # check only when max size is reached
        for n,t in name_list:
            name_freq[n]+=1
            if n not in date_dict: date_dict[n]=t # save first appearance moment
            if name_freq[n]>max_freq:
                max_freq = name_freq[n]
                max_name = n


    if max_name.lower() in ['','unknown'] or max_freq < thres-1:
        # no prediction made yet, add stuff back to queue
        for tu in name_list:
            NAME_BUFFER.put(tu)
        return None,None

    # there's a prediction => remove everything from BUFFER (by not putting things back)
    return max_name,date_dict[max_name]


def read_singlevalue_queue(q):
    result=None
    if not q.empty():
        result = q.get()
        q.put(result)
    return result
    
def write_singlevalue_queue(q,new_value):
    while not q.empty():
        _ = q.get()
    q.put(new_value)


def predict_async(frame,frm=None,args=None,frame_count=0):
    try:
        if frame is not None:
            frame_rsz = cv2.resize(frame, (0, 0), fx=frm.frame_resz, fy=frm.frame_resz)
            current_locations,current_names=[],[]
            # get locations every 2 frames
            if frame_count % 2 ==0:
                current_locations = face_locations(frame_rsz,
                                                    number_of_times_to_upsample=frm.upsample,
                                                    model=frm.model)
            
            if len(current_locations):
                if frame_count % frm.frame_skip==0:
                    current_encodings = face_encodings(frame_rsz,current_locations,
                                                        model=frm.enc_model_size,
                                                        num_jitters = frm.num_jitters)
                    # current_names = [get_names_from_encodings(enc,frm) for enc in current_encodings]
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        current_names = executor.map(partial(get_names_from_encodings,frm=frm),current_encodings)
                    current_names = list(current_names)
                    # print(current_names)
                    
                PRED_BUFFER.put((current_locations,current_names))

                if len(current_names):
                    if NAME_BUFFER.full():
                        # get rid of oldest item
                        _ = NAME_BUFFER.get()
                    # put new item in
                    for n in current_names:
                        NAME_BUFFER.put(n)
                    
                    pred_name,pred_time = get_single_prediction(args.max_size,args.min_pred)
                    if pred_name is not None:
                        # write_singlevalue_queue(CURRENT_PRED,(pred_name,pred_time))
                        print(f'Name: {pred_name}, Time: {pred_time}')


    except Exception as e:
        print('Something wrong')
        print(f'{e}')



def main(args,model_config):

    frame_count=0
    # Initiate and preprocess model
    resz = args.frame_resz # resolution to downsize

    frm = FaceRecModel(**model_config)
    frm.preprocess(args.enc_list)

    # FRAME_BUFFER,PRED_BUFFER,NAME_BUFFER,CURRENT_PRED,PREV_PRED = Queue(),Queue(),Queue(maxsize=args.max_size),Queue(),Queue()
    # pools = Pool(None, initializer=init_pool, initargs=(FRAME_BUFFER,PRED_BUFFER,NAME_BUFFER,CURRENT_PRED,PREV_PRED))

    FRAME_BUFFER,PRED_BUFFER,NAME_BUFFER= Queue(),Queue(),Queue(maxsize=args.max_size)
    pools = Pool(None, initializer=init_pool, initargs=(FRAME_BUFFER,PRED_BUFFER,NAME_BUFFER))
    
    # showing frame on 1 pool
    show_frame_aresult = pools.apply_async(show_frame_and_bb,args=(resz,))
    
    # write/read 

    # camera stream
    stream = cv2.VideoCapture(args.source)
    async_result_list=[]

    while True:
        ret,frame = stream.read()
        if ret:
            FRAME_BUFFER.put(frame)
            # make prediction using the rest of the pools
            aresult = pools.apply_async(predict_async,args=(frame,frm,args,frame_count))
            async_result_list.append(aresult)
            frame_count+=1
            if frame_count > 1000:
                frame_count=0
        else: 
            break
    
    print('Program ending...')
    # wait for all the frame-putting tasks to complete:
    for f in async_result_list:
        f.get() # Return the result when it arrives

    # signal the "show" task to end by placing None in the queue
    FRAME_BUFFER.put(None)
    show_frame_aresult.get()



if __name__=="__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--source", type=int, default=0,
        help="Webcam source (0 for first cam, 1 for second ...)")
    ap.add_argument("-c","--config", default="config/facerec.yaml",
        help="Configuration file for model")
    args = ap.parse_args()

    # read yaml file for config
    with open(str(Path(args.config))) as f:
        config = yaml.safe_load(f)

    for key in config:
        for k,v in config[key].items(): setattr(args,k,v)

    main(args,config['Model'])  