from concurrent.futures import ThreadPoolExecutor
from functools import partial
import cv2
from pathlib import Path
import argparse
import cv2
from models.FaceRecModel import FaceRecModel
import yaml
# from utils.util import are_eyes_closed
from utils.util import is_new_day,is_time_delta_meq_than,insert_to_spreadsheet
from multiprocessing import Pool, Queue, Lock
from datetime import datetime
import pytz
from utils.facerec_utils import *
from collections import defaultdict




# https://stackoverflow.com/questions/67567464/multi-threading-in-image-processing-video-python-opencv
def init_pool(d_a,d_b,d_c,d_d,d_e,d_f):
    global FRAME_BUFFER # Queue. Contain unaltered frame
    global PRED_BUFFER # Queue. Each item is list of tuple from single frame: [(location1,name1),(location2,name2),...]
    global NAMETIME_BUFFER # Queue. Each item is a tuple: (name1,timme1). Name is nonempty

    global CURRENT_PRED 
    # Queue. Should contain only 1 item (current 'prediction' cummulatively from NAMETIME_BUFFER),
    #   which is a tuple (name1,time1)

    global FINAL_PREDS # Queue. To store all cummulative prediction. Each item is a tuple from CURRENT_PRED.
    
    global PRED_DICT # Queue. This will match entries in spreadsheet
    # Store 1 item, which is a dictionary: {name1:time1,name2:time2}. Will be clear at the end of the day

    FRAME_BUFFER,PRED_BUFFER,NAMETIME_BUFFER,CURRENT_PRED,FINAL_PREDS,PRED_DICT = d_a,d_b,d_c,d_d,d_e,d_f


def show_frame_and_bb(resz):
    while True:
        try:
            frame = FRAME_BUFFER.get()
            if frame is not None: # maybe redundant
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
                
                current_name,current_time = read_singlevalue_queue(CURRENT_PRED,default_val=('',''))

                cv2.putText(frame, f"Welcome {current_name}, checking in at {current_time}", (4, 20), font, 0.6, (0, 0, 0), 1)
                cv2.imshow("Frame", frame)
            else:
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt

        except (KeyboardInterrupt, SystemExit):
            print("Exiting show_frame_and_bb")
            break
        except Exception as e:
            print('Something wrong when showing frames and bbox')
            print(f'{e}')
            break

def write_to_sheet(grace_minutes=5):
    while True:
        try:
            tmp = FINAL_PREDS.get()
            current_name,current_time = tmp
            current_day,current_timestamp = current_time.split()

            pred_dict = read_singlevalue_queue(PRED_DICT,{})
            # we won't add anything to PRED_DICT or sheet if:
            # it's still the same day, while his/her name is already in PRED_DICT, 
            #   and it hasn't been 5 minutes since his last timestamp
            if len(pred_dict):
                tmp_day = list(pred_dict.values())[0].split()[0]
                if not is_new_day(tmp_day):
                    if current_name in pred_dict:
                        prev_timestamp = pred_dict[current_name].split()[1]
                        if not is_time_delta_meq_than(prev_timestamp,current_timestamp,grace_minutes*60):
                            # print(f'{current_name} is still in grace period')
                            continue
                else:
                    # clear dictionary when it's a new day
                    pred_dict.clear()

            # At this point, it's either the new day, or this is a new name
            # or the name exists and he is checking in after the grace minutes
            # => add entry to PRED_DICT and sheet
            pred_dict[current_name] = current_time
            print(f'Writing {current_name} to PRED_DICT ...')
            write_singlevalue_queue(PRED_DICT,pred_dict)
            print(f'Writing {current_name} to sheet ...')
            insert_to_spreadsheet(current_name,current_day,current_timestamp)
            print(f'Write successfully')
            
        except (KeyboardInterrupt, SystemExit):
            print("Exiting write_to_sheet")
            break
        except Exception as e:
            print('Something wrong in writing to sheet')
            print(f'{e}')
            break


def get_names_from_encodings(enc,frm):
    name="Unknown"
    dt_HCM = datetime.now(pytz.timezone('Asia/Ho_Chi_Minh'))
    date_now = dt_HCM.strftime("%Y-%m-%d %H:%M:%S")
    face_distances = face_distance(frm.face_encs, enc)
    best_match_index = np.argmin(face_distances)
    if face_distances[best_match_index]<=frm.thres:
        name = frm.face_names[best_match_index]
        
    return name,date_now

def get_single_prediction(q,maxsize,min_pred):
    name_list=[]

    name_freq = defaultdict(int)
    date_dict = {}

    max_freq=0
    max_name=''

    while not q.empty():
        name_list.append(q.get())
    
    if len(name_list) >= maxsize: # do the check only when max size is reached
        for n,t in name_list:
            name_freq[n]+=1
            if n not in date_dict: date_dict[n]=t # save datetime of first appearance
            if name_freq[n]>max_freq:
                max_freq = name_freq[n]
                max_name = n


    if max_name.lower() in ['','unknown'] or max_freq < min_pred-1:
        # no prediction made yet, add stuff back to queue
        for tu in name_list:
            q.put(tu)
        return None,None

    # there's a prediction => remove everything from BUFFER (by not putting things back)
    return max_name,date_dict[max_name]


def read_singlevalue_queue(q,default_val=None):
    result=default_val
    if not q.empty():
        result = q.get()
        q.put(result)
    return result
    
def write_singlevalue_queue(q,new_value):
    while not q.empty():
        _ = q.get()
    q.put(new_value)

def update_dequeue(q,values):
    if q.full():
        # get rid of oldest
        _ = q.get()
    # put new item in
    for n in values:
        q.put(n)
    
def predict_async(frame,frm=None,args=None,frame_count=0):
    try:
        if frame is not None:
            frame_rsz = cv2.resize(frame, (0, 0), fx=frm.frame_resz, fy=frm.frame_resz)
            current_locations,current_names_and_times=[],[]
            # get locations every 2 frames
            if frame_count % 2 ==0:
                frame_rsz = frame_rsz[:,:,::-1]  # to rgb
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
                        current_names_and_times = executor.map(partial(get_names_from_encodings,frm=frm),current_encodings)
                    current_names_and_times = list(current_names_and_times)
                
                # PRED_BUFFER only put nonempty location list. There can be empty current_names list
                PRED_BUFFER.put((current_locations,current_names_and_times))
                


                if len(current_names_and_times):
                    print(f'Raw prediction: {current_names_and_times}')
                    update_dequeue(NAMETIME_BUFFER,current_names_and_times)
                    
                    pred_name,pred_time = get_single_prediction(NAMETIME_BUFFER,args.max_size,args.min_pred)
                    if pred_name is not None:
                        print(f'Cummulative prediction: {pred_name}, at {pred_time}')
                        FINAL_PREDS.put((pred_name,pred_time))
                        write_singlevalue_queue(CURRENT_PRED,(pred_name,pred_time))


    except Exception as e:
        print('Something wrong in making async prediction')
        print(f'{e}')



def main(args,model_config):

    frame_count=0

    # Initiate and preprocess model
    resz = args.frame_resz # resolution to downsize

    # initialize model and multiprocessing buffers
    frm = FaceRecModel(**model_config)
    frm.preprocess(args.enc_list)
    FRAME_BUFFER,PRED_BUFFER,NAMETIME_BUFFER = Queue(),Queue(),Queue(maxsize=args.max_size)
    CURRENT_PRED,FINAL_PREDS,PRED_DICT=Queue(maxsize=1),Queue(),Queue(maxsize=1)

    pools = Pool(None, initializer=init_pool, initargs=(FRAME_BUFFER,PRED_BUFFER,NAMETIME_BUFFER,
                                                        CURRENT_PRED,FINAL_PREDS,PRED_DICT))
    

    # showing frame on 1 pool
    show_frame_aresult = pools.apply_async(show_frame_and_bb,args=(resz,)) # remember to have the comma here
    
    # write/read files on 1 pool
    write_sheet_aresult = pools.apply_async(write_to_sheet,args=(5,))

    # camera stream
    stream = cv2.VideoCapture(args.source)
    async_result_list=[]
    while True:
        try:
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
        except (KeyboardInterrupt, SystemExit):
            print("Caught KeyboardInterrupt, terminating workers and ending programs...")
            stream.release()
            cv2.destroyAllWindows()
            pools.terminate()
            pools.join()
            break

    # wait for all the frame-putting tasks to complete:
    
    # for f in async_result_list:
    #     f.get() # Return the result when it arrives
    # FRAME_BUFFER.put(None)
    # PRED_BUFFER.put(None)
    # NAME_BUFFER.put(None)
    # FINAL_PREDS.put(None)
    # show_frame_aresult.get()
    # write_sheet_aresult.get()

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