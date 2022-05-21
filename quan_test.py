from multiprocessing import Pool, Queue
import time
import cv2

# intialize global variables for the pool processes:
def init_pool(d_a,d_b):
    global frame_buffer
    global pred_buffer
    frame_buffer,pred_buffer = d_a,d_b


def detect_object(frame):
    time.sleep(2)
    print('Calculation done')
    pred_buffer.put(frame)


def show():
    while True:
        print("Show frame")
        frame = frame_buffer.get()
        if frame is not None:
            cv2.imshow("Video", frame)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return


# required for Windows:
if __name__ == "__main__":

    frame_buffer,pred_buffer = Queue(),Queue()
    # 6 workers: 1 for the show task and 5 to process frames:
    pool = Pool(6, initializer=init_pool, initargs=(frame_buffer,pred_buffer))
    # run the "show" task:
    show_future_aresult = pool.apply_async(show)

    cap = cv2.VideoCapture(0)

    futures = []
    while True:
        ret, frame = cap.read()
        if ret:
            frame_buffer.put(frame)
            f_aresult = pool.apply_async(detect_object, args=(frame,))
            futures.append(f_aresult)
            # time.sleep(0.001)
        else:
            break
    # wait for all the frame-putting tasks to complete:
    for f in futures:
        f.get() # Return the result when it arrives

    # signal the "show" task to end by placing None in the queue
    frame_buffer.put(None)
    # pred_buffer.put(None)

    show_future_aresult.get()
    print("program ended")