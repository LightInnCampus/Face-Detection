import face_recognition
import cv2
import numpy as np
import time


# Get a reference to webcam #0 (the default one), or #1 (external one)
video_capture = cv2.VideoCapture(0)

prev_frame_time = 0

# Load a sample picture and learn how to recognize it.
face1_image = face_recognition.load_image_file("Database/Jenny.jpg")
face1_face_encoding = face_recognition.face_encodings(face1_image,model='large')[0]

# Load a second sample picture and learn how to recognize it.
face2_image = face_recognition.load_image_file("Database/Yen.jpg")
face2_face_encoding = face_recognition.face_encodings(face2_image,model='large')[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    face1_face_encoding,
    face2_face_encoding
]
known_face_names = [
    "Jenny",
    "Yen"
]

font = cv2.FONT_HERSHEY_DUPLEX

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
frame_resize = 0.2 # resize the frame. e.g. 0.2 means resizing frame to 20% its sisize
frame_skip = 0 # number of frame to skip. 0 means dont skip any frame
n_frame=0

while True:
    n_frame+=1

    start = time.time()

    # Grab a single frame of video
    ret, frame = video_capture.read()
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=frame_resize, fy=frame_resize)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    if n_frame%(frame_skip+1)==0:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame,number_of_times_to_upsample=1)
        # TODO: time-consuming task: convert frame (or face_location based on frame) to face encodings
        # face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations,model='large')
        face_names = []

        name="Unknown"
        # for face_encoding in face_encodings:
        #     # See if the face is a match for the known face(s)
        #     matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        #     name = "Unknown"

        #     # l2 norm
        #     face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        #     best_match_index = np.argmin(face_distances)
        #     # if matches[best_match_index]:
        #     if face_distances[best_match_index]<=0.4:
        #         name = known_face_names[best_match_index]

        #     face_names.append(name)


    # Display the results
    face_names = ["Temp"]*len(face_locations) # display temporary tag for each face location
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations 
        top = int(top/frame_resize)
        right = int(right/frame_resize)
        bottom = int(bottom/frame_resize)
        left = int(left/frame_resize)
        
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        # cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)



    # show frame per second
    end = time.time()
    cv2.putText(frame, str(1/(end-start)), (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()