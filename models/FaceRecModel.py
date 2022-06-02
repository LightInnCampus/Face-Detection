
from concurrent.futures import ThreadPoolExecutor
from http.client import EXPECTATION_FAILED
from tarfile import ENCODING
import face_recognition
from pathlib import Path
import numpy as np
import re
from utils.facerec_utils import *


DATA = Path('./Database/')
ENCODINGS = Path('./Weights/FaceRec_Encs/')

class FaceRecModel:
    def __init__(self,enc_model_size='large',frame_resz=0.2,upsample=1,
                model='hog',num_jitters=1,thres=0.4,enc_force_load=False,frame_skip=0):
        self.enc_model_size= enc_model_size
        self.frame_resz=frame_resz
        
        # How many times to upsample the image looking for faces. 
        # Higher numbers find smaller faces.
        self.upsample=upsample

        # “hog” is less accurate but faster on CPUs. 
        # “cnn” is a more accurate deep-learning model 
        # which is GPU/CUDA accelerated (if available)
        self.model=model

        # How many times to re-sample the face when calculating encoding. 
        # Higher is more accurate, but slower
        self.num_jitters = num_jitters

        self.thres = thres
        self.enc_force_load=enc_force_load

        self.frame_count=0
        self.frame_skip=frame_skip

        self.current_locations,self.current_names=[],[]

    def preprocess(self,enc_list=[]):
        '''
        Load data and/or pretrained model
        '''
        self.face_encs,self.face_names=[],[]

        # check path existence:
        if not DATA.exist():
            DATA.mkdir(parents=True)
            raise Exception(f"No images in database found. Please insert images in {str(DATA)}")
        if not ENCODING.exist():
            ENCODING.mkdir(parents=True)
        
        # empty database
        if self.enc_force_load or len(list(ENCODINGS.glob('**/*.npy')))==0:
            # remove all encodings
            print(f'Deleting all encodings in {ENCODINGS}')
            for f in ENCODINGS.glob('**/*.npy'):
                try: f.unlink()
                except OSError as e:
                    print(f'Error deleting {f}: {e}')

            files = [p for p in DATA.glob('**/*') if p.suffix in {'.png','.jpg','.jpeg'}]
            for f in files:
                print(f'Getting encodings from {f}...')
                img = face_recognition.load_image_file(f)
                enc = list(self.get_encodings(img))[0]
                name = Path(f).stem.lower()

                # write to encoding directory
                np.save(ENCODINGS/f'{name}.npy',enc)
                self.face_encs.append(enc)
                self.face_names.append(name)   

        else:
            for f in ENCODINGS.glob('**/*.npy'):
                print(f'Loading encodings from {f}...')
                name = Path(f).stem.lower()
                enc = np.load(ENCODINGS/f'{name}.npy')
                self.face_encs.append(enc)
                self.face_names.append(name)
            # Get encodings from extra list
            for name in enc_list.split():
                name = name.strip().lower()
                print(f'Get extra encodings for {name}...')
                f = [p for p in DATA.glob(f'**/{name}.*') if p.suffix in {'.png','.jpg','.jpeg'}]
                if len(f)==0 or len(f)>1:
                    print(f'Error getting encodings for {name}: too many images or image not found')
                    continue
                img = face_recognition.load_image_file(f[0])
                enc = list(self.get_encodings(img))[0]
                # write to encoding directory
                np.save(ENCODINGS/f'{name}.npy',enc)
                self.face_encs.append(enc)
                self.face_names.append(name) 

    def get_locations(self,frame):
        '''
        Return locations of faces from 1 single frame
        
        '''
        return face_locations(frame,number_of_times_to_upsample=self.upsample,model=self.model)
    
    def get_encodings(self,frame,locations=None):
        '''
        Return face encodings given frame and locations
        The most time-consuming task
        '''
        return face_encodings(frame,locations,model=self.enc_model_size,num_jitters = self.num_jitters)

    def predict(self,frame):
        '''
        Lower threshold => stricter to get a face in encoding database
        '''

        def get_names_from_encodings(enc):
            name="Unknown"
            face_distances = face_recognition.face_distance(self.face_encs, enc)
            best_match_index = np.argmin(face_distances)
            if face_distances[best_match_index]<=self.thres:
                name = self.face_names[best_match_index]
            return name

        frame = frame[:,:,::-1] # to rgb
    # upsample=1:
        # no get_locations: 202.25, with eye blink: 16.2
        # with get_locations : 
        #   no face: 12.6
        #   with face: 12.46, with eye blink: 8.65
        #  => massive reduction when adding get_locations
        #  => not much difference b/t with and without face.
    # upsample=0:
        # no get_locations: 277, with eye blink: 15
        # with get_locations 
        #   no face: 27.05
        #   with face: 28.6, with eye blink: 13.9
        # => >50% improvement with upsample 0

        current_names=[]
        current_locations = self.get_locations(frame)

        if self.frame_count%self.frame_skip==0:
            if len(current_locations):
                current_encodings = self.get_encodings(frame,current_locations)
                with ThreadPoolExecutor(max_workers=4) as executor:
                    current_names = executor.map(get_names_from_encodings,current_encodings)

        self.frame_count+=1
        if self.frame_count>=self.frame_skip*10:
            self.frame_count=0
        return current_locations,list(current_names)



    

