# Face Recognition Project

## Setup

### 1. Install the environment
- Install Anaconda
- Run ```conda env create --file environment.yml``` in terminal
- Activate conda environment using ```conda activate face_rec```

### 2. Create database
- Create a folder called 'Database'
- Put in pictures of people you want to recognize (1 picture per person) into `/Database` directory. The image name should be <person_name>.<image_extension>, for example:

![](imgs/demo1.png) 

### 3.Configuration file for face_recognition model
- For each model, there will be a <model_name>.yaml file in ```/config```, e.g. ```/config/facerec.yaml```
- In ```facerec.yaml```, you can change a few hyperparameters for face_recognition model. Important params:
    - ```thres``` (default: 0.45) Lower threshold => harder to get a face in encoding database, but more accurate
    - ```enc_force_load```: (default: False) Whether you want to rebuild the encoding database. Must set to `True` when you run the program for the first time
    - ```frame_skip```: (default: 8) Number of frame to skip before making a prediction. Higher frame_skip => smoother FPS. Good when the prediction time is high
    - ```enc_list```: To add new encodings without rebuilding the database. Let's say you add 2 more images (personA.jpg and personB.jpg), then you will put ```enc_list: "personA personB"```

### 4. Run the application
- Run ```python face_main.py```

Optional arguments for ```face_main.py``` include 
- `--display` (to show frame or not)
- `--source` (to choose webcam source, e.g. 0 for first webcam, 1 for second webcam...)
- `--framesize` to resize frame for speed-up

