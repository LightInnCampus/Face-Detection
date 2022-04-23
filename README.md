# Face Recognition Project

## Setup

### 1. Install the environment
- Install Anaconda
- Run ```conda env create --file environment.yml``` in terminal
- Activate conda environment using ```conda activate face_rec```

### 2. Create database
- Create a folder called 'Database'
- Put in pictures of people you want to recognize (1 picture per person). The image name should be <person_name>.<image_extension>, for example:

![](imgs/demo1.png) 


### 3. Run the application
- Run ```python face_main.py```

Optional arguments for ```face_main.py``` include `--display` (to show frame or not) and `--source` to choose webcam source (e.g. 0 for first webcam, 1 for second webcam...)

