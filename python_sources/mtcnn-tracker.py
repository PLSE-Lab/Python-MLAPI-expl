#!/usr/bin/env python
# coding: utf-8

# By skipping frames and using trackers, we can extract facial bounding boxes from video in the Deepfake Detection Challenge.
# 
# To begin, we install MTCNN, which has a dependency of opencv-python. 
# 
# Then we uninstall opencv-python so we can install opencv-contrib-python to use opencv's trackers.

# In[ ]:


get_ipython().system('pip install /kaggle/input/dfdc-packages/mtcnn-0.1.0-py3-none-any.whl')
get_ipython().system('pip uninstall numpy -y')
get_ipython().system('pip uninstall opencv-python -y')
get_ipython().system('pip install --upgrade --force-reinstall /kaggle/input/dfdc-packages/numpy-1.18.1-cp36-cp36m-manylinux1_x86_64.whl')
get_ipython().system('pip install --upgrade --force-reinstall --no-deps /kaggle/input/dfdc-packages/opencv_contrib_python-4.2.0.32-cp36-cp36m-manylinux1_x86_64.whl')


# In[ ]:


import os
import sys
import cv2
import csv
import time
import string
import random
from mtcnn.mtcnn import MTCNN
from datetime import timedelta
from joblib import Parallel, delayed


# We introduce a helper function to generate ids for each trackable face

# In[ ]:


def id_gen(chars=string.ascii_uppercase, id_len=6):
    return ''.join(random.choice(chars) 
             for x in range(id_len)).lower()


# Finally, the main function iterates through frames in the source video.
# 
# To process videos more quickly, we grab frames and only load every skip_frame to perform face detection.
# 
# Since trackers drift, we reinitialize trackers after an expiration period.
# 
# We write the frame number and corresponding bounding box to files for each trackable face.

# In[ ]:


def main(source, skip_frames=5, expiration=30):
    cap = cv2.VideoCapture(source)
    name = os.path.basename(source).replace('.mp4', '.csv')
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for idx in range(n_frames):
        ret = cap.grab()
        if ret:
            if not idx % skip_frames:
                ret, frame = cap.retrieve()
                if not idx % expiration:
                    file_dict, tracker_dict = {}, {}
                    faces = [tuple(face['box']) for face in detector.detect_faces(frame)]
                    for box in faces:
                        tracker_id = id_gen()
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(frame, box)
                        tracker_dict[tracker] = tracker_id
                        dataFile = open(DATA_DIR + '{}_{}'.format(tracker_id, name), 'w')
                        file_dict[tracker_id] = csv.writer(dataFile)
                else:
                    for tracker in list(tracker_dict.keys()):
                        (success, box) = tracker.update(frame)
                        box = list(map(int, box))
                        file_dict[tracker_dict[tracker]].writerow([idx] + box)
        else:
            break
    cap.release()
    return              


# In[ ]:


# Initialize face detector
detector = MTCNN()

INPUT_DIR = "/kaggle/input/deepfake-detection-challenge/test_videos/"
DATA_DIR = '/kaggle/working/'

start_time = time.time()
for vid_fl in os.listdir(INPUT_DIR):
    main(INPUT_DIR + vid_fl)
elapsed = time.time() - start_time
print("Elapsed time to process test set: ", str(timedelta(seconds=elapsed)))


# Processing video files sequentially is time consuming and inefficient. To speed up our processing we can use the ```Parallel``` module from the ```joblib``` library and take advantage of all cores available to process chunks of videos.

# In[ ]:


get_ipython().system('mkdir -p /kaggle/working/output_parallel/')

DATA_DIR = '/kaggle/working/output_parallel/'
test_videos = os.listdir(INPUT_DIR)
test_videos = [INPUT_DIR + fl for fl in test_videos]

start_time = time.time()
Parallel(n_jobs=4)(delayed(main)(fl) for fl in test_videos)
elapsed = time.time() - start_time
print("Elapsed time to process test set: ", str(timedelta(seconds=elapsed)))

