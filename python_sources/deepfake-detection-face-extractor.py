#!/usr/bin/env python
# coding: utf-8

# <a id="toc"></a>
# # Table of Contents
# 1. [Introduction](#introduction)
# 1. [Configure hyper-parameters](#configure_hyper_parameters)
# 1. [Install required libraries and tools](#install_required_libraries_and_tools)
# 1. [Import libraries](#import_libraries)
# 1. [Define FaceExtractor](#define_face_extractor)
# 1. [Get metadata](#get_metadata)
# 1. [Start the detection process](#start_the_detection_process)
# 1. [Save metadata.csv](#save_metadata_csv)
# 1. [Compress results](#compress_results)
# 1. [Conclusion](#conclusion)

# <a id="introduction"></a>
# # Introduction
# This notebook dedicated to extracting all faces appears in each video of the training dataset. The name of each face image corresponds to the index of the frame that this face appears, plus a suffix _2, or _3, etc. if the number of faces in a frame is greater than 1.
# 
# ---
# [Back to Table of Contents](#toc)

# <a id="configure_hyper_parameters"></a>
# # Configure hyper-parameters
# [Back to Table of Contents](#toc)

# In[ ]:


TRAIN_DIR = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/'
TMP_DIR = '/kaggle/tmp/'
ZIP_NAME = 'dfdc_train_faces_sample.zip'
METADATA_PATH = TRAIN_DIR + 'metadata.json'

SCALE = 0.25
N_FRAMES = None


# <a id="install_required_libraries_and_tools"></a>
# # Install required libraries and tools
# [Back to Table of Contents](#toc)

# In[ ]:


get_ipython().system('pip install facenet-pytorch > /dev/null 2>&1')
get_ipython().system('apt install zip > /dev/null 2>&1')


# <a id="import_libraries"></a>
# # Import libraries
# [Back to Table of Contents](#toc)

# In[ ]:


import os
import glob
import json
import torch
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from facenet_pytorch import MTCNN

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {device}')


# <a id="define_face_extractor"></a>
# # Define FaceExtractor
# [Back to Table of Contents](#toc)

# In[ ]:


class FaceExtractor:
    def __init__(self, detector, n_frames=None, resize=None):
        """
        Parameters:
            n_frames {int} -- Total number of frames to load. These will be evenly spaced
                throughout the video. If not specified (i.e., None), all frames will be loaded.
                (default: {None})
            resize {float} -- Fraction by which to resize frames from original prior to face
                detection. A value less than 1 results in downsampling and a value greater than
                1 result in upsampling. (default: {None})
        """

        self.detector = detector
        self.n_frames = n_frames
        self.resize = resize
    
    def __call__(self, filename, save_dir):
        """Load frames from an MP4 video, detect faces and save the results.

        Parameters:
            filename {str} -- Path to video.
            save_dir {str} -- The directory where results are saved.
        """

        # Create video reader and find length
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Pick 'n_frames' evenly spaced frames to sample
        if self.n_frames is None:
            sample = np.arange(0, v_len)
        else:
            sample = np.linspace(0, v_len - 1, self.n_frames).astype(int)

        # Loop through frames
        for j in range(v_len):
            success = v_cap.grab()
            if j in sample:
                # Load frame
                success, frame = v_cap.retrieve()
                if not success:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                
                # Resize frame to desired size
                if self.resize is not None:
                    frame = frame.resize([int(d * self.resize) for d in frame.size])

                save_path = os.path.join(save_dir, f'{j}.png')

                self.detector([frame], save_path=save_path)

        v_cap.release()


# <a id="get_metadata"></a>
# # Get metadata
# [Back to Table of Contents](#toc)

# In[ ]:


with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f)


# In[ ]:


train_df = pd.DataFrame(
    [
        (video_file, metadata[video_file]['label'], metadata[video_file]['split'], metadata[video_file]['original'] if 'original' in metadata[video_file].keys() else '')
        for video_file in metadata.keys()
    ],
    columns=['filename', 'label', 'split', 'original']
)

train_df.head()


# <a id="start_the_detection_process"></a>
# # Start the detection process
# [Back to Table of Contents](#toc)

# In[ ]:


# Load face detector
face_detector = MTCNN(margin=14, keep_all=True, factor=0.5, device=device).eval()


# In[ ]:


# Define face extractor
face_extractor = FaceExtractor(detector=face_detector, n_frames=N_FRAMES, resize=SCALE)


# In[ ]:


# Get the paths of all train videos
all_train_videos = glob.glob(os.path.join(TRAIN_DIR, '*.mp4'))


# In[ ]:


get_ipython().system('mkdir -p $TMP_DIR')


# In[ ]:


with torch.no_grad():
    for path in tqdm(all_train_videos):
        file_name = path.split('/')[-1]

        save_dir = os.path.join(TMP_DIR, file_name.split(".")[0])

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Detect all faces appear in the video and save them.
        face_extractor(path, save_dir)


# <a id="save_metadata_csv"></a>
# # Save metadata.csv
# [Back to Table of Contents](#toc)

# In[ ]:


cd $TMP_DIR


# In[ ]:


train_df.to_csv('metadata.csv', index=False)


# <a id="compress_results"></a>
# # Compress results
# [Back to Table of Contents](#toc)

# In[ ]:


get_ipython().system('zip -r -m -q /kaggle/working/$ZIP_NAME *')


# <a id="conclusion"></a>
# # Conclusion
# Data preprocessing is an integral part of machine learning and usually costs data scientists lots of time and effort. Sometimes, this process can even take weeks to be done before we can move on to the next step, and the huge dataset from the Deepfake Detection Challenge gives me the feeling that I must spend even months to digest all of it with my "Napoleon" PC. So, to worth the time and effort I've spent when struggling with this nearly-half-terabyte dataset, I think it's better to share all of my work, hope that they will be useful for many others.
# 
# Besides the result of this demo on the sample training dataset, I have finished some small parts of the whole dataset which download links are shared in my [*Other useful datasets*](https://www.kaggle.com/c/deepfake-detection-challenge/discussion/128954) discussion. I'll gradually update the other parts in the next few days and public a kernel to demonstrate how to load and prepare these data to train a simple classifier as soon as possible.
# 
# ---
# For some who want a demo of how to load and merge multiple datasets at once before training a classifier, let check this [*kernel*](https://www.kaggle.com/phunghieu/loading-merging-multiple-kaggle-datasets-demo).
# 
# For training and inference examples, let check [*DFDC-Multiface-Training*](https://www.kaggle.com/phunghieu/dfdc-multiface-training) and [*
# DFDC-Multiface-Inference*](https://www.kaggle.com/phunghieu/dfdc-multiface-inference).
# 
# ---
# If you think this kernel is worth your time of reading, consider to upvote it :3
# 
# Many thanks! ^^
# 
# ---
# [Back to Table of Contents](#toc)
