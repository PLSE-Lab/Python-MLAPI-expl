#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, sys, time
import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


import sys
sys.path.insert(0, "/kaggle/input/retinaface/RetinaFace_Dataset/")


# In[ ]:


from retinaface_faceextraction import RetinaFacesFaceExtractor
from retinaface_loadmodel import RetinaFaceLoadModel
from RetinaFace.retinaface import RetinaFace


# In[ ]:


RETINA_PATH = "/kaggle/input/retinaface/RetinaFace_Dataset/"


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trained_model_path = RETINA_PATH+"RetinaFace/weights/Final_Retinaface.pth"
model_best_path = RETINA_PATH+"RetinaFace/model_best.pth.tar"

RM = RetinaFaceLoadModel()

torch.set_grad_enabled(False)

#net and model
net = RetinaFace(model_best_path, phase="test")
net = RM.load_model(net , trained_model_path, device)
net.eval()
print("Finished loading model!")
net = net.to(device)


# In[ ]:


from read_video import VideoReader


# In[ ]:


frames_per_video = 16

video_reader = VideoReader()
video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)


# In[ ]:


vis_threshold = 0.9
dim = (200, 200)

rf = RetinaFacesFaceExtractor(video_read_fn, frames_per_video, net, vis_threshold,dim, device)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'faces_list, faces_landms_list = rf.process_video("/kaggle/input/deepfake-detection-challenge/test_videos/nymodlmxni.mp4")')


# In[ ]:


for i in range(len(faces_list)):
    plt.imshow(faces_list[i])
    plt.show()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'faces_list, faces_landms_list = rf.process_video("/kaggle/input/deepfake-detection-challenge/test_videos/pxjkzvqomp.mp4")')


# In[ ]:


for i in range(len(faces_list)):
    plt.imshow(faces_list[i])
    plt.show()


# In[ ]:




