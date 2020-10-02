#!/usr/bin/env python
# coding: utf-8

# # Detec face with dlib and recognition offline

# First import all dataset contain library you need. <br>
# * To install dlib without internet, add the dataset: https://www.kaggle.com/carlossouza/dlibpkg <br>
# * To install ace-recognition without internet, add the dataset: https://www.kaggle.com/minhtam/face-recognition

# Install dlib library

# In[ ]:


get_ipython().system("pip install '/kaggle/input/dlibpkg/dlib-19.19.0'")


# Install face-recognition without internet

# In[ ]:


get_ipython().system("pip install '/kaggle/input/face-recognition/face_recognition_models-0.3.0/face_recognition_models-0.3.0'")


# In[ ]:


get_ipython().system("pip install '/kaggle/input/face-recognition/face_recognition-0.1.5-py2.py3-none-any.whl'")


# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import face_recognition
import cv2
import matplotlib.pyplot as plt


# In[ ]:


v_cap = cv2.VideoCapture('/kaggle/input/deepfake-detection-challenge/test_videos/adohdulfwb.mp4')
for j in range(1):
    success, vframe = v_cap.read()


# In[ ]:


vframe = cv2.cvtColor(vframe,cv2.COLOR_BGR2RGB)
plt.imshow(vframe)


# In[ ]:



face_positions = face_recognition.face_locations(vframe)


# In[ ]:


for face_position in face_positions:
    y0,x1,y1,x0 = face_position
    img = cv2.rectangle(vframe,(x0,y0),(x1,y1),(255,0,0),5)
plt.imshow(img)


# In[ ]:




