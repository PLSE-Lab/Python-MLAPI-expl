#!/usr/bin/env python
# coding: utf-8

# # Face detection in a couple of lines with dlib

# To install dlib without internet, add the dataset: https://www.kaggle.com/carlossouza/dlibpkg

# In[ ]:


get_ipython().system("pip install '/kaggle/input/dlibpkg/dlib-19.19.0'")


# In[ ]:


import numpy as np
import cv2
from matplotlib import pyplot as plt
import dlib


# In[ ]:


get_ipython().run_cell_magic('time', '', "sample = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/aagfhgtpmv.mp4'\n\nreader = cv2.VideoCapture(sample)\n_, image = reader.read()\nimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)\n\nface_detector = dlib.get_frontal_face_detector()\ngray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\nfaces = face_detector(gray, 1)\nif len(faces) > 0:\n    face = faces[0]\n    \nface_image = image[face.top():face.bottom(), face.left():face.right()]")


# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
ax1.imshow(image)
ax1.xaxis.set_visible(False)
ax1.yaxis.set_visible(False)

ax2.imshow(face_image)
ax2.xaxis.set_visible(False)
ax2.yaxis.set_visible(False)

plt.grid(False)
plt.tight_layout()


# In[ ]:




