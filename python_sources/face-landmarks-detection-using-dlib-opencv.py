#!/usr/bin/env python
# coding: utf-8

# ## Face landmark detection using dlib, OpenCV
# 
# - In this task of facial landmarks detection, firstly, the face has to be detected in a given image then the face has to be analysed to obtain the face landmarks/keypoints.
# - Facial landmarks/keypoints are useful to know the alignment of face and face features positions.
# 
# ### Reference(s):
# - https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/

# ## 1. Install libraries which are not available

# In[ ]:


get_ipython().system('pip install dlib')


# In[ ]:


get_ipython().system('pip install imutils')


# ## 2. Some basic steps

# In[ ]:


import os
import cv2
import dlib
import imutils
from imutils import face_utils
import matplotlib.pyplot as plt

image_path = "../input/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/Chip_Knight/Chip_Knight_0001.jpg"
filter_path = "../input/dlib-landmarks-predictor/shape_predictor_68_face_landmarks.dat"

print(dlib.__version__)
print(imutils.__version__)


# In[ ]:


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(filter_path)


# ## 3. Load image for fetching its facial landmarks

# In[ ]:


image = cv2.imread(image_path)
image = imutils.resize(image, width=500)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
# detect faces in the image
rects = detector(image, 1)


# In[ ]:


for (i, rect) in enumerate(rects):
    shape = predictor(image, rect)
    shape = face_utils.shape_to_np(shape)
    
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 255, 0), 3)

plt.figure(figsize=(10,10))
plt.imshow(image)
plt.xticks([])
plt.yticks([])
plt.show()

