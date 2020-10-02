#!/usr/bin/env python
# coding: utf-8

# **VISIT MENTION URL FOR DETAILED EXPLANATION**
# https://www.thepythoncode.com/article/detect-faces-opencv-python
# https://becominghuman.ai/face-detection-using-opencv-with-haar-cascade-classifiers-941dbb25177

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[ ]:


#loading the test image
image = cv2.imread("/kaggle/input/facedetect/family.png")


# Before we detect faces in the image, we will first need to convert the image to grayscale, 
# that is because the function we gonna use to detect faces expects a grayscale image

# In[ ]:


#converting to grayscale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# In[ ]:


#initiliaze the face recognizer (dafault face haar cascade)
face_cascade = cv2.CascadeClassifier('/kaggle/input/facedetect/haarcascade_frontalface_default.xml')


# In[ ]:


#detect all faces in the image
faces = face_cascade.detectMultiScale(image_gray,
                                      scaleFactor=1.3,
                                      minNeighbors=3,
                                      minSize=(10, 10))


# In[ ]:


#print no of faces detected
print(len(faces))


# detectMultiScale() function takes an image as parameter and detects objects of different sizes as a list of rectangles, let's draw these rectangles in the image:

# In[ ]:


image_list = []
for x,y, width,height in faces:
    cv2.rectangle(image, (x,y),(x+width,y+height), color=(255,0,0), thickness=2)
    roi_color = image[y:y+height,x:x+width]
    cv2.imwrite(str(width)+str(height)+'_face.jpg',roi_color)
    image_list.append(str(width)+str(height)+'_face.jpg')


# In[ ]:


#save the image with rectangles
cv2.imwrite('box.jpg',image)


# Original Image

# In[ ]:


#from pylab import rcParams
#rcParams['figure.figsize'] = 5, 5


# In[ ]:


original_image = mpimg.imread('/kaggle/input/facedetect/family.png')
plt.imshow(original_image)


# Rectangle_image

# In[ ]:


box_image = mpimg.imread('box.jpg')
plt.imshow(box_image)


# Extracted Face

# In[ ]:


print(image_list)


# In[ ]:


extract_image1 = mpimg.imread('133133_face.jpg')
plt.imshow(extract_image1)


# In[ ]:


extract_image2 = mpimg.imread('139139_face.jpg')
plt.imshow(extract_image2)


# In[ ]:


extract_image3 = mpimg.imread('176176_face.jpg')
plt.imshow(extract_image3)

