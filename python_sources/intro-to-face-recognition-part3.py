#!/usr/bin/env python
# coding: utf-8

# **Face Encoding**: The process of taking an image of a face and turning into a set of measurements. Face encodings can be done automatically by using Deep Metric Learning.
# 
# **Deep Metric Learning**: Using deep learning to have a computer come up in a way to measure something that you don't know how to measure yourself.
# 

# # Training 
# We are going to check whether given person is in the picture or not.

# In[ ]:


get_ipython().system('pip install face_recognition')
#!pip install pillow
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import PIL.Image
import PIL.ImageDraw
import face_recognition

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


#person data loaded
image_of_person=face_recognition.load_image_file("../input/input-people/jennifer-aniston-as-rachel.jpg")
plt.imshow(image_of_person)
plt.show()


# In[ ]:


#face encoding process
person_face_encoding=face_recognition.face_encodings(image_of_person)[0]
known_face_encodings=[person_face_encoding] #can be multiple

#load the image that we want to check
unknown_image=face_recognition.load_image_file("../input/input-people/friends.jpg")
unknown_face_encodings=face_recognition.face_encodings(unknown_image)
plt.imshow(unknown_image)
plt.show()

name="Rachel"
for unknown_face_encoding in unknown_face_encodings:
    results=face_recognition.compare_faces(known_face_encodings,unknown_face_encoding)
    if results[0]:
        print("Found {} in the photo!".format(name))
    


# References:
# 
# 1.Linkedin Learning-Deep Learning: Face Recognition
