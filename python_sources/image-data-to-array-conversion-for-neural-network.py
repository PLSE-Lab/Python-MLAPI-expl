#!/usr/bin/env python
# coding: utf-8

# # Image Data to Array Conversion for Neural Network Input

# **We often find difficulties or do not understand the way how to convert image data to array. Well, this particular notebook is for doing this conversion in a simple yet powerful way. This particular notebook used cats and dogs images and get them ready to input in a convolutional neural network**

# **Import Image Data and Convert to Features and Labels**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
import os


# In[ ]:


DATADIR='/kaggle/input/cats-and-dogs-sentdex-tutorial/kagglecatsanddogs_3367a/PetImages'
CATEGORIES=['Cat','Dog']

IMG_SIZE=100


# In[ ]:


training_data=[]
def create_training_data():
    for category in CATEGORIES:
        path=os.path.join(DATADIR, category)
        class_num=CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
create_training_data()            


# In[ ]:


import random
random.shuffle(training_data)


# In[ ]:


X=[]
y=[]

for categories, label in training_data:
    X.append(categories)
    y.append(label)
X= np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# In[ ]:


X.shape


# **Let's visualize what we have done from image to array conversion**

# In[ ]:


X1=X.reshape(X.shape[0],IMG_SIZE,IMG_SIZE)
plt.imshow(X1[100]) #plotting a random image of index 100
plt.show() 


# In[ ]:


X1[100] #showing the respective array of the image


# **Please upvote if you like this or find this notebook useful.**
