#!/usr/bin/env python
# coding: utf-8

# I referred to this kernal: "https://www.kaggle.com/wesamelshamy/high-correlation-feature-image-classification-conf/data"
# To solve the problem of disk space, and being not able to analyze image of every images, I tried to make a loop, in which create directory, load image, generate score, and remove the directory. By doing this we could save the disk space but the problem is it takes too long. I think this is inevitable dilemma between storage capacity and processing capacity. I'm trying to use the score (prediction certainty) as one of the features. It seems that the images from the Zip file is not in the order of items in training data csv, so I'm trying to index them using image path. 

# In[39]:


import pandas as pd
import numpy as np
import os
from zipfile import ZipFile
import cv2
import numpy as np
import pandas as pd
from dask import bag, threaded
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
df = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

df_train = df
df_test = test


# In[40]:


# get filenames
zipped = ZipFile('../input/train_jpg.zip')
filenames = zipped.namelist()[1:] # exclude the initial directory listing
print(len(filenames))


# In[ ]:


#get blurrness score

def get_blurrness(file):
    exfile = zipped.read(file)
    arr = np.frombuffer(exfile, np.uint8)
    if arr.size > 0:   # exclude dirs and blanks
        imz = cv2.imdecode(arr, flags=cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(imz, cv2.CV_64F).var()
    else: 
        fm = -1
    return fm

blurrness = []
iteration = filenames[600000:750000]
for i in range(0, len(iteration)):
    print(i)
    blurrness.append(get_blurrness(iteration[i]))
    
frame = pd.DataFrame({"File" : filenames[600000:750000], "Score": blurrness})
frame.to_csv("7.csv", index = False)

