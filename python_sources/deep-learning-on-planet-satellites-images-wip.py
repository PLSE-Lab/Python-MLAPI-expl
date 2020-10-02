#!/usr/bin/env python
# coding: utf-8

# # I plan to apply deep learning to 10% of training data with pre-trained model.  I will update it step by step. 
# 
#  Now it is working in progress.  Give me a little time!!!

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ## Import libraries needed

# In[ ]:


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import csv
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.utils.np_utils import to_categorical
from keras import backend as K
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Create 10% subset of training data

# In[ ]:


lines =[]
with open('../input/train.csv') as csvfile:   
    reader = csv.reader(csvfile)
    for line in reader:
         lines.append(line)
lines= np.array(lines) 
np.shape(lines)


# In[ ]:


#### select 10% of training data randomly


# In[ ]:


msk = (np.random.rand(len(lines))< 0.1)
lines2=lines[msk]
print(np.shape(lines2))
lines2[0:10]


# ###  add index for each class

# In[ ]:


cl=np.unique(lines[:,1])
cll=list(cl)

clll=[]
for i in lines2[:,1]:
   clll.append(cll.index(i))


# In[ ]:


lines22=[]
for i in range(len(lines2)):
   ln=np.hstack((lines2[i],clll[i]))
   lines22.append(ln) 


# In[ ]:


lines22=np.array(lines22)
#np.shape(lines22)
lines22[0:20]


# In[ ]:


for line in lines2:   
    for i in range(3):
        source_path=line[i]
        tokens =source_path.split('/')
        filename = tokens[-1]
        local_path = ("/Users/TOSHI/Downloads/data/IMG/"+filename)
        #local_path = ("/Users/kugatoshifumi//Downloads/data/IMG/"+filename)
        image= cv2.imread(local_path)
        images.append(image)
        #print(image)
        #exit()

