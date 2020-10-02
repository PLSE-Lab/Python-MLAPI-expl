#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


y_test1 = pd.read_csv("../input/numberplate/trainVal.csv")


# In[ ]:


y_test2 = y_test1
y_test2.head()


# In[ ]:


y_test2 = y_test2.drop('image_path',axis = 1)


# In[ ]:


y_test2.head()


# In[ ]:


y_test2.shape


# In[ ]:


y_test2 = y_test2.drop('train',axis = 1)


# In[ ]:


y_test2.head()


# In[ ]:


index=range(0,140)
# indexy_test2 = y_test2

columns=['track_id','lp']
test = pd.DataFrame(index=index, columns=columns)


# In[ ]:


test.head()


# In[ ]:


j=0
for i in range(len(y_test2)):
    if i%2 != 0:
        test['track_id'][j] = y_test2['track_id'][i]
        test['lp'][j] = y_test2['lp'][i]
        j  += 1


# In[ ]:


test['lp'][140] = y_test2['lp'][280]
test['lp'][141] = y_test2['lp'][281]


# In[ ]:


len(test['lp']) == (len(y_test2['lp'])/2+1)


# In[ ]:


test.to_csv('trainValidation.csv')


# In[ ]:


test['lp'][140]


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebraX_train[0]
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.y_test1 = pd.read_csv("../input/numberplate/trainVal.csv")
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/vehicle-number-plate/h3m3/h3m3/crop_h3"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# y_test = os.listdir("../input/vehicle-number-plate/h3m3/h3m3/crop_h3"))
y_test = pd.read_csv("../input/number-plate-dataset/trainValidation (1).csv")
y_test.head(10)


# In[ ]:


y_test.tail(10)


# In[ ]:


import os, cv2, re, random
import numpy as np
import pandas as pd

TRAIN_DIR = '../input/vehicle-number-plate/h3m3/h3m3/crop_h3/'
train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]


# In[ ]:


train_images_list = []
for i in train_images:
    train_images_list.append(i)


# In[ ]:


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]


# In[ ]:


train_images_list.sort(key=natural_keys)


# In[ ]:


def prepare_data(list_of_images):
    x = [] 
    for image in list_of_images:
        '''Reshaping the images of number plates in 100x100x3 in order to reduce calculation done by neural network'''
        x.append(cv2.resize(cv2.imread(image), (100,100), interpolation=cv2.INTER_CUBIC))     
    return x


# In[ ]:


X_train = prepare_data(train_images_list)


# In[ ]:


X_train[0].shape


# In[ ]:


X_train[0]


# In[ ]:


train_images[0], train_images[0][40:]


# In[ ]:


import matplotlib.pyplot as plt 
plt.imshow(X_train[131])


# In[ ]:


from PIL import Image
Image.open(train_images[51])

