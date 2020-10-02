#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


path = '../input/ph-recognition/ph-data.csv'
df = pd.read_csv(path)
y = df['label']
from sklearn.preprocessing import StandardScaler
caler = StandardScaler()
xtrain = df.iloc[:,:-1]


# In[ ]:


df


# In[ ]:


caler.fit(xtrain)


# In[ ]:


xtrain  = caler.transform(xtrain)


# In[ ]:


from keras.utils import to_categorical


# In[ ]:


y = to_categorical(y)


# In[ ]:


from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV
logistic = linear_model.LogisticRegression()


# In[ ]:


penalty = ['l1', 'l2']
C = np.logspace(0, 4, 10)
hyperparameters = dict(C=C, penalty=penalty)


# In[ ]:


clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)


# In[ ]:


y.shape


# In[ ]:


clf.fit(xtrain,df['label'])


# In[ ]:


clf.best_params_


# In[ ]:


logistic = linear_model.LogisticRegression(C= 1.0, penalty='l2')


# In[ ]:


logistic.fit(xtrain,df['label'])


# In[ ]:


path_to_image = '../input/peatsoil/download.jpg'

import cv2
img = cv2.imread(path_to_image, cv2.IMREAD_COLOR)
bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# In[ ]:


bw_img.shape


# In[ ]:


red = bw_img[:,:,0]


# In[ ]:


green = bw_img[:,:,1]
blue = bw_img[:,:,2]


# In[ ]:


red = np.mean(red)
green = np.mean(green)
blue = np.mean(blue)


# In[ ]:


red


# In[ ]:


test = caler.transform([[blue,green,red]])


# In[ ]:


logistic.predict(test)

