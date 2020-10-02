#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


DATAQ = pd.read_csv('../input/training.csv')


# In[3]:


DATAQ.head(3)


# In[4]:


get_ipython().system('pip install geohash')


# **DECODE GEOHASH6**
# 

# In[5]:


import Geohash as geo


# In[6]:


LONG=[]
LAT=[]

for i in DATAQ['geohash6']:
    DUMMY = geo.decode_exactly(i)
    X,Y,E,R = DUMMY
    LONG.append(float(X))
    LAT.append(float(Y))
    
    


# MAKE NEW

# In[7]:


DATAQ['LONGITUDE']=LONG
DATAQ['LATITUDE']=LAT

DATAQ.tail(10)


# In[8]:


DATAQ.shape
DATAQi=DATAQ.drop(['geohash6'],axis=1)


# In[9]:


DATAQi.head(10)


# In[ ]:





# In[11]:


NEW_TIME = []

for xi in DATAQi['timestamp']:
    a,b = xi.split(':')
    a = int(a)
    b = int(b)
    c = a*60 + b
    #print(c)

    NEW_TIME.append(c)


# In[13]:


DATAQi['NEW_TIME'] = NEW_TIME


# In[19]:



DATAQi = DATAQi.drop(['timestamp'],axis=1)
DATAQi.head(10)


# In[37]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow import keras as keras
from keras.models import Sequential
from keras.layers import *

from mpl_toolkits.mplot3d import axes3d, Axes3D


# DIVIDE DATA INTO TEST & TRAINING

# In[38]:


DATA_TRAIN, DATA_TEST = train_test_split(DATAQi, test_size=0.2)


# In[41]:


print("INI SHAPE DATA_TRAIN {} DAN INI SHAPE DATA_TEST {}".format(DATA_TRAIN.shape, DATA_TEST.shape))


# **> SCALING AGAR DATA LEBIH KECIL**

# In[42]:


scaler = MinMaxScaler(feature_range = (0,1))

scale_training = scaler.fit_transform(DATA_TRAIN)
scale_test = scaler.transform(DATA_TEST)


# > **DATA_BARU HASIL SCALING**

# In[45]:


#Lihat hasil as sample

NEW_TRAINING = pd.DataFrame(scale_training, columns=DATA_TRAIN.columns.values)
NEW_TEST = pd.DataFrame(scale_test, columns=DATA_TEST.columns.values)

#write into file

#NEW_TRAINING.to_csv('../input/Training_OK.csv', index=False)
#NEW_TEST.to_csv('../input/Test_OK.csv', index=False)


# In[48]:


X = NEW_TRAINING.drop(['demand'],axis=1).values
Y = NEW_TRAINING[['demand']].values


# **> BUAT MODEL >**

# In[50]:


model = Sequential()
model.add(Dense(50, input_dim=4 , activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')


# In[51]:


model.fit(X,Y, epochs=50, shuffle=True, verbose=2)


# In[52]:



X_Test = DATA_TEST.drop(['demand'],axis=1).values
Y_Test = DATA_TEST[['demand']].values

Test_Error_Rate = model.evaluate(X_Test, Y_Test, verbose=0)
print("ini nilai MSE nya  --> {}".format(Test_Error_Rate))


# In[53]:


model.save("Bagus_DATA_TRAFFC_Model.h5")
print("Model Saved ")


# In[54]:


ls -l


# In[55]:


cp Bagus_DATA_TRAFFC_Model.h5 ../input/


# In[20]:


NEW_DATA = DATAQi[['LONGITUDE','LATITUDE','demand']]


# In[26]:


NEW_DATA.head(10)


# In[31]:


NEW_NORM = (NEW_DATA - NEW_DATA.mean())/(NEW_DATA.max() - NEW_DATA.min()) 


# In[32]:


NEW_NORM.head(5)


# In[33]:


fig = plt.figure(figsize=(8, 6))
ax = Axes3D(fig)


X = NEW_NORM['LONGITUDE'].values
Y = NEW_NORM['LATITUDE'].values
Z = NEW_NORM['demand'].values

Xi,yi = np.meshgrid(X,Y)

