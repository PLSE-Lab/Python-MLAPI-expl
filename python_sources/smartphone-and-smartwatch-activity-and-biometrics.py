#!/usr/bin/env python
# coding: utf-8

# Human Activity Tracker
# 
# Human Activity Detection is the problem of predicting what a person is doing based on a trace of their movement using sensors. The "Human Activity Detection #Dataset" includes data collected from 34 subjects, each of whom were asked to perform 18 tasks for 3 minutes each. Each subject had a smartwatch placed on his/her dominant hand and a smartphone in their pocket. The data collection was controlled by a custom-made app that ran on the #smartphone and #smartwatch. The sensor data that was collected was from the accelerometer and gyroscope on both the smartphone and smartwatch, yielding four total sensors. The sensor data was collected at a rate of 20 Hz (i.e., every 50ms). The smartphone was either the Google Nexus 5/5X or Samsung Galaxy S5 running Android 6.0 (Marshmallow). The smartwatch was the LG G Watch running Android Wear 1.5.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/smartphone-and-smartwatch-activity-and-biometrics/wisdm-dataset/wisdm-dataset/raw/phone/accel'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


from __future__ import print_function
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns
#import coremltools
from scipy import stats
from IPython.display import display, HTML

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils


# Phone Accelerometer Train/test

# In[ ]:


columns=['user','activity','time','x','y','z']

data_phone_accel_sum = pd.DataFrame(data=None,columns=columns)
for dirname, _, filenames in os.walk('/kaggle/input/smartphone-and-smartwatch-activity-and-biometrics/wisdm-dataset/wisdm-dataset/raw/phone/accel'):
    for filename in filenames:
        df = pd.read_csv('/kaggle/input/smartphone-and-smartwatch-activity-and-biometrics/wisdm-dataset/wisdm-dataset/raw/phone/accel/'+filename , sep=",", header=None)
        temp=pd.DataFrame(data=df.values, columns=columns)
        data_phone_accel_sum=pd.concat([data_phone_accel_sum,temp])


# In[ ]:


data_phone_accel_sum['z'] = data_phone_accel_sum['z'].str.replace(';','')
data_phone_accel_sum['activity'].value_counts()
data_phone_accel_sum['x']=data_phone_accel_sum['x'].astype('float')
data_phone_accel_sum['y']=data_phone_accel_sum['y'].astype('float')
data_phone_accel_sum['z']=data_phone_accel_sum['z'].astype('float')
data_phone_accel_sum.info()


# Phone Gyro files import Train/test

# In[ ]:



data_phone_gyro_sum = pd.DataFrame(data=None,columns=columns)
for dirname, _, filenames in os.walk('/kaggle/input/smartphone-and-smartwatch-activity-and-biometrics/wisdm-dataset/wisdm-dataset/raw/phone/gyro'):
    for filename in filenames:
        df = pd.read_csv('/kaggle/input/smartphone-and-smartwatch-activity-and-biometrics/wisdm-dataset/wisdm-dataset/raw/phone/gyro/'+filename , sep=",", header=None)
        temp=pd.DataFrame(data=df.values, columns=columns)
        data_phone_gyro_sum=pd.concat([data_phone_gyro_sum,temp])


# In[ ]:


data_phone_gyro_sum['z'] = data_phone_gyro_sum['z'].str.replace(';','')

data_phone_gyro_sum['x']=data_phone_gyro_sum['x'].astype('float')
data_phone_gyro_sum['y']=data_phone_gyro_sum['y'].astype('float')
data_phone_gyro_sum['z']=data_phone_gyro_sum['z'].astype('float')

data_phone_gyro_sum['activity'].value_counts()
data_phone_gyro_sum.info()


# Watch Gyro files import train/test

# In[ ]:



data_watch_gyro_sum = pd.DataFrame(data=None,columns=columns)
for dirname, _, filenames in os.walk('/kaggle/input/smartphone-and-smartwatch-activity-and-biometrics/wisdm-dataset/wisdm-dataset/raw/watch/gyro'):
    for filename in filenames:
        df = pd.read_csv('/kaggle/input/smartphone-and-smartwatch-activity-and-biometrics/wisdm-dataset/wisdm-dataset/raw/watch/gyro/'+filename , sep=",", header=None)
        temp=pd.DataFrame(data=df.values, columns=columns)
        data_watch_gyro_sum=pd.concat([data_watch_gyro_sum,temp])


# In[ ]:


data_watch_gyro_sum['z'] = data_watch_gyro_sum['z'].str.replace(';','')
data_watch_gyro_sum['x']=data_watch_gyro_sum['x'].astype('float')
data_watch_gyro_sum['y']=data_watch_gyro_sum['y'].astype('float')
data_watch_gyro_sum['z']=data_watch_gyro_sum['z'].astype('float')

data_watch_gyro_sum['activity'].value_counts()
data_watch_gyro_sum.info()


# Watch accelorometer files import train test

# In[ ]:



data_watch_accel_sum = pd.DataFrame(data=None,columns=columns)
for dirname, _, filenames in os.walk('/kaggle/input/smartphone-and-smartwatch-activity-and-biometrics/wisdm-dataset/wisdm-dataset/raw/watch/accel'):
    for filename in filenames:
        df = pd.read_csv('/kaggle/input/smartphone-and-smartwatch-activity-and-biometrics/wisdm-dataset/wisdm-dataset/raw/watch/accel/'+filename , sep=",", header=None)
        temp=pd.DataFrame(data=df.values, columns=columns)
        data_watch_accel_sum=pd.concat([data_watch_accel_sum,temp])


# In[ ]:


data_watch_accel_sum['z'] = data_watch_accel_sum['z'].str.replace(';','')
data_watch_accel_sum['x']=data_watch_accel_sum['x'].astype('float')
data_watch_accel_sum['y']=data_watch_accel_sum['y'].astype('float')
data_watch_accel_sum['z']=data_watch_accel_sum['z'].astype('float')

data_watch_accel_sum['activity'].value_counts()
data_watch_accel_sum.info()


# Combining Phone accel and gyro data

# In[ ]:


df_phone = pd.DataFrame(data=None, columns=columns)
df_phone['user']= data_phone_accel_sum['user'].head(3608635)
df_phone['activity']= data_phone_accel_sum['activity'].head(3608635)
df_phone['time']= data_phone_accel_sum['time'].head(3608635)
df_phone['x'] = data_phone_gyro_sum['x'].values + data_phone_accel_sum['x'].head(3608635).values
df_phone['y'] = data_phone_gyro_sum['y'].values + data_phone_accel_sum['y'].head(3608635).values
df_phone['z'] = data_phone_gyro_sum['z'].values + data_phone_accel_sum['z'].head(3608635).values


# Combining watch acccel and gyro data

# In[ ]:


df_watch = pd.DataFrame(data=None, columns=columns)
df_watch['user']= data_watch_accel_sum['user'].head(3440342)
df_watch['activity']= data_watch_accel_sum['activity'].head(3440342)
df_watch['time']= data_watch_accel_sum['time'].head(3440342)
df_watch['x'] = data_watch_gyro_sum['x'].values + data_watch_accel_sum['x'].head(3440342).values
df_watch['y'] = data_watch_gyro_sum['x'].values + data_watch_accel_sum['y'].head(3440342).values
df_watch['z'] = data_watch_gyro_sum['x'].values + data_watch_accel_sum['z'].head(3440342).values


# In[ ]:


df_phone['activity'].value_counts()


# In[ ]:


df_watch['activity'].value_counts()


# Combining Phone and Watch Data

# In[ ]:


df_phone_watch = pd.DataFrame(data=None, columns=columns)
df_phone_watch['user']= df_phone['user'].head(3440342)
df_phone_watch['activity']= df_phone['activity'].head(3440342)
df_phone_watch['time']= df_phone['time'].head(3440342)
df_phone_watch['x'] = df_watch['x'].values + df_phone['x'].head(3440342).values
df_phone_watch['y'] = df_watch['y'].values + df_phone['y'].head(3440342).values
df_phone_watch['z'] = df_watch['z'].values + df_phone['z'].head(3440342).values


# In[ ]:


df_phone_watch.info()


# In[ ]:


df_phone_watch['activity'].value_counts()


# In[ ]:


Fs = 20


# In[ ]:


activities = df_phone_watch['activity'].value_counts().index


# In[ ]:


df_phone_watch = df_phone_watch.drop(['user', 'time'], axis=1)


# In[ ]:


df_phone_watch['activity'].value_counts()


# In[ ]:


df_a = df_phone_watch[df_phone_watch['activity']=='A'].head(174604)
df_m = df_phone_watch[df_phone_watch['activity']=='M'].head(174604)
df_k = df_phone_watch[df_phone_watch['activity']=='K'].head(174604)
df_p = df_phone_watch[df_phone_watch['activity']=='P'].head(174604)
df_e = df_phone_watch[df_phone_watch['activity']=='E'].head(174604)
df_o = df_phone_watch[df_phone_watch['activity']=='O'].head(174604)
df_c = df_phone_watch[df_phone_watch['activity']=='C'].head(174604)
df_d = df_phone_watch[df_phone_watch['activity']=='D'].head(174604)
df_l = df_phone_watch[df_phone_watch['activity']=='L'].head(174604)
df_b = df_phone_watch[df_phone_watch['activity']=='B'].head(174604)
df_h = df_phone_watch[df_phone_watch['activity']=='H'].head(174604)
df_f = df_phone_watch[df_phone_watch['activity']=='F'].head(174604)
df_g = df_phone_watch[df_phone_watch['activity']=='G'].head(174604)
df_q = df_phone_watch[df_phone_watch['activity']=='Q'].head(174604)
df_r = df_phone_watch[df_phone_watch['activity']=='R'].head(174604)
df_s = df_phone_watch[df_phone_watch['activity']=='S'].head(174604)
df_i = df_phone_watch[df_phone_watch['activity']=='I'].head(174604)
df_j = df_phone_watch[df_phone_watch['activity']=='J']


# In[ ]:


balanced_data = pd.DataFrame()
balanced_data = balanced_data.append([df_a,df_m,df_k,df_p,df_e,df_o,df_c,df_d,df_l,df_b,df_h,df_f,df_g,df_q,df_r,df_s,df_i,df_j]) 


# In[ ]:


balanced_data['activity'].value_counts()


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


label = LabelEncoder()
balanced_data['label'] = label.fit_transform(balanced_data['activity']) 
balanced_data


# In[ ]:


label.classes_


# **Statndardize Data**

# In[ ]:


from sklearn.preprocessing import StandardScaler

x = balanced_data[['x','y','z']]
y = balanced_data['label']
scaler = StandardScaler()
x = scaler.fit_transform(x)

scaled_x = pd.DataFrame(data=x, columns=['x','y','z'])
scaled_x['label'] = y.values

scaled_x


# **Frame Preparation**

# In[ ]:


import scipy.stats as stats


# In[ ]:


Fs=20
frame_size = Fs*4 #80
hop_size = Fs*2 #40


# In[ ]:


def get_frames(df, frame_size, hop_size):
    
    N_FEATURES = 3
    frames = []
    labels = []
    for i in range(0,len(df )- frame_size, hop_size):
        x = df['x'].values[i: i+frame_size]
        y = df['y'].values[i: i+frame_size]
        z = df['z'].values[i: i+frame_size]
        
        label = stats.mode(df['label'][i: i+frame_size])[0][0]
        frames.append([x,y,z])
        labels.append(label)
        
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)
    
    return frames, labels


# In[ ]:


x,y = get_frames(scaled_x, frame_size, hop_size)


# In[ ]:


x.shape, y.shape


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.30, random_state = 0, stratify = y)


# In[ ]:


y_train.shape


# In[ ]:


x_train.shape, x_test.shape


# In[ ]:


x_train[0].shape, x_test[0].shape


# In[ ]:


x_train = x_train.reshape(55109, 80, 3,1)
x_test = x_test.reshape(23619, 80, 3,1)


# **2D CNN Model**

# In[ ]:


model = Sequential()
model.add(Conv2D(256, (2,2), activation = 'relu', input_shape = x_train[0].shape))
model.add(Dropout(0.1))

model.add(Conv2D(512, (2,2), activation = 'relu'))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(18, activation='softmax'))


# In[ ]:


model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]) 


# In[ ]:


history = model.fit(x_train, y_train, epochs = 25, validation_data=(x_test, y_test), verbose=1 )

