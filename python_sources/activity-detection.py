#!/usr/bin/env python
# coding: utf-8

# # Activity Detection
# 
# Human Activity Detection is the problem of predicting what a person is doing based on a trace of their
# movement using sensors.
# 

# The code is divided in 5 parts :
# 
# [1.Importing the libraries](#1)<br>
# [2. Loading and processing the Data](#2)<br>
# [3. Traning the model](#3)<br>
# [4. Predicting the output and calculating the accuracy](#4)<br>
# [5. Saving the model](#5)

# The "Human Activity Detection Dataset" includes data collected from 34 subjects, each of whom were
# asked to perform 18 tasks for 3 minutes each. Each subject had a smartwatch placed on his/her
# dominant hand and a smartphone in their pocket. The data collection was controlled by a custom-made
# app that ran on the smartphone and smartwatch. The sensor data that was collected was from the
# accelerometer and gyroscope on both the smartphone and smartwatch, yielding four total sensors. The
# sensor data was collected at a rate of 20 Hz (i.e., every 50ms). The smartphone was either the Google
# Nexus 5/5X or Samsung Galaxy S5 running Android 6.0 (Marshmallow). The smartwatch was the LG G
# Watch running Android Wear 1.5. The general characteristics of the data and data collection process are
# summarized in Table 1. More detailed information is presented later in this document.
# 
# ![image.png](attachment:image.png)

# Table 2 lists the 18 activities that were performed. The actual data files specify the activities using the
# code from Table 2. Similar activities are not necessarily grouped together (e.g., eating activities are not
# all together).
# ![image.png](attachment:image.png)
# 

# ![image.png](attachment:image.png)

# ## <a name="1"></a>1. Importing the libraries

# In[ ]:


# importing libraries 
import pandas as pd
import numpy
import os
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[ ]:


# library to save the model
import pickle


# ## About Data
# The raw sensor data is located in the directory. Each user has its own data file which is tagged with their<br>
# subject id, the sensor, and the device. Within the data file, each line is:<br>
# Subject-id, Activity Label, Timestamp, x, y, z
# 
# The features are defined as follows:
# * subject-id: Identifies the subject and is an integer value between 1600 and 1650.
# * activity-label: see Table 2 for a mapping from 18 characters to the activity name
# * timestamp: time that the reading was taken (Unix Time)
# * x: x sensor value (real valued)
# * y: y sensor value (real valued)
# * z: z sensor value (real valued)

# ## <a name="2"></a> 2. Loading and processing the Data

# In[ ]:


# to read the file from the location and process it to save it in a 2D list, with removing excess symbols like (';')
# each row in the list comprises of a separate data instance
def process(path_to_folder):
    train = []
    for root, dirs, files in os.walk(path_to_folder):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    text = f.read()
                    temp = text.split(';\n')
                    final = []
                    for i in range (len(temp)):
                        a = temp[i].split(',')
                        final.append(a)
                
                    train = train[:] + final
                    
    return train
    


# In[ ]:


# processing the data

trainphoneaccel = process('../input/prithviai-activitydetection/data/train/phone/accel')
trainphonegyro = process('../input/prithviai-activitydetection/data/train/phone/gyro')
trainwatchaccel = process('../input/prithviai-activitydetection/data/train/watch/accel')
trainwatchgyro = process('../input/prithviai-activitydetection/data/train/watch/gyro')


# In[ ]:


trainphoneaccel[:10]


# ### Comining all the data to store in a separate variable to be easily able to train the model

# In[ ]:


train = trainphoneaccel + trainphonegyro + trainwatchaccel + trainwatchgyro


# In[ ]:


len(train)


# ### Converting the data from the list into an dataframe

# In[ ]:


def transform(data):
    data = data[:-1]
    data = pd.DataFrame(data, columns = ['Subject-id', 'Activity Label', 'Timestamp', 'x', 'y', 'z'])
    return data


# In[ ]:


train = transform(train)


# In[ ]:


train.shape


# In[ ]:


# convert the elements of the dataframe from string to numeric
train = train.convert_objects(convert_numeric=True)


# In[ ]:


train.head()


# In[ ]:


# removing all the null values from the dataframe
train = train.dropna(subset = ['Subject-id','Timestamp', 'Activity Label','x', 'y', 'z'])


# ### converting the label from characters to integers

# In[ ]:


label = train['Activity Label'].unique()


# In[ ]:


l={}
n=0
for i in label:
    l[i] = n+1
    n+=1

train['Activity Label'] = train['Activity Label'].apply(lambda x: l[x])


# In[ ]:


train.head()


# ### preparing the test data just like we did train data 

# In[ ]:


testphoneaccel = process('../input/prithviai-activitydetection/data/test/phone/accel')
testphonegyro = process('../input/prithviai-activitydetection/data/test/phone/gyro')
testwatchaccel = process('../input/prithviai-activitydetection/data/test/watch/accel')
testwatchgyro = process('../input/prithviai-activitydetection/data/test/watch/gyro')


# In[ ]:


test = testphoneaccel + testphonegyro + testwatchaccel + testwatchgyro


# In[ ]:


test = transform(test)


# In[ ]:


test = test.convert_objects(convert_numeric=True)


# In[ ]:


test = test.dropna(subset = ['Subject-id','Timestamp', 'Activity Label','x', 'y', 'z'])


# In[ ]:


l={}
n=0
for i in label:
    l[i] = n+1
    n+=1

test['Activity Label'] = test['Activity Label'].apply(lambda x: l[x])


# In[ ]:


train['Timestamp'] = train['Timestamp'].apply(lambda x: x//1000000)
train['Timestamp'] = train['Timestamp'].apply(lambda x: datetime.fromtimestamp(x))
test['Timestamp'] = test['Timestamp'].apply(lambda x: x//1000000)
test['Timestamp'] = test['Timestamp'].apply(lambda x: datetime.fromtimestamp(x))
train.drop(columns="Subject-id",inplace=True)
test.drop(columns="Subject-id",inplace=True)


# In[ ]:


for time in ('year','month','week','day','hour','minute','second'):
    train[time] = getattr(train['Timestamp'].dt,time)
train.drop(columns="Timestamp",inplace=True)

for time in ('year','month','week','day','hour','minute','second'):
    test[time] = getattr(test['Timestamp'].dt,time)
test.drop(columns="Timestamp",inplace=True)


# In[ ]:


train.head()


# In[ ]:


train = train.sample(frac=1).reset_index(drop=True)
test = test.sample(frac=1).reset_index(drop=True)
data=pd.DataFrame()
data=pd.concat([train,test])


# In[ ]:


y=data["Activity Label"]
x=data.drop(columns="Activity Label")
x_train, x_test, y_train, y_test = train_test_split(x,y , train_size = 0.7, random_state =  42)


# ## <a name="3"></a>3. Traning the model

# In[ ]:


model = RandomForestClassifier()
model.fit(x_train, y_train)


# ## <a name="4"></a>4. Predicting the output and calculating the accuracy

# In[ ]:


y= model.predict(x_test)
acc = accuracy_score(y_test, y)


# In[ ]:


acc


# ## <a name="5"></a>5. Saving the model

# In[ ]:


filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))


# In[ ]:


df = pd.DataFrame(y)


# In[ ]:


df.to_csv('answer.csv')


# In[ ]:




