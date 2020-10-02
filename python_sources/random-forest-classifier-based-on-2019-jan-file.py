#!/usr/bin/env python
# coding: utf-8

# **SCOPE**
# 
# Since there it is not very clear of what "predict delays in flights" means,
# I am assuming here a generic model where I don't know exaclty which flight or company I will take,
# but I want to know if I have to expect delays due to the time slot in which I decide to flight
# (example: 7 to 8 AM). Of course, departure and arrival airports will be a feature as well.
# 
# Also, I don't care whether I will have delays in departure, but only if I will arrive late.
# 
# Given that, the target of my classifier will be the "ARR_DEL15" columns.
# 
# My features will be:
# 
# ARR_TIME_BLK: the time slot in which the flight will departure (i.e. 0700-0759)
# DAY_OF_WEEK: meant to be "working day" or "weekend"
# DEP_TIME_BLK:  the time slot in which the flight will arrive (i.e. 0700-0759)
# DEST_AIRPORT_ID	DISTANCE: the distance between the two airports
# ORIGIN_AIRPORT_ID: the origin airport
# DEST_AIRPORT_ID: the destination airport
# 
# n.b. I will consider only the Jan 2019 file

# In[ ]:


# import stuff
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import math
import pandas_datareader.data as web
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# read the csv
df = pd.read_csv('/kaggle/input/flight-delay-prediction/Jan_2019_ontime.csv')
df.head()


# In[ ]:


# original data length = 
original_len_dataset = len(df)
original_len_dataset


# In[ ]:


# columns with missing values are: 'TAIL_NUM', 'DEP_TIME', 'DEP_DEL15', 'ARR_TIME', 'ARR_DEL15', 'Unnamed: 21'
df.columns[df.isna().sum() != 0]


# In[ ]:


# Are ORIGIN_AIRPORT_ID and ORIGIN_AIRPORT_SEQ_ID  a duplication/different coding of ORIGIN ?
len(df[['ORIGIN','ORIGIN_AIRPORT_ID','ORIGIN_AIRPORT_SEQ_ID']].sort_values(by='ORIGIN')) == len(df.ORIGIN)


# In[ ]:


# Are DEST_AIRPORT_ID and DEST_AIRPORT_SEQ_ID are a duplication/different coding of DEST ?
len(df[['DEST','DEST_AIRPORT_ID','DEST_AIRPORT_SEQ_ID']].sort_values(by='DEST')) == len(df.DEST)


# In[ ]:


# it looks like not all the flights without departure or arrival time has been cancelled
print (len(df[df.CANCELLED == 1]), len(df[np.isnan(df.DEP_TIME)]) , len(df[np.isnan(df.ARR_TIME)]))
print ('percentage of cancelled flights:%.4f' %(len(df[df.CANCELLED == 1])*100/original_len_dataset))
print ('percentage of missing departure time: %.4f' %(len(df[np.isnan(df.DEP_TIME)])*100/original_len_dataset))
print ('percentage of missing arrival time: %.4f' %(len(df[np.isnan(df.ARR_TIME)])*100/original_len_dataset))
print ('percentage of missing arrival time delay: %.4f' %(len(df[np.isnan(df.ARR_DEL15)])*100/original_len_dataset))


# In[ ]:


# to generalize the data, I made this assumptions
# want to keep:
# 'DAY_OF_WEEK' because delay could depend on traffic in specific days: change to weekend, holiday, working_day
# 'ORIGIN_AIRPORT_ID'  because uniquely identify the origin airport (no need to encode)
# 'DEST_AIRPORT_ID' because uniquely identify the dest airport (no need to encode)
# 'DEP_TIME_BLK' it is full hour step
# 'ARR_TIME' must be generalized with the full hour step
# 'ARR_DEL15' this is the target value: who cares if the airplane departure is delayed: in real life the arrival time is what really matters
#

# want to remove:
# 'DAY_OF_MONTH' since all occurs in the same month
# 'ORIGIN_AIRPORT_SEQ_ID' 
# 'ORIGIN'
# 'OP_UNIQUE_CARRIER','OP_CARRIER_AIRLINE_ID', 'OP_CARRIER', 'TAIL_NUM', 'OP_CARRIER_FL_NUM' because are company dependent
# 'DEP_DEL15' doesn't care: the important is to arrive in time
# 'DEP_TIME' because we will provide DEP_TIME_BLK :the time block in which the flight departures
# 'CANCELLED' if it is canceled the target (delay) makes no sense
# 'DIVERTED' if it is diverted the target (delay) makes no sense


# In[ ]:


# start cleanup things:

# since all missing values are at most around 10% and the dataset is quite big, drop flights with missing data
df = df[ (df.CANCELLED != 1) & (df.DEP_TIME.isna() == False) & (df.ARR_TIME.isna() == False)]
len(df.CANCELLED.isna()), len(df.DEP_TIME.isna()), len(df.ARR_TIME.isna()), len(df)

# drop when target is NAN
df = df[ (df.ARR_DEL15.isna() == False)]

# drop 'Unnamed: 21' column since it is just full of NaN (why do I get this column?)
print(df['Unnamed: 21'].unique())
df.drop(['Unnamed: 21'], inplace=True, axis=1)


# In[ ]:


#'DAY_OF_WEEK' because delay could depend on traffic in specific days: classify it to weekend or working day
def get_day_category(day_of_week):
    if day_of_week <= 5:
        return 0 #'working day'
    elif day_of_week > 5:
        return 1 #'weekend'
    
df.DAY_OF_WEEK = df.DAY_OF_WEEK.apply(get_day_category)


# In[ ]:


# CREATE ARR_TIME_BLK ('ARR_TIME' must be generalized with the full hour step)

#generate block hours
blocks = []
for hour in range(0,24):
    hour_part = ('%02d' %(hour))
    blocks.append(hour_part + '00-' + hour_part + '59')
blocks

def get_arrival_time_blk(arr_time):
    arr_hour = str('%04d' %(arr_time))[:2]
    arr_block = None
    for block in blocks:
        #print (block,arr_hour)
        if block.startswith(arr_hour):
            arr_block = block
            break
    if arr_block == None and str(arr_time) == '2400.0':
        arr_block = '0000-0059'
        #print('Cannot find block for #' + str(arr_time) + '#: set block to #' + arr_block + '#')
    return arr_block

df['ARR_TIME_BLK'] = df.ARR_TIME.apply(get_arrival_time_blk)
# drop the no more useful ARR_TIME
df.drop(['ARR_TIME'], inplace=True, axis=1)


# In[ ]:


#it looks like some target values are set to NaN, so for those we assume that if the airplane departed in late,
#then it arrived in late (strong assumption)

def assume_arrival_delay(dep_delay, arr_delay):
    if np.isnan(arr_delay):
        return dep_delay
    else:
        return arr_delay

df['ARR_DEL15'] = df.apply(lambda row :assume_arrival_delay(row['DEP_DEL15'],row['ARR_DEL15']), axis = 1)


# In[ ]:


# drop all other feature I said I do not consider meaningful
features_to_be_dropped = ['DAY_OF_MONTH','ORIGIN_AIRPORT_SEQ_ID','ORIGIN','DEST', 'DEST_AIRPORT_SEQ_ID', 'OP_UNIQUE_CARRIER','OP_CARRIER_AIRLINE_ID', 'OP_CARRIER', 'TAIL_NUM', 'OP_CARRIER_FL_NUM','DEP_DEL15','DEP_TIME','DIVERTED','CANCELLED']
df.drop(features_to_be_dropped, inplace=True, axis=1)


# In[ ]:


# looks like DEP_TIME_BLK contains an invalid label (got ValueError: y contains previously unseen labels: '0001-0559')
# there is no reason why this should be treated differently so fix it to 0500-0559
df.loc[df['DEP_TIME_BLK'] == '0001-0559', 'DEP_TIME_BLK'] = '0500-0559'


# In[ ]:


# sort out columns
df = df.reindex(sorted(df.columns), axis=1)


# In[ ]:


# label encode ARR_TIME_BLK and DEP_TIME_BLK
le = LabelEncoder()
le.fit(blocks)
le.classes_
df['ARR_TIME_BLK'] = le.transform(df.ARR_TIME_BLK.values)
df['DEP_TIME_BLK'] = le.transform(df.DEP_TIME_BLK.values)


# In[ ]:


# show the head of the final dataset
df.head()


# In[ ]:


# split in train and test
Y = df['ARR_DEL15'].values
X = df.drop(['ARR_DEL15'], axis=1).values

X_train, X_test, Y_train, Y_test =  train_test_split(X,Y, test_size=0.3, random_state=1)


# In[ ]:


rfc = RandomForestClassifier(n_estimators=20)
rfc.fit(X_train,Y_train)

Y_train_pred = rfc.predict(X_train)
Y_test_pred = rfc.predict(X_test)

print('ACCURACY train: %.4f, test: %.4f' %(accuracy_score(Y_train,Y_train_pred), accuracy_score(Y_test,Y_test_pred)))

# On my pc I get ACCURACY train: 0.9328, test: 0.8969

