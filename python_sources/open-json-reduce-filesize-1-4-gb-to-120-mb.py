#!/usr/bin/env python
# coding: utf-8

# There many great kernels about flattening the JSON blobs, but we don't want to do this everytime we load the data. 

# In[ ]:


import numpy as np 
import pandas as pd 
import os
from pandas.io.json import json_normalize,loads
import time

def load_json(d):
    return loads(d)

def json_into_dataframe(dataframe,column):
    return json_normalize(dataframe[column].apply(load_json).tolist()).add_prefix(column +'.')
        
def open_flat(filepath,columns):
    counter = 0 
    data= pd.read_csv(filepath, low_memory = False)
    for column in columns :
        print ('Unpacking ' + column)
        temp = json_into_dataframe(dataframe = data, column = column)
        for item in temp.columns:
            if len(temp[item].unique()) == 1: # if a column has the same value for all rows is not significant
                temp.drop(item, axis = 1, inplace = True)
                print ('column '+ item + ' was dropped')
                counter += 1
        data = pd.concat([data,temp], axis = 1)
        data.drop([column],inplace = True, axis = 1)
    print ('Columns dropped :',counter)
    return data
train_path = "../input/train.csv"
test_path = "../input/test.csv"
columns = ['totals','device','geoNetwork','trafficSource']


# In[ ]:



t0 = time.time()
train = open_flat(filepath = train_path ,columns = columns)
t1 = time.time()
print ('time to load train set', t1-t0)


# In[ ]:


t0 = time.time()
test = open_flat(filepath = test_path ,columns = columns)
t1 = time.time()
print ('time to load test set ', t1-t0)


# Let's have a look at the shape of the resulting train and test set

# In[ ]:


print(train.shape)
print(test.shape)


# It seems there is a column in the train set ( other than the target feature totals.transactionRevenue ) , that it's not in the test set and we have to get rid of it.

# In[ ]:


for column in train.columns:
    if column not in test.columns:
        print(column)


# In[ ]:


train.drop('trafficSource.campaignCode' , axis = 1 , inplace = True)
print(train.shape)
print(test.shape)


# Now we can save our DataFrames into new excel files to save some space

# In[ ]:


t0 = time.time()
writer = pd.ExcelWriter('train.xlsx')
train.to_excel(writer,'Sheet1')
writer.save()
t1 = time.time()
print ('time to save train dataset to excel ', (t1-t0)/60.0 , ' mins') 


# In[ ]:


t0 = time.time()
writer = pd.ExcelWriter('test.xlsx')
test.to_excel(writer,'Sheet1')
writer.save()
t1 = time.time()
print ('time to save test dataset to excel ', (t1-t0)/60.0 , ' mins') 


# Let's see how much space we saved!

# In[ ]:


print ('File size of flat train set :' + str(((os.path.getsize("train.xlsx")/1024)/1024)) + ' MB')
print ('File size of original train set :' + str(((os.path.getsize("../input/train.csv")/1024)/1024)) + ' MB')


# In[ ]:


print ('File size of flat test set :' + str(((os.path.getsize("test.xlsx")/1024)/1024)) + ' MB')
print ('File size of original test set :' + str(((os.path.getsize("../input/test.csv")/1024)/1024)) + ' MB')

