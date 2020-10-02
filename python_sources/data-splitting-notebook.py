#!/usr/bin/env python
# coding: utf-8

# This is just an exercise to have a first go at creating a notebook in the Kaggle environement. 
# 
# The aim of this is to provide a data split facility to split the training data set, so that folks can either use
# this to create multiple hold-out sets, or just to check how our model will perform with different part of the original training set. 
# 
# So both stratified and non-stratified options are provided, would be interesting to see how CV score fare in both setting.
# Hope this is useful :) 
# 

# In[ ]:


from sklearn.cross_validation import train_test_split
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings 
warnings.filterwarnings("ignore")

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")


# Some ultility funcitons to save and load data in case you want to save the splits for later usage

# In[ ]:


def load_data(pickle_file):
    load_file = open(pickle_file,'rb')
    data = pickle.load(load_file)
    return data


def pickle_data(path, data):
    file = path
    save_file = open(file, 'wb')
    pickle.dump(data, save_file, -1)
    save_file.close()


# Generic functions that allow for splitting of dataset in either random or stratified manner.
# The dataset will be split into two portions: potentially one to be used as training set, and one for validation set.

# In[ ]:


def split_data(data, nrounds=5, train_size=0.8, stratified=True):
    # initialise random state
    rnd_state = np.random.RandomState(1234)
    split_results = []
    # label for stratified split, assump to be the last column of the input data
    y = data.iloc[:, data.shape[1]-1].as_matrix()
    # perform splitting runs
    for run in range(0, nrounds):
        if stratified:
            train_ix, val_ix = train_test_split(np.arange(data.shape[0]), train_size=train_size,                                                
                                                stratify=y, random_state=rnd_state)
        else:
            train_ix, val_ix = train_test_split(np.arange(data.shape[0]), train_size=train_size,
                                                random_state=rnd_state)
        data_train = data.ix[train_ix]
        data_val = data.ix[val_ix]
        train_zeros=data_train["TARGET"].value_counts()[0]
        train_ones=data_train["TARGET"].value_counts()[1]
        val_zeros=data_val["TARGET"].value_counts()[0]
        val_ones=data_val["TARGET"].value_counts()[1]

        print ("Run %d, zero in data_train: %d, one in data_train: %d" %(run, train_zeros, train_ones))
        print ("Run %d, zero in data_val: %d, one in data_val: %d" %(run, val_zeros, val_ones))

    return 0


# Stratified spltting of data 5 times, just to check that the split between 1s and 0s are the same every time.

# In[ ]:


print("Performing stratified data split")
stratified_splits = split_data(train, nrounds=5)


# Non-stratified spltting of data for 20 times, just to check each split is different for both
# - Train/val dataset split
# - label classes split

# In[ ]:


print("Performing non-stratified data split")
non_stratified_splits=split_data(train, nrounds=20, stratified=False)

