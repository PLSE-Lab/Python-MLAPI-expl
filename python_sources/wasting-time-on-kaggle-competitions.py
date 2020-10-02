#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Wasting time on Kaggle competitions

import pandas as pd

# load the data

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print(test_df.info())


# In[ ]:


# convert all of the features to numeric values

def convert_to_numeric(df):
    for col in ['Name', 'AnimalType', 'SexuponOutcome',
                'AgeuponOutcome', 'Breed', 'Color']:
        _col = "_%s" % (col)
        values = df[col].unique()
        _values = dict(zip(values, range(len(values))))
        df[_col] = df[col].map(_values).astype(int)
        df = df.drop(col, axis = 1)
    return df

train_df = convert_to_numeric(train_df)
test_df = convert_to_numeric(test_df)
print(test_df.info())


# In[ ]:


# fix the DateTime column

def fix_date_time(df):
    def extract_year(_df):
        return _df['DateTime'].map(lambda dt: int(dt[:4]))
    df['Year'] = extract_year(df)
    
    def extract_month(_df):
        return _df['DateTime'].map(lambda dt: int(dt[5:7]))
    df['Month'] = extract_month(df)
    
    def extract_day(_df):
        return _df['DateTime'].map(lambda dt: int(dt[8:10]))
    df['Day'] = extract_day(df)
    
    def extract_hour(_df):
        return _df['DateTime'].map(lambda dt: int(dt[11:13]))
    df['Hour'] = extract_hour(df)
    
    def extract_minute(_df):
        return _df['DateTime'].map(lambda dt: int(dt[14:16]))
    df['Minute'] = extract_minute(df)
    
    return df.drop(['DateTime'], axis = 1)

train_df = fix_date_time(train_df)
test_df = fix_date_time(test_df)
print(test_df.info())


# In[ ]:


# split the data into a training set (80%) and a validation set (20%)

cut = int(len(train_df) * 0.8)
validation_df = train_df[cut:]
train_df = train_df[:cut]

# build a classifier with scikit-learn

import sklearn
from sklearn.ensemble import AdaBoostClassifier

# ... need some sleep now, will continue wasting time tomorrow

