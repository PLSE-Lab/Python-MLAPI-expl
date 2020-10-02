#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
test = pd.read_csv("../input/titanic/test.csv")
train = pd.read_csv("../input/titanic/train.csv")

def IDA(df):
    global features # function output dataframe , MUST use global
    """ generic features """
    features = pd.DataFrame({'data_type' : df.dtypes})  # retunrs dataframe with feature name as column index and its dtype as column 
    features['feature_name'] = features.index # copy index to feature_name column
    features['unique'] = [df[col].is_unique for col in df] # True if column has all unique value and viceversa
    features['unique_count'] = [df[col].value_counts(dropna=False).count() for col in df] # number of unique values in the column
    features['null_count'] = [df[col].isnull().sum() for col in df] # number of null/NaN values in the column
    features['feature_type'] = "" # empty column for feature type 
    """ IDENTIFYING FEATURE TYPE """
    for val in features['feature_name']: # loop through each value of 'feature_name' column
        # df.at[row_index_name,col_name] which returns single value for comparision    
        if features.at[val,'unique_count'] <= 10 and features.at[val,'unique_count'] != len(df):
            # compare with len(df) to deal with small df with less than 10 rows otherwise it will consider all feature qualitative
            features.at[val,'feature_type'] = 'Qualitative' # assign "Qualitative" variable_type            
        else: # Quantitative variable need further seperation for easier processing in later stage
            if features.at[val,'data_type'] == 'int64' or features.at[val,'data_type'] == 'float64': # filter columns with data type int or float
                features.at[val,'feature_type'] = 'Quantitative-Numerical' # assign "Quantitative-Numerical" variable_type
            if features.at[val,'data_type'] == 'object': # filter columns with data type object
                features.at[val,'feature_type'] = 'Quantitative-Alphanumeric' # assign "Quantitative-Alphanumeric" variable_type
    features = features[['feature_name','data_type','feature_type','unique','unique_count','null_count']] # rearrange columns for better view
    # MUST be after re-arranging columns otherwise error while calling 'unique' column
    features.rename(columns={'unique':'unique_'+str(df.shape)},inplace=True) # edit column name and add size of df for comparision , MUST use inplace=True
IDA(test)
print(features)


# In[ ]:




