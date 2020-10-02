#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **IMPORTING DATA**

# Read Data from File Location (either locally or from a cloud location)
# and store as Process_Train, Process_Test and Process_Sample

# In[ ]:


Process_Train = pd.read_csv("/kaggle/input/intercampusai2019/train.csv")
Process_Test = pd.read_csv("/kaggle/input/intercampusai2019/test.csv")
Process_Sample = pd.read_csv("/kaggle/input/intercampusai2019/sample_submission2.csv")


# **Comments
# 1 The train and test data has **

# In[ ]:


Process_Train.info()


# In[ ]:


Process_Test.info()


# In[ ]:


Process_Sample.info()


# In[ ]:


Process_Train


# In[ ]:


Process_Test


# In[ ]:


Process_Sample


# **Handling the Missing Data.**
# 

# In[ ]:


Process_Train.isnull().sum()


# In[ ]:


Process_Test.isnull().sum()


# In[ ]:


Process_Train.dropna(axis=0,inplace=True)
print(Process_Train.shape)
Process_Train.isnull().sum()


# In[ ]:


Process_Test.dropna(axis=0,inplace=True)
print(Process_Test.shape)
Process_Test.isnull().sum()


# **Encoding Categorical Data.**

# In[ ]:


Process_Train.columns


# In[ ]:


Process_Train_Categorical_Columns = ['Division', 'Qualification', 'Gender', 
                                     'Channel_of_Recruitment', 'State_Of_Origin',
                                     'Foreign_schooled', 'Marital_Status', 'Past_Disciplinary_Action',
                                     'Previous_IntraDepartmental_Movement']


# In[ ]:


for column in Process_Train_Categorical_Columns:
    print(column)
    print(Process_Train[column].value_counts())
    print( )


# In[ ]:


sns.pairplot(Process_Train, hue='Promoted_or_Not');


# In[ ]:



# Import label encoder 
from sklearn import preprocessing 
  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'species'. 
for column in Process_Train_Categorical_Columns:
    Process_Train[column] = label_encoder.fit_transform(Process_Train[column])
    print(column)
    print(Process_Train[column].unique())
#return(Process_Train[column].unique())
#df['species'].unique() 


# In[ ]:


#Process_Train_Categorical_Columns.append('Promoted_or_Not')
sns.pairplot(Process_Train, hue='Promoted_or_Not');


# In[ ]:


Process_Train


# **Splitting the data set into test set and training set.**

# **
# Feature Scaling**
