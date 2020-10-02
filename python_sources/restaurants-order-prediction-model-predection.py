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


# ## Importing all packages

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ## Reading the dataset file

# In[ ]:


train_df=pd.read_csv("../input/train.csv")
test_df=pd.read_csv("../input/test.csv")


# In[ ]:


# Checking the Rows and column count for Test and Train Dataset
print("The no os rows and column of train datsets is",train_df.shape)
print("The no os rows and column of train datsets is",test_df.shape)


# In[ ]:


# Checking for Null Values
train_df.isnull().count()     # There is no Null value in train datasets
test_df.isnull().count()      # There is again no Null value in train datasets
# "Good to Go!!!!"


# In[ ]:


train_df.head(4)


# ### Building the Model

# In[ ]:


# Feature Engineering
#train_df['date_id']=pd.to_numeric(train_df['date_id'],errors='coerce')
y=train_df['orders']
X=train_df[['date_id']]


# In[ ]:


# Building the model with Decison Tree
from sklearn import tree
my_tree= tree.DecisionTreeRegressor(max_depth=2,random_state=42)
my_tree.fit(X,y)


# In[ ]:


test_df.head(2)


# In[ ]:


##Predicting Test dataset with the help of Train datasets
X_test =test_df[['date_id']]
y_test=test_df[['date_id']]
y_pred=my_tree.predict(X_test)


# ## We have Successfully predicted the Order for test Dataset from the input of Date_Id......................Thanks............................

# In[ ]:




