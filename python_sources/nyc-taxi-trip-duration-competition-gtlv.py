#!/usr/bin/env python
# coding: utf-8

# ### Loading libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn.neighbors as knn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import os
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split


from dateutil import parser
import io
import base64
from IPython.display import HTML
# from imblearn.under_sampling import RandomUnderSampler
# from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))


# ### Exploratory Data Analysis (EDA)

# In[ ]:


df = pd.read_csv('../input/train.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


# plt.hist(df['trip_duration'])
df.describe()


# In[ ]:


df[df['trip_duration']<3600].boxplot(column='trip_duration')


# In[ ]:


df.shape()
# df['trip_duration']


# In[ ]:


df['pickup_day'] = pd.DatetimeIndex(df['pickup_datetime']).day
df['pickup_month'] = pd.DatetimeIndex(df['pickup_datetime']).month
df['pickup_year'] = pd.DatetimeIndex(df['pickup_datetime']).year
df['pickup_hour'] = pd.DatetimeIndex(df['pickup_datetime']).hour
df['pickup_minute'] = pd.DatetimeIndex(df['pickup_datetime']).minute
df['pickup_dayofweek'] = pd.DatetimeIndex(df['pickup_datetime']).dayofweek
df['pickup_dayofyear'] = pd.DatetimeIndex(df['pickup_datetime']).dayofyear


# In[ ]:


X = df.drop(['id','pickup_minute','pickup_year','pickup_day','pickup_datetime','dropoff_datetime','trip_duration','store_and_fwd_flag'], axis=1)
y = df['trip_duration']


# In[ ]:


X.head()


# In[ ]:


plt.scatter(df[''],df[])


# #### Building Training and Test sets

# In[ ]:


# 70-30% of train and test
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.3)


# In[ ]:


from sklearn.preprocessing import StandardScaler

# Standarizing features
scaler = StandardScaler()
scaler.fit(Xtrain)
Xtrain = scaler.transform(Xtrain)
Xtest = scaler.transform(Xtest)


# #### KNN model

# In[ ]:


# KNN model (with k=3)
knn3 = knn.KNeighborsRegressor()
knn3.fit(Xtrain,ytrain)
y_pred = knn3.predict(Xtest)


# In[ ]:


knn3.score(Xtest,ytest)


# #### Running the model on Test.csv dataset

# In[ ]:


df2 = pd.read_csv('../input/test.csv')


# In[ ]:


df2['pickup_day'] = pd.DatetimeIndex(df2['pickup_datetime']).day
df2['pickup_month'] = pd.DatetimeIndex(df2['pickup_datetime']).month
df2['pickup_year'] = pd.DatetimeIndex(df2['pickup_datetime']).year
df2['pickup_hour'] = pd.DatetimeIndex(df2['pickup_datetime']).hour
df2['pickup_minute'] = pd.DatetimeIndex(df2['pickup_datetime']).minute
df2['pickup_dayofweek'] = pd.DatetimeIndex(df2['pickup_datetime']).dayofweek
df2['pickup_dayofyear'] = pd.DatetimeIndex(df2['pickup_datetime']).dayofyear


# In[ ]:


Xtesting = df2.drop(['id','pickup_minute','pickup_year','pickup_day','pickup_datetime','store_and_fwd_flag'], axis=1)


# In[ ]:


y_predicted_submission = knn3.predict(Xtesting)


# #### Preparing Submission File

# In[ ]:


df3 = pd.read_csv('../input/test.csv')

# Kaggle needs the submission to have a certain format;
# see https://www.kaggle.com/c/titanic-gettingStarted/download/gendermodel.csv
# for an example of what it's supposed to look like.
submission = pd.DataFrame({ 'id': df3.id,
                            'trip_duration': y_predicted_submission })
submission.to_csv("submission.csv", index=False)


# In[ ]:




