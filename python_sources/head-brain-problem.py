#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df =  pd.read_csv("../input/headbrain/headbrain.csv")


# In[ ]:


df['Gender'].unique()


# In[ ]:


df.columns


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# In[ ]:


### Feature Enginnering??? & Data Processing??
# Askign business about importance of exisiting data, ask them do they anything more??


# In[ ]:


## find co-relation
## Exploratory analysis
corr_df = df.corr()


# In[ ]:


corr_df.to_csv("correlation.csv")


# In[ ]:


X = df.drop('Brain Weight(grams)',axis = 1 )
y = df['Brain Weight(grams)']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(X_train, y_train)


# In[ ]:


y_pred = neigh.predict(X_test)


# In[ ]:


y_pred


# In[ ]:


from sklearn.metrics import mean_absolute_error , mean_squared_error


# In[ ]:


mae = mean_absolute_error(y_test,y_pred)
print("my mean absoulte error is == ",mae)


# In[ ]:


# my model predicts the brain weight with a error of 67 grams


# In[ ]:


mse = mean_squared_error(y_test,y_pred)
print("mean square error is == ",mse)


# In[ ]:


rmse =  np.sqrt(mse)
print("root mean square error is == ",rmse)


# In[ ]:


# mape: mean absoulte percentage error

mape = np.mean(np.abs((y_test - y_pred)/y_test))
print("mean absoulte percentage error is == ",mape)


# In[ ]:


print("accuracy is ",1-mape)


# In[ ]:


import joblib


# In[ ]:


###### Save the model file and have tea ####
filename = 'knn_model_headbrain.sav'
joblib.dump(neigh, filename)
 
# some time later...


# In[ ]:


filename = 'knn_model_headbrain.sav' 
# load the model from disk
loaded_model = joblib.load(filename)

#### pass any data --- test data
result = loaded_model.predict(X_test)
print(result)


# In[ ]:


X.columns

