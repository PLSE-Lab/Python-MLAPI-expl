#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# The aim here is to predict probability of student getting admission into university.

# In[ ]:


# import libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# set random seed
np.random.seed(100)


# ## Data overview
# We try to get an overview of the data by using some plots

# In[ ]:


df = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
df.columns


# In[ ]:


df.head()


# In[ ]:


df.plot(subplots=True,kind="box",figsize=(15,15))


# ## Simple Model
# Let's start with a simple model, without doing any preprocessing on data

# In[ ]:


X = df.drop(['Chance of Admit '], axis=1)
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)


# In[ ]:


# create a RandomForestRegressor
reg = RandomForestRegressor(n_estimators=50, max_depth = 5)
reg.fit(X_train,y_train)

print(reg.feature_importances_)
print('train accuracy ' + str(reg.score(X_train,y_train)))
print('test accuracy ' + str(reg.score(X_test,y_test)))


# In[ ]:




