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

# Any results you write to the current directory are saved as output.


# In[ ]:


df_Train = pd.read_csv("../input/train_V2.csv")

df_Test = pd.read_csv("../input/test_V2.csv")

print("Train data set :\n",df_Train.head())
print("Test data set :\n", df_Test.head())


# In[ ]:


df_Train.info()


# In[ ]:


total = df_Train.isnull().sum().sort_values(ascending= False)
percent_1= df_Train.isnull().sum()/df_Train.isnull().count()*100
percent_2= round(percent_1,1).sort_values(ascending= False)
missing_data = pd.concat([total,percent_2], axis= 1, keys = ['Total', '%']) 
print(missing_data)


# In[ ]:


df_Trial = df_Train[df_Train['winPlacePerc'].isna()==True]
print(df_Trial)


# In[ ]:


df_Train.loc[df_Train['winPlacePerc'].isna()==True, 'winPlacePerc']= 0.5


# In[ ]:


total = df_Test.isnull().sum().sort_values(ascending= False)
percent_1= df_Test.isnull().sum()/df_Test.isnull().count()*100
percent_2= round(percent_1,1).sort_values(ascending= False)
missing_data = pd.concat([total,percent_2], axis= 1, keys = ['Total', '%']) 
print(missing_data)


# In[ ]:


df_Train= df_Train.set_index(['Id'])
df_Test= df_Test.set_index(['Id'])
ColumnList = ['assists', 'boosts', 'damageDealt', 'DBNOs',
       'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
       'killStreaks', 'longestKill', 'maxPlace', 'numGroups', 'revives',
       'rideDistance', 'roadKills', 'swimDistance', 'teamKills',
       'vehicleDestroys', 'walkDistance', 'weaponsAcquired', 'winPoints',
       'winPlacePerc']
df_Train = df_Train[ColumnList]
ColumnList.remove('winPlacePerc')
df_Test = df_Test[ColumnList]

X_Column = ColumnList
Y_Column = 'winPlacePerc'
X_Train = df_Train[X_Column]
Y_Train = df_Train[Y_Column]
X_test = df_Test[X_Column]


# In[ ]:


X_Train = X_Train**0.25


# In[ ]:


X_test= X_test**0.25


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_Train= scaler.fit_transform(X_Train)
X_test = scaler.transform(X_test)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation = 'tanh'))
model.add(Dense(32, activation = 'tanh'))
model.add(Dense(128, activation = 'tanh'))
model.add(Dense(1, activation = 'relu'))
model.compile(optimizer= 'adam', loss= 'mae',metrics= ['mse'])


# In[ ]:


model.fit(X_Train, Y_Train.values, batch_size= 10, epochs = 10, verbose = 1)


# In[ ]:


y_pred = model.predict(X_test, batch_size= 10)
y_pred


# In[ ]:


# Restore some columns
df_Test["winPlacePerc"]= y_pred

# Edge case
df_Test.loc[(df_Test.winPlacePerc > 1.0), "winPlacePerc"] = 1.0
df_Test.loc[(df_Test.winPlacePerc < 0.0), "winPlacePerc"] = 0.0


# In[ ]:


df_sub = df_Test[["winPlacePerc"]].reset_index()
df_sub.head(10)


# In[ ]:


df_sub.to_csv("submission_adjusted.csv", index=False)

