#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


# We will start off by importing the FIFA dataset and cleaning it for using our Random Forest regressor
# to predict what attributes affect the Overall rating of a player

dat = pd.read_csv('../input/data.csv')
dat.shape
dat.columns


# In[3]:


# Change column name of the 1st column
dat.rename(columns={'Unnamed: 0':'Sl_num'},inplace=True)


# In[4]:


# Check if the name changed
dat.columns


# In[5]:


# Check the data summary
dat.describe()


# In[6]:


# Check for null values
dat.isnull().sum()


# In[7]:


#Get required coulmn numbers
dat.columns.get_loc("Crossing")


# In[8]:


dat.columns.get_loc("GKReflexes")


# In[9]:


# Find index of all missing values for a particular column
dat[dat['Dribbling'].isnull()].index.tolist()


# In[10]:


dat[dat['GKReflexes'].isnull()].index.tolist()


# In[11]:


# Drop all rows with empty / missing values 
dat.drop(dat.index[[13236,13237,13238,13239,13240,13241,13242,13243,13244,13245,13246,13247,13248,13249,13250,13251,13252,
13253,13254,13255,13256,13257,13258,13259,13260,13261,13262,13263,13264,13265,13266,13267,13268,13269,13270,13271,13272,13273,
13274,13275,13276,13277,13278,13279,13280,13281,13282,13283]], inplace= True)


# In[12]:


type(dat)


# In[13]:


# Check if delete was successful
dat[dat['GKReflexes'].isnull()].index.tolist()


# In[14]:


#Check for missing values again
dat.isnull().sum()


# In[15]:


#Select all attribute columns that affect the overall rating of the player
fifa=dat.iloc[:,54:88]


# In[16]:


fifa.head()


# In[17]:


type(fifa)


# In[18]:


#Check null values in the subsetted data
fifa.isnull().sum()


# In[19]:


X=fifa
X.head()


# In[21]:


#  Check for null values in the target variable
dat['Overall'].isnull().sum()


# In[22]:


# Y is our target variable
Y=dat['Overall']


# In[23]:


# Split data into train and test
import sklearn.model_selection as model_selection

X_train,X_test,Y_train,Y_test=model_selection.train_test_split(X,Y,test_size=0.2,random_state=400)


# In[24]:


# Estimate for 50 trees
from sklearn.ensemble import RandomForestRegressor
reg=RandomForestRegressor(n_estimators=50,max_depth=5,max_features='sqrt',oob_score=True)
reg.fit(X_train,Y_train)


# In[25]:


# Check fit score in the test data
reg.score(X_test,Y_test)


# In[27]:


# Check Ou-of-Bag score in the test data
reg.oob_score_


# In[28]:


# Check MSE value
from sklearn import metrics
metrics.mean_squared_error(Y_test,reg.predict(X_test))


# In[29]:


# Tuning the n_estimators
for i in range(50,160,10):
    reg=RandomForestRegressor(n_estimators=i,max_depth=5,max_features='sqrt',oob_score=True,random_state=1)
    reg.fit(X_train,Y_train)
    oob=reg.oob_score_
    print('For n_estimators = '+str(i))
    print('OOB score is '+str(oob))
    print('************************')


# In[30]:


#Find the score() at n_estimators=140
#Tuning the n_estimators
reg=RandomForestRegressor(n_estimators=140,max_depth=5,max_features='sqrt',oob_score=True,random_state=1)
reg.fit(X_test,Y_test)


# In[31]:


reg.score(X_test,Y_test)


# In[32]:


reg.oob_score_


# In[33]:


reg.feature_importances_


# In[34]:


imp_feat=pd.Series(reg.feature_importances_,index=X.columns.tolist())


# In[35]:


imp_feat.sort_values(ascending=False)


# In[36]:


graph=imp_feat.sort_values(ascending=False).plot(kind='bar')


# In[ ]:




