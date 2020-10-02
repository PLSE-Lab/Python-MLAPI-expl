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


# importing libraries
import numpy as np
import pandas as pd
from pandas import datetime
from datetime import datetime
from datetime import date
import calendar
import matplotlib.pyplot as plt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train=pd.read_csv('../input/train.csv')


# In[ ]:


test=pd.read_csv('../input/test.csv')


# In[ ]:


# shape of training and testing data
train.shape, test.shape


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


sn.heatmap(train.corr())


# In[ ]:


sn.heatmap(test.corr())


# In[ ]:


# distribution of count variable
sn.distplot(train["count"])


# In[ ]:


# distribution of count variable
sn.distplot(np.log(train["count"]))


# In[ ]:


sn.distplot(train["registered"])


# In[ ]:


# looking at the correlation between numerical variables
corr = train[["temp","atemp","casual","registered","humidity","windspeed","count"]].corr()
mask = np.array(corr)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sn.heatmap(corr, mask=mask,vmax=.9, square=True,annot=True, cmap="YlGnBu")


# In[ ]:


# looking for missing values in the datasaet
train.isnull().sum()


# In[ ]:


# looking for missing values in the datasaet
test.isnull().sum()


# In[ ]:


# extracting date, hour and month from the datetime
train["date"] = train.datetime.apply(lambda x : x.split()[0])
train["hour"] = train.datetime.apply(lambda x : x.split()[1].split(":")[0])
train["month"] = train.date.apply(lambda dateString : datetime.strptime(dateString,"%d-%m-%Y").month)


# In[ ]:


train=pd.read_csv('../input/train.csv')


# In[ ]:


training = train[train['datetime']<='2012-03-30 0:00:00']
validation = train[train['datetime']>'2012-03-30 0:00:00']


# In[ ]:


test=pd.read_csv('../input/test.csv')


# In[ ]:


train = train.drop(['datetime', 'atemp'],axis=1)
test = test.drop(['datetime', 'atemp'], axis=1)
training = training.drop(['datetime', 'atemp'],axis=1)
validation = validation.drop(['datetime', 'atemp'],axis=1)


#                         # model building****

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


# initialize the linear regression model
lModel = LinearRegression()


# In[ ]:


X_train = training.drop('count', 1)
y_train = np.log(training['count'])
X_val = validation.drop('count', 1)
y_val = np.log(validation['count'])


# In[ ]:


# checking the shape of X_train, y_train, X_val and y_val
X_train.shape, y_train.shape, X_val.shape, y_val.shape


# In[ ]:


# fitting the model on X_train and y_train
lModel.fit(X_train,y_train)


# In[ ]:


# making prediction on validation set
prediction = lModel.predict(X_val)


# In[ ]:


prediction


#                                                  # decision tree

# In[ ]:


from sklearn.tree import DecisionTreeRegressor


# In[ ]:


# defining a decision tree model with a depth of 5. You can further tune the hyperparameters to improve the score
dt_reg = DecisionTreeRegressor(max_depth=5)


# In[ ]:


dt_reg.fit(X_train, y_train)


# In[ ]:


predict = dt_reg.predict(X_val)


# In[ ]:


# defining a function which will return the rmsle score
def rmsle(y, y_):
    y = np.exp(y),   # taking the exponential as we took the log of target variable
    y_ = np.exp(y_)
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))


# In[ ]:


# calculating rmsle of the predicted values
rmsle(y_val, predict)


# In[ ]:


test_prediction = dt_reg.predict(test)


# In[ ]:


final_prediction = np.exp(test_prediction)


# In[ ]:


submission = pd.DataFrame()


# In[ ]:


# creating a count column and saving the predictions in it
submission['count'] = final_prediction


# submission.to_csv('add.csv file path in which you want to save predictions',header=True,index=False)

# In[ ]:




