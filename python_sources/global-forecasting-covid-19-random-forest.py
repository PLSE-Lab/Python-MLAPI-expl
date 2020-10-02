#!/usr/bin/env python
# coding: utf-8

# # Global Covid-19 Forecasting using a Random Forest
# 
# This is a very simple starter submission kernel using a random forest. Feature engineering and tuning will help performance.
# 
# ### As it turns out, it is very tough to make a RF algorithm properly extrapolate. Given the decision tree structure, conditional statements which recursively split the intpu space. There are ways to get random forests to predict values that fall outside the range of values of the targets in the training set, however I haven't become privy to these techniques. Take a look at the following: https://www.statworx.com/de/blog/time-series-forecasting-with-random-forest/
# 
# Nevertheless, I will leave this notebook posted as an illustration, and we can consider week 1's submission as a bit of an experiment :-)

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


import numpy as np
import pandas as pd


# ## Import Data

# In[ ]:


train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")
test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/test.csv")
submission = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/submission.csv")
train.tail()


# # Data Cleaning

# In[ ]:


# Format date
train["Date"] = train["Date"].apply(lambda x: x.replace("-",""))
train["Date"]  = train["Date"].astype(int)
train.head()


# ### Drop NaNs

# In[ ]:


# drop nan's
train = train.drop(['Province/State'],axis=1)
train = train.dropna()
train.isnull().sum()


# In[ ]:


# Do same to Test data
test["Date"] = test["Date"].apply(lambda x: x.replace("-",""))
test["Date"]  = test["Date"].astype(int)
# deal with nan's for lat and lon
#test = test.dropna()
test.isnull().sum()


# ### Prepare Training

# In[ ]:


x = train[['Lat', 'Long', 'Date']]
y1 = train[['ConfirmedCases']]
y2 = train[['Fatalities']]
x_test = test[['Lat', 'Long', 'Date']]


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
Tree_model = RandomForestClassifier(max_depth=200, random_state=0)


# ### Train Confirmed Cases Tree

# In[ ]:


##
Tree_model.fit(x,y1)
pred1 = Tree_model.predict(x_test)
pred1 = pd.DataFrame(pred1)
pred1.columns = ["ConfirmedCases_prediction"]


# In[ ]:


pred1.head()


# ### Train Deaths Tree

# In[ ]:




##
Tree_model.fit(x,y2)
pred2 = Tree_model.predict(x_test)
pred2 = pd.DataFrame(pred2)
pred2.columns = ["Death_prediction"]


# ### Prepare for Submission

# In[ ]:



Sub = pd.read_csv("../input/covid19-global-forecasting-week-1/submission.csv")
sub_new = Sub[["ForecastId"]]
sub_new


# In[ ]:


# submit

submit = pd.concat([pred1,pred2,sub_new],axis=1)
submit.head()


# In[ ]:


# Clean
submit.columns = ['ConfirmedCases', 'Fatalities', 'ForecastId']
submit = submit[['ForecastId','ConfirmedCases', 'Fatalities']]

submit["ConfirmedCases"] = submit["ConfirmedCases"].astype(int)
submit["Fatalities"] = submit["Fatalities"].astype(int)


# In[ ]:



submit.describe()


# In[ ]:


Sub = submit
Sub.to_csv('submission.csv', index=False)

