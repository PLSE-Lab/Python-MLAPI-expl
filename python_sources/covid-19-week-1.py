#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas_profiling as pp
from sklearn.ensemble import ExtraTreesClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")
test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/test.csv")


# # Using Pandas Profiling for analysis

# In[ ]:


pp.ProfileReport(train)


# In[ ]:


train = train.drop(["Province/State"], axis = "columns")


# ### It is found that aruba is the country that has missing Lat and Long so I took the real value of it.

# In[ ]:


train["Lat"] = train["Lat"].fillna(12.521)
train["Long"] = train["Long"].fillna(69.968)
train["Date"] = train["Date"].apply(lambda x: x.replace("-",""))
train["Date"] = train["Date"].astype(int)


# In[ ]:


test = test.drop(["Province/State"], axis = "columns")
test["Date"] = test["Date"].apply(lambda x: x.replace("-", ""))
test["Date"] = test["Date"].astype(int)
test["Lat"] = test["Lat"].fillna(12.521)
test["Long"] = test["Long"].fillna(69.968)


# In[ ]:


x = train[["Lat", "Long", "Date"]]
y = train[["ConfirmedCases", "Fatalities"]]


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
model1 = ExtraTreesClassifier()

model1.fit(x ,y["ConfirmedCases"])


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
model4 = ExtraTreesClassifier()

model4.fit(x ,y["Fatalities"])


# In[ ]:


test_x = test[["Lat", "Long", "Date"]]
cases_predict = model1.predict(test_x)
fat_predict = model4.predict(test_x)
ConfirmedCases = pd.DataFrame(cases_predict)
Fatalities = pd.DataFrame(fat_predict)
test = pd.concat([test, ConfirmedCases, Fatalities], axis = "columns")


# In[ ]:


test.head()


# In[ ]:


test = test.drop(["Country/Region", "Lat", "Long", "Date"], axis = "columns")
test.columns = ["ForecastId", "ConfirmedCases", "Fatalities"]


# In[ ]:


test.to_csv("submission.csv", index = False)


# In[ ]:




