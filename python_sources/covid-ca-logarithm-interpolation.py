#!/usr/bin/env python
# coding: utf-8

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


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_train = pd.read_csv("/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_train.csv")
df_train.head(5)


# In[ ]:


df_test = pd.read_csv("/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_test.csv")


# # Feature engineering

# In[ ]:


def from_date_to_day(train, test):
    date_train = pd.to_datetime(train["Date"])
    
    beginning = date_train.min()
    
    days_train = date_train - beginning
    train["Day"] = days_train.dt.days
    
    date_test = pd.to_datetime(test["Date"])
    days_test = date_test - beginning
    test["Day"] = days_test.dt.days


# In[ ]:


from_date_to_day(df_train, df_test)


# In[ ]:


# get rid of log(0) problem
def log_rectified(x):
    return np.log(max(x, 0) + 1.0)

np_log_rectified = np.vectorize(log_rectified)

def inv_log_rectified(x):
    return np.exp(x) - 1.0

np_inv_log_rectified = np.vectorize(inv_log_rectified)


# # Insights

# In[ ]:


def plot_log_cases(data):
    
    X = data["Day"].unique()
    y = np_log_rectified(data.groupby("Day").ConfirmedCases.sum())
    
    plt.plot(X, y, 'bo')


# In[ ]:


plot_log_cases(df_train)


# => linear interpolation is possible

# # Modelling

# In[ ]:


from scipy import interpolate
class Model:
    def __init__(self, train, target):
        X = np.arange(0, train.Day.max() + 1)
        y_log = np_log_rectified(train.groupby("Day")[target].sum()[X])
        
        self.f1 = interpolate.interp1d(X, y_log, fill_value="extrapolate", kind="linear")
    
    def predict(self, test):
        return self.f1(test)


# ## Visualize model error

# In[ ]:


def plot_prediction(data, model, target):
    X = np.linspace(0,65,100)
    y_pred = np_inv_log_rectified(model.predict(X))
    
    days = data["Day"].unique()
    y = data.groupby("Day")[target].sum()
    
    plt.plot(days, y, 'bo')
    plt.plot(X, y_pred)


# In[ ]:


def plot_error(data, model, target):
    days = data["Day"].unique()
    y = data.groupby("Day")[target].sum()
    
    y_pred = np_inv_log_rectified(model.predict(days))
    
    plt.plot(days, abs(y - y_pred))


# In[ ]:


model_case = Model(df_train, "ConfirmedCases")
plot_prediction(df_train, model_case, "ConfirmedCases")


# In[ ]:


plot_error(df_train, model_case, "ConfirmedCases")


# In[ ]:


model_fatal = Model(df_train, "Fatalities")
plot_prediction(df_train, model_fatal, "Fatalities")


# In[ ]:


plot_error(df_train, model_fatal, "Fatalities")


# # Submission

# In[ ]:


pred_cases = np_inv_log_rectified(model_case.predict(df_test.Day)).astype(int)
pred_fatal = np_inv_log_rectified(model_fatal.predict(df_test.Day)).astype(int)


# In[ ]:


submission = pd.concat(
    [ pd.Series(np.arange(1, df_test.ForecastId.max() + 1)),
     pd.Series(pred_cases),
     pd.Series(pred_fatal)],
    axis=1)


# In[ ]:


submission.head(5)


# In[ ]:


submission.to_csv('submission.csv', header=['ForecastId', 'ConfirmedCases', 'Fatalities'], index=False)

