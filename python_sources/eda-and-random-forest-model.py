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


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from subprocess import check_output
from datetime import time
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# **Exploratory data analysis**

# In[ ]:


df = pd.read_csv('../input/crowdedness-at-the-campus-gym/data.csv')


# In[ ]:


df.head()


# In[ ]:


df.describe()


# Overview

# In[ ]:


sns.pairplot(df)


# Correlation Matrix between different fearures and Heatmap

# In[ ]:


correlation = df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='viridis')


# In[ ]:


sns.distplot(df['temperature'], kde=True, rug=True)


# In[ ]:


sns.distplot(df['number_people'], kde=True, rug=True)


# **Data Pre-Processing**

# **Convert time format from 24h to 12h**

# In[ ]:


def time_to_seconds(time):
    return time.hour * 3600 + time.minute * 60 + time.second


# Encoding and Scale Data

# In[ ]:


df = df.drop("date", axis=1)
noon = time_to_seconds(time(12, 0, 0))
df.timestamp = df.timestamp.apply(lambda t: abs(noon - t))
# one hot encoding
columns = ["day_of_week", "month", "hour"]
df = pd.get_dummies(df, columns=columns)
df.head()


# Split dataframe to Train set and Test set

# In[ ]:


data = df.values
X = data[:, 1:]  # all rows, no label
y = data[:, 0]  # all rows, label only
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# Implement scaler on timestamp and Temperature

# In[ ]:


scaler1 = StandardScaler()
scaler1.fit(X_train[:, 0:1])
X_train[:, 0:1] = scaler1.transform(X_train[:, 0:1])
X_test[:, 0:1] = scaler1.transform(X_test[:, 0:1])
scaler2 = StandardScaler()
scaler2.fit(X_train[:, 3:4])
X_train[:, 3:4] = scaler2.transform(X_train[:, 3:4])
X_test[:, 3:4] = scaler2.transform(X_test[:, 3:4])


# In[ ]:


X_train


# **RANDOM FOREST REGRESSOR MODEL**

# In[ ]:


model = RandomForestRegressor(n_jobs=-1)


# In[ ]:


estimators = np.arange(50, 200, 10)
scores = []
for n in estimators:
    model.set_params(n_estimators=n)
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
    print('score = ', model.score(X_test, y_test))
plt.title("Effect of n_estimators")
plt.xlabel("n_estimator")
plt.ylabel("score")
plt.plot(estimators, scores)


# Evaluate Regression model

# In[ ]:


import sklearn.metrics as metrics


# In[ ]:


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true + 1 - y_pred) / (y_true + 1)) * 100)
def regression_results(y_true, y_pred):
    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('R^2: ', r2)
    print('MAE: ', mean_absolute_error)
    print('MSE: ', mse
    print('RMSE: ', np.sqrt(mse))
    print('MAPE: ', mean_absolute_percentage_error(y_true, y_pred), '%')


# In[ ]:


regression_results(y_test, model2.predict(X_test))

