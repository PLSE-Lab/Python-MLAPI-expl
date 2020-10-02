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


# It is a regression practice using admission data.
# 
# **Mohan S Acharya, Asfia Armaan, Aneeta S Antony : A Comparison of Regression Models for Prediction of Graduate Admissions, IEEE International Conference on Computational Intelligence in Data Science 2019**

# In[ ]:


df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
df.head()


# In[ ]:


df.info()


# In[ ]:


X = df.drop(['Serial No.', 'Chance of Admit '], axis=1)
y = df['Chance of Admit ']


# In[ ]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

lr = LinearRegression().fit(X_train, y_train)
y_pred = lr.predict(X_test)
mse_train = mean_squared_error(y_train, lr.predict(X_train))
mse_test = mean_squared_error(y_test, y_pred)
print("Training set's mean squared error:", mse_train)
print("Test set's mean squared error:", mse_test)


# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X.head())
print(X_scaled[:5])


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=123)

lr = LinearRegression().fit(X_train, y_train)
y_pred = lr.predict(X_test)
mse_train = mean_squared_error(y_train, lr.predict(X_train))
mse_test = mean_squared_error(y_test, y_pred)
print("Training set's mean squared error:", mse_train)
print("Test set's mean squared error:", mse_test)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

corr = df.drop('Serial No.', axis=1).corr()
sns.heatmap(corr, annot=True, cmap=plt.cm.Reds);


# In[ ]:




