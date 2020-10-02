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


from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Loading the data set
data_test=pd.read_csv("../input/salary/Salary.csv")
# print(data_test)

# Splitting the trainign and testing data
X = data_test['YearsExperience'].values
y = data_test['Salary'].values

X = X.reshape(-1,1)
y = y.reshape(-1,1)
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.33)
# print(x_train)

# plotting full data set
# plt.scatter(x_test,y_test)
# plt.show()

# reshaping the data 

# x_test.values.reshape(1,-1)
# y_test.values.reshape(1,-1)


# loading the model
print(np.shape(X))
print(np.shape(y))
reg=linear_model.LinearRegression()
reg.fit(x_train,y_train)
preds=reg.predict(x_test)
plt.scatter(x_train,y_train,c='b')
plt.scatter(x_test,y_test,c='r')
plt.plot(x_test,preds,'y',linewidth=2)
plt.show()

print("Success percentage is ")
print(reg.score(x_train,y_train))
print("on testing data")
print(reg.score(x_test,y_test))
# print("On predicted data")
# print(reg.score(x_test,preds))
# Plotting the model
# print(np.shape(x_train))
# x_train=x_train.values.reshape(1,-1)
# print(np.shape(x_train))
# print(np.shape(y_train))
# y_train=y_train.values.reshape(1,-1)
# print(np.shape(y_train))
# # Fitting the model
# reg.fit(x_train,y_train)
# print(np.shape(x_test))
# x_test=x_test.values.reshape(1,-1)
# print(np.shape(x_test))
# # preds=reg.predict(x_test)
# print("Score is ")
# print(reg.score(data_test[["YearsExperience"]],data_test[['Salary']]))

