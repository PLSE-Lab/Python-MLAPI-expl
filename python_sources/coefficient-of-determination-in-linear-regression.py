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


# # The coefficient of determination (denoted by R2) is a key output of regression analysis. It is interpreted as the proportion of the variance in the dependent variable that is predictable from the independent variable.

# # The coefficient of determination is the square of the correlation (r) between predicted y scores and actual y scores; thus, it ranges from 0 to 1.

# # An R2 of 0 means that the dependent variable cannot be predicted from the independent variable.
# # An R2 of 1 means the dependent variable can be predicted without error from the independent variable.

# ## I am explaining r2_score() with below example.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


# Importing the dataset
dataset = pd.read_csv('/kaggle/input/years-of-experience-and-salary-dataset/Salary_Data.csv')
X = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values


# In[ ]:


# As we need 2D Input 
X=X.reshape(-1, 1)


# In[ ]:


X.shape


# In[ ]:


y.shape


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


# In[ ]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[ ]:


lr.fit(x_train, y_train)


# In[ ]:


m=lr.coef_
b=lr.intercept_
print("slope=",m, "intercept=",b)


# In[ ]:


y_pred = lr.predict(x_test)


# In[ ]:


plt.scatter(x_train, y_train, color = "blue")
plt.plot(x_train, lr.predict(x_train), color = "red")
plt.title("Salary vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()


# In[ ]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))


# In[ ]:


# variance score: 1 is perfect prediction
from sklearn.metrics import mean_squared_error, r2_score
print('Variance score: %.2f' % r2_score(y_test, y_pred))


# In[ ]:


print(lr.predict([[6]]))


# # we are getting 96% score. It is a good result. you may get different score each time you run.you can achieve same result in Microsoft Azure Studio.
# 
