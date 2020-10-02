#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


randomGen = np.random.RandomState(1)
X1 = randomGen.randint(low=500, high=2000, size=50)
X2 = randomGen.randint(low=100, high=500, size=50)
X3 = X1 * 3 + randomGen.rand()
Y = X3 + X2

data = pd.DataFrame({
    'X1':X1,
    'X2':X2,
    'X3':X3,
    'Y':Y
})
curr1 = pd.DataFrame({
    'X1':X1,
    'Y':Y
})
curr2 = pd.DataFrame({
    'X2':X2,
    'Y':Y
})
curr3 = pd.DataFrame({
    'X3':X3,
    'Y':Y
})
data


# In[ ]:


curr1.corr()


# In[ ]:


curr2.corr()


# In[ ]:


curr3.corr()


# In[ ]:


plt.scatter(X1, Y)
plt.show()


# In[ ]:


plt.scatter(X2, Y)
plt.show()


# In[ ]:


from sklearn.model_selection import train_test_split 
from sklearn import linear_model
import statsmodels.api as sm


# In[ ]:


data2 = data.copy()


# In[ ]:


X1_data = data2[['X1']]
X2_data = data2[['X2']]
Y_data = data2['Y']
Y2_data = data2['Y']

X_train, X_test, y_train, y_test = train_test_split(X1_data, Y_data, test_size=0.30)
X2_train,X2_test, y2_train,y2_test = train_test_split(X2_data, Y2_data, test_size=0.30)
pd.DataFrame(X_test)


# In[ ]:


Regres = linear_model.LinearRegression()
model1 = Regres.fit(X_train,y_train)
model2 = Regres.fit(X2_train,y2_train)
prediction1= Regres.predict(pd.DataFrame(X_test))
prediction2 = Regres.predict(pd.DataFrame(X2_test))


# In[ ]:


plt.scatter(X_test,prediction1, color='blue')
plt.show()


# In[ ]:


plt.scatter(X2_test,prediction2, color='red')
plt.show()


# In[ ]:


import seaborn as sns


# In[ ]:



sns.set(style="whitegrid")

# Plot the residuals after fitting a linear model
sns.residplot(X_test, y_test, lowess=True, color="b")


# In[ ]:


sns.residplot(X2_test, y2_test, lowess=True, color="b")


# In[ ]:


sns.residplot(Regres.predict(X_train), Regres.predict(X_train)-y_train, lowess=True, color="r")
sns.residplot(Regres.predict(pd.DataFrame(X_test)), Regres.predict(pd.DataFrame(X_test))-y_test, lowess=True, color="g")
plt.title('Residual Plot using Training (red) and test (green) data ')
plt.ylabel('Residuals')


# In[ ]:


sns.residplot(Regres.predict(X2_train), Regres.predict(X2_train)-y_train, lowess=True, color="r")
sns.residplot(Regres.predict(pd.DataFrame(X2_test)), Regres.predict(pd.DataFrame(X2_test))-y_test, lowess=True, color="g")
plt.title('Residual Plot using Training (red) and test (green) data ')
plt.ylabel('Residuals')


# In[ ]:


from sklearn.metrics import r2_score
r2_score(y_test,prediction1)


# In[ ]:


from sklearn.metrics import r2_score
r2_score(y2_test,prediction2)


# In[ ]:


print(Regres.coef_)


# In[ ]:


print(Regres.intercept_)


# What is the impact of X1 and X2 on Y?
# 

# From your model, deduse the regression formula
# 
# Y1 = -2.159195798713163 * X1 + 1.86712932
# Y2 = -0.4061612823643841 * X2 + 1.86712932
# 

# In[ ]:


from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[ ]:


print("Mean squared error of X1: %.2f" % mean_squared_error(y_test, prediction1))
print("Mean squared error of X2: %.2f" % mean_squared_error(y2_test, prediction2))


# In[ ]:


print("Mean squared error of X1: %.2f" % mean_absolute_error(y_test, prediction1))
print("Mean squared error of X2: %.2f" % mean_absolute_error(y2_test, prediction2))


# In[ ]:


print("Mean squared error of X1: %.2f" % np.sqrt(((prediction1 - y_test) ** 2).mean()))
print("Mean squared error of X2: %.2f" % np.sqrt(((prediction2 - y2_test) ** 2).mean()))


# In[ ]:




