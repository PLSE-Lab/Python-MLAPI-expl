#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.api as sm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


rng= np.random.RandomState(1)
rng


# In[ ]:


X1=[rng.randint(500,2000) for i in rng.rand(50)]
X1


# In[ ]:


X2=[rng.randint(100,500) for i in rng.rand(50)]
X2


# In[ ]:


type(X2)


# Series is a one-dimensional labeled array capable of holding data of any type (integer, string, float, python objects, etc.). The axis labels are collectively called index.

# In[ ]:


pd.Series(X2)*3


# In[ ]:


X3=pd.Series(X2)*3+rng.rand(50)
X3


# In[ ]:


#testings
X4=pd.Series(X2)*3+rng.rand(50)
X4


# In[ ]:


rng.rand(50)+3*3


# X3 is missing a random vector.

# In[ ]:


Y=(X3*X2)*0.006
Y


# In[ ]:


dataset= pd.DataFrame({'X1':X1,
                      'X2':X2,
                      'X3':X3,
                      'Y':Y})
dataset


# In[ ]:


print("Correlation between X1 and Y")
dataset[['X1','Y']].corr()


# lienear correlation 1
# not correlation -1

# In[ ]:


print("Correlation between X1 and Y")
dataset[['X2','Y']].corr()


# In[ ]:


print("Correlation between X1 and Y")
dataset[['X3','Y']].corr()


# In[ ]:


dataset.plot(kind='scatter',x='X1',y='Y',
           title="Scatter plot of X1 against Y",
            figsize=(12,8))


# In[ ]:


dataset.plot(kind='scatter',x='X2',y='Y',
           title="Scatter plot of X2 against Y",
            figsize=(12,8))


# In[ ]:


# separate data into dependent (Y) and independent(X1 and X2) variables
X_data = dataset[['X1','X2']]
Y_data = dataset['Y']


# In[ ]:


reg= linear_model.LinearRegression()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data,test_size=0.3)


# In[ ]:


X_train.columns


# In[ ]:


reg.fit(X_train, y_train)


# In[ ]:


reg.coef_


# In[ ]:


print('Regression Coeffient')
pd.DataFrame(reg.coef_ ,index=X_train.columns,columns=['coeffient'])


# In[ ]:


reg.intercept_


# Y = -1250.5853581479378 + -0.071288 X1 + 10.583246 X2

# In[ ]:


""""from sklearn.model_selection import KFold # import KFold
kf = KFold(n_splits=2)
kf.get_n_splits(X1)          
print (kf)"""


# In[ ]:


# Perform 6-fold cross validation
""""scores = cross_val_score(model, df, y, cv=6)
print ('Cross validation:'), scores"""


# In[ ]:


# Make cross validated predictions
""""predictions = cross_val_predict(model, df, y, cv=6)
plt.scatter(y, predictions)"""


# In[ ]:


import seaborn
w = 12
h = 10
d = 70
plt.figure(figsize=(w, h), dpi=d)
seaborn.residplot(X1, Y)
plt.savefig("out.png")


# In[ ]:


reg.intercept_


# In[ ]:


#R^2
reg.score(X_test,y_test) 


# In[ ]:


predicted = reg.predict(X_test)
da = pd.DataFrame({'Actual': y_test, 'Predicted': predicted})
da


# In[ ]:


l=581
p=355
q = {'x':[l],'y':[p]}
new=pd.DataFrame(q)
reg.predict(new)


# In[ ]:


X_data
mean_absolute_error(y_test, predicted)


# In[ ]:


mean_squared_error(y_test, predicted)


# In[ ]:


rms = sqrt(mean_squared_error(y_test, predicted))
rms

