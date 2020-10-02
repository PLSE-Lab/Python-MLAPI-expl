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


data= pd.read_csv("../input/salary-data-simple-linear-regression/Salary_Data.csv")


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


data.tail()


# In[ ]:


data.info 


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


plt.scatter(data.YearsExperience , data.Salary , color="blue")
plt.xlabel("Years experience")
plt.ylabel("salary")
plt.show()


# In[ ]:


len(data)


# In[ ]:


msk = np.random.rand(len(data))< 0.8
train = data[msk]
test = data[~msk]


# In[ ]:


train


# In[ ]:


test


# In[ ]:


from sklearn import linear_model
regr=linear_model.LinearRegression()


# In[ ]:


train_x = np.asanyarray(train[["YearsExperience"]])
train_y = np.asanyarray(train[["Salary"]])


# In[ ]:


train_x


# In[ ]:


train_y


# In[ ]:


regr.fit(train_x , train_y)


# In[ ]:


print('intercept : ' , regr.intercept_ )

print('coefficient : ' , regr.coef_ ) 


# In[ ]:


plt.scatter(train_x , train_y , color= 'blue')
plt.plot(train_x , regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel('Years Experience')
plt.ylabel('Salary')
plt.show()


# In[ ]:


from sklearn.metrics import r2_score

test_x = np.asanyarray(test[["YearsExperience"]])
test_y = np.asanyarray(test[["Salary"]])


# In[ ]:


testing_y = regr.predict(test_x )


# In[ ]:


testing_y


# In[ ]:


test_y


# In[ ]:


print("Mean absolute error: %.2f" % np.mean(np.absolute(testing_y - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((testing_y - test_y) ** 2))
print("R2-score: %.2f" % r2_score(testing_y , test_y) )


# In[ ]:


print(testing_y)


# In[ ]:




