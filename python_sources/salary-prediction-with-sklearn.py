#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data = pd.read_csv('/kaggle/input/salary-data/DataSalary.csv')
data.head()


# In[ ]:


data.describe(include='all')


# ## Creating the regression

# In[ ]:


from sklearn.linear_model import LinearRegression
inputs = data['YearsExperience']
target = data['Salary']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(inputs,target,test_size=0.2,random_state=12)


# ## **Training the model**

# In[ ]:


x_train_matrix = x_train.values.reshape(-1,1)
reg = LinearRegression()
reg.fit(x_train_matrix,y_train)


# In[ ]:


reg.coef_


# In[ ]:


reg.intercept_


# In[ ]:


reg.score(x_train_matrix,y_train)*100


# *Our model explain about 95% of variablity in data*

# In[ ]:


x_test_matrix = x_test.values.reshape(-1,1)
y_hat_train = reg.predict(x_train_matrix)
plt.scatter(x_train,y_train)
fig = plt.plot(x_train,y_hat_train,c='red',label='linearregression')


# ## **Testing**

# In[ ]:


x_test_matrix = x_test.values.reshape(-1,1)
y_hat_test = reg.predict(x_test_matrix)
plt.scatter(x_test,y_test)
fig = plt.plot(x_test,y_hat_test,c='red',label='linearregression')


# In[ ]:


#Predicting Salary with New Data
new_data = pd.DataFrame({'Years of Experience':[1.2,5.3,8,15,10.1,3]})
new_data['Pridicted Salary'] = reg.predict(new_data)
new_data

