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


import pandas as pd


# In[ ]:


data=pd.read_csv('../input/positon-salaries1/Position_Salaries.csv')
data.head()


# In[ ]:


#apply polinomial regression to above data


# task is to predict the salary of persone which is lies between given levels

# 

# In[ ]:


#step 1: visualize our data points


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.scatter(data.Level,data.Salary)
plt.xlabel("levels")
plt.ylabel("salary")
plt.title("levels vs salary")
plt.show()


# from above fig it show  that our  data is non-linear

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures


# In[ ]:


poly=PolynomialFeatures(degree=4)


# In[ ]:


x=data['Level']


# In[ ]:


y=data["Salary"]
y.head()


# In[ ]:


import numpy as np


# In[ ]:


x=np.array(x)
y=np.array(y)


# In[ ]:


x_poly=poly.fit_transform(x[:,np.newaxis])


# In[ ]:


x_poly


# In[ ]:


plt.scatter(data.Level,data.Salary)
plt.plot(data.Level,model.predict(x_poly))
plt.show()


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


model=LinearRegression()


# In[ ]:


model.fit(x_poly,y)


# In[ ]:





# In[ ]:


x=np.array([6.5])


# In[ ]:


x.ndim


# In[ ]:


x


# In[ ]:


x1=poly.fit_transform(x[:,np.newaxis])


# In[ ]:


model.predict(x1)

