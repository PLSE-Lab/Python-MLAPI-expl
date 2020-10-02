#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


# In[ ]:


#loading the train and test data 
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")


# In[ ]:


train_data.head()


# In[ ]:


test_data.tail()


# In[ ]:


# lets visualize the data points using matplotlib
# simple regression 
# simple line function is sufficient to find the values of y.
# I can see only the liear relation between the Y and X so no need of transforming the Y value

plt.scatter(train_data['x'],train_data['y'])
plt.xlabel("x_values")
plt.ylabel("Y values")
plt.title("Linearly generated data")


# In[ ]:


# finding the null values and removing those
print(train_data.x.count())
print(train_data.y.count())

train_data = train_data[~train_data.y.isnull()]


# In[ ]:


#preparing the X and Y datasets
xData = np.array(train_data['x'],dtype=np.float64).reshape(-1,1)
yData = train_data.y 


# In[ ]:


xData


# In[ ]:


from sklearn import linear_model

model = linear_model.LinearRegression()


# In[ ]:


model.fit(xData, yData)


# In[ ]:


# lets test with the test data
x_testData = np.array(test_data.x,dtype=np.float32).reshape(-1,1)


# In[ ]:


y_testData = test_data.y


# In[ ]:


model.score(x_testData, y_testData)


# In[ ]:




