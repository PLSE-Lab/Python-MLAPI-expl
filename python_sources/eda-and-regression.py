#!/usr/bin/env python
# coding: utf-8

# # EDA and Regression
# 
# ## in progress...

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import os
print(os.listdir("../input"))


# Firstly fetch dataset into `data` variable.

# In[ ]:


data = pd.read_csv("../input/column_2C_weka.csv")


# And then take a glimpse of data, by using the method `data.head()`.

# In[ ]:


data.head()


# Let's examine the data, via `data.info()` method.

# In[ ]:


data.info()


# Each patient is represented in the data set by six biomechanical attributes derived from the shape and orientation of the pelvis and lumbar spine (each one is a column):  
# 
# * pelvic incidence
# * pelvic tilt
# * lumbar lordosis angle
# * sacral slope
# * pelvic radius
# * grade of spondylolisthesis

# In[ ]:


data.describe()


# In[ ]:


data.corr()


# In[ ]:


f,axis = plt.subplots(figsize=(12, 12))
sns.heatmap(data.corr(), annot=True, linewidths=.4, fmt= '.2f', ax = axis)
plt.show()


# In[ ]:


linear_reg = LinearRegression()

reshaped_x = data.pelvic_incidence.values.reshape(-1,1)
reshaped_y = data.sacral_slope.values.reshape(-1,1)

linear_reg.fit(reshaped_x,reshaped_y)

y_predicted = linear_reg.predict(reshaped_x)


# In[ ]:


plt.figure(figsize = (12,10))
plt.scatter(reshaped_x,reshaped_y,color='blue')
plt.plot(reshaped_x,y_predicted,color='red')
plt.show()


# In[ ]:


print(r2_score(y_predicted, reshaped_y))


# In[ ]:


data

