#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


# In[46]:


data = pd.read_csv("../input/Position_Salaries.csv")

data.head()


# In[47]:


x=data.iloc[:,1:2]
x


# In[48]:


y=data.iloc[:,2]
y


# In[49]:


model = DecisionTreeRegressor(random_state=0)
model.fit(x,y)


# In[50]:


a = model.predict(x)


# In[51]:


mae = mean_absolute_error(y,a)

mae


# In[54]:


fig,ax = plt.subplots()

ax.scatter(x,y)
ax.plot(x,a)

plt.xlabel("Position Level")
plt.ylabel("Salary")

plt.show()

