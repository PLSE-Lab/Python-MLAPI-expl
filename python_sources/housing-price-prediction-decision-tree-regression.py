#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split


# In[2]:


data = pd.read_csv("../input/melb_data.csv")

data.head()


# In[3]:


data.columns


# In[4]:


data.isnull().sum()
data = data.dropna(axis=0,how='any')
data.isnull().sum()


# In[5]:


data.describe()


# **Decision Tree Regression**

# In[6]:


y = data['Price']     #Prediction Target


# In[7]:


x = data[['Rooms','Bathroom','Landsize','Lattitude','Longtitude']]


# In[8]:


x.describe()


# In[ ]:





# In[9]:


x.head()


# In[10]:


model = DecisionTreeRegressor()

model.fit(x,y)


# In[11]:


print("Making predictions")
print(x)

print("Predictions : ")
a = model.predict(x)
a


# In[12]:


data.head()


# In[13]:


mean_absolute_error(y,a)


# In[14]:


train_x,val_x,train_y,val_y = train_test_split(x,y,random_state=0)


# In[15]:


model = DecisionTreeRegressor()

model.fit(train_x,train_y)


# In[16]:


a = model.predict(val_x)
print(a)


# In[17]:


print(mean_absolute_error(val_y,a))


# In[18]:


def get_mae(leaf_nodes,train_x,val_x,train_y,val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=leaf_nodes,random_state=0)
    model.fit(train_x,train_y)
    a = model.predict(val_x)
    mae = mean_absolute_error(val_y,a)
    return(mae)


# In[19]:


for leaf_nodes in np.arange(5,5000,500):
    mae = get_mae(leaf_nodes,train_x,val_x,train_y,val_y)
    print(leaf_nodes,"  >>>>>>>>>>  ",mae)


# **Random Forest**

# In[20]:


data.head()


# In[21]:


x = data[['Rooms','Bathroom','Landsize','Lattitude','Longtitude']]

y=data['Price']


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=1)
model.fit(train_x,train_y)
a=model.predict(val_x)
mean_absolute_error(val_y,a)

