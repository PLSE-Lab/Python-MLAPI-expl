#!/usr/bin/env python
# coding: utf-8

# ## Importing 
# 

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


dataset_dirty = pd.read_csv("../input/train.csv")
dataset = dataset_dirty.dropna()
testset_dirty = pd.read_csv("../input/test.csv")
testset = testset_dirty.dropna()
testset


# In[ ]:


x_train = dataset.iloc[:,0].values.reshape(-1, 1)
y_train = dataset.iloc[:,1].values


# # Data is currently clean so lets build the model 

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


model = LinearRegression()
model.fit(x_train, y_train)


# ## Now we Predict

# In[ ]:


x_test = testset.iloc[:,0].values.reshape(-1, 1)
y_test = testset.iloc[:,1].values


# In[ ]:


goal_test = model.predict(x_test)


# ## Plotting 

# In[ ]:


plt.plot(x_test, goal_test, x_test, y_test, '.')
plt.title("Accuracy of the model")
plt.show()


# and thats our model !! 

# In[ ]:




