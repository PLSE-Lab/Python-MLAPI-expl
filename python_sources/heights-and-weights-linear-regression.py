#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

print(os.listdir("../input"))


# **Reading data using pandas read_csv method**

# In[3]:


df = pd.read_csv("../input/data.csv")
df.head(5)


# Plotting graph.

# In[6]:


plt.title("Us data")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.scatter(df.Height,df.Weight,color='blue')


# **Now let's check if data contains NaN value or not.**

# In[8]:


np.where(np.isnan(df['Height']))


# No NaN entry in Height column, let's check for weight column

# In[9]:


np.where(np.isnan(df['Weight']))


# Here data is already clean. so now let's go for data split (training and testing set) 

# In[19]:


from sklearn.model_selection import train_test_split
x = df[['Height']]
y = df['Weight']


# In[23]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


#     Let's move to LinearRegression

# In[17]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[25]:


model.fit(x_train,y_train)


# **Let's see what our model will predict for first row of data.**

# In[29]:


df.head(1)


# In[33]:


print(model.predict([[1.47]]))


# **we are getting output as 51.05 it's near to actual value 52.21**

# **Let's see score of our model.**

# In[34]:


model.score(x_test,y_test)


# we are getting 0.97 accuracy (97%).
