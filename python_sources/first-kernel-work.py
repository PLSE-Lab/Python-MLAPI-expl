#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[3]:


data = pd.read_csv("../input/heart.csv")


# In[4]:


data.info()


# In[7]:


data.describe()


# In[8]:


data.corr()


# In[10]:


# correlation map
f,ax =plt.subplots(figsize=(12,12))
sns.heatmap(data.corr(), annot=True, linewidths=0.5, fmt=".1f", ax=ax)
plt.show()


# In[11]:


data.head(15)


# In[12]:


data.columns


# In[18]:


# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.trestbps.plot(kind="line", color="blue", label="trestbps", linewidth=1.5, alpha=0.5, grid=True, linestyle=":")
data.chol.plot(kind="line", color="red", label="chol", linewidth=1, alpha=0.5, grid=True, linestyle=":")
plt.legend(loc="upper left")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("Line Plot")
plt.show()


# In[20]:


# Scatter Plot 
# x = trestbps, y = thalach
plt.scatter(data.trestbps, data.thalach, color="orange", alpha=0.5)
plt.xlabel("trestbps")
plt.ylabel("thalach")
plt.title("Scatter Plot for trestbps and thalach")
plt.show()


# In[21]:


# Histogram
# bins = number of bar in figure
data.trestbps.plot(kind="hist", bins=50, figsize=(12,12))
plt.show()


# In[22]:


series = data["trestbps"]
print(type(series))
data_frame = data[["trestbps"]]
print(type(data_frame))


# In[32]:


x = data["trestbps"]>=180
data[x]


# In[35]:


data[(data["trestbps"]>=180) & (data["thalach"]>150)]


# In[36]:


# while loop
i = 0
while i != 5:
    print("i is",i)
    i += 1
print(i,"is not equal to 5")


# In[42]:


# for loop
liste = [5,4,3,2,1]

for index,value in enumerate(liste):
    print(index, " : ", value)
print("")
    
for index,value in data[["trestbps"]][0:2].iterrows():
    print(index, " : ", value)


# In[ ]:




