#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Hello,
# This is my first kagglel and data sicence project. Now, I will try to put into practice what I have learned so far.**
# 

# In[ ]:


#Firstly, the dataset was imported
data = pd.read_csv("../input/data.csv")


# In[ ]:


#Get information about the data set
data.info()


# In[ ]:


#correlation observation
#If  correlation is 1 between the two properties there is a direct proportion between them
#If correlation is 0 there is no relationship 
#If correlation is -1 there is negative (inverse) proportion
data.corr()


# In[ ]:


#correlation map
f,ax = plt.subplots(figsize = (20,20))
sns.heatmap(data.corr(), annot = True, linewidth = 0.2, fmt = ".2f", ax = ax)
plt.show()


# In[ ]:


#the first few lines of the data set
data.head(10)


# In[ ]:


#column names contained in the data set
data.columns


# In[ ]:


#Line Plot
data.radius_mean.plot(kind = "line", label = "Radius Mean", linewidth = 1 , color = "r", alpha = 0.7, grid = True, linestyle = ":" )
data.radius_worst.plot(label = "Radius Worst", linewidth = 1, color = "g", alpha = 0.7, grid= True, linestyle = "-.")
plt.legend(loc = "upper left")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("Line Plot")
plt.show()


# In[ ]:


#Scatter plot.
#Goal is to find the correlation between the two features
# x = "radius_mean" y="radius_worst"
data.plot(kind = "scatter", x = "radius_mean", y = "radius_worst", alpha = 0.5, color = "green" )
plt.xlabel("Radius Mean")
plt.ylabel("Radius Worst")
plt.title("Radius Mean - Radius Worst Scatter Plot")
plt.show()


# In[ ]:


#histogram representation
data.radius_mean.plot(kind = "hist", bins = 20, figsize = (10,10))
plt.show()


# In[ ]:


# clf() ----------> allows you to clean the drawn graphic
data.radius_mean.plot(kind = "hist", bins = 20)
plt.clf()


# In[ ]:


# pandas has three datatype. First is series, second is dataframe
series = data["radius_mean"]
print(type(series))
print("")
data_frame = data[["radius_mean"]]
print(type(data_frame))


# In[ ]:


#indicates whether the rows contained in the specified feature satisfy the condition
data["radius_mean"]>25


# In[ ]:


# filtering pandas data frame
x = data["radius_mean"]>25
data[x]


# In[ ]:


#filtering with logical_and 
data[np.logical_and(data["radius_mean"]>20,data["texture_mean"]<20)]


# In[ ]:


#using '&' for filtering
data[(data["radius_mean"]>20) & (data["texture_mean"]<20)]


# In[ ]:


#achieve index and value
for index,value in data[["radius_mean"]][0:3].iterrows():
    print(index,"****",value["radius_mean"])

