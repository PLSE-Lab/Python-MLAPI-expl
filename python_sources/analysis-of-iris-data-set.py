#!/usr/bin/env python
# coding: utf-8

# Hi everyone, it is my first kernel and I am new in data science but I have some knowledge about statistics and  machine learning. 
# In this study, I just analyzed the iris data set and I talked about the some statistical knowledge. I hope it would be useful for you. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/Iris.csv")
data.info()


# In[ ]:


data.head(10)


# We use the correlation to see the relationship between features. If the correlation between two features is close to 1 or -1  then they are related. If the correlation between two features is close to 0 then they are unrelated.

# In[ ]:


df = data.drop("Id",axis=1)
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# We can see the relationship between petal width and petal length is strong. On the other hand the relationship between sepal width and sepal length is weak because their correlation value is -0.1. 

# In[ ]:


plt.scatter(df.SepalLengthCm,df.PetalLengthCm,color="red")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.show()


# In[ ]:


plt.scatter(df.PetalLengthCm,df.PetalWidthCm,color="red")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.show()


# In[ ]:


plt.scatter(df.SepalWidthCm,df.SepalLengthCm,color="blue")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Sepal Length (cm)")
plt.show()


# We use the scatter plot to see the relationship between features. For example, we can see the 
# relationship between petal length and petal width is linear but there is no relation between sepal
# width and sepal length.

# We use the boxplot to see the distribution of data. It gives us five plot summary of data. The five plot summary contains, minimum value, first quartile, median, third quartile, maximum value. We can obtain the interquartile range and range from these values. We can use the boxplot to detect the outliers of data. Outliers are above from (3 * IQR(interquartile range) + third quartile) or  below from 
# (first quartile-3 * IQR(interquartile range) ) 
# The five plot summary is so important to obtain the distribution of data. 

# In[ ]:


df.boxplot()

