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


# **INTRODUCTION**
# 
# *You'll see on this project;*
# * EDA (Exploratory Data Analysis)
# * Data Cleaning
# * Logistic Regression with Sklearn
# * Results and Conclusion

# In[ ]:


#Firstly, we should import data from dataset
data = pd.read_csv("../input/seattleWeather_1948-2017.csv")


# In[ ]:


data.head()
#Shows us informations of first 5 weather conditions.


# In[ ]:


data.tail()
#Shows us informations of last 5 weather conditions.


# In[ ]:


data.columns #See columns in data


# In[ ]:


data.describe()


# **We will not use DATE feature in this practice. Because DATE feature doesn't effect our comparasion. Therefore, let's drop that feature**

# In[ ]:


data.drop(["DATE"],axis=1,inplace=True) #Drop processing


# In[ ]:


data.columns #As you can see, we droped DATE feature


# In[ ]:


data.info()


# **As we can see we have some missing values in there! We should clean that NaN values.**

# In[ ]:


data=data.dropna() #We drop all NaN values.
data.info()


# **As we can see; our RAIN feature is an object. We can not compare like this. There for we must convert this from object to integer.**
# 
# **I'll say 1 to TRUE values, 0 to FALSE values.**

# In[ ]:


data.RAIN = [1 if each == True  else 0 for each in data.RAIN.values]


# In[ ]:


data.info()
#You can see easily; RAIN values converted to integer.


# In[ ]:


y = data.RAIN.values
#our y axis is defined RAIN values because we want to learn how weather is? 
#So we are trying to get: the weather is rainy or not.


# In[ ]:


x_data = data.drop(["RAIN"],axis=1)
#x_data is all of features except RAIN in the data.
x_data


# In[ ]:


#Normalization
x = (x_data - np.min(x_data))/(np.max(x_data)).values
x


# **All of values are between 0 and 1.**

# In[ ]:


#Train-Test Datas Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 42)
#test_size=0.2 means %20 test datas, %80 train datas


# In[ ]:


x_train.head()


# In[ ]:


x_train.tail()


# In[ ]:


y_test


# **I'll transpose to values. This is optional, you don't have to do that.**

# In[ ]:


x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T


# **And finally, time to Logistic Regression!**

# In[ ]:


from sklearn import linear_model
logistic_reg = linear_model.LogisticRegression(random_state=42,max_iter=150)
#max_iter is optional parameter. You can write 10 or 3000 if you want.


# **Let's see accuracy of our regression**

# In[ ]:


print("Test accuracy {}".format(logistic_reg.fit(x_train.T,y_train.T).score(x_test.T,y_test.T)))
print("Train accuracy {}".format(logistic_reg.fit(x_train.T,y_train.T).score(x_train.T,y_train.T)))


# **CONCLUSION**
# 
# Logistic regression is used just a dataset which include 2 outputs (True-False, on-off, 1-2 etc.)
# We can write all methods by our hand but sklearn library is so useful for logistic regression.
