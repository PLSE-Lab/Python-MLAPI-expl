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


# This dataset gives information about us diabetes on Pima Indians.
# 
# Outcome = 0; not diabetes.
# Outcome = 1; diabetes.
# 
# Firstly, read dataset and save at "data" variable.

# In[ ]:


data = pd.read_csv("../input/diabetes.csv") 


# In[ ]:


data.info() # gives info about our data


# As we seen above, our dataset has 9 features (columns) and have 768 different data. Let's see correlation between of features and make its visualisation:

# In[ ]:


data.corr()


# In[ ]:


#correlation map 
f,ax = plt.subplots(figsize=(15, 10)) #figsize; sets the size of boxes
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
#annot = True; allows the use of correlation results in boxes
#linewidth; thickness of line between boxes
#fmt; sets the length of the decimal portion
plt.show()


# If correlation result is 1 or close to 1; we can say this features is revelant (positive revelant).
# If correlation result is 0 or close to 0; we can say this features is not revelant.
# If correlation result ise -1 or close to -1; we can say this features is revelant (negative revelant).
# 
# Based on this information, if we look at results, we can comment about features.  For example; Age and Pregnancies can be revelant because its correlation results is 0,5. An other example; Insulin and Age can't be revelant because its correlation result is 0. 
# 
# We can make more comment about this results.

# In[ ]:


print(data.head(20)) #Gives us top 20 of data.
print(data.tail(20)) #gives us last 20 of data.


# Now, we will see the histogram of our data. Firstly, we will view histogram of Age.

# In[ ]:


data.Age.plot(kind = 'hist',bins = 60,figsize = (15,15))
# bins = number of bar in figure
#x axis is Age. y axis is frequency of Age.
plt.show()


# If we look at histogram; there is the maximum data in the dataset is in the 21-23 age group. The minimum data in the dataset is in the ~65-66,~68-69,~72-73 and~80-82 ages and there is any data under ~22 years old and ~74-80 years old.

# In[ ]:


data.Outcome.plot(kind = 'hist',bins = 20,figsize = (5,5))
# bins = number of bar in figure
#x axis is Age. y axis is frequency of Age.
plt.show()


# We can see relevant of 2 features with scatter plot. For example; we can plot the relevant of glucose and insulin.

# In[ ]:


# x = pregnancy, y = outcome
data.plot(kind='scatter', x='Glucose', y='Insulin',alpha = 0.5,color = 'red')
#plt.scatter(data.Glucose,data.Insulin,alpha = 0.5,color = 'red') ## It is same as top row
plt.xlabel('Glucose')              
plt.ylabel('Insulin')
plt.title('Glucose and Insulin Relevant')            
plt.show()


# Now we can analyse the outcome. In out dataset; 0 is not diabetes patient and 1 is diabetes patient.
# As we can see, in out dataset; have diabetes patient about 260 people and have not about 500 diabetes patient. 
# 
# There is the describe of out dataset is in below.

# In[ ]:


data.describe()


# Let's analyse the describe results. 
# 
# The minimum age is 21 and maximum age is 81.  We saw this in histogram chart.
# The mean of our dataset is ~33.
# 
# In our dataset there is the maximum number of pregnancy is 17 and minimum number of pregnancy is 0. The mean of pregnancy is ~4.
# 
# The maximum glucose level is 199 and mean glucose level is ~120. 
# 
# The maximum blood pressure level is 122, minimum blood pressure level is ~19 and mean blood pressure level is ~69. 
# 

# We can make filters for see the data that we want.

# In[ ]:


filt = data.Age > data.Age.mean() #if da
filtered_data = data[filt]
print(filtered_data)


# In[ ]:


x = data['Glucose']>185 
data[x]


# In[ ]:


data[np.logical_and(data['Glucose']>180, data['Outcome'] == 1 )] 
#we have find the people who has glucose level over 180 and have diabetes.


# In[ ]:


data[np.logical_and(data['Age']>40, data['Outcome'] == 0 )]
#we have find the people who is over 40 years old and have not diabetes.


# **Conclusion**
# This is my first work while I am learning Data Science. So I can make mistakes or I may have missings.
# Please suggest me about this work. Thank you.
