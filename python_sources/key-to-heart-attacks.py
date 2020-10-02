#!/usr/bin/env python
# coding: utf-8

# * # Analyzing differnt reasons for heart attacks. 

# In[57]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import os
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# We will first analyze the dataset completely.

# In[58]:


data= pd.read_csv("../input/heart.csv")
print("there are", len(data.columns), "columns: ")
print (data.columns)
data.columns = [c.replace(' ', '_') for c in data.columns]
data.info()


# 

# In[59]:


#We will now analyze the genders first. 
#We can see that HEart attacks are more common in male than female. Lets do further analyses.


# In[60]:


data.head()


# In[61]:


sns.countplot(x="sex", data=data)
plt.show()


# In[62]:



plt.scatter(x=data.age[data.target==1], y=data.thalach[(data.target==1)], c ="red")
plt.scatter(x=data.age[data.target==0], y=data.thalach[(data.target==0)], c ="green")
plt.xlabel("age")
plt.ylabel("max heart rate")
plt.title("Heart Rate vs Age")
plt.show()


# #Here we can see the trend that people who have higher heart rate in early age (35-45) more prone to heart attack than peple with lower heart rate in early age.

# 

# #Now Lets analyze Chest Pain (CP) significance.

# In[ ]:





# In[63]:



sns.countplot(x="cp", data=data, hue="target")
plt.show()


# #We can see that maximum number of heart attack occur if CP is 2.

# In[64]:


sns.countplot(x="ca", data=data, hue="target")
plt.show()


# #Here we can see that number of vessel in blood is 0 in fluorosopy, then person has more chances of heart attack.

# In[ ]:





# In[65]:


sns.countplot(x="exang", data=data, hue="target")
plt.show()


# #WE can see that chest pain during exercise (exang) is not a big deal for a positive heart attack. People with exang 0(no chest pain) have higher heart attck rates.

# In[ ]:





# In[66]:


sns.countplot(x="ca", data=data, hue="target")
plt.show()


# In[ ]:





# In[67]:


data.describe()
#We can see visualization of statistical calculations.
data.boxplot(column="thalach", by="target")
# ages value by sex
plt.show()


# #This shows that person with more max heart rate are more prone to heart attck. Higher heart rate means more probabloty of heart attack.

# In[68]:


df3 = data[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']]


plt.figure(figsize = (15,15))
sns.pairplot(df3)
plt.show()


# In[69]:


plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,fmt='.1f')
plt.show()


# * # Please suggest me to improve this kernel further and upvote if you like it.
