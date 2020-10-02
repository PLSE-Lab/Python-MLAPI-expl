#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


# In[2]:


print(os.listdir("../input"))


# In[3]:


data = pd.read_csv("../input/StudentsPerformance.csv")


# In[4]:


data.head()


# In[5]:


data.sample(5)


# In[6]:


data.info()


# In[7]:


data.describe()


# In[8]:


data['gender'].value_counts


# In[9]:


sns.countplot(x='gender',data=data)
plt.ylabel('frequence')
plt.title('Gender Bar Plot')
plt.show()


# In[10]:


sns.countplot(x='parental level of education',data=data)
plt.xticks(rotation=45)
plt.ylabel('frequence')
plt.title('Parental level of education Bar Plot')
plt.show()


# In[11]:


sns.barplot(x='gender',y='reading score',data=data)
plt.show()


# In[12]:


sns.barplot(x='gender',y='reading score',hue='race/ethnicity',data=data)
plt.show()


# In[13]:


sns.distplot(data['math score'],bins =10,kde=True)


# In[14]:


sns.jointplot(x='math score',y='writing score',data=data)


# In[15]:


sns.pairplot(data)


# In[16]:


sns.boxplot(x='gender',y='math score',data=data)
plt.show()


# In[17]:


sns.boxplot(x='gender',y='writing score',data=data)
plt.show()


# In[18]:


sns.boxplot(x='race/ethnicity',y='writing score',data=data)
plt.show()


# In[19]:


data.corr()


# In[20]:


sns.heatmap(data.corr())


# In[21]:


g=sns.catplot(x='gender',y='writing score',hue='lunch',col_wrap=3,col='race/ethnicity',data=data,kind='bar',height=4,aspect=0.7)
plt.show()


# In[ ]:




