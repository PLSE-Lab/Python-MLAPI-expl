#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
print(os.listdir("../input"))
import matplotlib.pyplot as plt
# Any results you write to the current directory are saved as output.


# In[4]:


print("Number of images in the train images")
print(len(os.listdir("../input/train")))
print("Number of images in the test images")
print(len(os.listdir("../input/test")))


# The number of images in the test folder are greater than in the train folder.

# In[3]:


df = pd.read_csv('../input/train.csv')


# In[4]:


df.head()


# In[34]:


#histogram of 
df['Id'].value_counts().hist(bins=50)


# In[10]:


df['Id'].value_counts().head(7)


# In[19]:


print("Total values that are null")
print(df.isnull().sum().sort_values(ascending=False))
print("Total values that are ")
print(df.isnull().count())
total = df.isnull().sum().sort_values(ascending=False)
percent = df.isnull().sum().sort_values(ascending=False)/df.isnull().count().sort_values(ascending=False)


# In[24]:


pd.concat([total,percent],axis=1,keys=['Total','Percent'])


# There is no missing data.

# In[38]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Ids = le.fit_transform(df['Id'])
sns.distplot(Ids)
plt.title('Categorical Distribution')
plt.show()


# Most frequent whale classes

# In[77]:


temp = pd.DataFrame(df['Id'].value_counts().head(8))
temp.reset_index(inplace=True)
temp.columns = ['Id','Counts']


# In[81]:


plt.figure(figsize=(9,8))
sns.barplot(x='Id',y='Counts',data = temp)
plt.show()


# Least frequent whale classes

# In[92]:


temp = pd.DataFrame(df['Id'].value_counts().tail())
temp.reset_index(inplace=True)
temp.columns = ['Whale_id','Counts']


# In[93]:


sns.barplot(x='Whale_id',y='Counts',data=temp)


# **Reference:**
# *https://www.kaggle.com/codename007/a-very-extensive-landmark-exploratory-analysis*

# In[ ]:




