#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[23]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 

# In[24]:


data=pd.read_csv('../input/train.csv')
data1=pd.read_csv('../input/test.csv')


# In[27]:


data.head()


# In[26]:


data1.head()


# In[29]:


f,ax=plt.subplots(1,2,figsize=(10,4))
data['Survived'].value_counts().plot.pie(explode=[0.01,0.01],autopct='%1.1f%%',ax=ax[0])
ax[0].set_title('Survived passengers')
ax[0].set_ylabel('')
sns.countplot('Survived',data=data,ax=ax[1])
ax[1].set_title('Survived passengers')
plt.show()


# In[ ]:


data.groupby(['Sex','Survived'])['Survived'].count()

Grouping of categorial variables
# In[30]:


f,ax=plt.subplots(1,2,figsize=(10,4))
data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived people vs Sex')
sns.countplot('Sex',hue='Survived',data=data,ax=ax[1])
ax[1].set_title('Sex:Survived ones vs Dead ones')
plt.show()


# In[32]:


f,ax=plt.subplots(1,2,figsize=(10,4))
data['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])
ax[0].set_title('Number Of Passengers By Pclass')
ax[0].set_ylabel('Count')
sns.countplot('Pclass',hue='Survived',data=data,ax=ax[1])
ax[1].set_title('Pclass:Survived vs Dead')
plt.show()


# In[ ]:




