#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
hr = pd.read_csv("../input/human-resources-data-set/HRDataset_v13.csv")


# In[ ]:


hr.head(7)


# In[ ]:


hr.shape


# In[ ]:


hr['Sex'].unique()


# In[ ]:


hr['Sex'].replace('M ','Nam',inplace=True)
hr['Sex'].replace('F','Nu',inplace=True)


# In[ ]:


hr['Sex'].unique()


# In[ ]:


hr.head(5)


# In[ ]:


hr['MarriedID'].replace(0,'Doc Than',inplace=True)
hr['MarriedID'].replace(1,'ket Hon',inplace=True)


# In[ ]:


hr['MarriedID'].unique()


# In[ ]:


hr.dropna(subset=['Sex'],inplace=True)
hr.shape


# In[ ]:


hr['Sex'].unique()


# In[ ]:


hr['Sex'].value_counts()


# In[ ]:


import matplotlib.pyplot as plt
hr['Sex'].value_counts().plot(kind='bar')


# In[ ]:


import seaborn as sns
plt.figure(figsize=(5,5))
ax=sns.countplot(x=hr['MarriedID'],hue=hr['Sex'])


# In[ ]:


plt.figure(figsize=(5,8))
ax2=sns.countplot(x=hr['Department'],hue=hr['Sex'])


# In[ ]:


hr['Department'].value_counts().plot(kind='pie')


# In[ ]:


hr['Department'].value_counts()


# In[ ]:


hr['CitizenDesc'].value_counts().plot(kind='bar')


# In[ ]:


plt.figure(figsize=(16,9))
hr['Position'].value_counts().plot(kind='bar')


# In[ ]:


hr['PayRate'].unique()


# In[ ]:


hr['PayRate'].describe()


# In[ ]:


hr['Sex'].replace('Nu',1,inplace=True)
hr['Sex'].replace('Nam',0,inplace=True)


# In[ ]:


hr['Sex'].unique()
#replace the 'Sex'back to number for scatter plotting


# In[ ]:


hr.plot(x='Sex',y='PayRate',kind='scatter')
#I have no idea what this mean ? are 'Sex' & 'PayRate' related ?


# In[ ]:


hr['ManagerName'].unique()


# In[ ]:


#which manager perform best ?
sns.countplot(y=hr['PerformanceScore'],hue=hr['Sex'])
#looks like women do better than men 


# In[ ]:


plt.figure(figsize=(40,20))
sns.countplot(y=hr['PerformanceScore'],hue=hr['ManagerName'])
#Can you see anything ? I don't !


# In[ ]:


plt.figure(figsize=(20,80))
sns.countplot(y=hr['ManagerName'],hue=hr['PerformanceScore'])
#And the winner goes to David Stanley 


# In[ ]:


Dept_sum=hr.groupby('Department')['PayRate'].sum()
Dept_sum.plot(kind='bar')
#Production pays alot ! but for which position in Production ?


# In[ ]:


pos_sum=hr.groupby(['Position','Department'])['PayRate'].sum()
plt.figure(figsize=(20,5))
pos_sum.plot(kind='bar')
#They must have a lot of "Production technician I" position.


# In[ ]:


#hr['Department'].unique()
#hr['Position'].unique()
hr_IT_support=hr.loc[(hr['Department']=='IT/IS') & (hr['Position']=='IT Support')]
hr_IT_support
sns.countplot(y=hr_IT_support['Position'],hue=hr_IT_support['Sex'])
#Looks like most of them are women.


# In[ ]:


hr_IT=hr.loc[(hr['Department']=='IT/IS')] 
sns.countplot(y=hr_IT['Position'],hue=hr_IT['Sex'])
#Yeah, beside IT Support, women also dominent @ database admin & Data Analyst


# In[ ]:


hr_IT_sal=hr_IT.loc[(hr_IT['PayRate'].head(20))]
sns.countplot(y=hr_IT_sal['Position'])


# In[ ]:


hr_IT_sal1=hr_IT.groupby('Position')['PayRate'].sum()
hr_IT_sal1
#They spend a lot money on IT Support. But CIO earns most.

