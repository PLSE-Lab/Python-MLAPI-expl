#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
filepath='../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv'
data=pd.read_csv(filepath, index_col='sl_no')
data.shape


# In[ ]:


data.columns


# In[ ]:


data.head()


# In[ ]:


data.isnull().any()


# In[ ]:


data.isnull().sum()


# In[ ]:


mean=data.salary.mean
data.salary.fillna(mean, inplace=True)
data.isnull().any()


# In[ ]:


plt.figure(figsize=(10,6))
plt.bar(data.status.unique(), data.status.value_counts());
plt.title('Placement Status')
plt.xlabel('status');
plt.ylabel('Number of students')


# In[ ]:


data.status.value_counts()


# Among of 215 students,Placed (148), Not Placed (67)

# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.bar(data.gender.unique(), data.gender.value_counts());
plt.title('Gender Status');
plt.xlabel('Gender');
plt.ylabel('Number of students');


# In[ ]:


data.gender.value_counts()


# Among of 215 students,Mail (139), Femal(76)

# In[ ]:



plt.figure(figsize=(10,6))
sns.countplot(x = "gender", hue = "status", data = data)


# The number of male students are more than as compared to female.

# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x = "workex", hue = "status", data = data)


# The number of male students with  experience are almost fold  as compared to female. 

# In[ ]:


plt.figure(figsize=(10,6))
plt.bar(data.specialisation.unique(), data.specialisation.value_counts() );
plt.title('specialisation Status');
plt.xlabel('specialisation');
plt.ylabel('Number of students');


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x = "specialisation", hue = "status", data = data)


# In[ ]:


plt.figure(figsize=(10,6))

sns.kdeplot(data=data['hsc_p'], shade=True)
sns.kdeplot(data=data['ssc_p'], shade=True)
sns.kdeplot(data=data['mba_p'], shade=True)
sns.kdeplot(data=data['etest_p'], shade=True)


# Distribution of Data of students according their (ssc_p,hcs_,p,mba_p,etest_p)

# In[ ]:


sns.pairplot(data,  hue="status")


# Initial criteria of Placed/Not Placed:
# 
# Placed/Not Placed dependes primarly on (ssc_p,hsc_p,degree_p), not on (mba_p,etest_p)

# In[ ]:


plt.figure(figsize=(10,6))
data.status.replace('Placed', 1, inplace=True )
data.status.replace('Not Placed', 0, inplace=True)

sns.heatmap(data=data.corr(),annot=True)


# ssc_p, hsc_p, degree_p have a moderate correlation with status.

# In[ ]:



data.status.replace(1, 'Placed', inplace=True )
data.status.replace(0,'Not Placed', inplace=True)

sns.swarmplot(x=data['status'], y=data['etest_p'])


# The number of students were Placed while etest was low

# In[ ]:


sns.swarmplot(x=data['status'], y=data['ssc_p'])


# most Plased student have ssc_p Close to 60

# In[ ]:


sns.lmplot(x="ssc_p", y="etest_p", hue="status", data=data)
sns.lmplot(x="hsc_p", y="etest_p", hue="status", data=data)
sns.lmplot(x="degree_p", y="etest_p", hue="status", data=data)

