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


import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
print("Setup Complete")


# ## Import dataset

# In[ ]:


my_filepath = ('../input/mlcourse/adult_train.csv')
df=pd.read_csv(my_filepath)
df


# ## Check shape of dataset

# In[ ]:


df.shape


# > There are 32561 instances and 15 attributes in the data set.

# ## Preview dataset
# > Summary statistics of dataset

# In[ ]:


df.describe()


# ## Check for missing values

# In[ ]:


df.isnull().sum()


# In[ ]:


df['Workclass'].value_counts()


# In[ ]:


for col in df.columns:
    print(col, len(df[col].unique()))


# In[ ]:


for col in df.columns:
    print(col, df[col].value_counts())
    print('....................')


# ## Handle missing values

# In[ ]:


for col in ['Workclass','Occupation', 'Country']:
    df.fillna(df[col].value_counts().index[0], inplace=True)     #df.fillna(df[col].mode()[0], inplace=True)


# In[ ]:


df.isna().sum()


# In[ ]:


df.head()


# ## Data Visualization

# In[ ]:


df.Target.value_counts()


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x= 'Target',hue='Sex',data = df)


# > male who earned <=50000 is more than female and in total who earned >50000 is less than who earned <=50000

# In[ ]:





# In[ ]:


sns.pairplot(data=df)


# In[ ]:


#plt.figure(figsize = (10,6))
good_job = df.sort_values(by='Hours_per_week', ascending=False)[:10][['Age','Martial_Status','Hours_per_week']]
good_job


# In[ ]:


df.Sex.value_counts()


# In[ ]:


sns.set(style='dark')
plt.figure(figsize=(20,16))
sns.barplot(x=df['Martial_Status'], y=df['Age'])
plt.title('Relationship between Age and Martial Stauts')

